from flask import Flask, request
from pymycobot import MyCobot280, PI_PORT, PI_BAUD
from camera_detect import camera_detect
import numpy as np
import time, threading

# === 配置参数 ===
PORT = 5000
TARGET_ID_MAP = {"A": 0, "B": 1, "C": 2}

GRIPPER_Z_OFFSET   = 100
APPROACH_BUFFER    = 25
Z_OFFSET           = 45
LIFT_AFTER_GRASP   = 100
RETURN_LIFT        = 40

# === 初始化 ===
app = Flask(__name__)
print("[INFO] 初始化机械臂与相机...")
mc = MyCobot280(PI_PORT, PI_BAUD)
offset_j5 = -90 if mc.get_system_version() > 2 else 0

OBS_POSE = [-90, 5, -70, -40, 90 + offset_j5, 50]

def goto_observe(speed=40):
    try:
        mc.send_angles(OBS_POSE, speed)
        time.sleep(2)
    except Exception as e:
        print("[WARN] 回观察位异常：", e)

mc.send_angles(OBS_POSE, 40)
time.sleep(2)

camera_params = np.load("camera_params.npz")
mtx, dist = camera_params["mtx"], camera_params["dist"]
detector = camera_detect(0, 25, mtx, dist)

# === 状态互斥 ===
IS_BUSY = False
BUSY_LOCK = threading.Lock()

# === 夹爪控制 ===
def open_gripper():
    mc.set_gripper_state(0, 80)
    time.sleep(1.5)

def close_gripper():
    mc.set_gripper_state(1, 80)
    time.sleep(1.5)

# === 抓取逻辑 ===
def grasp_from_target_code(target_code: str):
    target_id = TARGET_ID_MAP.get(target_code.upper(), None)
    if target_id is None:
        print(f"[ERROR] 无效目标编号：{target_code}")
        return False

    print(f"[INFO] 准备抓取：{target_code} → STAG ID: {target_id}")
    marker_pos_pack, ids = detector.stag_identify()
    if ids is None or target_id not in ids.flatten():
        print("[WARN] 当前视野中未识别到指定目标")
        return False

    end_coords = mc.get_coords()
    while end_coords is None:
        end_coords = mc.get_coords()

    T_be = detector.Transformation_matrix(end_coords)
    position_cam  = np.append(marker_pos_pack[:3], 1.0)
    position_base = T_be @ detector.EyesInHand_matrix @ position_cam
    xyz_base = position_base[:3].flatten()
    rx, ry, rz = end_coords[3:6]

    grasp_coords = [xyz_base[0], xyz_base[1], xyz_base[2] + GRIPPER_Z_OFFSET, rx, ry, rz]
    detector.coord_limit(grasp_coords)

    above = grasp_coords.copy()
    above[2] += Z_OFFSET
    detector.coord_limit(above)

    approach = grasp_coords.copy()
    approach[2] += APPROACH_BUFFER
    detector.coord_limit(approach)

    try:
        open_gripper()
        mc.send_coords(above, 30);    time.sleep(1.6)
        mc.send_coords(approach, 20); time.sleep(1.6)
        mc.send_coords(grasp_coords, 12); time.sleep(1.6)
        close_gripper()

        lift = grasp_coords.copy()
        lift[2] += LIFT_AFTER_GRASP
        detector.coord_limit(lift)
        mc.send_coords(lift, 30);     time.sleep(1.6)

        if RETURN_LIFT > 0:
            lift_more = lift.copy()
            lift_more[2] += RETURN_LIFT
            detector.coord_limit(lift_more)
            mc.send_coords(lift_more, 30); time.sleep(1.2)

        open_gripper(); time.sleep(0.8)
        goto_observe(speed=40)
        print("[SUCCESS] 抓取并回位完成")
        return True

    except Exception as e:
        print(f"[ERROR] 抓取流程异常：{e}")
        try:
            goto_observe(speed=30)
        except:
            pass
        return False

# === HTTP 路由 ===
@app.route("/target", methods=["POST"])
def handle_target():
    global IS_BUSY
    target = (request.form.get("target") or "").strip().upper()
    print(f"📥 接收到目标编号：{target}")

    if IS_BUSY:
        print("[WARN] 忙碌中，拒绝请求")
        return "BUSY", 409

    with BUSY_LOCK:
        IS_BUSY = True
        try:
            success = grasp_from_target_code(target)
            if success:
                return "OK", 200
            else:
                return "FAIL", 500
        except Exception as e:
            print(f"[ERROR] 处理异常：{e}")
            return "ERROR", 500
        finally:
            IS_BUSY = False

# === 启动服务 ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
