# grasp_server.py

from flask import Flask, request
from pymycobot import MyCobot280, PI_PORT, PI_BAUD
from camera_detect import camera_detect
import numpy as np
import time

# === 配置参数 ===
PORT = 5000
TARGET_ID_MAP = {"A": 0, "B": 1, "C": 2}  # 编号 → STAG ID 映射
GRIPPER_Z_OFFSET = 110     # mm：夹爪前端长度补偿（末端提前停下）
APPROACH_BUFFER = 50       # mm：安全接近缓冲（用于预抓取阶段）
Z_OFFSET = 30              # mm：整体抬升高度
LIFT_AFTER_GRASP = 50      # mm：抓取后上抬验证高度

# === 初始化 ===
app = Flask(__name__)
print("[INFO] 初始化机械臂与相机...")
mc = MyCobot280(PI_PORT, PI_BAUD)
offset_j5 = -90 if mc.get_system_version() > 2 else 0
mc.send_angles([-90, 5, -104, 14, 90 + offset_j5, 60], 90)
time.sleep(2)

camera_params = np.load("camera_params.npz")
mtx, dist = camera_params["mtx"], camera_params["dist"]
detector = camera_detect(0, 40, mtx, dist)

# === 夹爪控制函数 ===
def open_gripper():
    print("[ACTION] 打开夹爪...")
    mc.set_gripper_state(0, 60)
    time.sleep(2)

def close_gripper():
    print("[ACTION] 闭合夹爪...")
    mc.set_gripper_state(1, 60)
    time.sleep(2)

# === 抓取函数（从目标编号） ===
def grasp_from_target_code(target_code: str):
    target_id = TARGET_ID_MAP.get(target_code.upper(), None)
    if target_id is None:
        print(f"[ERROR] 无效目标编号：{target_code}")
        return False

    print(f"[INFO] 准备抓取目标编号：{target_code} → STAG ID: {target_id}")
    marker_pos_pack, ids = detector.stag_identify()
    if ids is None or target_id not in ids.flatten():
        print("[WARN] 当前视野中未识别到指定目标")
        return False

    # === 获取当前末端姿态并转换目标位置 ===
    end_coords = mc.get_coords()
    while end_coords is None:
        end_coords = mc.get_coords()

    T_be = detector.Transformation_matrix(end_coords)
    position_cam = np.append(marker_pos_pack[:3], 1)
    position_base = T_be @ detector.EyesInHand_matrix @ position_cam
    xyz_base = position_base[:3].flatten()
    rx, ry, rz = end_coords[3:6]

    # === 构造抓取末端位姿（含夹爪Z补偿） ===
    grasp_coords = [
        xyz_base[0],
        xyz_base[1],
        xyz_base[2] + GRIPPER_Z_OFFSET,
        rx, ry, rz
    ]
    detector.coord_limit(grasp_coords)

    above = grasp_coords.copy()
    above[2] += Z_OFFSET
    detector.coord_limit(above)

    approach = grasp_coords.copy()
    approach[2] += APPROACH_BUFFER
    detector.coord_limit(approach)

    try:
        # === 动作序列 ===
        open_gripper()

        print("[ACTION] 移动到抬升点...")
        mc.send_coords(above, 30)
        time.sleep(2)

        print("[ACTION] 下降至预接近点...")
        mc.send_coords(approach, 20)
        time.sleep(2)

        print("[ACTION] 缓慢下降至抓取点...")
        mc.send_coords(grasp_coords, 10)
        time.sleep(2)

        close_gripper()

        print("[ACTION] 上抬以验证抓取...")
        lift = grasp_coords.copy()
        lift[2] += LIFT_AFTER_GRASP
        detector.coord_limit(lift)
        mc.send_coords(lift, 30)
        time.sleep(2)

        print("[SUCCESS] 抓取完成")
        return True

    except Exception as e:
        print(f"[ERROR] 抓取异常：{e}")
        return False

# === HTTP 路由 ===
@app.route("/target", methods=["POST"])
def handle_target():
    target = request.form.get("target")
    print(f"\U0001F4E5 接收到目标编号：{target}")
    success = grasp_from_target_code(target)
    return "OK" if success else "FAIL"

# === 启动服务 ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
