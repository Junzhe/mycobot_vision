from flask import Flask, request
from pymycobot import MyCobot280, PI_PORT, PI_BAUD
from camera_detect import camera_detect
import numpy as np
import time

# === 配置参数 ===
PORT = 5000
TARGET_ID_MAP = {"A": 0, "B": 1, "C": 2}  # 编号 → STAG ID 映射

GRIPPER_Z_OFFSET   = 100   # mm：夹爪前端长度补偿（末端提前停下）
APPROACH_BUFFER    = 25    # mm：安全接近缓冲（用于预抓取阶段）
Z_OFFSET           = 45    # mm：上方安全高度（物体正上方再高一些）
LIFT_AFTER_GRASP   = 100   # mm：抓取后上抬验证高度
RETURN_LIFT        = 40    # mm：回观察位前再抬一点，确保避障（可调为0~60）

# === 初始化 ===
app = Flask(__name__)
print("[INFO] 初始化机械臂与相机...")
mc = MyCobot280(PI_PORT, PI_BAUD)
offset_j5 = -90 if mc.get_system_version() > 2 else 0

# 观察位（与你初始化用的姿态一致；后面抓完回这里）
OBS_POSE = [-90, 5, -45, -40, 90, 50, 60]

def goto_observe(speed=40):
    """回观察位（安全返回）"""
    try:
        mc.send_angles(OBS_POSE, speed)
        time.sleep(2)
    except Exception as e:
        print("[WARN] 回观察位异常：", e)

# 启动时先去观察位
mc.send_angles(OBS_POSE, 40)
time.sleep(2)

camera_params = np.load("camera_params.npz")
mtx, dist = camera_params["mtx"], camera_params["dist"]
# 注意：这里 marker_size=25（毫米）按你给的最新版；如你的 STAG 方块是 40mm，请改回 40
detector = camera_detect(0, 25, mtx, dist)

# === 夹爪控制函数 ===
def open_gripper():
    print("[ACTION] 打开夹爪...")
    mc.set_gripper_state(0, 80)
    time.sleep(1.5)

def close_gripper():
    print("[ACTION] 闭合夹爪...")
    mc.set_gripper_state(1, 80)
    time.sleep(1.5)

# === 抓取函数（从目标编号） ===
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

    # === 获取当前末端姿态并转换目标位置 ===
    end_coords = mc.get_coords()
    while end_coords is None:
        end_coords = mc.get_coords()

    T_be = detector.Transformation_matrix(end_coords)
    position_cam  = np.append(marker_pos_pack[:3], 1.0)                  # 相机系齐次
    position_base = T_be @ detector.EyesInHand_matrix @ position_cam     # 到基座系
    xyz_base = position_base[:3].flatten()
    rx, ry, rz = end_coords[3:6]  # 保持当前末端姿态（抓取方向沿当前工具姿态）

    # === 构造抓取相关航点 ===
    grasp_coords = [
        xyz_base[0],
        xyz_base[1],
        xyz_base[2] + GRIPPER_Z_OFFSET,  # 物体表面 + 夹爪长度补偿
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

        print("[ACTION] 移动到上方安全位 ...")
        mc.send_coords(above, 30)   # 较快
        time.sleep(1.6)

        print("[ACTION] 下降至预接近点 ...")
        mc.send_coords(approach, 20)  # 稍慢
        time.sleep(1.6)

        print("[ACTION] 缓慢下降至抓取点 ...")
        mc.send_coords(grasp_coords, 12)  # 更慢，避免冲击
        time.sleep(1.6)

        close_gripper()

        print("[ACTION] 上抬验证抓取 ...")
        lift = grasp_coords.copy()
        lift[2] += LIFT_AFTER_GRASP
        detector.coord_limit(lift)
        mc.send_coords(lift, 30)
        time.sleep(1.6)

        # === 新增：释放并返回 ===
        # 可选：回程前再抬一点，进一步避障
        if RETURN_LIFT > 0:
            lift_more = lift.copy()
            lift_more[2] += RETURN_LIFT
            detector.coord_limit(lift_more)
            print("[ACTION] 进一步抬高以避障 ...")
            mc.send_coords(lift_more, 30)
            time.sleep(1.2)

        print("[ACTION] 在高位释放物体 ...")
        open_gripper()
        time.sleep(0.8)

        print("[ACTION] 返回观察位 ...")
        goto_observe(speed=40)

        print("[SUCCESS] 抓取并回位完成")
        return True

    except Exception as e:
        print(f"[ERROR] 抓取流程异常：{e}")
        # 出错也尽量回观察位，避免停在危险姿态
        try: goto_observe(speed=30)
        except: pass
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
