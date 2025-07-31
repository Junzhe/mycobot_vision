# grasp_server.py

from flask import Flask, request
from pymycobot import MyCobot280, PI_PORT, PI_BAUD
from camera_detect import camera_detect
import numpy as np
import time

# === 配置 ===
PORT = 5000
TARGET_ID_MAP = {"A": 0, "B": 1, "C": 2}  # 编号 → STAG ID 映射
Z_OFFSET = 30  # 抬升高度
LIFT_AFTER_GRASP = 50  # 抓取后上抬高度

# === 初始化 ===
app = Flask(__name__)
print("[INFO] 初始化机械臂与相机...")
mc = MyCobot280(PI_PORT, PI_BAUD)  # 使用树莓派端口
offset_j5 = -90 if mc.get_system_version() > 2 else 0
mc.send_angles([-90, 5, -104, 14, 90 + offset_j5, 0], 90)
time.sleep(2)

camera_params = np.load("camera_params.npz")
mtx, dist = camera_params["mtx"], camera_params["dist"]
detector = camera_detect(0, 40, mtx, dist)

# === 夹爪控制函数 ===
def open_gripper():
    print("[ACTION] 打开夹爪...")
    mc.set_gripper_state(0, 80)
    time.sleep(1)

def close_gripper():
    print("[ACTION] 闭合夹爪...")
    mc.set_gripper_state(1, 80)
    time.sleep(1)

# === 主抓取函数 ===
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

    # === 获取当前末端姿态并变换目标位姿 ===
    end_coords = mc.get_coords()
    while end_coords is None:
        end_coords = mc.get_coords()

    T_be = detector.Transformation_matrix(end_coords)
    position_cam = np.append(marker_pos_pack[:3], 1)
    position_base = T_be @ detector.EyesInHand_matrix @ position_cam
    xyz_base = position_base[:3].flatten()
    rx, ry, rz = end_coords[3:6]
    coords = [xyz_base[0], xyz_base[1], xyz_base[2], rx, ry, rz]
    detector.coord_limit(coords)

    # === 抬升位姿 ===
    above = coords.copy()
    above[2] += Z_OFFSET
    detector.coord_limit(above)

    try:
        # === 执行抓取动作 ===
        print("[ACTION] 移动到目标上方...")
        mc.send_coords(above, 30)
        time.sleep(2)

        print("[ACTION] 下移贴近目标...")
        mc.send_coords(coords, 30)
        time.sleep(2)

        print("[ACTION] 闭合夹爪...")
        close_gripper()
        time.sleep(1)

        print("[ACTION] 上抬以验证抓取...")
        lift = coords.copy()
        lift[2] += LIFT_AFTER_GRASP
        detector.coord_limit(lift)
        mc.send_coords(lift, 30)
        time.sleep(2)

        # 可选：上抬后打开夹爪
        # print("[ACTION] 打开夹爪以释放目标...")
        # open_gripper()

        print("[SUCCESS] 抓取完成")
        return True

    except Exception as e:
        print(f"[ERROR] 抓取异常：{e}")
        return False

# === HTTP 路由 ===
@app.route("/target", methods=["POST"])
def handle_target():
    target = request.form.get("target")
    print(f"📥 接收到目标编号：{target}")
    success = grasp_from_target_code(target)
    return "OK" if success else "FAIL"

# === 启动 Flask 服务 ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
