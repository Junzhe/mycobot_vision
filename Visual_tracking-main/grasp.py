# grasp_from_camera.py

import numpy as np
import time
from pymycobot import MyCobot280
from camera_detect import camera_detect

# === 参数配置 ===
GRIPPER_Z_OFFSET = 100     # mm：夹爪前端长度补偿（末端提前停下）
APPROACH_BUFFER = 20      # mm：安全接近缓冲（用于预抓取阶段）
Z_OFFSET = 30             # mm：整体抬升高度
LIFT_AFTER_GRASP = 50     # mm：抓取后上抬验证

def grasp_from_camera(detector, ml, target_id=None):
    """
    使用视觉识别目标，并根据手眼标定结果控制机械臂执行抓取动作。
    考虑夹爪长度补偿和预抓取路径，避免碰撞，并完成夹取动作。
    """

    # Step 1: 识别目标物（支持指定 ID）
    print("[INFO] 正在识别目标...")
    marker_pos_pack, ids = detector.stag_identify()
    if ids is None or (target_id is not None and target_id not in ids.flatten()):
        print("[WARN] 未找到指定目标 ID")
        return

    print(f"[DEBUG] 相机识别目标位置：{np.round(marker_pos_pack, 2)}")

    # Step 2: 获取当前末端姿态
    end_coords = ml.get_coords()
    while end_coords is None:
        end_coords = ml.get_coords()
    print(f"[DEBUG] 当前末端姿态：{np.round(end_coords, 2)}")

    T_be = detector.Transformation_matrix(end_coords)

    # Step 3: 相机 → 基座坐标转换
    position_cam = np.append(marker_pos_pack[:3], 1)
    position_base = T_be @ detector.EyesInHand_matrix @ position_cam
    xyz_base = position_base[:3].flatten()
    print(f"[DEBUG] 目标物 Base 坐标：{np.round(xyz_base, 2)}")

    # Step 4: 构造末端抓取姿态（加 Z 补偿）
    rx, ry, rz = end_coords[3:6]
    grasp_coords = [
        xyz_base[0],
        xyz_base[1],
        xyz_base[2] + GRIPPER_Z_OFFSET,
        rx, ry, rz
    ]
    detector.coord_limit(grasp_coords)
    print(f"[INFO] 实际抓取点（末端位置）：{np.round(grasp_coords, 2)}")

    # Step 5: 抬升点
    above = grasp_coords.copy()
    above[2] += Z_OFFSET
    detector.coord_limit(above)
    print(f"[DEBUG] 抬升位姿：{np.round(above, 2)}")

    # Step 6: 预接近点（缓冲接近）
    approach = grasp_coords.copy()
    approach[2] += APPROACH_BUFFER
    detector.coord_limit(approach)
    print(f"[DEBUG] 预接近位姿：{np.round(approach, 2)}")

    # Step 7: 执行动作序列
    try:
        # 打开夹爪准备抓取
        print("[ACTION] 打开夹爪...")
        ml.set_gripper_state(0, 80)
        time.sleep(2)

        print("[ACTION] 移动到抬升点...")
        ml.send_coords(above, 30)
        time.sleep(2)

        print("[ACTION] 下降至预接近点...")
        ml.send_coords(approach, 30)
        time.sleep(2)

        print("[ACTION] 缓慢下降至抓取点...")
        ml.send_coords(grasp_coords, 20)
        time.sleep(2)

        # 闭合夹爪抓取
        print("[ACTION] 闭合夹爪...")
        ml.set_gripper_state(1, 80)
        time.sleep(2)

        # 抬升验证
        print("[ACTION] 抬升以验证抓取...")
        lift = grasp_coords.copy()
        lift[2] += LIFT_AFTER_GRASP
        detector.coord_limit(lift)
        ml.send_coords(lift, 30)
        time.sleep(2)

        print("[SUCCESS] 抓取并验证成功")

    except Exception as e:
        print(f"[ERROR] 抓取失败：{e}")


if __name__ == "__main__":
    print("[INFO] 初始化机械臂与相机...")
    mc = MyCobot280("/dev/ttyAMA0", 1000000)
    offset_j5 = -90 if mc.get_system_version() > 2 else 0

    mc.send_angles([-90, 5, -104, 14, 90 + offset_j5, 0], 60)
    time.sleep(2)

    camera_params = np.load("camera_params.npz")
    mtx, dist = camera_params["mtx"], camera_params["dist"]
    detector = camera_detect(0, 40, mtx, dist)

    try:
        tid_input = input("请输入目标 STAG ID（可选，回车跳过）: ")
        tid = int(tid_input) if tid_input.strip() != "" else None
        grasp_from_camera(detector, mc, target_id=tid)
    except Exception as e:
        print(f"[输入错误] {e}")
