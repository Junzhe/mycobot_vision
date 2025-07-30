# grasp_from_camera.py

import numpy as np
import time
from pymycobot import MyCobot280
from camera_detect import camera_detect


def grasp_from_camera(detector, ml, target_id=None, z_offset=30):
    """
    使用视觉识别目标，并根据手眼标定结果控制机械臂执行抓取动作。
    使用当前末端姿态 [rx, ry, rz] 构造抓取点，避免引入固定角度。
    """

    # Step 1: 识别目标物（支持指定 ID）
    print("[INFO] 正在识别目标...")
    marker_pos_pack, ids = detector.stag_identify()
    if ids is None or (target_id is not None and target_id not in ids.flatten()):
        print("[WARN] 未找到指定目标 ID")
        return

    print(f"[DEBUG] 相机识别目标位置：{np.round(marker_pos_pack, 2)}")

    # Step 2: 获取当前末端位姿
    end_coords = ml.get_coords()
    while end_coords is None:
        end_coords = ml.get_coords()
    print(f"[DEBUG] 当前末端位姿：{np.round(end_coords, 2)}")

    T_be = detector.Transformation_matrix(end_coords)

    # Step 3: 相机 → 基座坐标系转换
    position_cam = np.append(marker_pos_pack[:3], 1)
    position_base = T_be @ detector.EyesInHand_matrix @ position_cam
    xyz_base = position_base[:3].flatten()
    print(f"[DEBUG] 计算后目标物位置（Base系）：{np.round(xyz_base, 2)}")

    # Step 4: 使用当前末端姿态构造目标位姿
    rx, ry, rz = end_coords[3:6]
    coords = [xyz_base[0], xyz_base[1], xyz_base[2], rx, ry, rz]
    detector.coord_limit(coords)
    print(f"[INFO] 最终抓取坐标: {np.round(coords, 2)}")

    # Step 5: 构造抬升位姿
    above = coords.copy()
    above[2] += z_offset
    detector.coord_limit(above)
    print(f"[DEBUG] 抬升预抓取坐标: {np.round(above, 2)}")

    # Step 6: 执行动作
    try:
        print("[ACTION] 移动到抓取上方...")
        ml.send_coords(above, 30)
        time.sleep(2)

        print("[ACTION] 移动至抓取位置...")
        ml.send_coords(coords, 30)
        time.sleep(2)

        print("[SUCCESS] 抓取流程完成")

    except Exception as e:
        print(f"[ERROR] 抓取失败：{e}")


if __name__ == "__main__":
    print("[INFO] 初始化机械臂与相机...")
    mc = MyCobot280("/dev/ttyAMA0", 1000000)
    offset_j5 = -90 if mc.get_system_version() > 2 else 0

    # 移动到初始观测位
    mc.send_angles([-90, 5, -104, 14, 90 + offset_j5, 0], 60)
    time.sleep(2)

    # 加载相机参数和初始化识别器
    camera_params = np.load("camera_params.npz")
    mtx, dist = camera_params["mtx"], camera_params["dist"]
    detector = camera_detect(0, 40, mtx, dist)

    # 执行抓取
    try:
        tid_input = input("请输入目标 STAG ID（可选，回车跳过）: ")
        tid = int(tid_input) if tid_input.strip() != "" else None
        grasp_from_camera(detector, mc, target_id=tid)
    except Exception as e:
        print(f"[输入错误] {e}")
