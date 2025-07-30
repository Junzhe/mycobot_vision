# grasp_from_camera.py

import numpy as np
import time
from pymycobot import MyCobot280
from camera_detect import camera_detect

def grasp_from_camera(detector, ml):
    """
    利用已标定的 EyesInHand_matrix，识别目标物在相机下的位置，
    计算其在基座坐标系下的位置，并控制机械臂抓取。
    """
    # === Step 1: 相机识别目标物 ===
    target_cam, ids = detector.stag_identify()  # 相机坐标系下目标物坐标 [x, y, z, rx, ry, rz]
    print("Detected (Camera coords):", target_cam)

    if detector.EyesInHand_matrix is None:
        print("Error: EyesInHand_matrix is not loaded or calibrated.")
        return

    # === Step 2: 获取当前末端姿态并计算 T_be ===
    end_coords = ml.get_coords()
    while end_coords is None:
        end_coords = ml.get_coords()

    print("Current end-effector coords:", end_coords)
    T_be = detector.Transformation_matrix(end_coords)

    # === Step 3: 相机坐标系下的目标物 → 齐次形式
    position_cam = np.append(target_cam[:3], 1)

    # === Step 4: 计算目标物在 Base 坐标系下的位置
    position_base = T_be @ detector.EyesInHand_matrix @ position_cam
    xyz_base = position_base[:3].flatten()
    print("Computed Base coords of target:", xyz_base)

    # === Step 5: 构造抓取位姿（保留当前姿态）
    rx, ry, rz = end_coords[3:6]
    target_pose = [xyz_base[0], xyz_base[1], xyz_base[2], rx, ry, rz]
    print("Planned grasp pose:", target_pose)

    # === Step 6: 使用逆解函数求解角度并发送控制命令 ===
    try:
        target_angles = ml._coords_to_angles(target_pose)
    except Exception as e:
        print("Inverse kinematics failed:", e)
        return

    print("Sending joint angles:", target_angles)
    ml.send_angles(target_angles, speed=40)
    print("Grasping motion executed.")


if __name__ == "__main__":
    # === 初始化 ===
    print("Connecting to MyCobot...")
    mc = MyCobot280("/dev/ttyAMA0", 1000000)
    camera_params = np.load("camera_params.npz")
    mtx, dist = camera_params["mtx"], camera_params["dist"]

    # 创建识别器对象（camera_detect 中已包含 EyesInHand_matrix）
    detector = camera_detect(camera_id=0, marker_size=40, mtx=mtx, dist=dist)

    # 执行抓取任务
    grasp_from_camera(detector, mc)
