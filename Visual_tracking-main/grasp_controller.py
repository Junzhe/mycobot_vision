import time
import numpy as np
from pymycobot import MyCobot280
from camera_detect import camera_detect
from target_map import TARGET_ID_MAP

def grasp_from_target_code(code: str):
    if code not in TARGET_ID_MAP:
        print(f"[ERROR] 未知编号 {code}")
        return False

    target_id = TARGET_ID_MAP[code]
    print(f"[INFO] 编号 {code} → STAG ID = {target_id}")

    # 初始化机械臂
    mc = MyCobot280("/dev/ttyAMA0", 1000000)
    offset_j5 = -90 if mc.get_system_version() > 2 else 0
    mc.send_angles([-90, 5, -104, 14, 90 + offset_j5, 0], 30)
    time.sleep(2)

    # 初始化相机
    camera_params = np.load("camera_params.npz")
    mtx, dist = camera_params["mtx"], camera_params["dist"]
    cd = camera_detect(0, 40, mtx, dist)  # 你已使用 40mm 标定

    # 识别目标
    print("[INFO] 相机识别中...")
    marker_pos_pack, ids = cd.stag_identify()
    if ids is None or target_id not in ids.flatten():
        print("[WARN] 目标 STAG ID 不在当前视野中")
        return False

    coords, _ = cd.stag_robot_identify(mc)
    cd.coord_limit(coords)

    # 设置抓取姿态
    coords[3:] = [-58, -2, -14 + offset_j5]
    print(f"[INFO] 基坐标系下的目标坐标: {np.round(coords, 2)}")

    # 执行抓取动作
    try:
        above = coords.copy()
        above[2] += 30
        mc.send_coords(above, 30)
        time.sleep(2)

        mc.send_coords(coords, 30)
        time.sleep(2)

        mc.set_gripper_state(1, 50)
        time.sleep(1)

        mc.send_coords(above, 30)
        time.sleep(2)

        print("[SUCCESS] 抓取完成")
        return True
    except Exception as e:
        print(f"[EXCEPTION] 抓取失败: {e}")
        return False
