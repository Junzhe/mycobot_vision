import time
from pymycobot import MyCobot
from camera_detect import camera_detect
import numpy as np

# ============ 目标编号与 STAG ID 映射 ============
TARGET_ID_MAP = {
    "A": 0,
    "B": 1,
    "C": 2
}

# ============ 抓取主流程 ============
def grasp_from_target_code(target_code):
    if target_code not in TARGET_ID_MAP:
        print(f"[ERROR] 未知编号：{target_code}")
        return False

    target_id = TARGET_ID_MAP[target_code]
    print(f"[INFO] 编号 {target_code} → STAG ID = {target_id}")
    return run_grasp_pipeline(target_id)


# ============ 执行抓取任务 ============
def run_grasp_pipeline(stag_id):
    try:
        # === 初始化机械臂与相机 ===
        mc = MyCobot("/dev/ttyAMA0", 1000000)
        camera_params = np.load("camera_params.npz")
        mtx, dist = camera_params["mtx"], camera_params["dist"]
        cam = camera_detect(0, 50, mtx, dist)


        time.sleep(1)

        # === 移动到观察位 ===
        mc.send_angles(cam.origin_mycbot_horizontal, 40)
        cam.wait()
        time.sleep(1)

        # === 识别目标 STAG ID ===
        max_try = 30
        found = False
        for _ in range(max_try):
            pos_list, id_list = cam.stag_identify()
            if stag_id in id_list:
                print(f"[INFO] 找到目标 STAG ID = {stag_id}")
                found = True
                break
            else:
                print("[INFO] 未找到目标，继续识别...")

        if not found:
            print(f"[ERROR] 超过最大尝试次数，未找到 STAG ID = {stag_id}")
            return False

        # === 获取基坐标系下的目标位姿 ===
        target_coords, _ = cam.stag_robot_identify(mc)
        cam.coord_limit(target_coords)
        print(f"[INFO] 抓取目标坐标: {target_coords}")

        # === 执行抓取动作 ===
        mc.send_coords(target_coords, 30)
        cam.wait()

        print("[SUCCESS] 抓取任务完成")
        return True

    except Exception as e:
        print(f"[ERROR] 抓取过程中异常: {e}")
        return False
