import time
import numpy as np
from pymycobot import MyCobot280
from camera_detect import camera_detect

def grasp_from_stag_id(target_id: int):
    print(f"[INFO] 手动指定 STAG ID = {target_id}")

    # 初始化机械臂
    mc = MyCobot280("/dev/ttyAMA0", 1000000)
    offset_j5 = -90 if mc.get_system_version() > 2 else 0
    mc.send_angles([-90, 5, -104, 14, 90 + offset_j5, 0], 60)
    time.sleep(2)

    # 初始化相机
    camera_params = np.load("camera_params.npz")
    mtx, dist = camera_params["mtx"], camera_params["dist"]
    cd = camera_detect(0, 40, mtx, dist)  # 使用 40mm STAG 标定尺寸

    # 识别目标
    print("[INFO] 相机识别中...")
    marker_pos_pack, ids = cd.stag_identify()
    if ids is None or target_id not in ids.flatten():
        print("[WARN] 目标 STAG ID 不在当前视野中")
        return False

    # === ✅ 打印相机坐标系下目标物位置
    print(f"[DEBUG] 相机识别到目标物坐标 (camera coords):\n{np.round(marker_pos_pack, 2)}")

    # 获取转换后的目标坐标
    coords, ids = cd.stag_robot_identify(mc)

    # === ✅ 强制替换末端姿态为固定姿态（防止姿态错误引发目标变换误差）
    coords[3:] = [-58, -2, -14 + offset_j5]

    # === ✅ 打印当前标定矩阵与转换后坐标
    print(f"[DEBUG] 当前使用的 EyesInHand_matrix:\n{np.round(cd.EyesInHand_matrix, 2)}")
    print(f"[DEBUG] 转换后的目标物坐标 (base coords):\n{np.round(coords[:3], 2)}")

    cd.coord_limit(coords)

    # 原始坐标日志
    raw_coords = coords.copy()
    print(f"[DEBUG] 原始识别坐标: {np.round(raw_coords, 2)}")

    print(f"[INFO] 最终抓取坐标: {np.round(coords, 2)}")

    try:
        above = coords.copy()
        above[2] += 30
        print(f"[DEBUG] 抬升预抓取坐标: {np.round(above, 2)}")

        # 移动到上方
        mc.send_coords(above, 30)
        time.sleep(2)
        print(f"[DEBUG] 当前坐标 after above: {np.round(mc.get_coords(), 2)}")

        # 下移抓取
        mc.send_coords(coords, 30)
        time.sleep(2)
        print(f"[DEBUG] 当前坐标 at grasp: {np.round(mc.get_coords(), 2)}")

        # 夹爪闭合
        mc.set_gripper_state(1, 50)
        time.sleep(1)

        # 回到上方
        mc.send_coords(above, 30)
        time.sleep(2)
        print(f"[DEBUG] 当前坐标 after lift: {np.round(mc.get_coords(), 2)}")

        print("[SUCCESS] 抓取完成")
        return True

    except Exception as e:
        print(f"[EXCEPTION] 抓取失败: {e}")
        return False

if __name__ == "__main__":
    try:
        tid = int(input("请输入目标 STAG ID（如0）: "))
        grasp_from_stag_id_
