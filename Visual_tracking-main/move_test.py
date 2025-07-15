import time
import numpy as np
from pymycobot import MyCobot280
from camera_detect import camera_detect

if __name__ == "__main__":
    try:
        # === 初始化 ===
        print("[INFO] 初始化机械臂...")
        mc = MyCobot280("/dev/ttyAMA0", 1000000)  # 确保串口正确
        offset_j5 = -90 if mc.get_system_version() > 2 else 0

        # === 加载相机标定信息 ===
        print("[INFO] 加载相机参数和标定矩阵...")
        camera_params = np.load("camera_params.npz")
        mtx, dist = camera_params["mtx"], camera_params["dist"]
        cd = camera_detect(camera_id=0, marker_size=32, mtx=mtx, dist=dist)

        # === 移动到推荐观察位姿 ===
        observe_pose = [-90, 5, -104, 14, 90 + offset_j5, 0]
        print("[INFO] 移动至观察位...")
        mc.send_angles(observe_pose, 30)
        time.sleep(3)

        # === 用户循环输入 ===
        while True:
            user_input = input("\n请输入目标 Stag ID（如0或1），输入-1退出: ")
            try:
                target_id = int(user_input)
            except ValueError:
                print("[ERROR] 输入无效，请输入数字。")
                continue

            if target_id == -1:
                print("[INFO] 退出程序。")
                break

            print("[INFO] 开始识别...")
            try:
                marker_pos, ids = cd.stag_identify()

                if ids is None or len(ids) == 0:
                    print("[WARN] 未检测到任何Stag码，请调整相机视角。")
                    continue

                ids = ids.flatten()
                if target_id not in ids:
                    print(f"[WARN] 未找到目标ID={target_id}，检测到的ID有: {ids}")
                    continue

                print(f"[INFO] 找到目标ID={target_id}，开始计算并移动...")

                # === 获取目标物位置（机器人基坐标系） ===
                coords, detected_ids = cd.stag_robot_identify(mc)
                cd.coord_limit(coords)
                print("[INFO] 计算得到目标位置:", coords)

                # === 替换目标姿态为测试可行的安全姿态 ===
                safe_rpy = [-58, -2, -14 + offset_j5]
                coords[3:] = safe_rpy

                # === 打印位姿差值用于调试 ===
                current = mc.get_coords()
                if current is None:
                    print("[ERROR] 无法获取当前机械臂坐标。")
                    continue

                delta = [round(coords[i] - current[i], 2) for i in range(6)]
                print("[DEBUG] 当前坐标:", current)
                print("[DEBUG] 差值:", delta)

                print("[INFO] 正在尝试移动...")
                mc.send_coords(coords, 30)
                time.sleep(1)

                # === 检查是否成功移动 ===
                if mc.is_moving():
                    print("[SUCCESS] 移动指令已触发，正在运动中...")
                else:
                    print("[WARNING] 未检测到移动，可能控制指令未触发或目标位置未变化。")

            except Exception as e:
                print(f"[EXCEPTION] 出现错误: {e}")

    except KeyboardInterrupt:
        print("\n[INFO] 用户中断，程序退出。")
