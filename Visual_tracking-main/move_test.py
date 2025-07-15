import time
import numpy as np
from pymycobot import MyCobot280
from camera_detect import camera_detect

if __name__ == "__main__":
    try:
        # === 初始化机械臂 ===
        print("[INFO] 初始化机械臂...")
        mc = MyCobot280("/dev/ttyAMA0", 1000000)  # 修改为实际串口
        offset_j5 = -90 if mc.get_system_version() > 2 else 0

        # === 加载相机参数和手眼标定矩阵 ===
        print("[INFO] 加载相机参数和标定矩阵...")
        camera_params = np.load("camera_params.npz")
        mtx, dist = camera_params["mtx"], camera_params["dist"]
        cd = camera_detect(camera_id=0, marker_size=32, mtx=mtx, dist=dist)

        # === 移动至观察位姿 ===
        observe_pose = [-90, 5, -104, 14, 90 + offset_j5, 0]
        print("[INFO] 移动至观察位...")
        mc.send_angles(observe_pose, 30)
        time.sleep(3)

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

                # 获取目标在基坐标系下的坐标
                coords, detected_ids = cd.stag_robot_identify(mc)
                cd.coord_limit(coords)

                # 使用当前末端角度
                current = mc.get_coords()
                if current is None:
                    print("[ERROR] 无法获取当前机械臂坐标。")
                    continue
                for i in range(3, 6):
                    coords[i] = current[i]

                print("[INFO] 计算得到目标位置:", coords)
                print("[INFO] 正在尝试移动...")

                mc.send_coords(coords, 30)
                time.sleep(1)

                # 检查是否触发运动
                if mc.is_moving() == 1:
                    print("[SUCCESS] 机械臂已开始移动。")
                else:
                    print("[WARNING] 未检测到移动，可能控制指令未触发或目标位置未变化。")

            except Exception as e:
                print(f"[EXCEPTION] 出现错误: {e}")

    except KeyboardInterrupt:
        print("\n[INFO] 用户中断，程序退出。")
