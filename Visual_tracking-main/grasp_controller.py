import time
import numpy as np
from pymycobot import MyCobot
from camera_detect import camera_detect  # 使用你已有的 camera_detect.py 模块

# 初始化机械臂与相机参数
mc = MyCobot("/dev/ttyAMA0", 1000000)

# 加载标定好的相机内参和畸变参数
camera_params = np.load("camera_params.npz")
mtx, dist = camera_params["mtx"], camera_params["dist"]

# 初始化相机识别对象，marker_size 单位 mm
cd = camera_detect(0, 50, mtx, dist)

# 抓取任务主程序
def main():
    mc.set_fresh_mode(1)
    mc.set_vision_mode(1)
    time.sleep(1)

    # 机械臂移动至观测初始位姿
    origin_angles = [42.36, -35.85, -52.91, 88.59, 90, 0.0]
    mc.send_angles(origin_angles, 50)
    wait(mc)
    time.sleep(1)

    origin_pose = mc.get_coords()
    while origin_pose is None:
        origin_pose = mc.get_coords()

    print("[INFO] 等待识别 STAG ID = 0 ...")
    while True:
        _, ids = cd.stag_identify()
        if ids is not None and ids[0] == 0:
            print("[INFO] 识别到 STAG ID = 0，开始计算目标位姿")
            target_coords, _ = cd.stag_robot_identify(mc)
            if target_coords is None:
                continue

            # 限制抓取坐标范围
            cd.coord_limit(target_coords)

            # 将姿态角度替换为当前末端姿态，避免识别抖动
            target_coords[3:] = origin_pose[3:]
            print(f"[INFO] 目标坐标: {target_coords}")

            # 移动到目标物前
            mc.send_coords(target_coords, 30)
            wait(mc)

            # 模拟夹取
            print("[INFO] 模拟执行夹爪动作（请手动控制或替换为抓取控制）")
            time.sleep(1)
            break

    print("[INFO] 抓取任务完成")

# 等待机械臂移动完成
def wait(mc):
    time.sleep(0.5)
    while mc.is_moving() == 1:
        time.sleep(0.2)

if __name__ == '__main__':
    main()
