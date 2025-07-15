import numpy as np
import cv2
import time
from camera_detect import camera_detect
from pymycobot import MyCobot280

# 初始化相机参数文件路径
PARAMS_FILE = "camera_params.npz"

# 初始化机械臂串口
MC_PORT = "/dev/ttyAMA0"
MC_BAUD = 1000000

# 初始化摄像头 ID 和 marker 尺寸（单位：mm）
CAMERA_ID = 0
MARKER_SIZE = 32

# 初始化观察位角度
OBSERVE_ANGLES = [0, 0, 2, -58, -2, -104]  # 注意根据版本添加 offset_j5


def move_to_observe_position(mc):
    print("[INFO] 移动至观察位...")
    mc.send_angles(OBSERVE_ANGLES, 30)
    time.sleep(3)

def main():
    print("[INFO] 初始化机械臂...")
    mc = MyCobot280(MC_PORT, MC_BAUD)
    time.sleep(1)
    mc.power_on()
    time.sleep(1)

    print("[INFO] 加载相机参数和标定矩阵...")
    try:
        params = np.load(PARAMS_FILE)
        mtx, dist = params["mtx"], params["dist"]
    except Exception as e:
        print("[ERROR] 相机参数加载失败:", e)
        return

    # 初始化视觉识别模块
    detector = camera_detect(CAMERA_ID, MARKER_SIZE, mtx, dist)

    while True:
        try:
            user_input = input("\n请输入目标 Stag ID（如0或1），输入-1退出: ")
            if user_input.strip() == "-1":
                break
            try:
                target_id = int(user_input)
            except ValueError:
                print("[WARN] 输入格式错误，请重新输入.")
                continue

            print("[INFO] 先移动至观察位...")
            move_to_observe_position(mc)

            print("[INFO] 开始识别...")
            marker_pos_pack, ids = detector.stag_identify()

            if ids is None or target_id not in ids:
                print(f"[WARN] 当前画面未检测到目标 ID={target_id}.")
                continue

            print(f"[INFO] 找到目标 ID={target_id}，计算并移动...")
            target_coords, found_ids = detector.stag_robot_identify(mc)
            print("[DEBUG] 目标位姿:", target_coords)

            # 获取当前机械臂位姿判断是否需要移动
            current_coords = mc.get_coords()
            if current_coords is None:
                print("[WARN] 无法获取当前机械臂坐标.")
                continue

            delta = np.linalg.norm(np.array(target_coords[:3]) - np.array(current_coords[:3]))
            if delta < 5.0:
                print("[INFO] 当前已接近目标，无需移动.")
            else:
                mc.send_coords(target_coords, 30)
                print("[INFO] 正在移动至目标位置...")

        except Exception as e:
            print("[EXCEPTION] 出现错误:", e)
            continue

if __name__ == "__main__":
    main()
