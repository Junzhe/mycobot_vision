import time
import numpy as np
from pymycobot.mycobot import MyCobot
from camera_detect import camera_detect

def move_to_observe_position(mc):
    print("[INFO] 移动至观察位...")
    version = mc.get_system_version()
    offset_j5 = -90 if version > 2 else 0
    observe_pose = [0, 0, 2, -58, -2, -14 + offset_j5]
    mc.send_angles(observe_pose, 30)
    time.sleep(3)

def is_far_enough(current, target, threshold=5.0):
    """比较坐标位置差异是否超过阈值（单位 mm）"""
    current = np.array(current[:3])
    target = np.array(target[:3])
    distance = np.linalg.norm(current - target)
    return distance > threshold

def main():
    print("[INFO] 初始化机械臂...")
    mc = MyCobot("/dev/ttyAMA0", 1000000)
    mc.power_on()
    time.sleep(1)

    print("[INFO] 加载相机参数和标定矩阵...")
    camera_params = np.load("camera_params.npz")
    mtx, dist = camera_params["mtx"], camera_params["dist"]
    detector = camera_detect(1, 32, mtx, dist)

    while True:
        try:
            target_id = int(input("\n请输入目标 Stag ID（如0或1），输入-1退出: "))
            if target_id == -1:
                break

            print("[INFO] 先移动至观察位...")
            move_to_observe_position(mc)

            print("[INFO] 开始识别...")
            for _ in range(30):
                marker_pack, ids = detector.stag_identify()
                if ids is not None and len(ids) > 0:
                    ids_list = ids.flatten().tolist()
                    if target_id in ids_list:
                        print(f"[INFO] 找到目标 ID={target_id}，计算并移动...")
                        target_coords, found_ids = detector.stag_robot_identify(mc)
                        if is_far_enough(mc.get_coords(), target_coords):
                            # 保持原角度，只移动位置坐标
                            current_coords = mc.get_coords()
                            for i in range(3, 6):
                                target_coords[i] = current_coords[i]
                            detector.coord_limit(target_coords)
                            mc.send_coords(target_coords, 30)
                        else:
                            print("[INFO] 当前已接近目标位置，跳过移动。")
                        break
                time.sleep(0.1)
            else:
                print("[WARN] 未检测到目标 ID。")
        except Exception as e:
            print(f"[EXCEPTION] 出现错误: {e}")
            continue

if __name__ == "__main__":
    main()
