import cv2
import time
from pymycobot.mycobot import MyCobot
from camera_detect import camera_detect

if __name__ == "__main__":
    print("[INFO] 初始化机械臂...")
    mc = MyCobot("/dev/ttyAMA0", 1000000)
    mc.power_on()
    time.sleep(1)

    print("[INFO] 加载相机参数和标定矩阵...")
    camera_params = cv2.FileStorage("camera_params.xml", cv2.FILE_STORAGE_READ)
    mtx = camera_params.getNode("camera_matrix").mat()
    dist = camera_params.getNode("distortion_coefficients").mat()
    detector = camera_detect(1, 32, mtx, dist)

    print("[INFO] 移动至观察位...")
    observe_pose = [0, 0, 2, -58, -2, -14 - 90]
    mc.send_angles(observe_pose, 30)
    time.sleep(3)

    while True:
        try:
            user_input = input("\n请输入目标 Stag ID（如0或1），输入-1退出: ")
            target_id = int(user_input)
            if target_id == -1:
                break

            print("[INFO] 先移动至观察位...")
            mc.send_angles(observe_pose, 30)
            time.sleep(2)

            print("[INFO] 开始识别...")
            marker_pos_pack, ids = detector.stag_identify()
            print("[DEBUG] 检测到 ID=", ids)
            if ids is None or target_id not in ids:
                print("[WARN] 未检测到指定 ID! 当前检测: ", ids)
                continue

            print(f"[INFO] 找到目标 ID={target_id}，计算并移动...")
            target_coords_all, ids_all = detector.stag_robot_identify(mc)
            print("[DEBUG] marker_pos_pack:", marker_pos_pack)
            print("[DEBUG] 计算后坐标 target_coords:", target_coords_all)

            if ids_all is None or target_id >= len(ids_all):
                print("[ERROR] 转换后目标 ID 不在识别结果中！")
                continue

            # 获取当前位置
            current_coords = mc.get_coords()
            if current_coords is None:
                print("[ERROR] 无法获取当前坐标！")
                continue

            # 打印当前位置
            print("[DEBUG] 当前坐标 current_coords:", current_coords)

            # 计算 xyz 距离差
            delta = sum([(target_coords_all[i] - current_coords[i]) ** 2 for i in range(3)]) ** 0.5
            print(f"[DEBUG] 目标与当前 XYZ 距离差：{delta:.2f} mm")

            if delta < 5:
                print("[INFO] 当前坐标已接近目标坐标，无需移动。")
                continue

            print("[ACTION] 移动机械臂至目标坐标...")
            mc.send_coords(target_coords_all, 30, 1)

            time.sleep(3)
        except Exception as e:
            print("[EXCEPTION] 出现错误:", e)
            continue
