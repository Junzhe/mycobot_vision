# encoding: UTF-8 
#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import time
import rospy
from pymycobot import MyCobot280, PI_PORT, PI_BAUD

# 相机与夹爪之间的偏移
gripper_offset_y = -55
gripper_offset_x = 15

# 可调映射比例（影响抓取动作方向）
ANGLE_SCALE_X = 0.25  # ArUco x 坐标 映射到 关节1（底座旋转）
ANGLE_SCALE_Y = 0.25  # ArUco y 坐标 映射到 关节2（手臂抬高）

# 初始姿态角度（适合俯视桌面）
BASE_ANGLES = [0, -30, 30, 0, 0, 0]

class DetectArucoGrasp:
    def __init__(self):
        self.mc = MyCobot280(PI_PORT, PI_BAUD)

        print("➡️ 打开夹爪准备抓取")
        self.mc.set_gripper_state(0, 80)
        time.sleep(1.0)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)

        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        self.camera_matrix = np.array([
            [781.33, 0., 347.53],
            [0., 783.79, 246.67],
            [0., 0., 1.]
        ])

        self.dist_coeffs = np.array(([[0.34, -2.52, -0.0012, 0.0067, 2.57]]))

        rospy.init_node("aruco_grasp_with_mapping", anonymous=True)

    def move_to_target_angles(self, x, y):
        print(f"📐 根据 tvec 坐标 x={x:.2f}, y={y:.2f} 计算目标角度")

        delta_j1 = x * ANGLE_SCALE_X
        delta_j2 = y * ANGLE_SCALE_Y

        target_angles = [
            BASE_ANGLES[0] + delta_j1,
            BASE_ANGLES[1] + delta_j2,
            BASE_ANGLES[2],
            BASE_ANGLES[3],
            BASE_ANGLES[4],
            BASE_ANGLES[5]
        ]

        print("🎯 移动到目标角度:", target_angles)
        self.mc.send_angles(target_angles, 25)
        time.sleep(2.5)

        print("🤖 闭合夹爪夹取目标")
        self.mc.set_gripper_state(1, 80)
        time.sleep(1.5)
        print("✅ 动作完成")

    def run(self):
        print("🚀 开始检测 ArUco 目标...")

        while cv2.waitKey(1) < 0:
            ret, img = self.cap.read()
            if not ret:
                print("❌ 无法获取摄像头图像")
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

            if ids is not None:
                print(f"✅ 检测到 ArUco ids: {ids.flatten()}")

                if 1 in ids:
                    index = list(ids.flatten()).index(1)
                    print(f"➡️ 选择目标 ID = 1, index = {index}")

                    ret = cv2.aruco.estimatePoseSingleMarkers(corners, 0.03, self.camera_matrix, self.dist_coeffs)
                    tvec = ret[1][index][0]

                    x = round(tvec[0] * 1000 + gripper_offset_y, 2)
                    y = round(tvec[1] * 1000 + gripper_offset_x, 2)

                    print(f"📍 ArUco ID 1 估计位置: X = {x}, Y = {y}")

                    self.move_to_target_angles(x, y)
                    break
                else:
                    print("⚠️ 检测到了 ArUco，但不是 ID=1")
            else:
                print("⚠️ 没有检测到任何 ArUco 标签")

            cv2.imshow("Aruco Detection", img)

if __name__ == '__main__':
    detect = DetectArucoGrasp()
    detect.run()
