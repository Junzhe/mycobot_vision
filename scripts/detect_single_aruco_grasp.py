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
from visualization_msgs.msg import Marker

# 摄像头与夹爪的偏移量（可微调）
gripper_offset_y = -55
gripper_offset_x = 15

class DetectArucoGrasp:
    def __init__(self):
        self.cache_x = self.cache_y = 0

        # 初始化 MyCobot（使用树莓派端口）
        self.mc = MyCobot280(PI_PORT, PI_BAUD)

        # 初始化夹爪（AG）
        print("➡️ 打开夹爪准备抓取")
        self.mc.set_gripper_state(0, 80)  # 打开
        time.sleep(1.0)

        # 初始化相机
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)

        # ArUco 字典和参数
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        # ✅ 相机内参矩阵（需根据标定值调整）
        self.camera_matrix = np.array([
            [781.33, 0., 347.53],
            [0., 783.79, 246.67],
            [0., 0., 1.]
        ])

        # 相机畸变参数（同样需标定后修改）
        self.dist_coeffs = np.array(([[0.34, -2.52, -0.0012, 0.0067, 2.57]]))

        rospy.init_node("aruco_single_grasp", anonymous=True)

    def move_to_target(self, x, y):
        print(f"➡️ 执行抓取动作 @ ({x:.1f}, {y:.1f})")
        approach = [x, y, 200, 178.99, -3.78, -62.9]  # 靠近上方
        grasp = [x, y, 65.5, 178.99, -3.78, -62.9]    # 抓取高度

        self.mc.send_coords(approach, 25, 0)
        time.sleep(2.5)

        self.mc.send_coords(grasp, 25, 0)
        time.sleep(2.5)

        print("🤖 闭合夹爪夹取目标")
        self.mc.set_gripper_state(1, 80)  # 闭合
        time.sleep(1.5)

        lift = [x, y, 180, 178.99, -3.78, -62.9]  # 提起
        self.mc.send_coords(lift, 25, 0)
        time.sleep(2.5)

        print("✅ 抓取动作完成")

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

                    print(f"🎯 ArUco ID 1 位姿坐标 X = {x}, Y = {y}")
                    self.move_to_target(x, y)
                    break
                else:
                    print("⚠️ 检测到了 ArUco，但不是 ID=1")
            else:
                print("⚠️ 没有检测到任何 ArUco 标签")

            cv2.imshow("Aruco Detection", img)

if __name__ == '__main__':
    detect = DetectArucoGrasp()
    detect.run()
