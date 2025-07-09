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

gripper_offset_y = -55
gripper_offset_x = 15

class DetectArucoGrasp:
    def __init__(self):
        self.cache_x = self.cache_y = 0
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

        rospy.init_node("aruco_single_grasp_angles", anonymous=True)

    def coords_to_angles(self, x, y):
        """
        将坐标（x, y）近似映射为关节角度（仅用于简化演示）
        实际项目中应使用反解或查表
        """
        base_angle = max(min(x * 0.3, 90), -90)
        shoulder_angle = max(min(y * -0.2, 45), -45)

        return [base_angle, shoulder_angle, 15, 0, 0, 0]

    def move_to_target(self, x, y):
        print(f"➡️ 拟合执行抓取动作 @ ({x:.1f}, {y:.1f})")

        approach_angles = self.coords_to_angles(x, y)
        self.mc.send_angles(approach_angles, 30)
        time.sleep(2.0)

        print("🤖 闭合夹爪夹取目标")
        self.mc.set_gripper_state(1, 80)
        time.sleep(1.5)

        print("✅ 抓取完成")

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
