# encoding: UTF-8
#!/usr/bin/env python3
import sys
import os
import time
import cv2
import numpy as np
import rospy
from pymycobot import MyCobot280, PI_PORT, PI_BAUD
from visualization_msgs.msg import Marker

# ----------------------------- 参数区域 -----------------------------
# 摄像头与夹爪的偏移（需微调）
gripper_offset_y = -55
gripper_offset_x = 15

# 相机内参矩阵（相机标定得到）
camera_matrix = np.array([
    [781.33, 0., 347.53],
    [0., 783.79, 246.67],
    [0., 0., 1.]
])
# 相机畸变参数
dist_coeffs = np.array(([[0.34, -2.52, -0.0012, 0.0067, 2.57]]))

# 初始俯视桌面的关节角度
BASE_ANGLES = [0, 0, 2, -58, -2, -14]  # 角度单位：度

# 角度映射系数（用于将 x/y 映射到关节增量）
ANGLE_MAP_COEFF = {
    "joint_1": 0.2,  # 控制基座旋转（左右）
    "joint_2": 0.2   # 控制前臂俯仰（上下）
}

# ----------------------------- 主体类 -----------------------------
class DetectArucoGrasp:
    def __init__(self):
        self.cache_x = self.cache_y = 0

        # 初始化 MyCobot
        self.mc = MyCobot280(PI_PORT, PI_BAUD)

        # 初始化夹爪（AG）
        print("\n➡️ 打开夹爪准备抓取")
        self.mc.set_gripper_state(0, 80)  # 打开
        time.sleep(1.0)

        # 初始化 ROS 节点
        rospy.init_node("aruco_single_grasp", anonymous=True)

        # 执行默认俯视姿态
        print("\n🤖 移动至初始俯视角度...\n")
        self.mc.send_angles(BASE_ANGLES, 30)
        time.sleep(3)

        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)

        # ArUco 字典与参数
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

    def map_position_to_angles(self, x, y):
        """将估算的 x/y 坐标映射为角度增量"""
        delta_j1 = x * ANGLE_MAP_COEFF["joint_1"]
        delta_j2 = y * ANGLE_MAP_COEFF["joint_2"]

        target_angles = [
            BASE_ANGLES[0] + delta_j1,
            BASE_ANGLES[1] + delta_j2,
            BASE_ANGLES[2],
            BASE_ANGLES[3],
            BASE_ANGLES[4],
            BASE_ANGLES[5]
        ]
        return target_angles

    def move_to_target(self, x, y):
        print(f"\n➡️ 执行角度控制抓取 @ x={x}, y={y}")

        target_angles = self.map_position_to_angles(x, y)
        print("[调试] 映射后角度：", [round(a, 2) for a in target_angles])

        # 执行移动
        self.mc.send_angles(target_angles, 30)
        time.sleep(3)

        # 执行夹爪抓取
        print("🤖 闭合夹爪夹取目标")
        self.mc.set_gripper_state(1, 80)
        time.sleep(1.5)

        print("✅ 抓取动作完成\n")

    def run(self):
        print("\n🚀 开始检测 ArUco 目标...\n")

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

                    ret = cv2.aruco.estimatePoseSingleMarkers(corners, 0.03, camera_matrix, dist_coeffs)
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
