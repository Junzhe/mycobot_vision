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

# 摄像头与夹爪的偏移量（可微调）
gripper_offset_y = -55
gripper_offset_x = 15

# 初始俯视视角的关节角（根据你调试确定）
DEFAULT_VIEW_ANGLES = [0, 0, 2, -58, -2, -14]

class DetectArucoGrasp:
    def __init__(self):
        self.target_id = 1  # 当前我们默认抓取 ID=1
        self.mc = MyCobot280(PI_PORT, PI_BAUD)

        print("打开夹爪准备抓取")
        self.mc.set_gripper_state(0, 80)  # 打开
        time.sleep(1.0)

        print("移动至初始俯视角度...")
        self.mc.send_angles(DEFAULT_VIEW_ANGLES, 30)
        time.sleep(3.5)

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

        rospy.init_node("aruco_grasp_live", anonymous=True)

    def move_to_target(self, x, y):
        print(f"执行角度控制抓取 @ x={x:.2f}, y={y:.2f}")

        # 构造一个角度映射策略：只映射到 joint1, joint2
        delta_j1 = x * 0.2
        delta_j2 = y * 0.2

        target_angles = [
            DEFAULT_VIEW_ANGLES[0] + delta_j1,
            DEFAULT_VIEW_ANGLES[1] + delta_j2,
            DEFAULT_VIEW_ANGLES[2],
            DEFAULT_VIEW_ANGLES[3],
            DEFAULT_VIEW_ANGLES[4],
            DEFAULT_VIEW_ANGLES[5]
        ]

        print(f"[调试] 映射后角度：{target_angles}")
        self.mc.send_angles(target_angles, 30)
        time.sleep(2.5)

        print("闭合夹爪夹取目标")
        self.mc.set_gripper_state(1, 80)
        time.sleep(1.5)
        print("抓取动作完成")

    def run(self):
        print("启动摄像头检测 ArUco 标签...")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("无法获取图像")
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

            if ids is not None:
                ids = ids.flatten()
                print(f"检测到 ArUco ids: {ids}")

                for i, id in enumerate(ids):
                    color = (0, 255, 0)
                    if id == self.target_id:
                        color = (0, 0, 255)  # 红色高亮

                        ret = cv2.aruco.estimatePoseSingleMarkers(corners, 0.03, self.camera_matrix, self.dist_coeffs)
                        tvec = ret[1][i][0]

                        x = round(tvec[0] * 1000 + gripper_offset_y, 2)
                        y = round(tvec[1] * 1000 + gripper_offset_x, 2)

                        print(f"ArUco ID {id} 位姿坐标 x = {x}, y = {y}")
                        self.move_to_target(x, y)

                    # 在图像中画框并标注
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    c = corners[i][0].astype(int)
                    cv2.polylines(frame, [c], True, color, 2)
                    cv2.putText(frame, f"ID:{id}", tuple(c[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                print("没有检测到任何 ArUco 标签")

            cv2.imshow("Aruco Detection Live", frame)
            key = cv2.waitKey(10)
            if key == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    detect = DetectArucoGrasp()
    detect.run()
