# encoding: UTF-8
#!/usr/bin/env python3

import cv2
import numpy as np
import time
import json
import rospy
from pymycobot import MyCobot280, PI_PORT, PI_BAUD

# ------------------ 参数 ------------------
TARGET_ID = 1
GRIPPER_OPEN = 0
GRIPPER_CLOSE = 1

# 加载标定矩阵
def load_hand_eye_matrix(path='EyesInHand_matrix.json'):
    with open(path, 'r') as f:
        matrix_data = json.load(f)
        return np.array(matrix_data['matrix']).reshape((4, 4))

# tvec → robot 坐标
def convert_to_robot_coords(tvec, matrix):
    cam_coords = np.array([tvec[0], tvec[1], tvec[2], 1]).reshape((4, 1))
    robot_coords = matrix @ cam_coords
    return robot_coords[:3].flatten()

# ------------------ 主类 ------------------
class ArucoGrasp:
    def __init__(self):
        self.mc = MyCobot280(PI_PORT, PI_BAUD)
        self.matrix = load_hand_eye_matrix()

        print("移动至俯视位")
        default_view = [0, 0, 2, -58, -2, -14]
        self.mc.send_angles(default_view, 30)
        time.sleep(3)

        print("打开夹爪")
        self.mc.set_gripper_state(GRIPPER_OPEN, 80)

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

        rospy.init_node("aruco_grasp_with_matrix", anonymous=True)

    def move_to_target(self, robot_coords):
        print("[抓取] 移动至坐标: ", robot_coords)
        target = [
            round(robot_coords[0], 2),
            round(robot_coords[1], 2),
            round(robot_coords[2], 2),
            0, 0, 0  # 可根据需求改姿态
        ]
        self.mc.send_coords(target, 40, 1)
        time.sleep(2.5)

        print("闭合夹爪...")
        self.mc.set_gripper_state(GRIPPER_CLOSE, 80)
        time.sleep(1.5)
        print("✅ 抓取完成")

    def run(self):
        print("启动摄像头识别 ArUco...")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("图像读取失败")
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

            if ids is not None:
                ids = ids.flatten()
                print(f"识别到 IDs: {ids}")

                for i, id in enumerate(ids):
                    color = (0, 255, 0)
                    if id == TARGET_ID:
                        color = (0, 0, 255)  # 红色高亮
                        ret = cv2.aruco.estimatePoseSingleMarkers(corners, 0.03, self.camera_matrix, self.dist_coeffs)
                        tvec = ret[1][i][0]

                        robot_coords = convert_to_robot_coords(tvec, self.matrix)
                        print(f"tvec: {tvec}, robot_coords: {robot_coords}")
                        self.move_to_target(robot_coords)

                    c = corners[i][0].astype(int)
                    cv2.polylines(frame, [c], True, color, 2)
                    cv2.putText(frame, f"ID:{id}", tuple(c[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                print("无 ArUco 检测")

            cv2.imshow("ArUco Grasp View", frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


# ------------------ 启动主程序 ------------------
if __name__ == '__main__':
    node = ArucoGrasp()
    node.run()
