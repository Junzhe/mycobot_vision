

import cv2
from uvc_camera import UVCCamera
import stag
import numpy as np
import json
import time
from marker_utils import *
from scipy.linalg import svd
from pymycobot import *

mc = MyCobot280("/dev/ttyAMA0", 1000000)
type = mc.get_system_version()
offset_j5 = -90 if type > 2 else 0

np.set_printoptions(suppress=True, formatter={'float_kind': '{:.2f}'.format})

class camera_detect:
    def __init__(self, camera_id, marker_size, mtx, dist):
        self.camera_id = camera_id
        self.mtx = mtx
        self.dist = dist
        self.marker_size = marker_size
        self.camera = UVCCamera(self.camera_id, self.mtx, self.dist)
        self.camera_open()

        self.observe_pose = [0, 0, 2, -58, -2, -14 + offset_j5]
        self.IDENTIFY_LEN = 300
        self.EyesInHand_matrix = None
        self.load_matrix()

    def save_matrix(self, filename="EyesInHand_matrix.json"):
        if self.EyesInHand_matrix is not None:
            with open(filename, 'w') as f:
                json.dump(self.EyesInHand_matrix.tolist(), f)

    def load_matrix(self, filename="EyesInHand_matrix.json"):
        try:
            with open(filename, 'r') as f:
                self.EyesInHand_matrix = np.array(json.load(f))
        except FileNotFoundError:
            print("Matrix file not found. EyesInHand_matrix will be initialized later.")

    def wait(self):
        time.sleep(0.5)
        while(mc.is_moving() == 1):
            time.sleep(0.2)

    def coord_limit(self, coords):
        min_coord = [-350, -350, 300]
        max_coord = [350, 350, 500]
        for i in range(3):
            coords[i] = max(min_coord[i], min(max_coord[i], coords[i]))

    def camera_open(self):
        self.camera.capture()

    def CvtRotationMatrixToEulerAngle(self, R):
        euler = np.zeros(3)
        euler[2] = np.arctan2(R[1, 0], R[0, 0])
        cos_r = np.cos(euler[2])
        sin_r = np.sin(euler[2])
        euler[1] = np.arctan2(-R[2, 0], R[0, 0]*cos_r + R[1, 0]*sin_r)
        euler[0] = np.arctan2(R[0, 2]*sin_r - R[1, 2]*cos_r, -R[0, 1]*sin_r + R[1, 1]*cos_r)
        return euler

    def Transformation_matrix(self, coord):
        position_robot = coord[:3]
        pose_robot = coord[3:]
        RBT = self.CvtEulerAngleToRotationMatrix(pose_robot)
        PBT = np.array([[position_robot[0]], [position_robot[1]], [position_robot[2]]])
        temp = np.concatenate((RBT, PBT), axis=1)
        array_1x4 = np.array([[0, 0, 0, 1]])
        return np.concatenate((temp, array_1x4), axis=0)

    def CvtEulerAngleToRotationMatrix(self, angles):
        sin_angle = np.sin(angles)
        cos_angle = np.cos(angles)
        R = np.zeros((3, 3))
        R[0, 0] = cos_angle[2] * cos_angle[1]
        R[0, 1] = cos_angle[2] * sin_angle[1] * sin_angle[0] - sin_angle[2] * cos_angle[0]
        R[0, 2] = cos_angle[2] * sin_angle[1] * cos_angle[0] + sin_angle[2] * sin_angle[0]
        R[1, 0] = sin_angle[2] * cos_angle[1]
        R[1, 1] = sin_angle[2] * sin_angle[1] * sin_angle[0] + cos_angle[2] * cos_angle[0]
        R[1, 2] = sin_angle[2] * sin_angle[1] * cos_angle[0] - cos_angle[2] * sin_angle[0]
        R[2, 0] = -sin_angle[1]
        R[2, 1] = cos_angle[1] * sin_angle[0]
        R[2, 2] = cos_angle[1] * cos_angle[0]
        return R

    def Eyes_in_hand(self, coord, camera, Matrix_TC):
        Position_Camera = np.transpose(camera[:3])
        Matrix_BT = self.Transformation_matrix(coord)
        Position_Camera = np.append(Position_Camera, 1)
        Position_B = Matrix_BT @ Matrix_TC @ Position_Camera
        return Position_B

    def calc_markers_base_position(self, corners, ids):
        if len(corners) == 0 or ids is None:
            return []
        rvecs, tvecs = solve_marker_pnp(corners, self.marker_size, self.mtx, self.dist)
        results = []
        for i, tvec, rvec in zip(ids, tvecs, rvecs):
            tvec = tvec.squeeze().tolist()
            rvec = rvec.squeeze().tolist()
            Rotation = cv2.Rodrigues(np.array([rvec]))[0]
            Euler = self.CvtRotationMatrixToEulerAngle(Rotation)
            coord = np.array([tvec[0], tvec[1], tvec[2], Euler[0], Euler[1], Euler[2]])
            results.append((int(i[0]), coord))
        return results

    def stag_identify(self):
        self.camera.update_frame()
        frame = self.camera.color_frame()
        corners, ids, _ = stag.detectMarkers(frame, 11)
        return self.calc_markers_base_position(corners, ids)

    def move_to_target_id(self, ml, target_id):
        print("[INFO] 先移动至观察位...")
        ml.send_angles(self.observe_pose, 30)
        self.wait()

        current_pose = ml.get_coords()
        while current_pose is None:
            current_pose = ml.get_coords()
        current_pose_rad = np.array(current_pose.copy())
        current_pose_rad[-3:] *= np.pi / 180

        for attempt in range(10):
            all_targets = self.stag_identify()
            if not all_targets:
                print("[WARN] 未识别到任何目标，重试中...")
                time.sleep(0.5)
                continue
            for id_val, cam_coord in all_targets:
                print(f"[DEBUG] 检测到 ID={id_val}")
                if id_val == target_id:
                    print(f"[INFO] 找到目标 ID={target_id}，计算并移动...")
                    base_coord = self.Eyes_in_hand(current_pose_rad, cam_coord, self.EyesInHand_matrix)
                    base_coord = base_coord[:3].tolist() + current_pose[3:6]
                    self.coord_limit(base_coord)
                    ml.send_coords(base_coord, 30)
                    self.wait()
                    return
            print("[INFO] 当前帧中未找到目标 ID，继续尝试...")
            time.sleep(0.5)
        print(f"[ERROR] 多次尝试后未能找到 ID={target_id}")

if __name__ == "__main__":
    camera_params = np.load("camera_params.npz")
    mtx, dist = camera_params["mtx"], camera_params["dist"]
    m = camera_detect(0, 32, mtx, dist)
    mc.set_vision_mode(0)

    while True:
        try:
            target_id = int(input("请输入目标 Stag ID（如0或1），输入-1退出: "))
            if target_id == -1:
                break
            m.move_to_target_id(mc, target_id)
        except Exception as e:
            print("[错误] 输入无效:", e)
