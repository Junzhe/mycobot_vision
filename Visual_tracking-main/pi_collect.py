from flask import Flask, request, jsonify
from pathlib import Path
import time, threading, h5py, numpy as np, cv2
from scipy.spatial.transform import Rotation as R
from pymycobot import MyCobot280, PI_PORT, PI_BAUD
from camera_detect import camera_detect
import stag

# ====== 路径与保存目录（项目内 data/）======
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
H5_DIR = str(DATA_DIR)

# ====== 配置 ======
PORT = 5055
TARGET_ID_MAP = {"A": 0, "B": 1, "C": 2}

# ====== 全局状态 ======
app = Flask(__name__)
state = {"target_id": None, "phase": "idle", "recording": False}

# ====== 初始化机械臂 / 观测位 ======
print("[INFO] 初始化机械臂与相机...")
mc = MyCobot280(PI_PORT, PI_BAUD)
time.sleep(0.5)
offset_j5 = -90 if mc.get_system_version() > 2 else 0
# 你当前使用的观察位（远离桌面，视角不变）
OBS_POSE = [-90, 5, -45, -40, 90 + offset_j5, 60]

def wait_stop(timeout=20.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if mc.is_moving() == 0:
                break
        except Exception:
            pass
        time.sleep(0.1)

def goto_observe(speed=60):
    try:
        mc.power_on()
    except Exception:
        pass
    time.sleep(0.5)
    mc.send_angles(OBS_POSE, speed)
    wait_stop(25.0)

# 启动回到观测位
goto_observe()

# ====== 相机 / 手眼 ======
CAM_PATH = ROOT / "camera_params.npz"
EIH_PATH = ROOT / "EyesInHand_matrix.json"
camera_params = np.load(str(CAM_PATH))
mtx, dist = camera_params["mtx"], camera_params["dist"]
detector = camera_detect(0, 25, mtx, dist)
if detector.EyesInHand_matrix is None and EIH_PATH.exists():
    detector.load_matrix(str(EIH_PATH))
T_ee_cam = detector.EyesInHand_matrix
assert T_ee_cam is not None, "未找到 EyesInHand_matrix.json，请先标定并放同目录"
cam = detector.camera

# ====== 末端基座位姿（mm/deg→m/rad）======
def get_ee_state_rad():
    """返回 [x,y,z,rx,ry,rz]（m/rad），用于 HDF5 的 state 通道。"""
    coords = mc.get_coords()
    while (coords is None) or (len(coords) < 6):
        time.sleep(0.01)
        coords = mc.get_coords()
    x, y, z, rx, ry, rz = coords
    return np.array([x*1e-3, y*1e-3, z*1e-3,
                     np.deg2rad(rx), np.deg2rad(ry), np.deg2rad(rz)],
                    dtype=np.float32)

def get_T_be_from_coords(coords):
    """基于你项目里的函数：输入 mm/deg，输出 4x4 齐次（平移单位与输入一致, 即 mm）。"""
    return detector.Transformation_matrix(coords)

# ====== 相机系Δ动作标签（由两帧末端位姿推导）======
def cam_delta_from_two_poses(prev_coords, now_coords):
    """
    输出上一帧相机系的 6D 增量：[dx,dy,dz, dRx,dRy,dRz]
    单位：平移=mm（与齐次矩阵一致），旋转=rad
    ——关键修正：不再对相机系增量重复旋转，直接用 inv(T_prev)@T_now 的结果。
    """
    T_be_prev = get_T_be_from_coords(prev_coords)  # mm/deg → 4x4(mm)
    T_be_now  = get_T_be_from_coords(now_coords)
    T_bc_prev = T_be_prev @ T_ee_cam
    T_bc_now  = T_be_now  @ T_ee_cam

    # 这就是“上一帧相机坐标系”表达的相机相对位姿
    T_delta = np.linalg.inv(T_bc_prev) @ T_bc_now

    dp_C = T_delta[:3, 3].astype(np.float32)                         # mm
    dw_C = R.from_matrix(T_delta[:3, :3]).as_rotvec().astype(np.float32)  # rad
    return np.r_[dp_C,_]()_
