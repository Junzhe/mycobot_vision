# -*- coding: utf-8 -*-
# Pi 端数据采集服务：录制 RGB + mask + 末端状态 + 相机系Δ动作 → HDF5
# 依赖你的 camera_detect（STAG/手眼）、pymycobot

from flask import Flask, request, jsonify
import time, threading, json, h5py, numpy as np, cv2
from scipy.spatial.transform import Rotation as R
from pymycobot import MyCobot280, PI_PORT, PI_BAUD
from camera_detect import camera_detect

# ====== 配置 ======
PORT = 5055
TARGET_ID_MAP = {"A": 0, "B": 1, "C": 2}   # 与你抓取程序保持一致
H5_DIR = "/home/pi/bci_runtime/data"

# ====== 全局状态 ======
app = Flask(__name__)
state = {"target_id": None, "phase": "idle", "recording": False}

# ====== 初始化设备 / 相机 / 手眼 ======
print("[INFO] 初始化机械臂与相机...")
mc = MyCobot280(PI_PORT, PI_BAUD)
time.sleep(1)

camera_params = np.load("camera_params.npz")
mtx, dist = camera_params["mtx"], camera_params["dist"]
detector = camera_detect(0, 40, mtx, dist)   # 直接复用你的类
# 你的类会在 __init__ 时尝试 load_matrix("EyesInHand_matrix.json")
T_ee_cam = detector.EyesInHand_matrix        # 这是 T^T_C（相机->末端），正是我们要的
assert T_ee_cam is not None, "未找到 EyesInHand_matrix.json，请先完成手眼标定"

cam = detector.camera  # 复用 UVCCamera，避免重复打开设备

# ====== 读取末端基座位姿（mm/deg→m/rad）======
def get_ee_state_rad():
    """返回 [x,y,z, rx,ry,rz]，位置 m，姿态 rad（与 HDF5 低维状态一致）"""
    coords = mc.get_coords()
    while (coords is None) or (len(coords) < 6):
        time.sleep(0.01)
        coords = mc.get_coords()
    x, y, z, rx, ry, rz = coords
    return np.array([x*1e-3, y*1e-3, z*1e-3,
                     np.deg2rad(rx), np.deg2rad(ry), np.deg2rad(rz)],
                    dtype=np.float32)

def get_T_be_from_coords(coords):
    """用你自己的变换构造函数（内部会做角度→弧度）"""
    return detector.Transformation_matrix(coords)  # coords: [x,y,z,rx,ry,rz] (mm/deg)

# ====== 相机系Δ动作标签（由两帧末端位姿推导）======
def cam_delta_from_two_poses(prev_coords, now_coords):
    """
    输入两帧末端 pose（mm/deg），利用手眼 T_ee_cam 计算相机系 6D 增量 [dx,dy,dz, droll,dpitch,dyaw]
    单位：m / rad
    """
    T_be_prev = get_T_be_from_coords(prev_coords)
    T_be_now  = get_T_be_from_coords(now_coords)
    T_bc_prev = T_be_prev @ T_ee_cam
    T_bc_now  = T_be_now  @ T_ee_cam

    T_delta_B = np.linalg.inv(T_bc_prev) @ T_bc_now
    dR = R.from_matrix(T_delta_B[:3, :3])
    rotvec_B = dR.as_rotvec()
    dp_B = T_delta_B[:3, 3]

    # 旋到“上一帧相机系”
    R_cb = T_bc_prev[:3, :3].T
    dp_C = R_cb @ dp_B
    dw_C = R_cb @ rotvec_B
    return np.r_[dp_C, dw_C].astype(np.float32)

# ====== 生成“选中ID”的二值掩码 ======
def detect_mask_bgr(frame_bgr, target_id_or_code):
    h, w = frame_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    if target_id_or_code is None:
        return mask

    # 统一为数字 ID
    if isinstance(target_id_or_code, str):
        tid = TARGET_ID_MAP.get(target_id_or_code.upper(), None)
    else:
        tid = int(target_id_or_code)
    if tid is None:
        return mask

    # 优先 STAG（与你的工程一致）
    try:
        # 你的 camera_detect.stag_identify() 走了 PnP，不返回角点；这里直接用 stag.detectMarkers
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.stag.detectMarkers(gray, 11)  # 11 与你一致
        if ids is not None:
            for c, i in zip(corners, ids):
                if int(np.array(i).flatten()[0]) == tid:
                    poly = np.asarray(c).reshape(-1, 2).astype(np.int32)
                    cv2.fillConvexPoly(mask, poly, 255)
                    break
    except Exception:
        pass

    # 可选回退：ArUco（需 opencv-contrib）——如果你只用 STAG，可删
    if mask.sum() == 0:
        try:
            aruco = cv2.aruco
            dict4 = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
            detector_aruco = aruco.ArucoDetector(dict4, aruco.DetectorParameters())
            corners, ids, _ = detector_aruco.detectMarkers(gray)
            if ids is not None:
                for c, i in zip(corners, ids):
                    if int(np.array(i).flatten()[0]) == tid:
                        poly = c.reshape(-1, 2).astype(np.int32)
                        cv2.fillConvexPoly(mask, poly, 255)
                        break
        except Exception:
            pass

    if mask.sum() > 0:
        k = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.dilate(mask, k, iterations=1)
    return mask

# ====== 采集线程：循环读相机/末端，条件录入缓冲 ======
buf = {"rgb": [], "mask": [], "state": [], "action": [],
       "cond_target": [], "cond_phase": [], "ts": []}
lock = threading.Lock()
h5file = None
running = True

def reset_buffers():
    for k in buf: buf[k].clear()

def writer_loop():
    # 用原生 coords（mm/deg）计算相机系Δ更稳
    prev_coords = mc.get_coords()
    while prev_coords is None or len(prev_coords) < 6:
        time.sleep(0.01)
        prev_coords = mc.get_coords()

    t0 = time.time()
    while running:
        # 相机帧
        cam.update_frame()
        frame = cam.color_frame()   # BGR
        if frame is None:
            time.sleep(0.01); continue

        # 末端本帧 coords（mm/deg）
        now_coords = mc.get_coords()
        if now_coords is None or len(now_coords) < 6:
            time.sleep(0.005); continue

        # 相机系 Δ（6D）标签
        d6 = cam_delta_from_two_poses(prev_coords, now_coords)  # m / rad
        prev_coords = now_coords

        # 低维状态（m/rad + gripper）
        st = get_ee_state_rad()  # [x,y,z,rx,ry,rz]
        g = 0.0  # 若你后续对接真实夹爪读数，这里替换
        lowdim = np.r_[st, g].astype(np.float32)  # 7 + 1

        # 掩码（选中ID涂白）
        mask = detect_mask_bgr(frame, state["target_id"])

        # 目标 one-hot
        onehot = np.zeros(3, np.float32)
        if isinstance(state["target_id"], str) and state["target_id"].upper() in TARGET_ID_MAP:
            idx = TARGET_ID_MAP[state["target_id"].upper()]
            onehot[idx] = 1.0

        with lock:
            if state["recording"] and h5file is not None:
                buf["rgb"].append(frame[..., ::-1])               # RGB
                buf["mask"].append(mask)
                buf["state"].append(lowdim)
                buf["action"].append(np.r_[d6, 0.0].astype(np.float32))  # 6D + gripperΔ
                buf["cond_target"].append(onehot)
                buf["cond_phase"].append({"idle":0,"selected":1,"confirmed":2}.get(state["phase"],0))
                buf["ts"].append(time.time() - t0)
        time.sleep(0.001)

thr = threading.Thread(target=writer_loop, daemon=True)
thr.start()

# ====== HTTP API ======
@app.post("/bci/target")
def api_target():
    d = request.get_json(force=True)
    state["target_id"] = d.get("id")
    state["phase"] = d.get("phase", "selected")
    return jsonify(ok=True, target=state["target_id"], phase=state["phase"])

@app.post("/record/start")
def api_start():
    global h5file
    fname = f"{H5_DIR}/epi_{int(time.time())}.hdf5"
    with lock:
        reset_buffers()
        h5file = h5py.File(fname, "w")
        state["recording"] = True
        h5file.attrs["target_id"] = state["target_id"] or ""
    return jsonify(ok=True, file=fname)

@app.post("/record/stop")
def api_stop():
    global h5file
    with lock:
        state["recording"] = False
        if h5file is None:
            return jsonify(ok=False, msg="no open file")
        g = h5file.create_group("frames")
        def save(name, arr, dtype=None):
            a = np.asarray(arr, dtype=dtype) if dtype else np.asarray(arr)
            g.create_dataset(name, data=a, compression="gzip")
        save("images/rgb",  buf["rgb"],  np.uint8)
        save("images/mask", buf["mask"], np.uint8)
        save("state",       buf["state"], np.float32)     # [x,y,z,rx,ry,rz,g]
        save("action",      buf["action"], np.float32)    # [dx,dy,dz, dR, dP, dY, dGrip]
        save("cond/target", buf["cond_target"], np.float32)
        save("cond/phase",  buf["cond_phase"], np.int32)
        save("time/ts",     buf["ts"], np.float64)
        path = h5file.filename
        h5file.close(); h5file = None
    return jsonify(ok=True, file=path)

@app.get("/health")
def api_health():
    return jsonify(ok=True, **state)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, threaded=True)
