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
OBS_POSE = [-90, 5, -45, -40, 90 + offset_j5, 60]  # 与你抓取程序一致

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
    try: mc.power_on()
    except Exception: pass
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
detector = camera_detect(0, 40, mtx, dist)
if detector.EyesInHand_matrix is None and EIH_PATH.exists():
    detector.load_matrix(str(EIH_PATH))
T_ee_cam = detector.EyesInHand_matrix
assert T_ee_cam is not None, "未找到 EyesInHand_matrix.json，请先标定并放同目录"
cam = detector.camera

# ====== 末端基座位姿（mm/deg→m/rad）======
def get_ee_state_rad():
    coords = mc.get_coords()
    while (coords is None) or (len(coords) < 6):
        time.sleep(0.01)
        coords = mc.get_coords()
    x, y, z, rx, ry, rz = coords
    return np.array([x*1e-3, y*1e-3, z*1e-3,
                     np.deg2rad(rx), np.deg2rad(ry), np.deg2rad(rz)],
                    dtype=np.float32)

def get_T_be_from_coords(coords):
    return detector.Transformation_matrix(coords)

# ====== 相机系Δ动作标签（由两帧末端位姿推导）======
def cam_delta_from_two_poses(prev_coords, now_coords):
    T_be_prev = get_T_be_from_coords(prev_coords)
    T_be_now  = get_T_be_from_coords(now_coords)
    T_bc_prev = T_be_prev @ T_ee_cam
    T_bc_now  = T_be_now  @ T_ee_cam
    T_delta_B = np.linalg.inv(T_bc_prev) @ T_bc_now
    dR = R.from_matrix(T_delta_B[:3, :3])
    rotvec_B = dR.as_rotvec()
    dp_B = T_delta_B[:3, 3]
    R_cb = T_bc_prev[:3, :3].T
    dp_C = R_cb @ dp_B
    dw_C = R_cb @ rotvec_B
    return np.r_[dp_C, dw_C].astype(np.float32)

# ====== 掩码（仅涂选中 STAG ID）======
def detect_mask_bgr(frame_bgr, target_id_or_code):
    h, w = frame_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    if target_id_or_code is None: return mask
    if isinstance(target_id_or_code, str):
        tid = TARGET_ID_MAP.get(target_id_or_code.upper(), None)
    else:
        tid = int(target_id_or_code)
    if tid is None: return mask
    try:
        corners, ids, _ = stag.detectMarkers(frame_bgr, 11)
        if ids is not None:
            for c, i in zip(corners, ids):
                if int(np.array(i).flatten()[0]) == tid:
                    poly = np.asarray(c).reshape(-1, 2).astype(np.int32)
                    cv2.fillConvexPoly(mask, poly, 255)
                    break
    except Exception:
        pass
    if mask.sum() > 0:
        k = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.dilate(mask, k, iterations=1)
    return mask

# ====== 采集线程 ======
buf = {"rgb": [], "mask": [], "state": [], "action": [],
       "cond_target": [], "cond_phase": [], "ts": []}
lock = threading.Lock()
h5file = None
running = True

def reset_buffers():
    for k in buf: buf[k].clear()

def writer_loop():
    prev_coords = mc.get_coords()
    while prev_coords is None or len(prev_coords) < 6:
        time.sleep(0.01)
        prev_coords = mc.get_coords()

    t0 = time.time()
    while running:
        cam.update_frame()
        frame = cam.color_frame()
        if frame is None:
            time.sleep(0.01); continue

        now_coords = mc.get_coords()
        if now_coords is None or len(now_coords) < 6:
            time.sleep(0.005); continue

        d6 = cam_delta_from_two_poses(prev_coords, now_coords)
        prev_coords = now_coords

        st = get_ee_state_rad()
        g = 0.0
        lowdim = np.r_[st, g].astype(np.float32)

        mask = detect_mask_bgr(frame, state["target_id"])

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

# ====== API：目标、录制、回观测位、健康 ======
@app.post("/bci/target")
def api_target():
    d = request.get_json(silent=True) or {}
    if not d and request.form:
        d = {"id": request.form.get("target", None), "phase": "selected"}
    tid = d.get("id")
    ph  = d.get("phase", "selected")
    if tid is None:
        return jsonify(ok=False, msg="need id or form 'target'"), 400
    state["target_id"] = tid
    state["phase"] = ph
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
        save("action",      buf["action"], np.float32)    # [dx,dy,dz, dR,dP,dY, dGrip]
        save("cond/target", buf["cond_target"], np.float32)
        save("cond/phase",  buf["cond_phase"], np.int32)
        save("time/ts",     buf["ts"], np.float64)
        path = h5file.filename
        h5file.close(); h5file = None
    return jsonify(ok=True, file=path)

@app.post("/goto_obs")
def api_goto_obs():
    sp = int((request.get_json(silent=True) or {}).get("speed", 60))
    goto_observe(sp)
    return jsonify(ok=True, pose=OBS_POSE)

# ====== IK 老师抓取（只做“抓取”，不负责放回）======
GRIPPER_Z_OFFSET = 100
APPROACH_BUFFER  = 20
Z_OFFSET         = 30
LIFT_AFTER_GRASP = 60

def _open_gripper(sp=80):  mc.set_gripper_state(0, sp)
def _close_gripper(sp=80): mc.set_gripper_state(1, sp)

def _find_target_base_xyz(selected_id: int):
    cam.update_frame()
    frame = cam.color_frame()
    corners, ids, _ = stag.detectMarkers(frame, 11)
    if ids is None: return None
    ids = np.array(ids).flatten()
    idxs = np.where(ids == selected_id)[0]
    if len(idxs) == 0: return None
    idx = int(idxs[0])
    one_corners = [corners[idx]]
    one_ids = np.array([[selected_id]], dtype=np.int32)
    target_cam = detector.calc_markers_base_position(one_corners, one_ids)
    if target_cam is None or len(target_cam) < 3: return None
    end_coords = mc.get_coords()
    while end_coords is None: time.sleep(0.01); end_coords = mc.get_coords()
    T_be = detector.Transformation_matrix(end_coords)
    p_cam = np.array([target_cam[0], target_cam[1], target_cam[2], 1.0], dtype=float)
    p_base = (T_be @ T_ee_cam @ p_cam).flatten()[:3]
    return p_base, end_coords

@app.post("/demo_grasp")
def api_demo_grasp():
    d = request.get_json(silent=True) or {}
    sp = int(d.get("speed", 30))

    tid_code = state.get("target_id")
    if isinstance(tid_code, str):
        tid = TARGET_ID_MAP.get(tid_code.upper(), None)
    else:
        tid = int(tid_code) if tid_code is not None else None
    if tid is None:
        return jsonify(ok=False, msg="no target id, call /bci/target first"), 400

    found = _find_target_base_xyz(tid)
    if found is None:
        return jsonify(ok=False, msg="stag id not found in view"), 404
    xyz_base, end_coords = found
    rx, ry, rz = end_coords[3:6]

    grasp = [float(xyz_base[0]), float(xyz_base[1]), float(xyz_base[2] + GRIPPER_Z_OFFSET),
             float(rx), float(ry), float(rz)]
    above = grasp.copy();    above[2]    += Z_OFFSET
    approach = grasp.copy(); approach[2] += APPROACH_BUFFER
    lift = grasp.copy();     lift[2]     += LIFT_AFTER_GRASP

    detector.coord_limit(grasp); detector.coord_limit(above)
    detector.coord_limit(approach); detector.coord_limit(lift)

    try:
        mc.power_on(); time.sleep(0.5)
        _open_gripper()
        mc.send_coords(above,    sp); time.sleep(2)
        mc.send_coords(approach, sp); time.sleep(2)
        mc.send_coords(grasp,    max(15, sp//2)); time.sleep(2)
        _close_gripper()
        mc.send_coords(lift,     sp); time.sleep(2)
        return jsonify(ok=True)
    except Exception as e:
        return jsonify(ok=False, msg=str(e)), 500

@app.get("/health")
def api_health():
    return jsonify(ok=True, **state)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, threaded=True)
