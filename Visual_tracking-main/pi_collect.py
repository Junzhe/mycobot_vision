from flask import Flask, request, jsonify, send_file
from pathlib import Path
import io, time, threading, h5py, numpy as np, cv2
from scipy.spatial.transform import Rotation as R
from pymycobot import MyCobot280, PI_PORT, PI_BAUD
from camera_detect import camera_detect
import stag

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
H5_DIR = str(DATA_DIR)

PORT = 5055
TARGET_ID_MAP = {"A": 0, "B": 1, "C": 2}
# 抓取参数
GRIPPER_Z_OFFSET = 100     # mm：末端提前停下（抓取前的高度补偿）
APPROACH_BUFFER  = 20      # mm：预接近
Z_OFFSET         = 30      # mm：上方准备点抬升
LIFT_AFTER_GRASP = 60      # mm：闭合后上抬验证
FIND_TAG_DICT_ID = 11      # STAG 字典号

# ========= 全局状态 =========
app = Flask(__name__)
state = {
    "target_id": None,   # "A"/"B"/"C" or None
    "phase": "idle",     # idle/selected/confirmed
    "recording": False
}

# ========= 初始化机械臂 =========
print("[INFO] 初始化机械臂与相机...")
mc = MyCobot280(PI_PORT, PI_BAUD)
time.sleep(0.5)
offset_j5 = -90 if mc.get_system_version() > 2 else 0
OBS_POSE = [-90, 5, -45, -40, 90 + offset_j5, 60]   # 观测位（可按需调整）

def now_mono() -> float:
    return float(time.monotonic())

def wait_stop(timeout=20.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if mc.is_moving() == 0:
                break
        except Exception:
            pass
        time.sleep(0.05)

# ========= 相机/手眼 =========
CAM_PATH = ROOT / "camera_params.npz"
EIH_PATH = ROOT / "EyesInHand_matrix.json"
camera_params = np.load(str(CAM_PATH))
mtx, dist = camera_params["mtx"], camera_params["dist"]
detector = camera_detect(0, 25, mtx, dist)  # 目标 25fps
if detector.EyesInHand_matrix is None and EIH_PATH.exists():
    detector.load_matrix(str(EIH_PATH))
T_ee_cam = detector.EyesInHand_matrix
assert T_ee_cam is not None, "未找到 EyesInHand_matrix.json，请先标定并放同目录"
cam = detector.camera

# ========= 读当前传感 =========
def get_coords_mmdeg() -> np.ndarray:
    """末端 mm/deg（阻塞获取）"""
    coords = mc.get_coords()
    while (coords is None) or (len(coords) < 6):
        time.sleep(0.005)
        coords = mc.get_coords()
    return np.array(coords, dtype=np.float32)

def get_angles_deg() -> np.ndarray:
    """6关节角 deg（阻塞获取）"""
    ang = mc.get_angles()
    while (ang is None) or (len(ang) < 6):
        time.sleep(0.005)
        ang = mc.get_angles()
    return np.array(ang, dtype=np.float32)

def T_from_coords_mmdeg(coords: np.ndarray) -> np.ndarray:
    """输入末端 coords(mm/deg) → 4x4 齐次（平移 mm）"""
    return detector.Transformation_matrix(coords)

def rvec_deg_from_rotm(Rm: np.ndarray) -> np.ndarray:
    rv = R.from_matrix(Rm[:3, :3]).as_rotvec()   # rad
    return np.rad2deg(rv).astype(np.float32)

# ========= 目标检测 =========
def detect_mask_bgr(frame_bgr, target_id_or_code):
    h, w = frame_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    if target_id_or_code is None:
        return mask
    if isinstance(target_id_or_code, str):
        tid = TARGET_ID_MAP.get(target_id_or_code.upper(), None)
    else:
        tid = int(target_id_or_code)
    if tid is None:
        return mask
    try:
        corners, ids, _ = stag.detectMarkers(frame_bgr, FIND_TAG_DICT_ID)
        if ids is not None:
            ids = np.array(ids).flatten()
            for c, i in zip(corners, ids):
                if int(i) == tid:
                    poly = np.asarray(c).reshape(-1, 2).astype(np.int32)
                    cv2.fillConvexPoly(mask, poly, 255)
                    break
    except Exception:
        pass
    if mask.sum() > 0:
        k = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.dilate(mask, k, iterations=1)
    return mask

def find_target_positions(frame_bgr, T_be):
    """
    返回:
      visible: int(0/1)
      p_cam: (3,) 目标在相机系(mm)
      p_base: (3,) 目标在基座系(mm)
    检测不到时 (0, nan, nan)
    """
    visible = 0
    p_cam = np.full(3, np.nan, np.float32)
    p_base = np.full(3, np.nan, np.float32)

    tid_code = state.get("target_id")
    if isinstance(tid_code, str):
        tid = TARGET_ID_MAP.get(tid_code.upper(), None)
    else:
        tid = int(tid_code) if tid_code is not None else None
    if tid is None:
        return visible, p_cam, p_base

    try:
        corners, ids, _ = stag.detectMarkers(frame_bgr, FIND_TAG_DICT_ID)
        if ids is None:
            return visible, p_cam, p_base
        ids = np.array(ids).flatten()
        idxs = np.where(ids == tid)[0]
        if len(idxs) == 0:
            return visible, p_cam, p_base

        one_corners = [corners[int(idxs[0])]]
        one_ids = np.array([[tid]], dtype=np.int32)
        target_cam = detector.calc_markers_base_position(one_corners, one_ids)
        if target_cam is None or len(target_cam) < 3:
            return visible, p_cam, p_base

        p_cam = np.array(target_cam[:3], dtype=np.float32)
        p4 = np.r_[p_cam.astype(float), 1.0]
        p_base = (T_be @ T_ee_cam @ p4)[:3].astype(np.float32)
        visible = 1
        return visible, p_cam, p_base
    except Exception:
        return 0, p_cam, p_base

# ========= 记录缓冲 =========
buf = {
    "rgb": [], "mask": [],
    "state": [],
    "action_cam": [], "action_base": [],
    "vel_cam": [], "vel_base": [],
    "joints": [], "joints_vel": [],
    "T_be": [], "T_bc": [],
    "target_cam": [], "target_base": [], "target_vis": [],
    "cond_target": [], "cond_phase": [],
    "ts": [], "dt": [],
    "cmd_type": [], "cmd_speed": [], "cmd_coords": [], "cmd_angles": [], "cmd_gripper": [], "cmd_time": []
}
lock = threading.Lock()
h5file = None
running = True

# —— 目标保持/随夹爪 —— #
last_target_base = None
last_target_cam  = None

# ========= 真实命令封装 =========
# type: 0=none, 1=send_coords, 2=send_angles, 3=gripper
last_cmd = {
    "type": 0,
    "speed": np.float32(np.nan),
    "coords": np.full(6, np.nan, np.float32),
    "angles": np.full(6, np.nan, np.float32),
    "gripper": np.int32(-1),     # -1=none, 0=open, 1=close
    "time": np.float64(np.nan)
}
cmd_lock = threading.Lock()

def snapshot_cmd():
    with cmd_lock:
        return {
            "type": int(last_cmd["type"]),
            "speed": float(last_cmd["speed"]),
            "coords": last_cmd["coords"].copy(),
            "angles": last_cmd["angles"].copy(),
            "gripper": int(last_cmd["gripper"]),
            "time": float(last_cmd["time"]),
        }

def send_coords_logged(coords_mmdeg, speed=30):
    coords = [float(v) for v in coords_mmdeg]
    mc.send_coords(coords, int(speed))
    with cmd_lock:
        last_cmd["type"] = 1
        last_cmd["speed"] = np.float32(speed)
        last_cmd["coords"] = np.array(coords, dtype=np.float32)
        last_cmd["angles"] = np.full(6, np.nan, np.float32)
        last_cmd["gripper"] = np.int32(-1)
        last_cmd["time"] = np.float64(now_mono())

def send_angles_logged(angles_deg, speed=30):
    angles = [float(v) for v in angles_deg]
    mc.send_angles(angles, int(speed))
    with cmd_lock:
        last_cmd["type"] = 2
        last_cmd["speed"] = np.float32(speed)
        last_cmd["coords"] = np.full(6, np.nan, np.float32)
        last_cmd["angles"] = np.array(angles, dtype=np.float32)
        last_cmd["gripper"] = np.int32(-1)
        last_cmd["time"] = np.float64(now_mono())

def set_gripper_logged(open_or_close, speed=80):
    try:
        mc.set_gripper_state(int(open_or_close), int(speed))
    except Exception:
        pass
    with cmd_lock:
        last_cmd["type"] = 3
        last_cmd["speed"] = np.float32(speed)
        last_cmd["coords"] = np.full(6, np.nan, np.float32)
        last_cmd["angles"] = np.full(6, np.nan, np.float32)
        last_cmd["gripper"] = np.int32(open_or_close)  # 0=open,1=close
        last_cmd["time"] = np.float64(now_mono())

# ========= 采集线程 =========
def reset_buffers():
    for k in buf:
        buf[k].clear()

def writer_loop():
    global last_target_base, last_target_cam
    # 初始
    prev_coords = get_coords_mmdeg()
    prev_angles = get_angles_deg()
    prev_T_be = T_from_coords_mmdeg(prev_coords)
    prev_T_bc = prev_T_be @ T_ee_cam
    prev_t = now_mono()

    while running:
        cam.update_frame()
        frame = cam.color_frame()
        if frame is None:
            time.sleep(0.003); continue

        now_coords = get_coords_mmdeg()
        now_angles = get_angles_deg()
        now_T_be = T_from_coords_mmdeg(now_coords)
        now_T_bc = now_T_be @ T_ee_cam

        now_t = now_mono()
        dt = float(max(1e-4, now_t - prev_t))
        prev_t = now_t

        # Δpose（相机系/基座系）
        T_delta_cam = np.linalg.inv(prev_T_bc) @ now_T_bc
        dpos_cam = T_delta_cam[:3, 3].astype(np.float32)
        dang_cam = rvec_deg_from_rotm(T_delta_cam)

        T_delta_base = np.linalg.inv(prev_T_be) @ now_T_be
        dpos_base = T_delta_base[:3, 3].astype(np.float32)
        dang_base = rvec_deg_from_rotm(T_delta_base)

        vel_cam = np.r_[dpos_cam, dang_cam] / dt
        vel_base = np.r_[dpos_base, dang_base] / dt

        # 关节速度
        joints_vel = (now_angles - prev_angles) / dt

        # 目标
        vis, p_cam, p_base = find_target_positions(frame, now_T_be)
        phase = state.get("phase", "idle")
        if vis == 0:
            if phase in ("idle", "selected") and (last_target_base is not None):
                # 保持上一帧（桌面静止）
                p_base = last_target_base.copy()
                inv_T = np.linalg.inv(now_T_be @ T_ee_cam)
                p4 = np.r_[p_base.astype(float), 1.0]
                p_cam = (inv_T @ p4)[:3].astype(np.float32)
        else:
            last_target_base = p_base.copy()
            last_target_cam  = p_cam.copy()

        if phase == "confirmed":
            # 抓后随夹爪（可加固定偏移）
            p_base = now_T_be[:3, 3].astype(np.float32)
            p_cam  = (T_ee_cam @ np.array([0,0,0,1.0])).astype(np.float32)[:3]
            vis = 0

        # 低维状态 [x,y,z,rx,ry,rz,g]
        x, y, z, rx, ry, rz = now_coords.tolist()
        g = 0.0
        lowdim = np.array([x, y, z, rx, ry, rz, g], dtype=np.float32)

        # 掩码 & 条件
        mask = detect_mask_bgr(frame, state["target_id"])
        onehot = np.zeros(3, np.float32)
        if isinstance(state["target_id"], str) and state["target_id"].upper() in TARGET_ID_MAP:
            idx = TARGET_ID_MAP[state["target_id"].upper()]
            onehot[idx] = 1.0

        # 命令快照
        cmd = snapshot_cmd()

        with lock:
            if state["recording"] and h5file is not None:
                buf["rgb"].append(frame[..., ::-1])  # RGB
                buf["mask"].append(mask)
                buf["state"].append(lowdim)
                buf["action_cam"].append(np.r_[dpos_cam, dang_cam, 0.0].astype(np.float32))
                buf["action_base"].append(np.r_[dpos_base, dang_base, 0.0].astype(np.float32))
                buf["vel_cam"].append(vel_cam.astype(np.float32))
                buf["vel_base"].append(vel_base.astype(np.float32))
                buf["joints"].append(now_angles.astype(np.float32))
                buf["joints_vel"].append(joints_vel.astype(np.float32))
                buf["T_be"].append(now_T_be.astype(np.float32))
                buf["T_bc"].append(now_T_bc.astype(np.float32))
                buf["target_cam"].append(p_cam.astype(np.float32))
                buf["target_base"].append(p_base.astype(np.float32))
                buf["target_vis"].append(np.int32(vis))
                buf["cond_target"].append(onehot)
                buf["cond_phase"].append({"idle":0,"selected":1,"confirmed":2}.get(state["phase"],0))
                buf["ts"].append(now_t)
                buf["dt"].append(dt)
                buf["cmd_type"].append(np.int32(cmd["type"]))
                buf["cmd_speed"].append(np.float32(cmd["speed"]))
                buf["cmd_coords"].append(cmd["coords"].astype(np.float32))
                buf["cmd_angles"].append(cmd["angles"].astype(np.float32))
                buf["cmd_gripper"].append(np.int32(cmd["gripper"]))
                buf["cmd_time"].append(np.float64(cmd["time"]))

        # 滚动
        prev_coords = now_coords
        prev_angles = now_angles
        prev_T_be   = now_T_be
        prev_T_bc   = now_T_bc
        time.sleep(0.001)

# ========= 夹爪（封装版） =========
def _open_gripper(sp=80):
    set_gripper_logged(0, sp)

def _close_gripper(sp=80):
    set_gripper_logged(1, sp)

# ========= demo_grasp（使用封装命令） =========
def _find_target_base_xyz(selected_id: int):
    cam.update_frame()
    frame = cam.color_frame()
    corners, ids, _ = stag.detectMarkers(frame, FIND_TAG_DICT_ID)
    if ids is None:
        return None
    ids = np.array(ids).flatten()
    idxs = np.where(ids == selected_id)[0]
    if len(idxs) == 0:
        return None
    idx = int(idxs[0])
    one_corners = [corners[idx]]
    one_ids = np.array([[selected_id]], dtype=np.int32)
    target_cam = detector.calc_markers_base_position(one_corners, one_ids)
    if target_cam is None or len(target_cam) < 3:
        return None

    end_coords = get_coords_mmdeg()
    T_be = T_from_coords_mmdeg(end_coords)
    p_cam  = np.array([target_cam[0], target_cam[1], target_cam[2], 1.0], dtype=float)
    p_base = (T_be @ T_ee_cam @ p_cam).flatten()[:3]
    return p_base, end_coords

def goto_observe(speed=60):
    try: mc.power_on()
    except Exception: pass
    time.sleep(0.3)
    send_angles_logged(OBS_POSE, speed)
    wait_stop(25.0)

# ========= 线程启动 =========
running = True
thr = threading.Thread(target=writer_loop, daemon=True)
thr.start()

# ========= API =========

@app.get("/health")
def api_health():
    return jsonify(ok=True, **state)

@app.get("/state")
def api_state():
    return jsonify(ok=True, target=state["target_id"], phase=state["phase"], recording=state["recording"])

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
        for k in buf: buf[k].clear()
        h5file = h5py.File(fname, "w")
        f = h5file
        # 文件级元数据（统一 mm/deg）
        f.attrs["target_id"]   = state["target_id"] or ""
        f.attrs["st_pos_unit"] = "mm"
        f.attrs["st_rot_unit"] = "deg"
        f.attrs["ac_pos_unit"] = "mm"
        f.attrs["ac_rot_unit"] = "deg"
        f.attrs["T_ee_cam_mm"] = T_ee_cam.astype(np.float32)
        f.attrs["success"]     = np.bool_(True)
        f.attrs["collision"]   = np.bool_(False)
        state["recording"] = True
    return jsonify(ok=True, file=fname)

@app.post("/record/stop")
def api_stop():
    """
    停录后：写盘 → （默认）松爪 → 回观察位
    可用 JSON 覆盖：{"release": false, "back_to_obs": false, "success": true, "collision": false}
    """
    d = request.get_json(silent=True) or {}
    do_release = d.get("release", True)
    do_back    = d.get("back_to_obs", True)
    success    = bool(d.get("success", True))
    collision  = bool(d.get("collision", False))

    global h5file
    with lock:
        state["recording"] = False
        if h5file is None:
            return jsonify(ok=False, msg="no open file"), 400

        f = h5file
        f.attrs["success"] = np.bool_(success)
        f.attrs["collision"] = np.bool_(collision)
        g = f.create_group("frames")

        def save(name, arr, dtype=None):
            a = np.asarray(arr, dtype=dtype) if dtype else np.asarray(arr)
            g.create_dataset(name, data=a, compression="gzip")

        # 图像
        save("images/rgb",  buf["rgb"],  np.uint8)
        save("images/mask", buf["mask"], np.uint8)

        # 末端状态/动作/速度
        save("state",         buf["state"],        np.float32)     # [x,y,z,rx,ry,rz,g] (mm/deg)
        save("action_cam",    buf["action_cam"],   np.float32)     # [dx,dy,dz,dα,dβ,dγ,dGrip]
        save("action_base",   buf["action_base"],  np.float32)
        save("vel_cam",       buf["vel_cam"],      np.float32)
        save("vel_base",      buf["vel_base"],     np.float32)

        # 关节
        save("joints/angles_deg", buf["joints"],     np.float32)   # [N,6]
        save("joints/vel_deg_s",  buf["joints_vel"], np.float32)   # [N,6]

        # 位姿矩阵
        save("poses/T_be_mm", buf["T_be"], np.float32)             # [N,4,4]
        save("poses/T_bc_mm", buf["T_bc"], np.float32)

        # 目标
        save("target/cam_mm",   buf["target_cam"],  np.float32)    # [N,3]
        save("target/base_mm",  buf["target_base"], np.float32)    # [N,3]
        save("target/visible",  buf["target_vis"],  np.int32)

        # 条件
        save("cond/target", buf["cond_target"], np.float32)
        save("cond/phase",  buf["cond_phase"],  np.int32)

        # 时间
        save("time/ts", buf["ts"], np.float64)
        save("time/dt", buf["dt"], np.float64)

        # 真实命令
        save("cmd/type",        buf["cmd_type"],   np.int32)
        save("cmd/speed",       buf["cmd_speed"],  np.float32)
        save("cmd/coords_mmdeg",buf["cmd_coords"], np.float32)
        save("cmd/angles_deg",  buf["cmd_angles"], np.float32)
        save("cmd/gripper",     buf["cmd_gripper"],np.int32)
        save("cmd/time",        buf["cmd_time"],   np.float64)

        path = h5file.filename
        h5file.close(); h5file = None

    if do_release:
        _open_gripper(sp=80)
        time.sleep(0.4)
    if do_back:
        goto_observe(speed=60)

    return jsonify(ok=True, file=path, released=do_release, back_to_obs=do_back)

@app.post("/goto_obs")
def api_goto_obs():
    sp = int((request.get_json(silent=True) or {}).get("speed", 60))
    goto_observe(sp)
    return jsonify(ok=True, pose=OBS_POSE)

# ========= 命令接口（可选，便于测试） =========
@app.post("/cmd/coords")
def api_cmd_coords():
    d = request.get_json(silent=True) or {}
    coords = d.get("coords", None)   # [x,y,z,rx,ry,rz] (mm/deg)
    sp = float(d.get("speed", 30))
    if not coords or len(coords) != 6:
        return jsonify(ok=False, msg="need coords[6]"), 400
    send_coords_logged(coords, sp)
    return jsonify(ok=True)

@app.post("/cmd/angles")
def api_cmd_angles():
    d = request.get_json(silent=True) or {}
    angles = d.get("angles", None)   # [j1..j6] deg
    sp = float(d.get("speed", 30))
    if not angles or len(angles) != 6:
        return jsonify(ok=False, msg="need angles[6]"), 400
    send_angles_logged(angles, sp)
    return jsonify(ok=True)

@app.post("/cmd/gripper")
def api_cmd_gripper():
    d = request.get_json(silent=True) or {}
    mode = int(d.get("mode", 1))     # 0=open, 1=close
    sp = float(d.get("speed", 80))
    set_gripper_logged(mode, sp)
    return jsonify(ok=True)

# ========= 抓取演示（会自动记录 cmd/*） =========
@app.post("/demo_grasp")
def api_demo_grasp():
    d = request.get_json(silent=True) or {}
    sp = int(d.get("speed", 30))

    # 解析目标 id
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

    grasp = [float(xyz_base[0]),
             float(xyz_base[1]),
             float(xyz_base[2] + GRIPPER_Z_OFFSET),
             float(rx), float(ry), float(rz)]
    above = grasp.copy();    above[2]    += Z_OFFSET
    approach = grasp.copy(); approach[2] += APPROACH_BUFFER
    lift = grasp.copy();     lift[2]     += LIFT_AFTER_GRASP

    detector.coord_limit(grasp); detector.coord_limit(above)
    detector.coord_limit(approach); detector.coord_limit(lift)

    try:
        mc.power_on(); time.sleep(0.3)
        _open_gripper()
        send_coords_logged(above,    sp); time.sleep(2)
        send_coords_logged(approach, sp); time.sleep(2)
        send_coords_logged(grasp,    max(15, sp//2)); time.sleep(2)
        _close_gripper()
        send_coords_logged(lift,     sp); time.sleep(2)
        # 你也可以在这里把 state['phase'] 设为 confirmed
        state["phase"] = "confirmed"
        return jsonify(ok=True)
    except Exception as e:
        return jsonify(ok=False, msg=str(e)), 500

# ========= 预览：PC 端 obs_viewer 用 =========
@app.get("/obs_npz")
def api_obs_npz():
    """
    返回一个 npz：rgba(HxWx4)、coords(mm/deg)
    """
    size = int(request.args.get("size", 384))
    tg   = request.args.get("target", "")
    try:
        cam.update_frame()
        frame = cam.color_frame()
        if frame is None:
            raise RuntimeError("no frame")
        coords = get_coords_mmdeg()
        mask = detect_mask_bgr(frame, tg or state.get("target_id"))
        rgb = frame[..., ::-1]  # BGR->RGB

        if size > 0:
            H, W = rgb.shape[:2]
            scale = size / max(H, W)
            nh, nw = int(H*scale), int(W*scale)
            rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

        rgba = np.dstack([rgb, mask]).astype(np.uint8)
        bio = io.BytesIO()
        np.savez_compressed(bio, rgba=rgba, coords=coords.astype(np.float32))
        bio.seek(0)
        return send_file(bio, mimetype="application/octet-stream",
                         as_attachment=False, download_name="obs.npz")
    except Exception:
        rgba = np.zeros((size, size, 4), np.uint8)
        bio = io.BytesIO()
        np.savez_compressed(bio, rgba=rgba, coords=np.zeros(6, np.float32))
        bio.seek(0)
        return send_file(bio, mimetype="application/octet-stream",
                         as_attachment=False, download_name="obs.npz")

# ========= 启动 =========
if __name__ == "__main__":
    # 上电后回到观测位（记录 cmd/*）
    try: mc.power_on()
    except Exception: pass
    time.sleep(0.3)
    send_angles_logged(OBS_POSE, 60)
    wait_stop(25.0)
    app.run(host="0.0.0.0", port=PORT, threaded=True)
