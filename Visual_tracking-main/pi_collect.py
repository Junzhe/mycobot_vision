from flask import Flask, request, jsonify, Response, render_template_string
from pathlib import Path
import time, threading, h5py, numpy as np, cv2, io
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

app = Flask(__name__)
state = {"target_id": None, "phase": "idle", "recording": False}

print("[INFO] 初始化机械臂与相机...")
mc = MyCobot280(PI_PORT, PI_BAUD)
time.sleep(0.5)
offset_j5 = -90 if mc.get_system_version() > 2 else 0
OBS_POSE = [-90, 5, -45, -40, 90 + offset_j5, 60]   # 观测位(关节角, deg)

def wait_stop(timeout=20.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if mc.is_moving() == 0:
                break
        except Exception:
            pass
        time.sleep(0.1)

_cmd_lock = threading.Lock()
_last_cmd = {"type": None, "time": 0.0, "payload": {}}

def _log_cmd(cmd_type, **payload):
    with _cmd_lock:
        _last_cmd["type"] = cmd_type
        _last_cmd["time"] = time.time()
        _last_cmd["payload"] = payload

def send_angles_logged(angles, speed):
    mc.send_angles(angles, speed)
    _log_cmd("send_angles", angles=list(map(float, angles)), speed=float(speed))

def send_coords_logged(coords, speed):
    mc.send_coords(coords, speed)
    _log_cmd("send_coords", coords=list(map(float, coords)), speed=float(speed))

def set_gripper_state_logged(state_val, speed):
    mc.set_gripper_state(state_val, speed)
    _log_cmd("gripper", state=int(state_val), speed=float(speed))

def goto_observe(speed=60):
    try:
        mc.power_on()
    except Exception:
        pass
    time.sleep(0.5)
    send_angles_logged(OBS_POSE, speed)  # deg
    wait_stop(25.0)

goto_observe()

# ====== 相机 / 手眼 ======
CAM_PATH = ROOT / "camera_params.npz"
EIH_PATH = ROOT / "EyesInHand_matrix.json"
camera_params = np.load(str(CAM_PATH))
mtx, dist = camera_params["mtx"], camera_params["dist"]
detector = camera_detect(0, 25, mtx, dist)  # cam_id=0, 25fps
if detector.EyesInHand_matrix is None and EIH_PATH.exists():
    detector.load_matrix(str(EIH_PATH))
T_ee_cam = detector.EyesInHand_matrix
assert T_ee_cam is not None, "未找到 EyesInHand_matrix.json，请先标定并放同目录"
cam = detector.camera

# ====== 工具：姿态/矩阵 ======
def get_ee_state_mmdeg():
    """返回 [x,y,z,rx,ry,rz]（mm/deg）。"""
    coords = mc.get_coords()
    while (coords is None) or (len(coords) < 6):
        time.sleep(0.01)
        coords = mc.get_coords()
    x, y, z, rx, ry, rz = coords
    return np.array([x, y, z, rx, ry, rz], dtype=np.float32)

def get_T_be_from_coords(coords):
    """输入 mm/deg，输出 4x4（平移单位 mm）。"""
    return detector.Transformation_matrix(coords)

def deltas_from_two_poses(prev_coords, now_coords):
    """
    返回：(cam_delta_mmdeg, base_delta_mmdeg, T_be_mm, T_bc_mm)
    - cam_delta: inv(T_bc_prev) @ T_bc_now   -> [dx,dy,dz(mm), dR,dP,dY(以rotvec分量转deg)]
    - base_delta: inv(T_be_prev) @ T_be_now  -> 同上
    旋转：先用旋转向量(rad)，再逐分量转为“deg 表达”（与mm/deg统一）。
    """
    T_be_prev = get_T_be_from_coords(prev_coords)  # mm
    T_be_now  = get_T_be_from_coords(now_coords)
    T_bc_prev = T_be_prev @ T_ee_cam
    T_bc_now  = T_be_now  @ T_ee_cam

    d_cam   = np.linalg.inv(T_bc_prev) @ T_bc_now
    d_base  = np.linalg.inv(T_be_prev) @ T_be_now

    # 平移：mm；旋转：rotvec(rad)->deg（按分量）
    def _to_mmdeg(dT):
        dp_mm = dT[:3, 3].astype(np.float64)
        rv_rad = R.from_matrix(dT[:3, :3]).as_rotvec().astype(np.float64)
        rv_deg = np.rad2deg(rv_rad)
        return np.r_[dp_mm, rv_deg].astype(np.float32)

    return _to_mmdeg(d_cam), _to_mmdeg(d_base), T_be_now.astype(np.float32), T_bc_now.astype(np.float32)

def detect_mask_bgr(frame_bgr, target_id_or_code):
    h, w = frame_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    if target_id_or_code is None:
        return mask
    if isinstance(target_id_or_code, str):
        tid = TARGET_ID_MAP.get(target_id_or_code.upper(), None)
    else:
        try:
            tid = int(target_id_or_code)
        except Exception:
            tid = None
    if tid is None:
        return mask
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
        k = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.dilate(mask, k, iterations=1)
    return mask

# ====== 采集缓冲（全量）======
buf = {
    "rgb": [], "mask": [],
    "state": [],                    # [x,y,z,rx,ry,rz,g]  (mm/deg)
    "action_cam": [], "action_base": [],      # Δpose (mm/deg) + dGrip
    "vel_cam": [], "vel_base": [],            # 速度 (mm/s, deg/s)
    "cond_target": [], "cond_phase": [],
    "ts": [], "dt": [],

    "joints_deg": [], "jvel_deg_s": [],       # 关节角/速度
    "T_be_mm": [], "T_bc_mm": [],             # 4x4
    "target_cam_mm": [], "target_base_mm": [],

    # 真实命令（逐帧快照）
    "cmd_type": [], "cmd_speed": [],
    "cmd_coords_mmdeg": [], "cmd_angles_deg": [],
    "cmd_gripper": [], "cmd_time": []
}
lock = threading.Lock()
h5file = None
running = True

# 给 /stream 的最新可视化帧
latest_overlay = None

def reset_buffers():
    for k in buf:
        buf[k].clear()

def _snapshot_last_cmd():
    """把最近一次控制命令快照化（单位统一 mm/deg）。"""
    with _cmd_lock:
        typ = _last_cmd["type"]
        payload = dict(_last_cmd["payload"])
        tcmd = float(_last_cmd["time"])
    typ_map = {"send_coords": 1, "send_angles": 2, "gripper": 3}
    code = np.int8(typ_map.get(typ, 0))
    speed = np.float32(payload.get("speed", np.nan))

    coords_mmdeg = np.full(6, np.nan, np.float32)
    if "coords" in payload:
        cc = payload["coords"]  # [x,y,z,rx,ry,rz] mm/deg
        if len(cc) >= 6:
            coords_mmdeg[:6] = np.array(cc[:6], np.float32)

    angles_deg = np.full(6, np.nan, np.float32)
    if "angles" in payload:
        aa = payload["angles"]
        if len(aa) >= 6:
            angles_deg[:6] = np.array(aa[:6], np.float32)

    grip = np.int8(payload.get("state", -1))  # -1=无, 0=open,1=close
    return code, speed, coords_mmdeg, angles_deg, grip, np.float64(tcmd)

def writer_loop():
    # 起始参考
    prev_coords = mc.get_coords()
    while prev_coords is None or len(prev_coords) < 6:
        time.sleep(0.01)
        prev_coords = mc.get_coords()

    prev_joints_deg = mc.get_angles() or [np.nan]*6
    t0_wall = time.time()
    last_ts_rel = 0.0

    while running:
        cam.update_frame()
        frame = cam.color_frame()  # BGR
        if frame is None:
            time.sleep(0.01); continue

        now_coords = mc.get_coords()
        if now_coords is None or len(now_coords) < 6:
            time.sleep(0.005); continue

        # Δpose（相机/基座） & 当前 4x4
        cam_d6, base_d6, T_be_now_mm, T_bc_now_mm = deltas_from_two_poses(prev_coords, now_coords)
        prev_coords = now_coords

        # 低维状态（mm/deg + g）
        st = get_ee_state_mmdeg()
        g = 0.0
        lowdim = np.r_[st, g].astype(np.float32)

        # 时间
        now_wall = time.time()
        ts_rel = now_wall - t0_wall
        dt_val = ts_rel - last_ts_rel if last_ts_rel > 0 else 1/30.0
        last_ts_rel = ts_rel
        dt_val = max(dt_val, 1e-3)

        # 关节角与角速度（deg/s）
        joints_deg = mc.get_angles() or [np.nan]*6
        joints_deg = np.array(joints_deg[:6], np.float32)
        prev_deg   = np.array(prev_joints_deg[:6], np.float32)
        jvel = (joints_deg - prev_deg) / dt_val
        prev_joints_deg = joints_deg

        # 线速度/角速度（mm/s, deg/s）
        vel_cam  = np.r_[cam_d6[:3]/dt_val,  cam_d6[3:]/dt_val].astype(np.float32)
        vel_base = np.r_[base_d6[:3]/dt_val, base_d6[3:]/dt_val].astype(np.float32)

        # 目标坐标（相机/基座，mm；不可见时 NaN）
        target_cam = np.full(3, np.nan, np.float32)
        target_base = np.full(3, np.nan, np.float32)
        try:
            corners, ids, _ = stag.detectMarkers(frame, 11)
            if ids is not None and state["target_id"] is not None:
                tid = TARGET_ID_MAP.get(str(state["target_id"]).upper(), None) \
                      if isinstance(state["target_id"], str) else int(state["target_id"])
                if tid is not None:
                    ids1 = np.array(ids).flatten()
                    idxs = np.where(ids1 == tid)[0]
                    if len(idxs) > 0:
                        one_c = [corners[int(idxs[0])]]
                        one_i = np.array([[tid]], dtype=np.int32)
                        cam_pos = detector.calc_markers_base_position(one_c, one_i)  # 相机系（常为 mm）
                        if cam_pos is not None and len(cam_pos) >= 3:
                            p_cam = np.array([cam_pos[0], cam_pos[1], cam_pos[2]], np.float64)  # mm
                            target_cam = p_cam.astype(np.float32)
                            p_cam_h = np.r_[p_cam, 1.0]
                            p_base_mm = (T_be_now_mm @ T_ee_cam @ p_cam_h)[:3]
                            target_base = p_base_mm.astype(np.float32)
        except Exception:
            pass

        # 掩码 & 条件
        mask = detect_mask_bgr(frame, state["target_id"])
        onehot = np.zeros(3, np.float32)
        if isinstance(state["target_id"], str) and state["target_id"].upper() in TARGET_ID_MAP:
            onehot[TARGET_ID_MAP[state["target_id"].upper()]] = 1.0

        # 实时预览叠加
        try:
            vis = frame.copy()
            corners, ids, _ = stag.detectMarkers(frame, 11)
            if ids is not None:
                for c in corners:
                    poly = np.asarray(c).reshape(-1, 2).astype(np.int32)
                    cv2.polylines(vis, [poly], True, (0, 255, 0), 2)
            if mask is not None and mask.sum() > 0:
                overlay = vis.copy()
                overlay[mask > 0] = (0, 0, 255)
                vis = cv2.addWeighted(vis, 0.8, overlay, 0.2, 0)
            h, w = vis.shape[:2]
            cv2.drawMarker(vis, (w//2, h//2), (255,255,255),
                           markerType=cv2.MARKER_CROSS, markerSize=18, thickness=1)
            global latest_overlay
            latest_overlay = vis
        except Exception:
            pass

        # 命令快照
        ctype, cspeed, ccoords, cangles, cgrip, ctime = _snapshot_last_cmd()

        # 写入缓冲
        with lock:
            if state["recording"] and h5file is not None:
                buf["rgb"].append(frame[..., ::-1])                 # RGB
                buf["mask"].append(mask)
                buf["state"].append(lowdim)                         # [mm/deg, g]

                buf["action_cam"].append(np.r_[cam_d6, 0.0])        # Δ + dGrip
                buf["action_base"].append(np.r_[base_d6, 0.0])
                buf["vel_cam"].append(vel_cam)
                buf["vel_base"].append(vel_base)

                buf["cond_target"].append(onehot)
                buf["cond_phase"].append({"idle":0,"selected":1,"confirmed":2}.get(state["phase"],0))
                buf["ts"].append(np.float64(ts_rel))
                buf["dt"].append(np.float32(dt_val))

                buf["joints_deg"].append(joints_deg.astype(np.float32))
                buf["jvel_deg_s"].append(jvel.astype(np.float32))
                buf["T_be_mm"].append(T_be_now_mm.astype(np.float32))
                buf["T_bc_mm"].append(T_bc_now_mm.astype(np.float32))
                buf["target_cam_mm"].append(target_cam)
                buf["target_base_mm"].append(target_base)

                buf["cmd_type"].append(ctype)
                buf["cmd_speed"].append(cspeed)
                buf["cmd_coords_mmdeg"].append(ccoords)
                buf["cmd_angles_deg"].append(cangles)
                buf["cmd_gripper"].append(cgrip)
                buf["cmd_time"].append(ctime)

        time.sleep(0.001)

thr = threading.Thread(target=writer_loop, daemon=True)
thr.start()

# ====== 夹爪工具 ======
def _open_gripper(sp=80):
    try: set_gripper_state_logged(0, sp)
    except Exception: pass

def _close_gripper(sp=80):
    try: set_gripper_state_logged(1, sp)
    except Exception: pass

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
        # 文件级元数据（单位声明）
        h5file.attrs["target_id"]   = state["target_id"] or ""
        h5file.attrs["st_pos_unit"] = "mm"
        h5file.attrs["st_rot_unit"] = "deg"
        h5file.attrs["ac_pos_unit"] = "mm"
        h5file.attrs["ac_rot_unit"] = "deg"
        # 标定快照（可选）
        h5file.attrs["camera_mtx"]  = mtx.astype(np.float32)
        h5file.attrs["camera_dist"] = dist.astype(np.float32)
        h5file.attrs["T_ee_cam_mm"] = T_ee_cam.astype(np.float32)
        state["recording"] = True
    return jsonify(ok=True, file=fname)

@app.post("/record/stop")
def api_stop():
    """
    停录后：写盘 → （默认）松爪 → 回观察位
    可用 JSON：
      {"release": false, "back_to_obs": false, "success": true/false, "collision": false/true}
    """
    d = request.get_json(silent=True) or {}
    do_release = d.get("release", True)
    do_back    = d.get("back_to_obs", True)
    success    = d.get("success", None)
    collision  = d.get("collision", None)

    global h5file
    with lock:
        state["recording"] = False
        if h5file is None:
            return jsonify(ok=False, msg="no open file")
        g = h5file.create_group("frames")

        def save(name, arr, dtype=None):
            a = np.asarray(arr, dtype=dtype) if dtype else np.asarray(arr)
            g.create_dataset(name, data=a, compression="gzip")

        # 图像与条件
        save("images/rgb",  buf["rgb"],  np.uint8)
        save("images/mask", buf["mask"], np.uint8)
        save("cond/target", buf["cond_target"], np.float32)
        save("cond/phase",  buf["cond_phase"], np.int32)

        # 状态/动作/速度/时间
        save("state",          buf["state"],        np.float32)  # [x,y,z,rx,ry,rz,g] (mm/deg)
        save("action_cam",     buf["action_cam"],   np.float32)  # Δ (mm/deg) + dGrip
        save("action_base",    buf["action_base"],  np.float32)
        save("vel_cam",        buf["vel_cam"],      np.float32)  # (mm/s, deg/s)
        save("vel_base",       buf["vel_base"],     np.float32)
        save("time/ts",        buf["ts"],           np.float64)
        save("time/dt",        buf["dt"],           np.float32)

        # 关节/变换/目标
        save("joints/angles_deg", buf["joints_deg"],  np.float32)
        save("joints/vel_deg_s",  buf["jvel_deg_s"],  np.float32)
        save("poses/T_be_mm",     buf["T_be_mm"],     np.float32)  # [N,4,4]
        save("poses/T_bc_mm",     buf["T_bc_mm"],     np.float32)
        save("target/cam_mm",     buf["target_cam_mm"],  np.float32)
        save("target/base_mm",    buf["target_base_mm"], np.float32)

        # 真实命令（逐帧快照）
        save("cmd/type",            buf["cmd_type"],           np.int8)     # 0/1/2/3
        save("cmd/speed",           buf["cmd_speed"],          np.float32)
        save("cmd/coords_mmdeg",    buf["cmd_coords_mmdeg"],   np.float32)  # [x,y,z,rx,ry,rz]
        save("cmd/angles_deg",      buf["cmd_angles_deg"],     np.float32)  # [6]
        save("cmd/gripper",         buf["cmd_gripper"],        np.int8)
        save("cmd/time",            buf["cmd_time"],           np.float64)

        # Episode 级标签
        if success is not None:   h5file.attrs["success"]   = bool(success)
        if collision is not None: h5file.attrs["collision"] = bool(collision)

        path = h5file.filename
        h5file.close(); h5file = None

    # —— 停录后的动作（先松爪再回观测位）——
    if do_release:
        _open_gripper(sp=80); time.sleep(0.5)
    if do_back:
        goto_observe(speed=60)

    return jsonify(ok=True, file=path, released=do_release, back_to_obs=do_back)

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
    while end_coords is None:
        time.sleep(0.01); end_coords = mc.get_coords()

    T_be = detector.Transformation_matrix(end_coords)  # mm
    p_cam  = np.array([target_cam[0], target_cam[1], target_cam[2], 1.0], dtype=float)
    p_base = (T_be @ T_ee_cam @ p_cam).flatten()[:3]   # mm
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
    rx, ry, rz = end_coords[3:6]  # deg

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
        mc.power_on(); time.sleep(0.5)
        _open_gripper()
        send_coords_logged(above,    sp); time.sleep(2)
        send_coords_logged(approach, sp); time.sleep(2)
        send_coords_logged(grasp,    max(15, sp//2)); time.sleep(2)
        _close_gripper()
        send_coords_logged(lift,     sp); time.sleep(2)
        return jsonify(ok=True)
    except Exception as e:
        return jsonify(ok=False, msg=str(e)), 500

# ====== 健康与只读查看接口 ======
@app.get("/health")
def api_health():
    return jsonify(ok=True, **state)

@app.get("/state")
def api_state():
    """最小状态：target 与当前末端 coords(mm/deg)"""
    try:
        coords = mc.get_coords() or []
    except Exception:
        coords = []
    return jsonify(ok=True, target=state.get("target_id"), coords=coords)

@app.get("/obs_npz")
def api_obs_npz():
    """
    返回一帧观察数据（压缩 npz）：
      - rgba: HxWx4 (RGB + mask 0/255)
      - coords: (6,) 末端位姿 mm/deg
    可选参数：?size=256&target=A/B/C/""/auto
    """
    size = int(request.args.get("size", 256))
    tg   = request.args.get("target", "")
    if tg == "auto":
        tg = state.get("target_id") or ""

    cam.update_frame()
    frame = cam.color_frame()  # BGR
    if frame is None:
        return jsonify(ok=False, msg="no frame"), 503

    mask = detect_mask_bgr(frame, tg)

    # 等比例缩放：短边 = size
    h, w = frame.shape[:2]
    if min(h, w) != size:
        if h <= w:
            nh, nw = size, int(size * w / h)
        else:
            nh, nw = int(size * h / w), size
        frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        mask  = cv2.resize(mask,  (nw, nh), interpolation=cv2.INTER_NEAREST)

    rgb  = frame[..., ::-1]                      # BGR->RGB
    rgba = np.dstack([rgb, mask.astype(np.uint8)])
    coords = np.array(mc.get_coords() or [0,0,0,0,0,0], dtype=np.float32)  # mm/deg

    bio = io.BytesIO()
    np.savez_compressed(bio, rgba=rgba.astype(np.uint8), coords=coords)
    bio.seek(0)
    return Response(bio.read(), mimetype="application/octet-stream")

def _mjpeg_generator():
    global latest_overlay
    while True:
        img = latest_overlay
        if img is None:
            time.sleep(0.03); continue
        ok, jpg = cv2.imencode('.jpg', img)
        if not ok:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')
        time.sleep(0.03)

@app.get("/stream")
def stream_overlay():
    return Response(_mjpeg_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

HTML_VIEW = """
<!doctype html>
<title>Robot Cam Preview</title>
<style>
body{margin:0;background:#111;color:#ddd;font-family:system-ui}
.wrap{display:flex;gap:12px;padding:12px}
.card{background:#1a1a1a;padding:12px;border-radius:10px}
img{max-width:100%;height:auto;border-radius:8px}
small{opacity:.7}
</style>
<div class="wrap">
  <div class="card">
    <h3>Overlay Stream</h3>
    <img src="/stream">
    <small>/stream（叠加已识别 STAG 边框与掩码）</small>
  </div>
</div>
"""
@app.get("/view")
def view():
    return render_template_string(HTML_VIEW)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, threaded=True)
