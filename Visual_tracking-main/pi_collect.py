from flask import Flask, request, jsonify, Response
from pathlib import Path
import time, threading, io, json
import h5py, numpy as np, cv2
from scipy.spatial.transform import Rotation as R
from pymycobot import MyCobot280, PI_PORT, PI_BAUD
from camera_detect import camera_detect
import stag

# ========= 路径/目录 =========
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
H5_DIR = str(DATA_DIR)

# ========= 配置 =========
PORT = 5055
TARGET_ID_MAP = {"A": 0, "B": 1, "C": 2}

# 观察位
OBS_POSE_BASE = [-90, 5, -45, -40, 90, 50]   
GRASP_LOCK_OFFSET_MM = np.array([0.0, 0.0, -100.0], dtype=np.float32)

# 末端-相机外参存放路径
CAM_PATH = ROOT / "camera_params.npz"
EIH_PATH = ROOT / "EyesInHand_matrix.json"

# ========= 全局状态 =========
app = Flask(__name__)
state = {
    "target_id": None,           # "A"|"B"|"C"|None
    "phase": "idle",             # "idle"|"selected"|"approach"|"confirmed"
    "recording": False,
    "locked": False,             # 是否已抓取后锁定
    "last_cmd": None,            # 最近一次命令快照（字典）
    "gripper_closed": False,     # 最近一次夹爪状态（0/1）
}

# ========= 初始化机械臂/相机 =========
print("[INFO] 初始化机械臂与相机...")
mc = MyCobot280(PI_PORT, PI_BAUD)
time.sleep(0.5)
try:
    offset_j5 = -90 if mc.get_system_version() > 2 else 0
except Exception:
    offset_j5 = 0
OBS_POSE = OBS_POSE_BASE.copy()
OBS_POSE[4] = OBS_POSE[4] + offset_j5

def wait_stop(timeout=20.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if mc.is_moving() == 0:
                break
        except Exception:
            pass
        time.sleep(0.05)

def goto_observe(speed=60):
    try:
        mc.power_on()
    except Exception:
        pass
    time.sleep(0.3)
    mc.send_angles(OBS_POSE, speed)
    wait_stop(25.0)

# 启动回到观测位
goto_observe()

# 相机与手眼
camera_params = np.load(str(CAM_PATH))
mtx, dist = camera_params["mtx"], camera_params["dist"]
# 第二个参数是曝光/增益/帧率相关控制，按你项目需求设置；这里用 25
detector = camera_detect(0, 25, mtx, dist)
if detector.EyesInHand_matrix is None and EIH_PATH.exists():
    detector.load_matrix(str(EIH_PATH))
T_ee_cam = detector.EyesInHand_matrix
assert T_ee_cam is not None, "未找到 EyesInHand_matrix.json，请先完成手眼标定并放在同目录"
T_ee_cam = np.array(T_ee_cam, dtype=np.float32).reshape(4,4)
cam = detector.camera

# ========= 常用工具 =========
def get_coords_blocking():
    c = mc.get_coords()
    while (c is None) or (len(c) < 6):
        time.sleep(0.01)
        c = mc.get_coords()
    return c  # [x,y,z,rx,ry,rz] mm/deg

def get_angles_blocking():
    a = mc.get_angles()
    while (a is None) or (len(a) < 6):
        time.sleep(0.01)
        a = mc.get_angles()
    return a  # [j1..j6], deg

def trans_from_coords_mmdeg(coords):
    """输入 coords=[x,y,z,rx,ry,rz] (mm/deg) → 4x4 齐次（平移单位 mm）"""
    return detector.Transformation_matrix(coords)  # 已是 mm/deg→4x4(mm)

def stag_mask(frame_bgr, wanted_id):
    h, w = frame_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    if wanted_id is None:
        return mask
    if isinstance(wanted_id, str):
        tid = TARGET_ID_MAP.get(wanted_id.upper(), None)
    else:
        tid = int(wanted_id)
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

def stag_target_cam(frame_bgr, wanted_id):
    """返回 (pos_cam_mm[3], visible:bool, mask)"""
    mask = stag_mask(frame_bgr, wanted_id)
    try:
        corners, ids, _ = stag.detectMarkers(frame_bgr, 11)
        if ids is None:
            return (np.full(3, np.nan, np.float32), False, mask)
        ids = np.array(ids).flatten()
        tid = TARGET_ID_MAP.get(wanted_id, None) if isinstance(wanted_id, str) else wanted_id
        if tid is None:
            return (np.full(3, np.nan, np.float32), False, mask)
        idxs = np.where(ids == tid)[0]
        if len(idxs) == 0:
            return (np.full(3, np.nan, np.float32), False, mask)
        idx = int(idxs[0])
        one_corners = [corners[idx]]
        one_ids = np.array([[tid]], dtype=np.int32)
        pos_cam = detector.calc_markers_base_position(one_corners, one_ids)  # 相机系 3d(mm)
        if pos_cam is None or len(pos_cam) < 3:
            return (np.full(3, np.nan, np.float32), False, mask)
        return (np.array(pos_cam[:3], dtype=np.float32), True, mask)
    except Exception:
        return (np.full(3, np.nan, np.float32), False, mask)

def se3_delta(T_prev, T_now):
    """Δ = inv(T_prev) @ T_now → [dx,dy,dz, dRx,dRy,dRz] (mm, rad)"""
    Td = np.linalg.inv(T_prev) @ T_now
    dp = Td[:3, 3]
    dw = R.from_matrix(Td[:3, :3]).as_rotvec()
    return np.r_[dp, dw].astype(np.float32)

# ========= 采集缓存 =========
buf = {
    "images/rgb": [],       # [N,H,W,3] uint8 RGB
    "images/mask": [],      # [N,H,W]   uint8 (0/255)
    "state": [],            # [N,7]  [x,y,z,rx,ry,rz,g] (mm/deg + gripper)
    "time/ts": [],          # [N]  float64
    "time/dt": [],          # [N]  float64
    # joints
    "joints/angles_deg": [],    # [N,6]
    "joints/vel_deg_s": [],     # [N,6]
    # poses
    "poses/T_be_mm": [],        # [N,4,4]
    "poses/T_bc_mm": [],        # [N,4,4]
    # actions (cam/base Δpose) + velocities
    "action_cam": [],           # [N,7]  dx,dy,dz(mm),dRx,dRy,dRz(rad), dGrip
    "action_base": [],          # [N,7]
    "vel_cam": [],              # [N,7]  ≈ action/dt
    "vel_base": [],             # [N,7]
    # command snapshots per-frame
    "cmd/type": [],             # [N] int {0:none,1:coords,2:angles,3:gripper}
    "cmd/speed": [],            # [N] float
    "cmd/coords_mmdeg": [],     # [N,6]  last send_coords goal
    "cmd/angles_deg": [],       # [N,6]  last send_angles goal
    "cmd/gripper": [],          # [N] int -1|0|1
    "cmd/time": [],             # [N] float (monotonic)
    # condition
    "cond/target": [],          # [N,3] onehot
    "cond/phase": [],           # [N] int {0:idle,1:selected,2:approach,3:confirmed}
    # target
    "target/cam_mm": [],        # [N,3]  may be NaN
    "target/base_mm": [],       # [N,3]  hold/lock策略后
    "target/visible": [],       # [N] uint8 0/1
}
lock = threading.Lock()
h5file = None
running = True

# 运行中变量（上一帧）
last = {
    "ts": None,                 # monotonic
    "coords": None,             # mm/deg
    "angles": None,             # deg
    "T_be": None,               # 4x4 mm
    "T_bc": None,               # 4x4 mm
    "jvel": np.zeros(6, np.float32),
    "grip": 0.0,
    "target_base_hold": None,   # 抓取前不可见时的静态保持（基座系）
}

def reset_buffers():
    for k in buf:
        buf[k].clear()

def snapshot_cmd(now_t):
    """将 state['last_cmd'] 快照到 buf 的 cmd/*（每帧都填）"""
    lc = state.get("last_cmd") or {}
    ctype = int(lc.get("type", 0))
    speed = float(lc.get("speed", np.nan)) if ctype in (1,2) else (float(lc.get("speed", np.nan)) if ctype==3 else np.nan)
    coords = np.array(lc.get("coords_mmdeg", [np.nan]*6), dtype=np.float32)
    angles = np.array(lc.get("angles_deg", [np.nan]*6), dtype=np.float32)
    grip   = int(lc.get("gripper", -1))
    ctime  = float(lc.get("time", now_t))
    buf["cmd/type"].append(ctype)
    buf["cmd/speed"].append(speed)
    buf["cmd/coords_mmdeg"].append(coords)
    buf["cmd/angles_deg"].append(angles)
    buf["cmd/gripper"].append(grip)
    buf["cmd/time"].append(ctime)

def phase_int(p):
    return {"idle":0,"selected":1,"approach":2,"confirmed":3}.get(p,0)

def writer_loop():
    global h5file
    # 初始化上一帧
    last["coords"] = get_coords_blocking()
    last["angles"] = np.array(get_angles_blocking(), dtype=np.float32)
    last["T_be"]   = trans_from_coords_mmdeg(last["coords"])
    last["T_bc"]   = last["T_be"] @ T_ee_cam
    last["ts"]     = time.monotonic()

    t0 = time.monotonic()
    while running:
        cam.update_frame()
        frame = cam.color_frame()
        if frame is None:
            time.sleep(0.003); continue

        # 读取当前状态
        now_coords = get_coords_blocking()
        now_angles = np.array(get_angles_blocking(), dtype=np.float32)
        T_be = trans_from_coords_mmdeg(now_coords)
        T_bc = T_be @ T_ee_cam

        now_t = time.monotonic()
        dt = max(1e-4, now_t - (last["ts"] or now_t))

        # 关节速度（差分）
        jvel = (now_angles - last["angles"]) / dt

        # 动作：相机/基座系 Δpose（上一帧→当前帧）
        d_cam  = se3_delta(last["T_bc"], T_bc)
        d_base = se3_delta(last["T_be"], T_be)

        # 末端低维（mm/deg + gripper）
        g = 1.0 if state.get("gripper_closed", False) else 0.0
        st = np.r_[now_coords, g].astype(np.float32)

        # 目标观测与 hold/lock
        tid = state.get("target_id")
        pos_cam, visible, mask = stag_target_cam(frame, tid)

        # 计算 target/base_mm
        if not state.get("locked", False):
            # 抓取前：可见→投到基座；不可见→静态保持
            if visible:
                p_cam_h = np.r_[pos_cam.astype(np.float32), 1.0]
                p_base = (T_be @ T_ee_cam @ p_cam_h)[:3].astype(np.float32)
                last["target_base_hold"] = p_base.copy()
            else:
                # 如果未可见，保持上一帧基座位置（桌面假设静止）
                if last["target_base_hold"] is None:
                    # 初始不可见时，置 NaN
                    p_base = np.full(3, np.nan, np.float32)
                else:
                    p_base = last["target_base_hold"].copy()

            # 触发锁定：phase>=confirmed 或者 最近一次夹爪命令=闭合
            if phase_int(state.get("phase","idle")) >= 3 or state.get("gripper_closed", False):
                state["locked"] = True
        else:
            # 抓取后锁定：目标 = 法兰原点 + R_be @ 偏移(末端坐标系)
            R_be = T_be[:3,:3].astype(np.float32)
            t_be = T_be[:3, 3].astype(np.float32)
            p_base = (t_be + R_be @ GRASP_LOCK_OFFSET_MM).astype(np.float32)
            # 抓取后，pos_cam 多半不可见，允许 pos_cam=NaN
            visible = False
            pos_cam = np.full(3, np.nan, np.float32)

        # 相机/基座“速度”（≈ Δ/dt）
        v_cam  = np.r_[d_cam[:6] / dt, 0.0].astype(np.float32)
        v_base = np.r_[d_base[:6] / dt, 0.0].astype(np.float32)

        # 条件 onehot
        onehot = np.zeros(3, np.float32)
        if isinstance(tid, str) and tid.upper() in TARGET_ID_MAP:
            onehot[TARGET_ID_MAP[tid.upper()]] = 1.0

        # 组包写缓存
        with lock:
            if state.get("recording", False) and (h5file is not None):
                # 图像
                buf["images/rgb"].append(frame[..., ::-1].astype(np.uint8))  # RGB
                buf["images/mask"].append(mask.astype(np.uint8))
                # 末端+时间
                buf["state"].append(st)
                buf["time/ts"].append(now_t - t0)
                buf["time/dt"].append(dt)
                # joints/poses
                buf["joints/angles_deg"].append(now_angles)
                buf["joints/vel_deg_s"].append(jvel.astype(np.float32))
                buf["poses/T_be_mm"].append(T_be.astype(np.float32))
                buf["poses/T_bc_mm"].append(T_bc.astype(np.float32))
                # action/vel
                buf["action_cam"].append(np.r_[d_cam, 0.0].astype(np.float32))
                buf["action_base"].append(np.r_[d_base, 0.0].astype(np.float32))
                buf["vel_cam"].append(v_cam)
                buf["vel_base"].append(v_base)
                # cmd 快照
                snapshot_cmd(now_t)
                # 条件与目标
                buf["cond/target"].append(onehot)
                buf["cond/phase"].append(phase_int(state.get("phase","idle")))
                buf["target/cam_mm"].append(pos_cam.astype(np.float32))
                buf["target/base_mm"].append(p_base.astype(np.float32))
                buf["target/visible"].append(1 if visible else 0)

        # 滚动上一帧
        last["ts"] = now_t
        last["coords"] = now_coords
        last["angles"] = now_angles
        last["T_be"]   = T_be
        last["T_bc"]   = T_bc
        last["jvel"]   = jvel.astype(np.float32)

        time.sleep(0.001)

thr = threading.Thread(target=writer_loop, daemon=True)
thr.start()

# ========= 夹爪封装 =========
def _open_gripper(sp=80):
    try:
        mc.set_gripper_state(0, sp)
        state["gripper_closed"] = False
    except Exception:
        pass

def _close_gripper(sp=80):
    try:
        mc.set_gripper_state(1, sp)
        state["gripper_closed"] = True
    except Exception:
        pass

def record_cmd_snapshot(cmd_type, speed=None, coords=None, angles=None, grip=None):
    state["last_cmd"] = {
        "type": cmd_type,           # 0/1/2/3
        "speed": speed if speed is not None else np.nan,
        "coords_mmdeg": coords if coords is not None else [np.nan]*6,
        "angles_deg": angles if angles is not None else [np.nan]*6,
        "gripper": grip if grip is not None else -1,
        "time": time.monotonic(),
    }

# ========= API：健康/状态/相位 =========
@app.get("/health")
def api_health():
    return jsonify(ok=True, target_id=state["target_id"], phase=state["phase"],
                   recording=state["recording"], locked=state["locked"])

@app.get("/state")
def api_state():
    # 简单状态给 PC viewer 用
    return jsonify(ok=True, target=state.get("target_id"), phase=state.get("phase"),
                   locked=state.get("locked"), gripper=int(state.get("gripper_closed", False)))

@app.post("/bci/target")
def api_target():
    d = request.get_json(silent=True) or request.form
    tid = d.get("id", None)
    ph  = d.get("phase", "selected")
    if tid is None:
        return jsonify(ok=False, msg="need id (A/B/C)"), 400
    state["target_id"] = tid
    state["phase"] = ph
    if ph in ("idle","selected"):
        state["locked"] = False
    return jsonify(ok=True, target=state["target_id"], phase=state["phase"])

@app.post("/bci/phase")
def api_phase():
    d = request.get_json(silent=True) or request.form
    ph = d.get("phase", None)
    if ph is None:
        return jsonify(ok=False, msg="need phase"), 400
    state["phase"] = ph
    if phase_int(ph) < 3:
        state["locked"] = False
    return jsonify(ok=True, phase=state["phase"], locked=state["locked"])

# ========= 录制 =========
@app.post("/record/start")
def api_start():
    global h5file
    with lock:
        reset_buffers()
        fname = f"{H5_DIR}/epi_{int(time.time())}.hdf5"
        h5file = h5py.File(fname, "w")
        # 文件级属性（单位）
        h5file.attrs["target_id"]   = (state["target_id"] or "").encode("utf-8")
    # 单位标注（统一 mm/deg）
    h5file.attrs["st_pos_unit"] = "mm"
    h5file.attrs["st_rot_unit"] = "deg"
    h5file.attrs["ac_pos_unit"] = "mm"
    h5file.attrs["ac_rot_unit"] = "rad"   # rotvec Δ 的单位
    h5file.attrs["T_ee_cam_mm"] = T_ee_cam.astype(np.float32)
    h5file.attrs["GRASP_LOCK_OFFSET_MM"] = GRASP_LOCK_OFFSET_MM.astype(np.float32)
    state["recording"] = True
    state["locked"] = (phase_int(state["phase"]) >= 3) or state.get("gripper_closed", False)
    return jsonify(ok=True, file=h5file.filename)

@app.post("/record/stop")
def api_stop():
    """停止录制并写盘。可附带元标签 success/collision 与动作（松爪/回观察位）"""
    d = request.get_json(silent=True) or {}
    do_release = bool(d.get("release", True))
    do_back    = bool(d.get("back_to_obs", True))
    success    = d.get("success", True)
    collision  = d.get("collision", False)

    global h5file
    with lock:
        state["recording"] = False
        if h5file is None:
            return jsonify(ok=False, msg="no open file")
        g = h5file.create_group("frames")

        def save(name, arr, dtype=None):
            a = np.asarray(arr, dtype=dtype) if dtype else np.asarray(arr)
            g.create_dataset(name, data=a, compression="gzip")

        # 图像
        save("images/rgb",  buf["images/rgb"],  np.uint8)
        save("images/mask", buf["images/mask"], np.uint8)

        # 时间
        save("time/ts", np.asarray(buf["time/ts"], dtype=np.float64))
        save("time/dt", np.asarray(buf["time/dt"], dtype=np.float64))

        # 末端低维
        save("state", np.asarray(buf["state"], dtype=np.float32))

        # joints/poses
        save("joints/angles_deg", np.asarray(buf["joints/angles_deg"], dtype=np.float32))
        save("joints/vel_deg_s",  np.asarray(buf["joints/vel_deg_s"], dtype=np.float32))
        save("poses/T_be_mm",     np.asarray(buf["poses/T_be_mm"], dtype=np.float32))
        save("poses/T_bc_mm",     np.asarray(buf["poses/T_bc_mm"], dtype=np.float32))

        # action/vel
        save("action_cam", np.asarray(buf["action_cam"], dtype=np.float32))
        save("action_base",np.asarray(buf["action_base"], dtype=np.float32))
        save("vel_cam",    np.asarray(buf["vel_cam"], dtype=np.float32))
        save("vel_base",   np.asarray(buf["vel_base"], dtype=np.float32))

        # cmd/*
        save("cmd/type",   np.asarray(buf["cmd/type"], dtype=np.int32))
        save("cmd/speed",  np.asarray(buf["cmd/speed"], dtype=np.float32))
        save("cmd/coords_mmdeg", np.asarray(buf["cmd/coords_mmdeg"], dtype=np.float32))
        save("cmd/angles_deg",   np.asarray(buf["cmd/angles_deg"], dtype=np.float32))
        save("cmd/gripper", np.asarray(buf["cmd/gripper"], dtype=np.int32))
        save("cmd/time",    np.asarray(buf["cmd/time"], dtype=np.float64))

        # 条件/目标
        save("cond/target", np.asarray(buf["cond/target"], dtype=np.float32))
        save("cond/phase",  np.asarray(buf["cond/phase"], dtype=np.int32))
        save("target/cam_mm",   np.asarray(buf["target/cam_mm"], dtype=np.float32))
        save("target/base_mm",  np.asarray(buf["target/base_mm"], dtype=np.float32))
        save("target/visible",  np.asarray(buf["target/visible"], dtype=np.uint8))

        # 文件级元标签
        h5file.attrs["success"]   = bool(success)
        h5file.attrs["collision"] = bool(collision)

        path = h5file.filename
        h5file.close(); h5file = None

    # 后处理动作
    if do_release:
        _open_gripper(sp=80); time.sleep(0.3)
    if do_back:
        goto_observe(speed=60)

    return jsonify(ok=True, file=path, released=do_release, back_to_obs=do_back,
                   success=bool(success), collision=bool(collision))

@app.post("/goto_obs")
def api_goto_obs():
    sp = int((request.get_json(silent=True) or {}).get("speed", 60))
    goto_observe(sp)
    return jsonify(ok=True, pose=OBS_POSE)

# ========= 命令端点（务必使用这些端点才能记录到 cmd/*） =========
@app.post("/cmd/coords")
def api_cmd_coords():
    d = request.get_json(silent=True) or request.form
    try:
        x = float(d.get("x")); y=float(d.get("y")); z=float(d.get("z"))
        rx=float(d.get("rx")); ry=float(d.get("ry")); rz=float(d.get("rz"))
        sp = int(d.get("speed", 30))
    except Exception:
        return jsonify(ok=False, msg="need x,y,z,rx,ry,rz"), 400
    coords = [x,y,z,rx,ry,rz]
    record_cmd_snapshot(1, speed=sp, coords=coords)
    try:
        mc.power_on(); time.sleep(0.2)
        mc.send_coords(coords, sp)
    except Exception as e:
        return jsonify(ok=False, msg=str(e)), 500
    return jsonify(ok=True)

@app.post("/cmd/angles")
def api_cmd_angles():
    d = request.get_json(silent=True) or request.form
    try:
        j = [float(d.get(f"j{k+1}")) for k in range(6)]
        sp = int(d.get("speed", 30))
    except Exception:
        return jsonify(ok=False, msg="need j1..j6"), 400
    record_cmd_snapshot(2, speed=sp, angles=j)
    try:
        mc.power_on(); time.sleep(0.2)
        mc.send_angles(j, sp)
    except Exception as e:
        return jsonify(ok=False, msg=str(e)), 500
    return jsonify(ok=True)

@app.post("/cmd/gripper")
def api_cmd_gripper():
    d = request.get_json(silent=True) or request.form
    try:
        st = int(d.get("state"))  # 0 open / 1 close
        sp = int(d.get("speed", 80))
    except Exception:
        return jsonify(ok=False, msg="need state 0|1"), 400
    record_cmd_snapshot(3, speed=sp, grip=st)
    try:
        if st == 1:
            _close_gripper(sp)
        else:
            _open_gripper(sp)
    except Exception as e:
        return jsonify(ok=False, msg=str(e)), 500
    return jsonify(ok=True, gripper=st)

# ========= demo_grasp（示例 IK 抓取：只做抓，不放回） =========
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

    end_coords = get_coords_blocking()
    T_be = trans_from_coords_mmdeg(end_coords)
    p_cam  = np.array([target_cam[0], target_cam[1], target_cam[2], 1.0], dtype=float)
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

    grasp    = [float(xyz_base[0]), float(xyz_base[1]), float(xyz_base[2] + GRIPPER_Z_OFFSET), float(rx), float(ry), float(rz)]
    above    = grasp.copy();    above[2]    += Z_OFFSET
    approach = grasp.copy();    approach[2] += APPROACH_BUFFER
    lift     = grasp.copy();    lift[2]     += LIFT_AFTER_GRASP

    detector.coord_limit(grasp); detector.coord_limit(above)
    detector.coord_limit(approach); detector.coord_limit(lift)

    try:
        mc.power_on(); time.sleep(0.2)
        # 打开夹爪
        record_cmd_snapshot(3, speed=80, grip=0); _open_gripper(80)
        # 上到 above
        record_cmd_snapshot(1, speed=sp, coords=above);    mc.send_coords(above, sp);    time.sleep(2)
        # 到 approach
        record_cmd_snapshot(1, speed=sp, coords=approach); mc.send_coords(approach, sp); time.sleep(2)
        # 下到 grasp
        record_cmd_snapshot(1, speed=max(15, sp//2), coords=grasp); mc.send_coords(grasp, max(15, sp//2)); time.sleep(2)
        # 闭合夹爪
        record_cmd_snapshot(3, speed=80, grip=1); _close_gripper(80)
        # 抬升
        record_cmd_snapshot(1, speed=sp, coords=lift);     mc.send_coords(lift, sp);     time.sleep(2)
        # 标记阶段：已抓取
        state["phase"] = "confirmed"
        state["locked"] = True
        return jsonify(ok=True)
    except Exception as e:
        return jsonify(ok=False, msg=str(e)), 500

# ========= 预览（给 PC 端 obs_viewer.py 使用） =========
@app.get("/obs_npz")
def api_obs_npz():
    try:
        size = int(request.args.get("size", 256))
        tg   = request.args.get("target", "") or ""
    except Exception:
        size = 256; tg = ""
    cam.update_frame()
    frame = cam.color_frame()
    if frame is None:
        # 返回空
        pack = io.BytesIO()
        np.savez_compressed(pack, rgba=np.zeros((size,size,4), np.uint8), coords=np.zeros(6,np.float32))
        pack.seek(0)
        return Response(pack.getvalue(), mimetype="application/octet-stream")

    if tg not in ("A","B","C"): tg = state.get("target_id") or ""
    mask = stag_mask(frame, tg if tg else None)
    rgb  = frame[..., ::-1]
    H, W = rgb.shape[:2]
    if H != size or W != size:
        # 等比最小边缩放到 size
        scale = size / max(H, W)
        nh, nw = int(H * scale), int(W * scale)
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
        pad_h, pad_w = size - nh, size - nw
        rgb  = cv2.copyMakeBorder(rgb, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    rgba = np.dstack([rgb, mask]).astype(np.uint8)
    coords = np.array(get_coords_blocking(), dtype=np.float32)
    pack = io.BytesIO()
    np.savez_compressed(pack, rgba=rgba, coords=coords)
    pack.seek(0)
    return Response(pack.getvalue(), mimetype="application/octet-stream")

# ========= 主程序 =========
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, threaded=True)
