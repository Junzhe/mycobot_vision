import time, io, json, numpy as np, cv2, stag
from flask import Flask, request, jsonify, Response
from pymycobot import MyCobot280, PI_PORT, PI_BAUD
from camera_detect import camera_detect

PORT = 5066
TARGET_ID_MAP = {"A":0,"B":1,"C":2}

app = Flask(__name__)
current_target = None

print("[INFO] init robot & camera ...")
mc = MyCobot280(PI_PORT, PI_BAUD)
time.sleep(1.0)
try:
    mc.send_angles([-90, 5, -45, -40, 90, 50], 60)
    time.sleep(2.0)
except Exception:
    pass

cam_params = np.load("camera_params.npz")
mtx, dist = cam_params["mtx"], cam_params["dist"]
det = camera_detect(0, 25, mtx, dist)
assert det.EyesInHand_matrix is not None, "EyesInHand_matrix.json 未找到，请先手眼标定"
camera = det.camera

def latest_frame():
    camera.update_frame()
    return camera.color_frame()

def stag_mask(frame_bgr, target_code):
    tid = TARGET_ID_MAP.get((target_code or "").upper(), None)
    if tid is None: return np.zeros(frame_bgr.shape[:2], np.uint8)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = stag.detectMarkers(gray, 11)
    H,W = gray.shape
    mask = np.zeros((H,W), np.uint8)
    if ids is not None:
        for c,i in zip(corners, ids):
            if int(np.array(i).flatten()[0]) == tid:
                poly = np.asarray(c).reshape(-1,2).astype(np.int32)
                cv2.fillConvexPoly(mask, poly, 255)
                break
    return mask

@app.get("/health")
def api_health():
    c = mc.get_coords()
    return jsonify(ok=True, target=(current_target or ""), coords=c)

@app.post("/bci/target")
def api_bci_target():
    global current_target
    d = request.get_json(silent=True) or {}
    tg = (d.get("id") or d.get("target") or "").upper()
    if tg not in TARGET_ID_MAP: return jsonify(ok=False, msg="bad target"), 400
    current_target = tg
    return jsonify(ok=True, target=current_target)

@app.get("/state")
def api_state():
    return jsonify(ok=True, target=(current_target or ""))

@app.get("/eih")
def api_eih():
    return jsonify(ok=True, T_ee_cam=det.EyesInHand_matrix.tolist())

@app.get("/coords")
def api_coords():
    c = mc.get_coords()
    while (c is None) or (len(c)<6):
        time.sleep(0.01); c = mc.get_coords()
    return jsonify(ok=True, coords=c)

@app.post("/send_coords")
def api_send_coords():
    d = request.get_json(force=True)
    coords = d.get("coords", None)
    speed  = int(d.get("speed", 30))
    if (coords is None) or (len(coords)<6):
        return jsonify(ok=False, msg="need coords[6]"), 400
    # 夹紧 + z保底
    coords = list(map(float, coords))
    coords[0] = max(-350.0, min(350.0, coords[0]))
    coords[1] = max(-350.0, min(350.0, coords[1]))
    coords[2] = max(60.0,   coords[2])       # 底面安全
    try:
        det.coord_limit(coords)               # 你已有的边界函数
    except Exception:
        pass
    try:
        mc.send_coords(coords, speed)
        return jsonify(ok=True)
    except Exception as e:
        return jsonify(ok=False, msg=str(e)), 500

@app.post("/gripper")
def api_gripper():
    d = request.get_json(force=True)
    open_ = d.get("open", True)
    spd   = int(d.get("speed", 80))
    mc.set_gripper_state(0 if open_ else 1, spd)
    return jsonify(ok=True)

@app.get("/frame.jpg")
def api_frame():
    f = latest_frame()
    ok, buf = cv2.imencode(".jpg", f, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return Response(buf.tobytes(), mimetype="image/jpeg")

@app.get("/mask.png")
def api_mask():
    target = request.args.get("target", None) or current_target
    f = latest_frame()
    m = stag_mask(f, target)
    ok, buf = cv2.imencode(".png", m)
    return Response(buf.tobytes(), mimetype="image/png")

@app.get("/obs_npz")
def api_obs_npz():
    target = (request.args.get("target") or (current_target or "")).upper()
    size   = int(request.args.get("size", 128))
    f = latest_frame()
    m = stag_mask(f, target)
    rgb = cv2.resize(f[..., ::-1], (size,size), cv2.INTER_AREA)
    m   = cv2.resize(m, (size,size), cv2.INTER_NEAREST)
    rgba = np.dstack([rgb.astype(np.uint8), m.astype(np.uint8)])
    c = mc.get_coords()
    bio = io.BytesIO()
    np.savez_compressed(bio, rgba=rgba, coords=np.array(c, np.float32))
    bio.seek(0)
    return Response(bio.read(), mimetype="application/octet-stream")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, threaded=True)
