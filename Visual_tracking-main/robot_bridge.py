# 5066robot_bridge.py — Pi 端：接收 HTTP 命令并驱动机械臂
import time, io, numpy as np, cv2, stag
from flask import Flask, request, jsonify, Response
from pymycobot import MyCobot280, PI_PORT, PI_BAUD
from camera_detect import camera_detect

PORT = 5066
TARGET_ID_MAP = {"A":0,"B":1,"C":2}

app = Flask(__name__)
current_target = None

print("[INFO] init robot & camera ...")
mc = MyCobot280(PI_PORT, PI_BAUD); time.sleep(1.0)
try:
    mc.power_on(); time.sleep(0.3)
    mc.send_angles([-90, 5, -45, -40, 0, 60], 40); time.sleep(2.0)
except Exception:
    pass

cam_params = np.load("camera_params.npz")
mtx, dist = cam_params["mtx"], cam_params["dist"]
det = camera_detect(0, 40, mtx, dist)
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
    H,W = gray.shape; mask = np.zeros((H,W), np.uint8)
    if ids is not None:
        for c,i in zip(corners, ids):
            if int(np.array(i).flatten()[0]) == tid:
                poly = np.asarray(c).reshape(-1,2).astype(np.int32)
                cv2.fillConvexPoly(mask, poly, 255); break
    return mask

@app.get("/health")
def api_health(): return jsonify(ok=True)

@app.post("/bci/target")
def api_bci_target():
    global current_target
    d = request.get_json(silent=True) or {}
    tg = (d.get("id") or d.get("target") or "").upper()
    if tg not in TARGET_ID_MAP: return jsonify(ok=False, msg="bad target"), 400
    current_target = tg
    return jsonify(ok=True, target=current_target)

@app.get("/state")
def api_state(): return jsonify(ok=True, target=(current_target or ""))

@app.get("/eih")
def api_eih():
    return jsonify(ok=True, T_ee_cam=det.EyesInHand_matrix.tolist())

@app.get("/coords")
def api_coords():
    c = mc.get_coords()
    while (c is None) or (len(c) < 6):
        time.sleep(0.01); c = mc.get_coords()
    return jsonify(ok=True, coords=c)

@app.post("/send_coords")
def api_send_coords():
    d = request.get_json(force=True)
    coords = d.get("coords", None)
    speed  = int(d.get("speed", 30))
    if (coords is None) or (len(coords) < 6):
        return jsonify(ok=False, msg="need coords[6]"), 400
    det.coord_limit(coords)
    try:
        mc.power_on(); time.sleep(0.2)
    except Exception:
        pass
    print("[MOVE] ->", [round(v,2) for v in coords[:6]], " sp=", speed)
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
    try:
        mc.power_on(); time.sleep(0.2)
    except Exception:
        pass
    mc.set_gripper_state(0 if open_ else 1, spd)
    print("[GRIP]", "open" if open_ else "close", " sp=", spd)
    return jsonify(ok=True)

@app.get("/frame.jpg")
def api_frame():
    f = latest_frame()
    ok, buf = cv2.imencode(".jpg", f, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return Response(buf.tobytes(), mimetype="image/jpeg")

@app.get("/mask.png")
def api_mask():
    target = request.args.get("target", None) or current_target
    f = latest_frame(); m = stag_mask(f, target)
    ok, buf = cv2.imencode(".png", m)
    return Response(buf.tobytes(), mimetype="image/png")

@app.get("/obs_npz")
def api_obs_npz():
    target = (request.args.get("target") or (current_target or "")).upper()
    try: size = int(request.args.get("size", 128))
    except Exception: size = 128
    f = latest_frame()
    m = stag_mask(f, target)
    rgb = cv2.resize(f[..., ::-1], (size,size), cv2.INTER_AREA)   # BGR->RGB
    m   = cv2.resize(m, (size,size), cv2.INTER_NEAREST)
    rgba = np.dstack([rgb.astype(np.uint8), m.astype(np.uint8)])
    c = mc.get_coords() or [-90,5,-45,-40,0,60]
    bio = io.BytesIO(); np.savez_compressed(bio, rgba=rgba, coords=np.array(c, np.float32))
    bio.seek(0)
    return Response(bio.read(), mimetype="application/octet-stream")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, threaded=True)
