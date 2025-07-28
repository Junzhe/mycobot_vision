import numpy as np
from pymycobot import MyCobot280
from camera_detect import camera_detect

# === 实测数据 ===
pos_real = np.array([5.0, 68.0, -135.0])  # 单位 mm

# === 初始化设备 ===
mc = MyCobot280("/dev/ttyAMA0", 1000000)
camera_params = np.load("camera_params.npz")
mtx, dist = camera_params["mtx"], camera_params["dist"]
cd = camera_detect(0, 40, mtx, dist)

# === 当前末端位置 ===
tool_pose = np.array(mc.get_coords())  # [x, y, z, rx, ry, rz]
tool_position = tool_pose[:3]
print("[INFO] 当前末端在 base 下位置 =", np.round(tool_position, 2))

# === 推算二维码真实位置（实测方式）===
pos_real_in_base = tool_position + pos_real
print("[INFO] 实测推算二维码位置（base坐标系） =", np.round(pos_real_in_base, 2))

# === 程序识别二维码位置（基于手眼标定）===
predicted_pose, _ = cd.stag_robot_identify(mc)
pos_pred = np.array(predicted_pose[:3])
print("[INFO] 程序预测二维码位置 =", np.round(pos_pred, 2))

# === 比较误差 ===
error = pos_pred - pos_real_in_base
print("\n[RESULT] 误差（单位 mm） =", np.round(error, 2))

threshold = 10  # mm 误差容限
if np.all(np.abs(error) < threshold):
    print("[✅] 标定精度良好，误差在 ±10mm 范围内")
else:
    print("[❌] 标定误差较大，建议重新校准或调整姿态")
