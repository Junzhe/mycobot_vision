import numpy as np

# === Step 1: 加载当前标定矩阵（EyesInHand_matrix.npy） ===
try:
    T_tool_cam = np.load("EyesInHand_matrix.json")
    print("[INFO] 原始标定矩阵加载成功:\n", np.round(T_tool_cam, 2))
except Exception as e:
    print("[ERROR] 无法加载 EyesInHand_matrix.npy，请确认路径正确:", e)
    exit()

# === Step 2: 修复旋转矩阵的 Z 轴方向 ===
R = T_tool_cam[:3, :3].copy()
R[:, 2] = -R[:, 2]  # 取反 Z 轴方向

# === Step 3: 重新归一化旋转矩阵（确保正交性） ===
z = R[:, 2]
y = R[:, 1]
x = np.cross(y, z)
x = x / np.linalg.norm(x)
y = np.cross(z, x)
y = y / np.linalg.norm(y)
z = z / np.linalg.norm(z)

R_fixed = np.column_stack((x, y, z))
T_tool_cam[:3, :3] = R_fixed

# === Step 4: 保存修正后的矩阵 ===
np.save("EyesInHand_matrix_fixed.json", T_tool_cam)
print("\n[✅] 修正后的 EyesInHand_matrix 已保存为 EyesInHand_matrix_fixed.json")
print("\n[RESULT] 修正后矩阵:\n", np.round(T_tool_cam, 2))
