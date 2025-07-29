import numpy as np
import json

# === Step 1: 加载当前标定矩阵（EyesInHand_matrix.json） ===
try:
    with open("EyesInHand_matrix.json", "r") as f:
        data = json.load(f)
        # 支持纯数组或字典格式两种
        if isinstance(data, list):
            T_tool_cam = np.array(data)
        elif isinstance(data, dict) and "matrix" in data:
            T_tool_cam = np.array(data["matrix"])
        else:
            raise ValueError("JSON格式不正确，必须是列表或包含'matrix'键的字典")
    print("[INFO] 原始标定矩阵加载成功:\n", np.round(T_tool_cam, 2))
except Exception as e:
    print("[ERROR] 无法加载 EyesInHand_matrix.json，请确认路径正确:", e)
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

# === Step 4: 保存修正后的矩阵为 JSON ===
output_data = {"matrix": T_tool_cam.tolist()}
with open("EyesInHand_matrix_fixed.json", "w") as f:
    json.dump(output_data, f, indent=4)

print("\n[✅] 修正后的 EyesInHand_matrix 已保存为 EyesInHand_matrix_fixed.json")
print("\n[RESULT] 修正后矩阵:\n", np.round(T_tool_cam, 2))
