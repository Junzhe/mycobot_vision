import time
import numpy as np
from pymycobot import MyCobot280

def test_send_coords_accuracy():
    mc = MyCobot280("/dev/ttyAMA0", 1000000)
    time.sleep(1)

    # 获取当前位置作为目标位置
    target_pose = mc.get_coords()
    while not target_pose or len(target_pose) != 6:
        print("[WARN] 获取初始坐标失败，重试中...")
        time.sleep(0.5)
        target_pose = mc.get_coords()
    print(f"[INFO] 初始坐标: {np.round(target_pose, 2)}")

    # 发出指令回到该点
    print("[INFO] 使用 send_coords() 移动回当前位置...")
    mc.send_coords(target_pose, speed=30)
    time.sleep(2)

    # 再次获取当前坐标
    actual_pose = mc.get_coords()
    while not actual_pose or len(actual_pose) != 6:
        print("[WARN] 获取执行后坐标失败，重试中...")
        time.sleep(0.5)
        actual_pose = mc.get_coords()
    print(f"[INFO] 实际到达坐标: {np.round(actual_pose, 2)}")

    # 计算误差
    position_error = np.array(actual_pose[:3]) - np.array(target_pose[:3])
    angle_error = np.array(actual_pose[3:]) - np.array(target_pose[3:])
    print(f"[RESULT] 位置误差（mm）: {np.round(position_error, 2)}")
    print(f"[RESULT] 姿态误差（deg）: {np.round(angle_error, 2)}")

    # 简单判断是否在误差容限内
    if np.any(np.abs(position_error) > 5) or np.any(np.abs(angle_error) > 5):
        print("[WARNING] ❌ 误差超过阈值，send_coords() 精度可能不可靠")
    else:
        print("[SUCCESS] ✅ send_coords() 执行精度在可接受范围内")

if __name__ == "__main__":
    test_send_coords_accuracy()
