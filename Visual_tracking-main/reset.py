# reset_arm.py
import time
from pymycobot import MyCobot280, PI_PORT, PI_BAUD

def reset_robot():
    print("[INFO] 正在连接机械臂...")
    mc = MyCobot280(PI_PORT, PI_BAUD)
    offset_j5 = -90 if mc.get_system_version() > 2 else 0

    print("[INFO] 释放夹爪...")
    mc.set_gripper_state(0, 80)  # 打开夹爪
    time.sleep(1)

    print("[INFO] 复位机械臂到默认安全位置...")
    safe_angles = [-90, 5, -104, 14, 90 + offset_j5, 60]
    mc.send_angles(safe_angles, 50)
    time.sleep(3)

    print("[SUCCESS] 机械臂已复原至默认位置")

if __name__ == "__main__":
    reset_robot()
