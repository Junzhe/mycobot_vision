#!/usr/bin/env python3
# encoding: utf-8

import time
from pymycobot import MyCobot

def main():
    print("[INFO] 正在连接机械臂...")
    mc = MyCobot("/dev/ttyAMA0", 115200)  # 确保串口正确
    
    print("[INFO] 上电中...")
    mc.power_on()
    time.sleep(1)

    version = mc.get_system_version()
    print(f"[INFO] 固件版本: {version}")
    
    offset_j5 = -90 if version > 2 else 0

    observe_pose = [0, 0, 2, -58, -2, -14 + offset_j5]

    print("[INFO] 正在复位至观察位...")
    mc.send_angles(observe_pose, 30)
    time.sleep(3)

    print("[SUCCESS] 已复位完毕。")

if __name__ == "__main__":
    main()
