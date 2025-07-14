#!/usr/bin/env python3
# encoding: utf-8

import time
from pymycobot import MyCobot

def main():
    print("[INFO] 正在连接机械臂...")
    mc = MyCobot("/dev/ttyAMA0", 1000000)

    # 检查设备是否连接成功
    version = mc.get_system_version()
    print(f"[INFO] 固件版本: {version}")
    
    # 根据设备类型判断是否是 MyCobot280
    offset_j5 = -90 if version > 2 else 0

    # 观察位姿（单位：角度）
    observe_pose = [0, 0, 2, -58, -2, -14 + offset_j5]

    print("[INFO] 正在复位至俯视观察位姿...")
    mc.send_angles(observe_pose, 30)
    time.sleep(3)

    print("[SUCCESS] 机械臂已移动到观察位。可放置二维码执行手眼标定。")

if __name__ == "__main__":
    main()
