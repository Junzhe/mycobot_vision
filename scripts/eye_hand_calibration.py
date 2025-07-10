# encoding: UTF-8
#!/usr/bin/env python3

import time
import json
import numpy as np
from pymycobot.mycobot import MyCobot

# 👇 添加此段以修复 vision 模块找不到问题
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vision.camera_detect import camera_detect

# ------------------ Step 1: Run hand-eye calibration ------------------

def run_eye_hand_calibration():
    mc = MyCobot("/dev/ttyAMA0", 115200)

    # Step 1: Set camera parameters and create detector
    camera_params = np.load("camera_params.npz")
    mtx, dist = camera_params['mtx'], camera_params['dist']
    m = camera_detect(camera_id=0, marker_size=47, mtx=mtx, dist=dist)

    # 设置夹爪参考与末端类型（工具中心点）
    mc.set_tool_reference([0, 20, 0, 0, 0, 0])
    mc.set_end_type(1)

    # Step 2: 移动到俯视位
    observe_pos = [0, 0, 2, -58, -2, -14]  # 与主程序一致的俯视位
    print("[INFO] 移动机械臂至俯视桌面位置...")
    mc.send_angles(observe_pos, 30)
    time.sleep(3)

    print("[INFO] 请将 Stag 标签放在桌面中心位置，确保摄像头可以看到")
    input("确认 Stag 已放好并稳定后，按回车开始标定...")

    # Step 3: 执行手眼标定
    print("[INFO] 正在执行手眼标定过程，请勿移动标签...")
    m.eyes_in_hand_calibration(mc)
    print("[SUCCESS] 标定完成，手眼矩阵已保存至 EyesInHand_matrix.json")

# ------------------ 主程序入口 ------------------

if __name__ == '__main__':
    print("[模式 1] 手眼标定模式 (自动保存标定矩阵)")
    run_eye_hand_calibration()
