cmake_minimum_required(VERSION 3.0.2)
project(mycobot_vision)

## 添加 Python 支持
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  visualization_msgs
)

catkin_package()

## 安装 Python 脚本
catkin_install_python(PROGRAMS
  scripts/detect_single_aruco_grasp.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## 声明消息/服务路径等（如需要）
