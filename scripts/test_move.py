from pymycobot import MyCobot280
from pymycobot import PI_PORT, PI_BAUD
import time

mc = MyCobot280(PI_PORT, PI_BAUD)

print("正在尝试发送移动指令...")
mc.send_coords([0, 0, 150, 180, 0, 0], 25, 0)
time.sleep(3)
print("当前关节角度：", mc.get_angles())
print("已发送移动指令")
