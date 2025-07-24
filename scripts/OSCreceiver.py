from pythonosc import dispatcher
from pythonosc import osc_server

# 回调函数：接收到 /target_info 时打印目标编号
def target_info_handler(address, *args):
    print(f"📥 接收到目标编号：{args[0]} (来自 {address})")

# 创建 dispatcher 并绑定地址
dispatcher = dispatcher.Dispatcher()
dispatcher.map("/target_info", target_info_handler)

# 设置本机监听的 IP 和端口（确保与 Unity 配置一致）
ip = "0.0.0.0"         # 表示接收任意来源
port = 8000            # Unity 的 Remote Port 也应为 8000

print(f"🟢 开始监听 OSC 指令：{ip}:{port}")
server = osc_server.ThreadingOSCUDPServer((ip, port), dispatcher)
server.serve_forever()
