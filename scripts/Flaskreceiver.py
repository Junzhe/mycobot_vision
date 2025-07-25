from flask import Flask, request

app = Flask(__name__)

@app.route('/target', methods=['POST'])
def handle_target():
    target = request.form.get("target")
    print(f"📥 接收到目标编号：{target}")
    # TODO: 可在此处调用目标识别 + 抓取程序
    return "OK"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
