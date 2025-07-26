

from flask import Flask, request
from grasp_controller import grasp_from_target_code

app = Flask(__name__)

@app.route('/target', methods=['POST'])
def handle_target():
    target_code = request.form.get("target")
    print(f"📥 接收到目标编号：{target_code}")

    if not target_code:
        return "Missing 'target'", 400

    success = grasp_from_target_code(target_code)
    return "OK" if success else "Failed", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
