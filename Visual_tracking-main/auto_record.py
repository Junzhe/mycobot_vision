#!/usr/bin/env python3
# auto_record.py —— 单次录制：设目标 → start → demo_grasp → stop（一次就退出）
import argparse, time, requests, sys

def post(url, json=None, data=None, timeout=60):
    try:
        r = requests.post(url, json=json, data=data, timeout=timeout)
        r.raise_for_status()
        try: return True, r.json()
        except Exception: return True, {"raw": r.text}
    except Exception as e:
        return False, {"error": str(e)}

def get(url, timeout=10):
    try:
        r = requests.get(url, timeout=timeout); r.raise_for_status()
        return True, r.json()
    except Exception as e:
        return False, {"error": str(e)}

def wait_ready(base, max_wait=15):
    import time
    t0 = time.time()
    while time.time() - t0 < max_wait:
        ok, _ = get(f"{base}/health", timeout=2)
        if ok: return True
        time.sleep(0.5)
    return False

def main():
    ap = argparse.ArgumentParser(description="One-shot record with pi_collect.py")
    ap.add_argument("--base", default="http://127.0.0.1:5055", help="pi_collect 服务地址")
    ap.add_argument("--target", default="A", choices=["A","B","C"], help="目标编号")
    ap.add_argument("--speed", type=int, default=30, help="demo_grasp 速度")
    ap.add_argument("--pre-wait", type=float, default=0.5, help="start 后等待秒数")
    ap.add_argument("--post-wait", type=float, default=0.5, help="stop 前等待秒数")
    args = ap.parse_args()

    base = args.base.rstrip("/")
    print(f"[INFO] 服务：{base}")

    if not wait_ready(base):
        print("[ERR] /health 不可用，请确认 pi_collect.py 已运行")
        sys.exit(1)

    # 设目标
    ok, resp = post(f"{base}/bci/target", json={"id": args.target, "phase":"confirmed"})
    if not ok or not resp.get("ok", False):
        print(f"[ERR] 设目标失败: {resp}"); sys.exit(2)
    print(f"[OK] 目标已设为 {args.target}")

    # 开始录制
    ok, resp = post(f"{base}/record/start")
    if not ok or not resp.get("ok", False):
        print(f"[ERR] start 失败: {resp}"); sys.exit(3)
    print(f"[OK] start: {resp.get('file','(no path)')}")
    time.sleep(args.pre_wait)

    # 抓取
    ok, resp = post(f"{base}/demo_grasp", json={"speed": args.speed}, timeout=120)
    if not ok or not resp.get("ok", False):
        print(f"[WARN] demo_grasp 失败或未找到标签: {resp}")

    # 停止录制
    time.sleep(args.post_wait)
    ok, resp = post(f"{base}/record/stop")
    if not ok or not resp.get("ok", False):
        print(f"[ERR] stop 失败: {resp}"); sys.exit(4)
    print(f"[OK] stop: {resp.get('file','(no path)')}")
    print("\n[DONE] 本次录制完成。请手动将物块放回原位，然后再次运行本脚本开始下一段。")

if __name__ == "__main__":
    main()
