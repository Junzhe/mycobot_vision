#!/usr/bin/env python3
# auto_record.py — 本机批量录制：设目标 → start → demo_grasp → stop，循环 N 次
import argparse, time, requests, sys

def post(url, json=None, data=None, timeout=30):
    try:
        r = requests.post(url, json=json, data=data, timeout=timeout)
        r.raise_for_status()
        try:
            return True, r.json()
        except Exception:
            return True, {"raw": r.text}
    except Exception as e:
        return False, {"error": str(e)}

def get(url, timeout=10):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return True, r.json()
    except Exception as e:
        return False, {"error": str(e)}

def wait_ready(base, max_wait=10):
    t0 = time.time()
    while time.time() - t0 < max_wait:
        ok, _ = get(f"{base}/health", timeout=2)
        if ok: return True
        time.sleep(0.5)
    return False

def main():
    ap = argparse.ArgumentParser(description="Batch record episodes with pi_collect.py")
    ap.add_argument("--base", default="http://127.0.0.1:5055", help="pi_collect服务地址")
    ap.add_argument("--target", default="A", choices=["A","B","C"], help="目标编号")
    ap.add_argument("--episodes", "-n", type=int, default=10, help="录制次数")
    ap.add_argument("--speed", type=int, default=30, help="demo_grasp 速度")
    ap.add_argument("--pre-wait", type=float, default=0.5, help="start 后等待秒数")
    ap.add_argument("--post-wait", type=float, default=0.5, help="stop 前等待秒数")
    ap.add_argument("--set-target-each", action="store_true",
                   help="每次循环都重新设置目标（默认只设置一次）")
    args = ap.parse_args()

    base = args.base.rstrip("/")
    print(f"[INFO] 使用服务：{base}")

    # 1) 等待服务就绪
    if not wait_ready(base, max_wait=15):
        print("[ERR] /health 不可用，请确认 pi_collect.py 已在运行")
        sys.exit(1)

    # 2) 设置目标（默认只设一次）
    def set_target():
        ok, resp = post(f"{base}/bci/target", json={"id": args.target, "phase":"confirmed"})
        if not ok or not resp.get("ok", False):
            print(f"[ERR] 设目标失败: {resp}")
            sys.exit(2)
        print(f"[OK] 目标已设为 {args.target}")

    set_target()
    for ep in range(1, args.episodes+1):
        if args.set_target_each:
            set_target()

        print(f"\n===== Episode {ep}/{args.episodes} =====")

        # 3) 开始录制
        ok, resp = post(f"{base}/record/start")
        if not ok or not resp.get("ok", False):
            print(f"[ERR] start 失败: {resp}")
            sys.exit(3)
        print(f"[OK] start: {resp.get('file','(no path)')}")
        time.sleep(args.pre_wait)

        # 4) 自动抓取（IK 老师）
        ok, resp = post(f"{base}/demo_grasp", json={"speed": args.speed}, timeout=120)
        if not ok or not resp.get("ok", False):
            print(f"[WARN] demo_grasp 失败或未找到标签: {resp}")
            # 失败也尝试停止录制，避免卡住
        else:
            print("[OK] demo_grasp 完成")

        # 5) 停止录制
        time.sleep(args.post_wait)
        ok2, resp2 = post(f"{base}/record/stop")
        if not ok2 or not resp2.get("ok", False):
            print(f"[ERR] stop 失败: {resp2}")
            sys.exit(4)
        print(f"[OK] stop: {resp2.get('file','(no path)')}")

        time.sleep(1.0)

    print("\n[DONE] 全部录制完成。数据在脚本同目录下的 data/ 里。")

if __name__ == "__main__":
    main()
