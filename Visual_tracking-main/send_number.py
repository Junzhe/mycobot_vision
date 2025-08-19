# target_test.py  —— 发目标编号到 pi_collect（端口 5055）
import time, argparse, requests

ap = argparse.ArgumentParser()
ap.add_argument("--pi", required=True, help="http://<pi-ip>:5055")
ap.add_argument("--target", required=True, choices=["A","B","C"])
ap.add_argument("--repeat", type=int, default=1)
ap.add_argument("--interval", type=float, default=1.0)
args = ap.parse_args()

s = requests.Session()
for i in range(args.repeat):
    r = s.post(f"{args.pi}/bci/target",
               json={"id": args.target, "phase":"confirmed"}, timeout=2)
    print(f"{args.target} -> {r.status_code} {r.text}")
    time.sleep(args.interval)
