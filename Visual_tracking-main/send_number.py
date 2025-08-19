# target_sender.py
import argparse, time, requests, sys

def send_target(pi_base: str, code: str) -> bool:
    pi_base = pi_base.rstrip("/")
    code = code.strip().upper()
    r = requests.post(f"{pi_base}/target", data={"target": code}, timeout=5)
    ok = (r.status_code == 200) and ("OK" in r.text.upper())
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] POST {pi_base}/target target={code} -> {r.status_code} {r.text.strip()}")
    return ok

def main():
    ap = argparse.ArgumentParser(description="Send target code A/B/C to grasp_server.py")
    ap.add_argument("--pi", required=True, help="Pi URL, e.g. http://10.7.182.37:5000")
    ap.add_argument("--target", default="A",
                    help="A/B/C 或逗号序列如 A,B,C（配合 --round 轮询）")
    ap.add_argument("--repeat", type=int, default=1, help="发送次数")
    ap.add_argument("--interval", type=float, default=1.5, help="两次发送间隔秒")
    ap.add_argument("--round", action="store_true",
                    help="对 --target 给出的序列做轮询（如 A,B,C,A,B,C...）")
    args = ap.parse_args()

    valid = {"A","B","C"}
    seq = [s.strip().upper() for s in args.target.split(",") if s.strip()]
    if not seq or any(s not in valid for s in seq):
        print("ERROR: --target 只允许 A/B/C（或它们的逗号序列）")
        sys.exit(1)

    for i in range(args.repeat):
        code = seq[i % len(seq)] if (args.round or len(seq) > 1) else seq[0]
        try:
            send_target(args.pi, code)
        except Exception as e:
            print("REQUEST ERROR:", e)
        if i < args.repeat - 1:
            time.sleep(args.interval)

if __name__ == "__main__":
    main()

