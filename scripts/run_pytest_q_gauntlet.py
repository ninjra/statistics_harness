from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def _tail(path: Path, lines: int = 200) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""
    buf = text.splitlines()
    return "\n".join(buf[-lines:])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="/tmp/stat_harness_pytest_q.log")
    parser.add_argument("--heartbeat-seconds", type=int, default=30)
    args = parser.parse_args()

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        try:
            log_path.unlink()
        except Exception:
            pass

    env = dict(os.environ)
    # Reduce chances of interactive plugins altering output; keep deterministic.
    env.setdefault("PYTHONUNBUFFERED", "1")

    cmd = [sys.executable, "-m", "pytest", "-q"]
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        tick = 0
        while True:
            rc = proc.poll()
            if rc is not None:
                break
            tick += 1
            print(f"pytest -q running... tick={tick} (log={log_path})", flush=True)
            time.sleep(max(5, int(args.heartbeat_seconds)))

    rc = int(proc.returncode or 0)
    print(f"pytest -q exit_code={rc}", flush=True)
    if rc != 0:
        tail = _tail(log_path, lines=250)
        if tail:
            print("pytest -q failed; tailing log:", flush=True)
            print(tail, flush=True)
        else:
            print("pytest -q failed; log was empty", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

