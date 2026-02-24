#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "release_evidence" / "openplanter_pack_release_gate.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_pytest(args: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *args],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    return int(proc.returncode), str(proc.stdout or ""), str(proc.stderr or "")


def _parse_summary(text: str) -> dict[str, int]:
    summary = {"passed": 0, "failed": 0, "skipped": 0, "xfailed": 0, "xpassed": 0, "errors": 0}
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    tail = lines[-1] if lines else ""
    for key in ("passed", "failed", "skipped", "xfailed", "xpassed", "error", "errors"):
        token = f" {key}"
        if token not in f" {tail}":
            continue
        parts = tail.replace(",", " ").split()
        for i, part in enumerate(parts[:-1]):
            if part.isdigit() and parts[i + 1].startswith(key):
                mapped = "errors" if key in {"error", "errors"} else key
                summary[mapped] += int(part)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Run OpenPlanter cross-dataset release gate checks.")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--allow-skips", action="store_true")
    ap.add_argument("--allow-xfail", action="store_true")
    ap.add_argument("--pytest-args", default="-q")
    args = ap.parse_args()

    parsed_pytest_args = shlex.split(str(args.pytest_args))
    code, out, err = _run_pytest(parsed_pytest_args)
    summary = _parse_summary(out + "\n" + err)
    blocked = bool(code != 0)
    if not args.allow_skips and int(summary.get("skipped", 0)) > 0:
        blocked = True
    if not args.allow_xfail and (int(summary.get("xfailed", 0)) > 0 or int(summary.get("xpassed", 0)) > 0):
        blocked = True
    payload: dict[str, Any] = {
        "generated_at_utc": _now(),
        "command": [sys.executable, "-m", "pytest", *parsed_pytest_args],
        "exit_code": code,
        "summary": summary,
        "allow_skips": bool(args.allow_skips),
        "allow_xfail": bool(args.allow_xfail),
        "blocked": bool(blocked),
        "DO_NOT_SHIP": bool(blocked),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(str(out_path))
    return 1 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
