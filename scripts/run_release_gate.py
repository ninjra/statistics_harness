#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "docs" / "release_evidence"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _run_step(args: list[str]) -> dict[str, Any]:
    proc = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "cmd": args,
        "rc": int(proc.returncode),
        "stdout": str(proc.stdout or ""),
        "stderr": str(proc.stderr or ""),
    }


def _bundle_out_path(out_dir: Path, run_id: str) -> Path:
    return out_dir / f"bundle_{run_id}.json"


def _release_gate_payload(
    *,
    run_id: str,
    before_run_id: str,
    known_issues_mode: str,
    state_db: str,
    pytest_args: str,
    top_n: int,
    out_dir: Path,
) -> dict[str, Any]:
    steps: dict[str, dict[str, Any]] = {}
    files: dict[str, str] = {}

    docs_step = _run_step([sys.executable, "scripts/verify_docs_and_plugin_matrices.py"])
    steps["verify_docs_and_plugin_matrices"] = docs_step

    pytest_gate_out = out_dir / "openplanter_pack_release_gate.json"
    files["pytest_gate_json"] = str(pytest_gate_out)
    pytest_step = _run_step(
        [
            sys.executable,
            "scripts/verify_openplanter_pack_release_gate.py",
            f"--pytest-args={pytest_args}",
            "--out",
            str(pytest_gate_out),
        ]
    )
    steps["verify_openplanter_pack_release_gate"] = pytest_step

    if run_id:
        bundle_step = _run_step(
            [
                sys.executable,
                "scripts/build_post_run_bundle.py",
                "--run-id",
                run_id,
                "--before-run-id",
                before_run_id,
                "--known-issues-mode",
                known_issues_mode,
                "--state-db",
                state_db,
                "--top-n",
                str(max(1, int(top_n))),
                "--strict",
            ]
        )
        steps["build_post_run_bundle"] = bundle_step
        files["post_run_bundle_json"] = str(_bundle_out_path(out_dir, run_id))

    ok = all(int((step or {}).get("rc", 1)) == 0 for step in steps.values())
    return {
        "schema_version": "release_gate.v1",
        "generated_at_utc": _now_iso(),
        "ok": bool(ok),
        "run_id": run_id or None,
        "before_run_id": before_run_id or None,
        "known_issues_mode": known_issues_mode,
        "steps": steps,
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic release gates: docs matrix verification, pytest gate, "
            "and optional post-run bundle validation."
        )
    )
    parser.add_argument("--run-id", default="")
    parser.add_argument("--before-run-id", default="")
    parser.add_argument("--known-issues-mode", choices=("any", "on", "off"), default="any")
    parser.add_argument("--state-db", default=str(ROOT / "appdata" / "state.sqlite"))
    parser.add_argument("--pytest-args", default="-q")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    out_dir = Path(str(args.out_dir))
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = _release_gate_payload(
        run_id=str(args.run_id).strip(),
        before_run_id=str(args.before_run_id).strip(),
        known_issues_mode=str(args.known_issues_mode).strip(),
        state_db=str(args.state_db).strip(),
        pytest_args=str(args.pytest_args).strip(),
        top_n=int(args.top_n),
        out_dir=out_dir,
    )

    explicit_out = str(args.out).strip()
    if explicit_out:
        out_path = Path(explicit_out)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
    else:
        slug = str(args.run_id).strip() or _slug_now()
        out_path = out_dir / f"release_gate_{slug}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(str(out_path))
    return 0 if bool(payload.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
