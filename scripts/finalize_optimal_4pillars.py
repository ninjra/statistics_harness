#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "appdata" / "state.sqlite"
DEFAULT_OUT_DIR = ROOT / "docs" / "release_evidence"


def _run_status(db_path: Path, run_id: str) -> str | None:
    with sqlite3.connect(str(db_path)) as con:
        row = con.execute("select status from runs where run_id=?", (run_id,)).fetchone()
    if row is None:
        return None
    value = row[0]
    return str(value) if value is not None else None


def _wait_until_not_running(db_path: Path, run_id: str, poll_seconds: int) -> str:
    while True:
        status = _run_status(db_path, run_id)
        if status is None:
            raise SystemExit(f"run_id_not_found:{run_id}")
        if status != "running":
            return status
        time.sleep(max(1, int(poll_seconds)))


def _run_cmd(args: list[str]) -> None:
    cp = subprocess.run(args, cwd=str(ROOT), check=False, text=True, capture_output=True)
    if cp.returncode != 0:
        raise RuntimeError(
            f"command_failed:{' '.join(args)}\nstdout:\n{cp.stdout}\nstderr:\n{cp.stderr}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Wait for optimized run completion and emit before/after diff artifacts.")
    ap.add_argument("--before-run-id", required=True)
    ap.add_argument("--after-run-id", required=True)
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--poll-seconds", type=int, default=60)
    args = ap.parse_args()

    before_run = str(args.before_run_id).strip()
    after_run = str(args.after_run_id).strip()
    db_path = Path(args.db)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    final_status = _wait_until_not_running(db_path, after_run, poll_seconds=int(args.poll_seconds))

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    compare_outputs_json = out_dir / f"optimal_compare_outputs_{before_run}_to_{after_run}_{ts}.json"
    compare_action_json = out_dir / f"optimal_compare_actionability_{before_run}_to_{after_run}_{ts}.json"
    hotspots_md = out_dir / f"optimal_hotspots_{after_run}_{ts}.md"
    contract_after_json = out_dir / f"optimal_contract_{after_run}_{ts}.json"
    contract_compare_json = out_dir / f"optimal_contract_compare_{before_run}_to_{after_run}_{ts}.json"
    summary_json = out_dir / f"optimal_finalize_summary_{after_run}_{ts}.json"

    _run_cmd(
        [
            sys.executable,
            "scripts/compare_run_outputs.py",
            "--run-before",
            before_run,
            "--run-after",
            after_run,
            "--output-json",
            str(compare_outputs_json),
        ]
    )
    _run_cmd(
        [
            sys.executable,
            "scripts/compare_plugin_actionability_runs.py",
            "--before-run-id",
            before_run,
            "--after-run-id",
            after_run,
            "--out",
            str(compare_action_json),
        ]
    )
    _run_cmd(
        [
            sys.executable,
            "scripts/run_hotspots_report.py",
            "--run-id",
            after_run,
            "--top-n",
            "40",
            "--out-md",
            str(hotspots_md),
        ]
    )
    _run_cmd(
        [
            sys.executable,
            "scripts/verify_agent_execution_contract.py",
            "--run-id",
            after_run,
            "--expected-known-issues-mode",
            "any",
            "--recompute-recommendations",
            "always",
            "--out",
            str(contract_after_json),
        ]
    )
    _run_cmd(
        [
            sys.executable,
            "scripts/verify_agent_execution_contract.py",
            "--run-id",
            after_run,
            "--compare-run-id",
            before_run,
            "--expected-known-issues-mode",
            "any",
            "--recompute-recommendations",
            "always",
            "--out",
            str(contract_compare_json),
        ]
    )

    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "before_run_id": before_run,
        "after_run_id": after_run,
        "after_final_status": final_status,
        "artifacts": {
            "compare_run_outputs_json": str(compare_outputs_json),
            "compare_plugin_actionability_json": str(compare_action_json),
            "hotspots_md": str(hotspots_md),
            "contract_after_json": str(contract_after_json),
            "contract_compare_json": str(contract_compare_json),
        },
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"after_run_id={after_run}")
    print(f"after_final_status={final_status}")
    print(f"summary_json={summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
