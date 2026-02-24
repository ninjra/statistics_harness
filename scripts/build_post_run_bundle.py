#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "appdata" / "state.sqlite"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def _run_row(conn: sqlite3.Connection, run_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT run_id, status, dataset_version_id, created_at
        FROM runs
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchone()
    return dict(row) if row else None


def _latest_completed_before(
    conn: sqlite3.Connection, *, dataset_version_id: str, run_id: str
) -> str | None:
    row = conn.execute(
        """
        SELECT run_id
        FROM runs
        WHERE dataset_version_id = ?
          AND run_id <> ?
          AND status = 'completed'
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (dataset_version_id, run_id),
    ).fetchone()
    return str(row["run_id"]) if row else None


def _run_subprocess(args: list[str]) -> dict[str, Any]:
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


def _write_top_recommendations(run_id: str, out_path: Path, limit: int) -> bool:
    rec_path = (
        ROOT
        / "appdata"
        / "runs"
        / run_id
        / "artifacts"
        / "analysis_recommendation_dedupe_v2"
        / "recommendations.json"
    )
    if not rec_path.exists():
        return False
    try:
        payload = json.loads(rec_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    items = payload.get("summary_recommendations")
    if not isinstance(items, list):
        return False
    top_items = [row for row in items if isinstance(row, dict)][: max(1, int(limit))]
    out = {
        "schema_version": "post_run_bundle_top_recommendations.v1",
        "run_id": run_id,
        "count": int(len(top_items)),
        "items": top_items,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a deterministic post-run evidence bundle in one command."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--before-run-id", default="")
    parser.add_argument("--known-issues-mode", choices=("any", "on", "off"), default="any")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--state-db", default=str(DEFAULT_DB))
    parser.add_argument("--out-dir", default="docs/release_evidence")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    db_path = Path(str(args.state_db)).resolve()
    if not db_path.exists():
        raise SystemExit(f"Missing state DB: {db_path}")

    out_dir = Path(str(args.out_dir))
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = str(args.run_id).strip()
    before_run_id = str(args.before_run_id).strip()

    con = _connect(db_path)
    try:
        run = _run_row(con, run_id)
        if not isinstance(run, dict):
            raise SystemExit(f"Run not found: {run_id}")
        dataset_version_id = str(run.get("dataset_version_id") or "").strip()
        if not before_run_id and dataset_version_id:
            before_run_id = _latest_completed_before(
                con,
                dataset_version_id=dataset_version_id,
                run_id=run_id,
            ) or ""
    finally:
        con.close()

    files: dict[str, str] = {}
    steps: dict[str, dict[str, Any]] = {}

    if before_run_id:
        diff_path = out_dir / f"diff_{before_run_id}_to_{run_id}.json"
        files["run_diff_json"] = str(diff_path)
        steps["compare_run_outputs"] = _run_subprocess(
            [
                sys.executable,
                "scripts/compare_run_outputs.py",
                "--run-before",
                before_run_id,
                "--run-after",
                run_id,
                "--output-json",
                str(diff_path),
            ]
        )

        actionability_diff_path = out_dir / f"actionability_{before_run_id}_to_{run_id}.json"
        files["actionability_diff_json"] = str(actionability_diff_path)
        steps["compare_plugin_actionability"] = _run_subprocess(
            [
                sys.executable,
                "scripts/compare_plugin_actionability_runs.py",
                "--before-run-id",
                before_run_id,
                "--after-run-id",
                run_id,
                "--out",
                str(actionability_diff_path),
            ]
        )

    hotspots_path = out_dir / f"hotspots_{run_id}.md"
    files["hotspots_md"] = str(hotspots_path)
    steps["run_hotspots_report"] = _run_subprocess(
        [
            sys.executable,
            "scripts/run_hotspots_report.py",
            "--run-id",
            run_id,
            "--top-n",
            str(max(1, int(args.top_n))),
            "--out-md",
            str(hotspots_path),
        ]
    )

    contract_path = out_dir / f"contract_{run_id}.json"
    files["execution_contract_json"] = str(contract_path)
    steps["verify_execution_contract"] = _run_subprocess(
        [
            sys.executable,
            "scripts/verify_agent_execution_contract.py",
            "--run-id",
            run_id,
            "--expected-known-issues-mode",
            str(args.known_issues_mode),
            "--recompute-recommendations",
            "never",
            "--out",
            str(contract_path),
        ]
    )

    plugin_contract_path = out_dir / f"plugin_contract_{run_id}.json"
    files["plugin_contract_json"] = str(plugin_contract_path)
    steps["verify_plugin_result_contract"] = _run_subprocess(
        [
            sys.executable,
            "scripts/verify_plugin_result_contract.py",
            "--run-id",
            run_id,
            "--out-json",
            str(plugin_contract_path),
        ]
    )

    burndown_json_path = out_dir / f"actionability_burndown_{run_id}.json"
    burndown_md_path = out_dir / f"actionability_burndown_{run_id}.md"
    files["actionability_burndown_json"] = str(burndown_json_path)
    files["actionability_burndown_md"] = str(burndown_md_path)
    burndown_cmd = [
        sys.executable,
        "scripts/actionability_burndown.py",
        "--run-id",
        run_id,
        "--out-json",
        str(burndown_json_path),
        "--out-md",
        str(burndown_md_path),
    ]
    if before_run_id:
        burndown_cmd.extend(["--before-run-id", before_run_id])
    steps["actionability_burndown"] = _run_subprocess(burndown_cmd)

    if before_run_id:
        reason_delta_json_path = out_dir / f"non_actionable_reason_{before_run_id}_to_{run_id}.json"
        reason_delta_md_path = out_dir / f"non_actionable_reason_{before_run_id}_to_{run_id}.md"
        files["reason_delta_json"] = str(reason_delta_json_path)
        files["reason_delta_md"] = str(reason_delta_md_path)
        steps["reason_code_burndown_delta"] = _run_subprocess(
            [
                sys.executable,
                "scripts/actionability_burndown.py",
                "--run-id",
                run_id,
                "--before-run-id",
                before_run_id,
                "--out-json",
                str(reason_delta_json_path),
                "--out-md",
                str(reason_delta_md_path),
            ]
        )

    top_path = out_dir / f"top_recommendations_{run_id}.json"
    files["top_recommendations_json"] = str(top_path)
    top_ok = _write_top_recommendations(run_id, top_path, int(args.top_n))
    steps["top_recommendations"] = {
        "rc": 0 if top_ok else 1,
        "detail": "ok" if top_ok else "missing recommendations artifact",
    }

    ok = all(int((step or {}).get("rc", 1)) == 0 for step in steps.values())
    payload = {
        "schema_version": "post_run_bundle.v1",
        "generated_at_utc": _now_iso(),
        "run_id": run_id,
        "before_run_id": before_run_id or None,
        "state_db": str(db_path),
        "ok": bool(ok),
        "steps": steps,
        "files": files,
    }

    manifest = out_dir / f"bundle_{run_id}.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(str(manifest))
    if bool(args.strict) and not bool(ok):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
