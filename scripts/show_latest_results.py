#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
APPDATA = REPO_ROOT / "appdata"
DB_PATH = APPDATA / "state.sqlite"


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def _latest_dataset_version_id(con: sqlite3.Connection) -> str | None:
    row = con.execute(
        "select dataset_version_id from dataset_versions order by row_count desc, created_at desc limit 1"
    ).fetchone()
    return str(row["dataset_version_id"]) if row else None


def _latest_run_id_for_dataset(
    con: sqlite3.Connection, dataset_version_id: str, status: str | None
) -> str | None:
    if status:
        row = con.execute(
            "select run_id from runs where dataset_version_id=? and status=? order by created_at desc limit 1",
            (dataset_version_id, status),
        ).fetchone()
    else:
        row = con.execute(
            "select run_id from runs where dataset_version_id=? order by created_at desc limit 1",
            (dataset_version_id,),
        ).fetchone()
    return str(row["run_id"]) if row else None


def _run_row(con: sqlite3.Connection, run_id: str) -> dict[str, Any]:
    row = con.execute(
        "select run_id, status, created_at, completed_at, dataset_version_id, input_filename from runs where run_id=?",
        (run_id,),
    ).fetchone()
    return dict(row) if row else {}


def _plugin_status_counts(con: sqlite3.Connection, run_id: str) -> dict[str, int]:
    rows = con.execute(
        "select status, count(*) as c from plugin_results where run_id=? group by status order by status",
        (run_id,),
    ).fetchall()
    out: dict[str, int] = {}
    for r in rows:
        out[str(r["status"] or "")] = int(r["c"] or 0)
    return out


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _top_recommendations(run_dir: Path, limit: int = 15) -> list[dict[str, Any]]:
    payload_path = run_dir / "artifacts" / "analysis_recommendation_dedupe_v2" / "recommendations.json"
    if not payload_path.exists():
        return []
    payload = _read_json(payload_path)
    if not isinstance(payload, dict):
        return []
    items = payload.get("summary_recommendations") or []
    if isinstance(items, list):
        return [i for i in items if isinstance(i, dict)][: int(limit)]
    return []


def _ideaspace_block(report: dict[str, Any]) -> dict[str, Any]:
    plugins = report.get("plugins")
    if not isinstance(plugins, dict):
        return {}
    gap = plugins.get("analysis_ideaspace_normative_gap")
    act = plugins.get("analysis_ideaspace_action_planner")
    return {
        "normative_gap": gap if isinstance(gap, dict) else {},
        "action_planner": act if isinstance(act, dict) else {},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-version-id", default="")
    ap.add_argument("--prefer-status", default="completed", choices=["completed", "partial", "running", "aborted", "error", "any"])
    ap.add_argument("--recommendations", type=int, default=15)
    args = ap.parse_args()

    if not DB_PATH.exists():
        raise SystemExit(f"Missing DB: {DB_PATH}")

    with _connect() as con:
        dataset_version_id = str(args.dataset_version_id or "").strip()
        if not dataset_version_id:
            dataset_version_id = _latest_dataset_version_id(con) or ""
        if not dataset_version_id:
            raise SystemExit("No dataset_version_id found.")

        status = None if args.prefer_status == "any" else args.prefer_status
        run_id = _latest_run_id_for_dataset(con, dataset_version_id, status=status) or ""
        if not run_id and status is not None:
            run_id = _latest_run_id_for_dataset(con, dataset_version_id, status=None) or ""
        if not run_id:
            raise SystemExit(f"No runs found for dataset_version_id={dataset_version_id}")

        run = _run_row(con, run_id)
        counts = _plugin_status_counts(con, run_id)

    run_dir = APPDATA / "runs" / run_id
    outputs = {
        "run_dir": str(run_dir),
        "report_md": str(run_dir / "report.md"),
        "report_json": str(run_dir / "report.json"),
        "answers_summary_json": str(run_dir / "answers_summary.json"),
        "answers_recommendations_md": str(run_dir / "answers_recommendations.md"),
        "business_summary_md": str(run_dir / "business_summary.md"),
        "engineering_summary_md": str(run_dir / "engineering_summary.md"),
    }
    present = {k: Path(v).is_file() for k, v in outputs.items() if k != "run_dir"}

    created = _parse_iso(str(run.get("created_at") or "")) or None
    completed = _parse_iso(str(run.get("completed_at") or "")) or None
    elapsed_min = None
    if created:
        end = completed or _now_utc()
        elapsed_min = max(0.0, (end - created).total_seconds() / 60.0)

    report = {}
    if (run_dir / "report.json").exists():
        try:
            report = _read_json(run_dir / "report.json")
        except Exception:
            report = {}

    payload: dict[str, Any] = {
        "dataset_version_id": dataset_version_id,
        "run": {
            "run_id": run_id,
            "status": run.get("status"),
            "created_at": run.get("created_at"),
            "completed_at": run.get("completed_at"),
            "elapsed_minutes": elapsed_min,
            "input_filename": run.get("input_filename"),
        },
        "plugin_status_counts": counts,
        "outputs": outputs,
        "outputs_present": present,
        "recommendations_top": _top_recommendations(run_dir, limit=int(args.recommendations)),
        "ideaspace": _ideaspace_block(report) if isinstance(report, dict) else {},
    }

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

