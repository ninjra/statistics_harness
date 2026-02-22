#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "appdata" / "state.sqlite"
DEFAULT_OUT_DIR = ROOT / "docs" / "release_evidence"


@dataclass(frozen=True)
class PluginRow:
    plugin_id: str
    status: str
    duration_ms: int
    max_rss_kb: int
    exit_code: int | None
    stderr: str


def _connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    return con


def _fetch_rows(con: sqlite3.Connection, run_id: str) -> list[PluginRow]:
    rows = con.execute(
        """
        SELECT plugin_id, status, duration_ms, max_rss, exit_code, COALESCE(stderr, '') AS stderr
        FROM plugin_executions
        WHERE run_id = ?
        ORDER BY plugin_id
        """,
        (run_id,),
    ).fetchall()
    out: list[PluginRow] = []
    for row in rows:
        out.append(
            PluginRow(
                plugin_id=str(row["plugin_id"] or ""),
                status=str(row["status"] or ""),
                duration_ms=int(row["duration_ms"] or 0),
                max_rss_kb=int(row["max_rss"] or 0),
                exit_code=int(row["exit_code"]) if row["exit_code"] is not None else None,
                stderr=str(row["stderr"] or ""),
            )
        )
    return out


def _reason(row: PluginRow) -> str:
    stderr = row.stderr.lower()
    if row.status in {"error", "aborted"}:
        if "memoryerror" in stderr or "unable to allocate" in stderr:
            return "memory_bound_failure"
        if row.exit_code == -9:
            return "killed_or_time_budget_exhausted"
        return "runtime_failure"
    if row.duration_ms >= 300_000:
        return "high_duration_hotspot"
    if row.max_rss_kb >= 1_500_000:
        return "high_memory_hotspot"
    return "healthy"


def _recommended_next_step(reason: str) -> str:
    if reason == "memory_bound_failure":
        return "Replace full-frame boolean slices with index-based access and column-projected reads."
    if reason == "killed_or_time_budget_exhausted":
        return "Bound algorithmic complexity (pair checks, tokens, rows) and add deterministic caps."
    if reason == "runtime_failure":
        return "Capture plugin-specific traceback, add guardrails, and force deterministic fallback result."
    if reason == "high_duration_hotspot":
        return "Add bounded-window execution path and lightweight approximation mode for large inputs."
    if reason == "high_memory_hotspot":
        return "Stream/batch large computations and avoid materializing wide intermediate DataFrames."
    return "No immediate action required."


def _priority_score(row: PluginRow, reason: str) -> float:
    status_weight = 0.0
    if row.status in {"error", "aborted"}:
        status_weight = 1_000_000.0
    elif row.status == "degraded":
        status_weight = 500_000.0
    reason_weight = {
        "memory_bound_failure": 200_000.0,
        "killed_or_time_budget_exhausted": 180_000.0,
        "runtime_failure": 150_000.0,
        "high_duration_hotspot": 70_000.0,
        "high_memory_hotspot": 60_000.0,
        "healthy": 0.0,
    }.get(reason, 0.0)
    runtime_weight = float(max(0, row.duration_ms))
    memory_weight = float(max(0, row.max_rss_kb))
    return status_weight + reason_weight + runtime_weight + memory_weight


def _markdown(payload: dict[str, Any]) -> str:
    lines = []
    lines.append("# Optimal 4-Pillars Triage")
    lines.append("")
    lines.append(f"- run_id: `{payload['run_id']}`")
    lines.append(f"- generated_at: `{payload['generated_at']}`")
    lines.append(
        f"- totals: plugins={payload['totals']['plugins']} "
        f"ok={payload['totals']['ok']} error={payload['totals']['error']} "
        f"running={payload['totals']['running']} degraded={payload['totals']['degraded']}"
    )
    lines.append("")
    lines.append("## Priority Queue")
    lines.append("")
    lines.append("| rank | plugin_id | status | reason | duration_ms | max_rss_kb | next_step |")
    lines.append("|---:|---|---|---|---:|---:|---|")
    for item in payload["priority_queue"]:
        lines.append(
            f"| {item['rank']} | `{item['plugin_id']}` | `{item['status']}` | "
            f"`{item['reason']}` | {item['duration_ms']} | {item['max_rss_kb']} | "
            f"{item['recommended_next_step']} |"
        )
    lines.append("")
    lines.append("## Workflow")
    lines.append("")
    lines.append("1. Fix top queue items first (failure > runtime > memory).")
    lines.append("2. Run targeted tests for touched plugins.")
    lines.append("3. Run one full gauntlet (`--plugin-set full --force`).")
    lines.append("4. Compare before/after with `scripts/compare_run_outputs.py` and `scripts/compare_plugin_actionability_runs.py`.")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Build deterministic 4-pillars plugin triage from a run.")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--top-n", type=int, default=40)
    args = ap.parse_args()

    run_id = str(args.run_id).strip()
    db_path = Path(args.db)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with _connect(db_path) as con:
        rows = _fetch_rows(con, run_id)
    if not rows:
        raise SystemExit(f"run_id_not_found:{run_id}")

    queue: list[dict[str, Any]] = []
    for row in rows:
        reason = _reason(row)
        score = _priority_score(row, reason)
        queue.append(
            {
                "plugin_id": row.plugin_id,
                "status": row.status,
                "reason": reason,
                "duration_ms": row.duration_ms,
                "max_rss_kb": row.max_rss_kb,
                "exit_code": row.exit_code,
                "priority_score": score,
                "recommended_next_step": _recommended_next_step(reason),
            }
        )
    queue.sort(
        key=lambda item: (
            -float(item["priority_score"]),
            str(item["plugin_id"]),
        )
    )
    top_n = max(1, int(args.top_n))
    top = queue[:top_n]
    for idx, item in enumerate(top, start=1):
        item["rank"] = idx

    payload = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "totals": {
            "plugins": len(rows),
            "ok": sum(1 for r in rows if r.status == "ok"),
            "error": sum(1 for r in rows if r.status == "error"),
            "running": sum(1 for r in rows if r.status == "running"),
            "degraded": sum(1 for r in rows if r.status == "degraded"),
            "aborted": sum(1 for r in rows if r.status == "aborted"),
        },
        "priority_queue": top,
    }

    json_path = out_dir / f"optimal_4pillars_triage_{run_id}.json"
    md_path = out_dir / f"optimal_4pillars_triage_{run_id}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")

    print(f"run_id={run_id}")
    print(f"out_json={json_path}")
    print(f"out_md={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

