#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from statistic_harness.core.recommendation_filters import is_specific_process_target


ROOT = Path(__file__).resolve().parents[1]


DIRECT_PROCESS_ACTION_TYPES = {
    "add_server",
    "batch_input",
    "batch_group_candidate",
    "batch_or_cache",
    "batch_input_refactor",
    "throttle_or_dedupe",
    "dedupe_or_cache",
    "orchestrate_macro",
    "ideaspace_action",
    "route_process",
    "reschedule",
    "tune_schedule",
    "schedule_shift_target",
    "reduce_spillover_past_eom",
    "add_upload_linkage",
    "reduce_close_cycle_slowdown",
    "isolate_process",
    "cap_concurrency",
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _recommendations(report: dict[str, Any]) -> list[dict[str, Any]]:
    recs = report.get("recommendations")
    if not isinstance(recs, dict):
        return []
    items = recs.get("items")
    if not isinstance(items, list):
        return []
    return [row for row in items if isinstance(row, dict)]


def _process_hint(item: dict[str, Any]) -> str:
    for key in ("process_norm", "process_id", "process"):
        token = str(item.get(key) or "").strip()
        if token:
            return token
    where = item.get("where")
    if isinstance(where, dict):
        for key in ("process_norm", "process_id", "process"):
            token = str(where.get(key) or "").strip()
            if token:
                return token
    return ""


def audit_process_targeting(run_dir: Path) -> dict[str, Any]:
    report = _load_json(run_dir / "report.json")
    rows: list[dict[str, Any]] = []
    violation_count = 0

    for item in _recommendations(report):
        action_type = str(item.get("action_type") or item.get("action") or "").strip().lower()
        process_hint = _process_hint(item)
        violations: list[str] = []
        if action_type in DIRECT_PROCESS_ACTION_TYPES and not is_specific_process_target(process_hint):
            violations.append("DIRECT_ACTION_MISSING_SPECIFIC_PROCESS_TARGET")
        if violations:
            violation_count += 1
        rows.append(
            {
                "plugin_id": str(item.get("plugin_id") or "").strip() or None,
                "action_type": action_type or None,
                "process_hint": process_hint or None,
                "violations": violations,
            }
        )

    return {
        "schema_version": "process_targeting_audit.v1",
        "run_dir": str(run_dir),
        "recommendation_count": len(rows),
        "violation_count": violation_count,
        "ok": violation_count == 0,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit direct-action recommendations for process targeting.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-root", default=str(ROOT / "appdata" / "runs"))
    parser.add_argument("--out-json", default="")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    run_dir = Path(str(args.runs_root)) / str(args.run_id).strip()
    payload = audit_process_targeting(run_dir)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    out_json = str(args.out_json).strip()
    if out_json:
        out_path = Path(out_json)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if bool(args.strict) and not bool(payload.get("ok")):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

