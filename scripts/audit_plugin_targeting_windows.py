#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


WINDOW_FIELDS = (
    "delta_hours_accounting_month",
    "delta_hours_close_static",
    "delta_hours_close_dynamic",
    "efficiency_gain_pct_accounting_month",
    "efficiency_gain_pct_close_static",
    "efficiency_gain_pct_close_dynamic",
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _recommendation_items(report: dict[str, Any]) -> list[dict[str, Any]]:
    recs = report.get("recommendations")
    if not isinstance(recs, dict):
        return []
    items = recs.get("items")
    if not isinstance(items, list):
        return []
    return [row for row in items if isinstance(row, dict)]


def _has_window_metrics(item: dict[str, Any]) -> bool:
    for key in WINDOW_FIELDS:
        if isinstance(item.get(key), (int, float)):
            return True
    return False


def audit_targeting_windows(run_dir: Path) -> dict[str, Any]:
    report = _load_json(run_dir / "report.json")
    rows: list[dict[str, Any]] = []
    missing = 0
    for item in _recommendation_items(report):
        plugin_id = str(item.get("plugin_id") or "").strip()
        action_type = str(item.get("action_type") or item.get("action") or "").strip().lower()
        has_windows = _has_window_metrics(item)
        violations: list[str] = []
        if not has_windows:
            violations.append("MISSING_WINDOW_METRICS")
            missing += 1
        rows.append(
            {
                "plugin_id": plugin_id or None,
                "action_type": action_type or None,
                "has_window_metrics": has_windows,
                "violations": violations,
            }
        )

    return {
        "schema_version": "targeting_window_audit.v1",
        "run_dir": str(run_dir),
        "recommendation_count": len(rows),
        "missing_window_metric_count": missing,
        "ok": missing == 0,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit recommendation window targeting fields.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-root", default=str(ROOT / "appdata" / "runs"))
    parser.add_argument("--out-json", default="")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    run_dir = Path(str(args.runs_root)) / str(args.run_id).strip()
    payload = audit_targeting_windows(run_dir)
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

