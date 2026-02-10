#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
APPDATA = REPO_ROOT / "appdata"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _items_from_recs(recs: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(recs, dict):
        return [], []
    if "known" in recs or "discovery" in recs:
        known = recs.get("known") or {}
        disc = recs.get("discovery") or {}
        known_items = [i for i in (known.get("items") or []) if isinstance(i, dict)]
        disc_items = [i for i in (disc.get("items") or []) if isinstance(i, dict)]
        return known_items, disc_items
    return [], [i for i in (recs.get("items") or []) if isinstance(i, dict)]


def _fmt_where(where: Any) -> str:
    if not isinstance(where, dict) or not where:
        return ""
    # Keep it stable and compact.
    keys = ("process_norm", "process", "process_id", "activity", "parent_process", "child_process")
    out = {k: where.get(k) for k in keys if k in where}
    if not out:
        out = where
    try:
        return json.dumps(out, sort_keys=True)
    except Exception:
        return str(out)


def _ranked_actionables(disc_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not disc_items:
        return []

    def _priority(item: dict[str, Any]) -> int:
        plugin_id = str(item.get("plugin_id") or "")
        kind = str(item.get("kind") or "")
        action_type = str(item.get("action_type") or item.get("action") or "")
        if plugin_id == "analysis_actionable_ops_levers_v1" or kind == "actionable_ops_lever":
            return 6
        if plugin_id.startswith("analysis_ideaspace_"):
            return 5
        if "sequence" in plugin_id or "bottleneck" in plugin_id or "conformance" in plugin_id:
            return 4
        if plugin_id == "analysis_upload_linkage":
            return 3
        if action_type and action_type not in ("review", "tune_threshold"):
            return 2
        if plugin_id in ("analysis_queue_delay_decomposition", "analysis_busy_period_segmentation_v2"):
            return 1
        return 0

    def _score(item: dict[str, Any]) -> float:
        for key in ("relevance_score", "impact_hours", "modeled_delta"):
            try:
                v = item.get(key)
                if isinstance(v, (int, float)):
                    return float(v)
            except Exception:
                continue
        return 0.0

    # Sort highest priority and score first. Don't let a single plugin drown out all others;
    # selection-time caps handle breadth.
    return sorted(disc_items, key=lambda i: (_priority(i), _score(i)), reverse=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--top-n", type=int, default=25)
    ap.add_argument("--max-per-plugin", type=int, default=5)
    args = ap.parse_args()

    run_id = str(args.run_id).strip()
    run_dir = APPDATA / "runs" / run_id
    report_path = run_dir / "report.json"
    if not report_path.exists():
        raise SystemExit(f"Missing report.json: {report_path}")

    report = _read_json(report_path)
    recs = report.get("recommendations") if isinstance(report, dict) else None
    _, disc_items = _items_from_recs(recs)
    ranked = _ranked_actionables(disc_items)
    items: list[dict[str, Any]] = []
    per_plugin: dict[str, int] = {}
    for item in ranked:
        pid = str(item.get("plugin_id") or "")
        per_plugin[pid] = int(per_plugin.get(pid, 0)) + 1
        if per_plugin[pid] > int(args.max_per_plugin):
            continue
        items.append(item)
        if len(items) >= int(args.top_n):
            break

    print("# Actionable Results")
    print("")
    print(f"- run_id: {run_id}")
    print(f"- run_dir: {run_dir}")
    for rel in ("report.md", "answers_recommendations.md", "answers_summary.json"):
        p = run_dir / rel
        print(f"- {rel}: {p if p.exists() else '(missing)'}")

    print("")
    if not items:
        print("No discovery recommendations found in report.json.")
        return 0

    print(f"## Top {min(int(args.top_n), len(items))} Recommendations")
    for item in items[: int(args.top_n)]:
        txt = str(item.get("recommendation") or item.get("title") or "").strip()
        if not txt:
            continue
        plugin_id = str(item.get("plugin_id") or "")
        kind = str(item.get("kind") or "")
        where = _fmt_where(item.get("where"))
        delta = item.get("modeled_delta")
        if delta is None:
            delta = item.get("impact_hours")
        delta_txt = ""
        if isinstance(delta, (int, float)) and float(delta) != 0.0:
            delta_txt = f" delta_hours~={float(delta):.2f}"
        where_txt = f" where={where}" if where else ""
        print(f"- {txt} (plugin={plugin_id} kind={kind}{delta_txt}{where_txt})")
    print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
