#!/usr/bin/env python3
"""Regenerate report.json and report.md from cached plugin results.

Usage:
    python scripts/regenerate_report.py --run-id <run_id>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from statistic_harness.core.storage import Storage
from statistic_harness.core.report import build_report, write_report

APPDATA = REPO_ROOT / "appdata"
SCHEMA = REPO_ROOT / "docs" / "report.schema.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    run_id = args.run_id
    run_dir = APPDATA / "runs" / run_id
    db_path = APPDATA / "state.sqlite"

    if not run_dir.exists():
        print(f"ERROR: run dir not found: {run_dir}")
        sys.exit(1)
    if not db_path.exists():
        print(f"ERROR: db not found: {db_path}")
        sys.exit(1)

    print(f"Regenerating report for {run_id} ...")
    storage = Storage(db_path, mode="ro")
    report = build_report(storage, run_id, run_dir, SCHEMA)
    write_report(report, run_dir)
    print(f"Done. Output: {run_dir}/report.json")

    # Print top-20 recommendations summary
    recs = report.get("recommendations", {})
    known_items = []
    disc_items = []
    if "known" in recs or "discovery" in recs:
        known = recs.get("known", {})
        disc = recs.get("discovery", {})
        known_items = known.get("items", []) if isinstance(known, dict) else []
        disc_items = disc.get("items", []) if isinstance(disc, dict) else []
    else:
        disc_items = recs.get("items", []) if isinstance(recs, dict) else []

    all_items = known_items + disc_items

    print(f"\n=== TOP RECOMMENDATIONS ({len(all_items)} total) ===\n")

    # Count distinct plugins
    plugin_ids = set()
    kind_counts: dict[str, int] = {}
    for item in all_items:
        if isinstance(item, dict):
            pid = item.get("plugin_id", "")
            plugin_ids.add(pid)
            k = item.get("kind", "unknown")
            kind_counts[k] = kind_counts.get(k, 0) + 1

    print(f"Distinct plugins contributing: {len(plugin_ids)}")
    print(f"Kind distribution: {json.dumps(kind_counts, indent=2)}\n")

    for i, item in enumerate(all_items[:20], 1):
        if not isinstance(item, dict):
            continue
        title = item.get("title", "?")
        plugin = item.get("plugin_id", "?")
        kind = item.get("kind", "?")
        score = item.get("weighted_rank_score", 0)
        vs2 = item.get("value_score_v2", 0)
        action = item.get("action_type", "?")
        process = item.get("primary_process", "?")
        delta_h = item.get("modeled_delta_hours", 0)
        print(f"  {i:>2}. [{kind}] {title[:70]}")
        print(f"      plugin={plugin}  score={score:.4f}  value_v2={vs2}  action={action}")
        print(f"      process={process}  delta_h={delta_h}")
        print()


if __name__ == "__main__":
    main()
