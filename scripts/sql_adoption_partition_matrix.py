#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

GROUP_A = "group_A_shared_layer_covered"
GROUP_B = "group_B_direct_sql_benefit"
GROUP_C = "group_C_not_applicable_reclassify"
GROUP_D = "group_D_optional_defer"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _is_registry_wrapper(plugin_text: str) -> bool:
    return (
        "statistic_harness.core.stat_plugins.registry" in plugin_text
        and "run_plugin(" in plugin_text
    )


def _assignment_group(
    *,
    sql_intent: str,
    contracts: set[str],
    is_registry: bool,
) -> tuple[str, str]:
    if sql_intent == "not_applicable":
        return (
            GROUP_C,
            "Intent is not_applicable; classify for intent cleanup instead of SQL implementation.",
        )
    if is_registry:
        return (
            GROUP_A,
            "Registry-backed wrapper; shared stat-plugin SQL path should supersede local SQL edits.",
        )
    if "dataset_loader" in contracts:
        return (
            GROUP_B,
            "Plugin reads through dataset_loader directly; local SQL path is a direct optimization candidate.",
        )
    return (
        GROUP_D,
        "Low-confidence SQL benefit signal; keep deferred until higher-value candidates are complete.",
    )


def generate_payload(root: Path) -> dict[str, Any]:
    sql_adoption = _load_json(root / "docs" / "sql_assist_adoption_matrix.json")
    access_matrix = _load_json(root / "docs" / "plugin_data_access_matrix.json")

    access_by_plugin = {
        row.get("plugin_id"): row
        for row in (access_matrix.get("plugins") or [])
        if isinstance(row, dict) and isinstance(row.get("plugin_id"), str)
    }

    assignments: list[dict[str, Any]] = []
    groups: dict[str, list[str]] = {
        GROUP_A: [],
        GROUP_B: [],
        GROUP_C: [],
        GROUP_D: [],
    }

    for row in sql_adoption.get("plugins") or []:
        if not isinstance(row, dict):
            continue
        plugin_id = row.get("plugin_id")
        if not isinstance(plugin_id, str):
            continue
        sql_intent = str(row.get("sql_intent") or "").strip()
        uses_sql = bool(
            row.get("uses_sql_effective")
            or row.get("uses_sql")
            or row.get("uses_sql_exec")
        )
        if sql_intent != "recommended" or uses_sql:
            continue

        access_row = access_by_plugin.get(plugin_id, {})
        contracts = set(access_row.get("access_contracts") or [])
        uses_loader_unbounded = bool(access_row.get("uses_dataset_loader_unbounded"))

        plugin_path = root / "plugins" / plugin_id / "plugin.py"
        plugin_text = plugin_path.read_text(encoding="utf-8") if plugin_path.exists() else ""
        is_registry = _is_registry_wrapper(plugin_text)

        group, reason = _assignment_group(
            sql_intent=sql_intent,
            contracts=contracts,
            is_registry=is_registry,
        )
        groups[group].append(plugin_id)
        assignments.append(
            {
                "plugin_id": plugin_id,
                "group": group,
                "reason": reason,
                "signals": {
                    "sql_intent": sql_intent,
                    "uses_sql": uses_sql,
                    "is_registry_wrapper": is_registry,
                    "uses_dataset_loader_unbounded": uses_loader_unbounded,
                    "access_contracts": sorted(contracts),
                },
            }
        )

    for values in groups.values():
        values.sort()
    assignments.sort(key=lambda row: row["plugin_id"])

    candidate_ids = sorted([row["plugin_id"] for row in assignments])
    grouped_ids = [pid for group in groups.values() for pid in group]
    overlap_count = len(grouped_ids) - len(set(grouped_ids))
    unclassified = sorted(set(candidate_ids) - set(grouped_ids))
    missing_from_candidates = sorted(set(grouped_ids) - set(candidate_ids))

    payload = {
        "generated_by": "scripts/sql_adoption_partition_matrix.py",
        "source_files": [
            "docs/sql_assist_adoption_matrix.json",
            "docs/plugin_data_access_matrix.json",
        ],
        "candidate_count": len(candidate_ids),
        "candidate_plugin_ids": candidate_ids,
        "groups": groups,
        "assignments": assignments,
        "coverage": {
            "classified_count": len(grouped_ids),
            "unique_classified_count": len(set(grouped_ids)),
            "overlap_count": overlap_count,
            "unclassified_plugin_ids": unclassified,
            "unexpected_grouped_plugin_ids": missing_from_candidates,
            "is_complete": len(unclassified) == 0 and len(missing_from_candidates) == 0,
            "is_exclusive": overlap_count == 0,
        },
    }
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-json",
        default="docs/sql_adoption_partition_matrix.json",
    )
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    payload = generate_payload(ROOT)
    out_json = (ROOT / args.out_json).resolve()
    json_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"

    if args.verify:
        if not out_json.exists() or out_json.read_text(encoding="utf-8") != json_text:
            return 2
        return 0

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json_text, encoding="utf-8")
    print(f"out_json={out_json}")
    print(f"candidate_count={payload.get('candidate_count')}")
    print(f"group_A={len(payload['groups'].get(GROUP_A, []))}")
    print(f"group_B={len(payload['groups'].get(GROUP_B, []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
