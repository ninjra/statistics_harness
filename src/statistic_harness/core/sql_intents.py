from __future__ import annotations

from typing import Any


def default_sql_intents(schema_snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a stable list of query intents for text2sql generation.

    Intents describe *what* we want to compute; they do not prescribe the exact SQL.
    """

    tables = schema_snapshot.get("tables") if isinstance(schema_snapshot, dict) else None
    colset: set[str] = set()
    if isinstance(tables, list):
        for t in tables:
            cols = t.get("columns") if isinstance(t, dict) else None
            if not isinstance(cols, list):
                continue
            for c in cols:
                if isinstance(c, dict) and isinstance(c.get("name"), str):
                    colset.add(c["name"])

    intents: list[dict[str, Any]] = [
        {
            "id": "eventlog_core_projection",
            "purpose": "Project core event log columns used by most analyses (process, timestamps, duration).",
            "required_columns_any": [["PROCESS_ID", "PROCESS"], ["QUEUE_DT"], ["START_DT"], ["END_DT"]],
            "output_columns": ["PROCESS_ID", "QUEUE_DT", "START_DT", "END_DT"],
        },
        {
            "id": "per_process_wait_to_start_stats",
            "purpose": "Compute per-process wait-to-start distribution and over-threshold totals.",
            "required_columns_any": [["PROCESS_ID", "PROCESS"], ["QUEUE_DT"], ["START_DT"]],
            "output_columns": ["process", "n", "p50_wait_sec", "p90_wait_sec", "over_threshold_wait_sec_total"],
        },
        {
            "id": "per_process_hourly_medians",
            "purpose": "Compute per-process median duration by hour-of-day to find scheduling windows.",
            "required_columns_any": [["PROCESS_ID", "PROCESS"], ["QUEUE_DT", "START_DT"], ["START_DT"]],
            "output_columns": ["process", "hour", "n", "median_duration_sec"],
        },
        {
            "id": "dependency_hotspots",
            "purpose": "Find parent->child process pairs with high over-threshold wait contribution.",
            "required_columns_any": [["PROCESS_ID", "PROCESS"], ["DEPENDENCY_PROCESS_ID", "DEPENDENCY_PROCESS"], ["QUEUE_DT"], ["START_DT"]],
            "output_columns": ["parent_process", "child_process", "n", "over_threshold_wait_sec_total"],
        },
    ]

    for it in intents:
        required = it.get("required_columns_any") if isinstance(it, dict) else None
        present = []
        if isinstance(required, list):
            for group in required:
                if isinstance(group, list):
                    present.append([c for c in group if c in colset])
        it["present_candidates"] = present
    return intents

