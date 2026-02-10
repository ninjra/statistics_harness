from __future__ import annotations

from typing import Any

from statistic_harness.core.storage import Storage


def build_run_lineage(storage: Storage, run_id: str) -> dict[str, Any]:
    """Build a portable lineage block for a run.

    This is intentionally small and JSON-serializable so it can be embedded in
    `report.json` and exported without requiring DB access.
    """

    run_row = storage.get_run(run_id)
    if not run_row:
        return {"run_id": run_id, "status": "missing"}

    plugins: dict[str, Any] = {}
    for row in storage.list_plugin_results(run_id):
        plugin_id = row.get("plugin_id") or ""
        if not plugin_id:
            continue
        plugins[str(plugin_id)] = {
            "plugin_id": str(plugin_id),
            "plugin_version": row.get("plugin_version"),
            "completed_at": row.get("completed_at"),
            "status": row.get("status"),
            "summary": row.get("summary") or "",
            "execution_fingerprint": row.get("execution_fingerprint"),
        }

    return {
        "run_id": run_id,
        "created_at": run_row.get("created_at"),
        "completed_at": run_row.get("completed_at"),
        "status": run_row.get("status"),
        "input_hash": run_row.get("input_hash"),
        "dataset_id": run_row.get("dataset_id"),
        "dataset_version_id": run_row.get("dataset_version_id"),
        "plugins": plugins,
    }

