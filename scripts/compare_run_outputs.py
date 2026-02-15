#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from statistic_harness.core.storage import Storage
from statistic_harness.core.tenancy import get_tenant_context

_COMPONENT_KEYS = (
    "summary",
    "metrics_json",
    "findings_json",
    "artifacts_json",
    "error_json",
    "budget_json",
    "references_json",
    "debug_json",
)
_MATERIAL_COMPONENTS = {
    "metrics_json",
    "findings_json",
    "artifacts_json",
    "error_json",
    "budget_json",
    "references_json",
}


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "unknown")
        counts[status] = int(counts.get(status, 0)) + 1
    return counts


def _fetch_dataset_row(
    storage: Storage, dataset_version_id: str
) -> dict[str, Any] | None:
    tenant_id = storage._tenant_id()  # noqa: SLF001 - script-level read-only helper
    with storage.connection() as conn:
        cur = conn.execute(
            """
            SELECT dataset_version_id, dataset_id, created_at, row_count, column_count,
                   data_hash, source_classification
            FROM dataset_versions
            WHERE tenant_id = ? AND dataset_version_id = ?
            """,
            (tenant_id, dataset_version_id),
        )
        row = cur.fetchone()
    return dict(row) if row else None


def _resolve_latest_run_for_dataset(
    storage: Storage, dataset_version_id: str, statuses: list[str]
) -> dict[str, Any] | None:
    if not statuses:
        raise ValueError("statuses must not be empty")
    tenant_id = storage._tenant_id()  # noqa: SLF001 - script-level read-only helper
    placeholders = ",".join(["?"] * len(statuses))
    params: list[str] = [tenant_id, dataset_version_id]
    params.extend(statuses)
    with storage.connection() as conn:
        cur = conn.execute(
            f"""
            SELECT run_id, created_at, completed_at, status, requested_run_seed, run_seed,
                   dataset_version_id, input_filename
            FROM runs
            WHERE tenant_id = ? AND dataset_version_id = ? AND status IN ({placeholders})
            ORDER BY created_at DESC, run_id DESC
            LIMIT 1
            """,
            tuple(params),
        )
        row = cur.fetchone()
    return dict(row) if row else None


def _summarize_changed_plugin(
    plugin_id: str,
    row_a: dict[str, Any],
    row_b: dict[str, Any],
    changed_components: list[str],
) -> dict[str, Any]:
    return {
        "plugin_id": plugin_id,
        "status_before": str(row_a.get("status") or ""),
        "status_after": str(row_b.get("status") or ""),
        "changed_components": changed_components,
        "summary_before": str(row_a.get("summary") or ""),
        "summary_after": str(row_b.get("summary") or ""),
        "signature_before": _digest(
            "|".join(_as_text(row_a.get(key)) for key in _COMPONENT_KEYS)
        ),
        "signature_after": _digest(
            "|".join(_as_text(row_b.get(key)) for key in _COMPONENT_KEYS)
        ),
    }


def compare_runs(
    storage: Storage,
    run_before: str,
    run_after: str,
    *,
    ignore_components: set[str] | None = None,
    max_changed_plugins: int = 200,
) -> dict[str, Any]:
    ignore = set(ignore_components or set())
    row_before = storage.fetch_run(run_before)
    row_after = storage.fetch_run(run_after)
    if not row_before:
        raise ValueError(f"Run not found: {run_before}")
    if not row_after:
        raise ValueError(f"Run not found: {run_after}")

    results_before = {r["plugin_id"]: r for r in storage.fetch_plugin_results(run_before)}
    results_after = {r["plugin_id"]: r for r in storage.fetch_plugin_results(run_after)}
    plugin_ids = sorted(set(results_before.keys()) | set(results_after.keys()))

    added_plugins: list[str] = []
    removed_plugins: list[str] = []
    status_changes: list[dict[str, Any]] = []
    changed_plugins: list[dict[str, Any]] = []
    material_changed_plugins: list[str] = []
    component_change_counts = {k: 0 for k in _COMPONENT_KEYS}

    for plugin_id in plugin_ids:
        before = results_before.get(plugin_id)
        after = results_after.get(plugin_id)
        if before is None:
            added_plugins.append(plugin_id)
            continue
        if after is None:
            removed_plugins.append(plugin_id)
            continue

        status_before = str(before.get("status") or "")
        status_after = str(after.get("status") or "")
        if status_before != status_after:
            status_changes.append(
                {
                    "plugin_id": plugin_id,
                    "status_before": status_before,
                    "status_after": status_after,
                }
            )

        changed_components: list[str] = []
        for key in _COMPONENT_KEYS:
            if key in ignore:
                continue
            if _as_text(before.get(key)) != _as_text(after.get(key)):
                component_change_counts[key] += 1
                changed_components.append(key)
        if changed_components:
            if any(key in _MATERIAL_COMPONENTS for key in changed_components):
                material_changed_plugins.append(plugin_id)
            changed_plugins.append(
                _summarize_changed_plugin(plugin_id, before, after, changed_components)
            )

    changed_plugins.sort(
        key=lambda item: (
            0
            if item["status_before"] != item["status_after"]
            else 1,
            item["plugin_id"],
        )
    )

    return {
        "mode": "run_to_run",
        "run_before": {
            "run_id": run_before,
            "status": row_before.get("status"),
            "created_at": row_before.get("created_at"),
            "completed_at": row_before.get("completed_at"),
            "dataset_version_id": row_before.get("dataset_version_id"),
            "run_seed": row_before.get("run_seed"),
            "requested_run_seed": row_before.get("requested_run_seed"),
        },
        "run_after": {
            "run_id": run_after,
            "status": row_after.get("status"),
            "created_at": row_after.get("created_at"),
            "completed_at": row_after.get("completed_at"),
            "dataset_version_id": row_after.get("dataset_version_id"),
            "run_seed": row_after.get("run_seed"),
            "requested_run_seed": row_after.get("requested_run_seed"),
        },
        "plugin_counts": {
            "before": len(results_before),
            "after": len(results_after),
            "added": len(added_plugins),
            "removed": len(removed_plugins),
            "status_changed": len(status_changes),
            "payload_changed": len(changed_plugins),
            "material_payload_changed": len(material_changed_plugins),
        },
        "status_counts": {
            "before": _status_counts(list(results_before.values())),
            "after": _status_counts(list(results_after.values())),
        },
        "added_plugins": added_plugins,
        "removed_plugins": removed_plugins,
        "status_changes": status_changes,
        "component_change_counts": component_change_counts,
        "material_changed_plugins": sorted(material_changed_plugins),
        "changed_plugins": changed_plugins[: max(1, int(max_changed_plugins))],
    }


def compare_datasets(
    storage: Storage,
    dataset_before: str,
    dataset_after: str,
    *,
    statuses: list[str] | None = None,
    ignore_components: set[str] | None = None,
    max_changed_plugins: int = 200,
) -> dict[str, Any]:
    statuses = [s.strip() for s in (statuses or ["completed"]) if s.strip()]
    if not statuses:
        raise ValueError("At least one status is required for dataset comparison")

    dataset_row_before = _fetch_dataset_row(storage, dataset_before)
    dataset_row_after = _fetch_dataset_row(storage, dataset_after)
    if not dataset_row_before:
        raise ValueError(f"Dataset version not found: {dataset_before}")
    if not dataset_row_after:
        raise ValueError(f"Dataset version not found: {dataset_after}")

    run_row_before = _resolve_latest_run_for_dataset(storage, dataset_before, statuses)
    run_row_after = _resolve_latest_run_for_dataset(storage, dataset_after, statuses)
    if not run_row_before:
        raise ValueError(
            f"No runs found for dataset {dataset_before} with statuses={statuses}"
        )
    if not run_row_after:
        raise ValueError(
            f"No runs found for dataset {dataset_after} with statuses={statuses}"
        )

    comparison = compare_runs(
        storage,
        str(run_row_before["run_id"]),
        str(run_row_after["run_id"]),
        ignore_components=ignore_components,
        max_changed_plugins=max_changed_plugins,
    )
    comparison["mode"] = "dataset_to_dataset"
    comparison["dataset_before"] = dataset_row_before
    comparison["dataset_after"] = dataset_row_after
    comparison["dataset_run_selection"] = {
        "statuses": statuses,
        "run_before": run_row_before,
        "run_after": run_row_after,
    }
    return comparison


def _parse_csv(raw: str) -> list[str]:
    return [piece.strip() for piece in raw.replace(";", ",").split(",") if piece.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare plugin outputs either between two runs or between two dataset "
            "versions (latest run per dataset)."
        )
    )
    parser.add_argument("--run-before", default="")
    parser.add_argument("--run-after", default="")
    parser.add_argument("--dataset-before", default="")
    parser.add_argument("--dataset-after", default="")
    parser.add_argument(
        "--dataset-statuses",
        default="completed",
        help="Comma-separated run statuses used when selecting runs for dataset mode.",
    )
    parser.add_argument(
        "--ignore-components",
        default="",
        help=(
            "Comma-separated result fields to ignore when diffing payloads. "
            "Example: summary,debug_json"
        ),
    )
    parser.add_argument(
        "--max-changed-plugins",
        type=int,
        default=200,
        help="Maximum changed plugin entries to keep in the output payload.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional output path. If omitted, JSON is printed to stdout.",
    )
    args = parser.parse_args()

    run_before = str(args.run_before or "").strip()
    run_after = str(args.run_after or "").strip()
    dataset_before = str(args.dataset_before or "").strip()
    dataset_after = str(args.dataset_after or "").strip()

    run_mode = bool(run_before and run_after)
    dataset_mode = bool(dataset_before and dataset_after)
    if run_mode == dataset_mode:
        raise SystemExit(
            "Choose exactly one mode: (--run-before + --run-after) OR "
            "(--dataset-before + --dataset-after)."
        )

    ctx = get_tenant_context()
    storage = Storage(ctx.db_path, ctx.tenant_id)
    ignore_components = set(_parse_csv(str(args.ignore_components or "")))

    if run_mode:
        payload = compare_runs(
            storage,
            run_before,
            run_after,
            ignore_components=ignore_components,
            max_changed_plugins=max(1, int(args.max_changed_plugins)),
        )
    else:
        payload = compare_datasets(
            storage,
            dataset_before,
            dataset_after,
            statuses=_parse_csv(str(args.dataset_statuses or "")),
            ignore_components=ignore_components,
            max_changed_plugins=max(1, int(args.max_changed_plugins)),
        )

    output_json = str(args.output_json or "").strip()
    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(str(out_path))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
