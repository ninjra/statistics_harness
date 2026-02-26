from __future__ import annotations

from pathlib import Path

from statistic_harness.core.payout_report import build_payout_report
from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json


def _loader_column_names(ctx) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    storage = getattr(ctx, "storage", None)
    dataset_version_id = getattr(ctx, "dataset_version_id", None)
    if storage is None or not dataset_version_id:
        return []

    # First choice: use template field names, because dataset_loader validates
    # selected columns against template field names.
    dataset_template: dict | None = None
    try:
        dataset_template = storage.fetch_dataset_template(dataset_version_id)
    except Exception:
        dataset_template = None
    template_id = int(dataset_template.get("template_id")) if dataset_template and dataset_template.get("template_id") else None
    if template_id is not None:
        try:
            for row in storage.fetch_template_fields(template_id) or []:
                name = str(row.get("name") or "").strip()
                if name and name not in seen:
                    names.append(name)
                    seen.add(name)
        except Exception:
            pass
    if names:
        return names

    # Fallback: dataset original names, used when template mapping is missing.
    try:
        with storage.connection() as conn:
            rows = storage.fetch_dataset_columns(dataset_version_id, conn) or []
    except TypeError:
        rows = storage.fetch_dataset_columns(dataset_version_id) or []
    except Exception:
        rows = []
    for row in rows:
        name = str(row.get("original_name") or "").strip()
        if name and name not in seen:
            names.append(name)
            seen.add(name)
    return names


def _payout_report_columns(ctx) -> list[str]:
    # Keep payout report memory-safe by loading only columns needed for
    # process/param/time/source inference.
    exact_order = [
        "PROCESS_ID",
        "process_id",
        "process_norm",
        "activity",
        "ACTIVITY",
        "PARAM_DESCR_LIST",
        "param_descr_list",
        "PARAMS",
        "params",
        "QUEUE_DT",
        "queue_dt",
        "START_DT",
        "start_dt",
        "END_DT",
        "end_dt",
        "__source_file",
    ]
    available = _loader_column_names(ctx)
    if not available:
        return []
    selected: list[str] = []
    selected_set: set[str] = set()

    def _add(name: str) -> None:
        if name and name in available and name not in selected_set:
            selected.append(name)
            selected_set.add(name)

    for name in exact_order:
        _add(name)

    pattern_buckets = {
        "process": ("process",),
        "param": ("param", "period", "batch", "payout", "pay", "run_key", "key"),
        "queue": ("queue",),
        "start": ("start",),
        "end": ("end", "finish", "completed"),
        "source": ("source", "file"),
    }
    bucket_limits = {"process": 2, "param": 2, "queue": 1, "start": 1, "end": 1, "source": 1}
    bucket_counts = {key: 0 for key in pattern_buckets}
    for name in available:
        lowered = name.lower()
        for bucket, tokens in pattern_buckets.items():
            if bucket_counts[bucket] >= bucket_limits[bucket]:
                continue
            if any(token in lowered for token in tokens):
                _add(name)
                bucket_counts[bucket] += 1
                break
    if selected:
        return selected
    # Never fall back to full-table load. Use a bounded prefix subset so this
    # plugin remains memory-safe even when schema names are opaque.
    return available[:32]


class Plugin:
    def run(self, ctx) -> PluginResult:
        # Report plugins run late; keep logic local and deterministic.
        selected_columns = _payout_report_columns(ctx)
        if not selected_columns:
            return PluginResult(
                status="ok",
                summary="Payout report unavailable: no dataset columns available",
                metrics={"rows_seen": 0, "rows_used": 0, "cols_used": 0, "selected_column_count": 0},
                findings=[
                    {
                        "kind": "payout_report",
                        "summary": "No dataset columns available for payout report",
                        "metrics": {},
                        "measurement_type": "measured",
                    }
                ],
                artifacts=[],
                error=None,
            )
        try:
            df = ctx.dataset_loader(columns=selected_columns)
        except ValueError as exc:
            # Deterministic fallback: return an explanatory result instead of
            # hard-failing this report plugin when loader/template schemas drift.
            if "Unknown columns" in str(exc):
                return PluginResult(
                    status="ok",
                    summary="Payout report unavailable: loader columns unresolved",
                    metrics={"rows_seen": 0, "rows_used": 0, "cols_used": 0, "selected_column_count": int(len(selected_columns))},
                    findings=[
                        {
                            "kind": "payout_report",
                            "summary": "Payout report not generated because selected columns are not available in dataset loader mapping",
                            "metrics": {"selected_column_count": int(len(selected_columns))},
                            "measurement_type": "measured",
                        }
                    ],
                    artifacts=[],
                    error=None,
                )
            raise
        if df is None or df.empty:
            return PluginResult(
                status="ok",
                summary="Payout report unavailable: empty dataset",
                metrics={"rows_seen": 0, "rows_used": 0, "cols_used": 0, "selected_column_count": int(len(selected_columns))},
                findings=[
                    {
                        "kind": "payout_report",
                        "summary": "No rows available for payout report generation",
                        "metrics": {"selected_column_count": int(len(selected_columns))},
                        "measurement_type": "measured",
                    }
                ],
                artifacts=[],
                error=None,
            )

        regex = str(ctx.settings.get("payout_process_regex") or "")
        report = build_payout_report(df, payout_process_regex=regex) if regex else build_payout_report(df)

        artifacts_dir = ctx.artifacts_dir("report_payout_report_v1")
        json_path = artifacts_dir / "payout_report.json"
        csv_path = artifacts_dir / "payout_report.csv"

        # JSON (canonical)
        write_json(json_path, report)
        # CSV (human / spreadsheet friendly)
        rows = []
        overall = dict(report.get("metrics") or {})
        rows.append({"scope": "overall", **overall})
        for row in report.get("per_source") or []:
            if isinstance(row, dict):
                rows.append({"scope": "source", **row})
        import pandas as pd

        pd.DataFrame(rows).to_csv(csv_path, index=False)

        artifacts = [
            PluginArtifact(path=str(json_path.relative_to(ctx.run_dir)), type="json", description="payout_report.json"),
            PluginArtifact(path=str(csv_path.relative_to(ctx.run_dir)), type="csv", description="payout_report.csv"),
        ]
        findings = [
            {
                "kind": "payout_report",
                "summary": str(report.get("summary") or ""),
                "metrics": report.get("metrics") or {},
                "measurement_type": "measured",
            }
        ]
        return PluginResult(
            status="ok",
            summary="Generated payout report artifacts",
            metrics={
                "rows_seen": int(len(df)),
                "rows_used": int(len(df)),
                "cols_used": int(len(df.columns)),
                "selected_column_count": int(len(selected_columns)),
            },
            findings=findings,
            artifacts=artifacts,
            error=None,
        )
