from __future__ import annotations

from typing import Any

import pandas as pd

from statistic_harness.core.column_inference import choose_timestamp_column
from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json


def _pick_column(
    preferred: str | None,
    columns: list[str],
    role_by_name: dict[str, str],
    roles: set[str],
    patterns: list[str],
    lower_names: dict[str, str],
) -> str | None:
    if preferred and preferred in columns:
        return preferred
    for col in columns:
        if role_by_name.get(col) in roles:
            return col
    for col in columns:
        name = lower_names[col]
        if any(pattern in name for pattern in patterns):
            return col
    return None


def _pick_timestamp_column(
    preferred: str | None,
    columns: list[str],
    role_by_name: dict[str, str],
    roles: set[str],
    patterns: list[str],
    lower_names: dict[str, str],
    df: pd.DataFrame,
) -> str | None:
    candidates: list[str] = []
    if preferred and preferred in columns:
        candidates.append(preferred)
    for col in columns:
        if role_by_name.get(col) in roles and col not in candidates:
            candidates.append(col)
    for col in columns:
        name = lower_names[col]
        if any(pattern in name for pattern in patterns) and col not in candidates:
            candidates.append(col)
    if not candidates:
        return None
    return choose_timestamp_column(df, candidates)


def _series_from_frame(frame: pd.DataFrame, column: str | None) -> pd.Series | None:
    if not column:
        return None
    data = frame[column]
    if isinstance(data, pd.DataFrame):
        return data.iloc[:, 0]
    return data


class Plugin:
    def run(self, ctx) -> PluginResult:
        df = ctx.dataset_loader()
        if df.empty:
            return PluginResult("skipped", "Empty dataset", {}, [], [], None)

        columns_meta = []
        role_by_name: dict[str, str] = {}
        if ctx.dataset_version_id:
            dataset_template = ctx.storage.fetch_dataset_template(ctx.dataset_version_id)
            if dataset_template and dataset_template.get("status") == "ready":
                fields = ctx.storage.fetch_template_fields(
                    int(dataset_template["template_id"])
                )
                columns_meta = fields
                role_by_name = {
                    field["name"]: (field.get("role") or "") for field in fields
                }
            else:
                columns_meta = ctx.storage.fetch_dataset_columns(ctx.dataset_version_id)
                role_by_name = {
                    col["original_name"]: (col.get("role") or "")
                    for col in columns_meta
                }

        columns = list(df.columns)
        lower_names = {col: str(col).lower() for col in columns}

        process_col = _pick_column(
            ctx.settings.get("process_column"),
            columns,
            role_by_name,
            {"process_name", "process_id"},
            ["process", "activity", "event", "step", "task", "action", "job"],
            lower_names,
        )
        module_col = _pick_column(
            ctx.settings.get("module_column"),
            columns,
            role_by_name,
            {"module_code"},
            ["module", "mod"],
            lower_names,
        )
        queue_col = _pick_timestamp_column(
            ctx.settings.get("queue_column"),
            columns,
            role_by_name,
            {"queue_time"},
            ["queue", "queued", "enqueue"],
            lower_names,
            df,
        )
        eligible_col = _pick_timestamp_column(
            ctx.settings.get("eligible_column"),
            columns,
            role_by_name,
            {"eligible", "ready", "available"},
            ["eligible", "ready", "available"],
            lower_names,
            df,
        )
        start_col = _pick_timestamp_column(
            ctx.settings.get("start_column"),
            columns,
            role_by_name,
            {"start_time", "start"},
            ["start", "begin"],
            lower_names,
            df,
        )
        end_col = _pick_timestamp_column(
            ctx.settings.get("end_column"),
            columns,
            role_by_name,
            {"end_time", "end"},
            ["end", "finish", "complete", "stop"],
            lower_names,
            df,
        )

        if not start_col or (not queue_col and not eligible_col):
            return PluginResult(
                "ok",
                "Percentile analysis not applicable",
                {"groups": 0},
                [
                    {
                        "kind": "percentile_stats",
                        "measurement_type": "not_applicable",
                        "reason": "Missing queue/eligible and start columns.",
                        "columns": [
                            col
                            for col in [queue_col, eligible_col, start_col, end_col]
                            if col
                        ],
                    }
                ],
                [],
                None,
            )

        selected = [
            col
            for col in [
                process_col,
                module_col,
                queue_col,
                eligible_col,
                start_col,
                end_col,
            ]
            if col
        ]
        selected = list(dict.fromkeys(selected))
        work = df.loc[:, selected].copy()
        eligible_series = _series_from_frame(work, eligible_col)
        eligible_ts = (
            pd.to_datetime(eligible_series, errors="coerce", utc=False)
            if eligible_series is not None
            else None
        )
        if eligible_ts is None:
            queue_series = _series_from_frame(work, queue_col)
            eligible_ts = pd.to_datetime(queue_series, errors="coerce", utc=False)
            eligible_basis = "queue"
        else:
            eligible_basis = "eligible"
        start_series = _series_from_frame(work, start_col)
        start_ts = pd.to_datetime(start_series, errors="coerce", utc=False)
        end_series = _series_from_frame(work, end_col)
        end_ts = (
            pd.to_datetime(end_series, errors="coerce", utc=False)
            if end_series is not None
            else None
        )
        work["__eligible_ts"] = eligible_ts
        work["__start_ts"] = start_ts
        if end_ts is not None:
            work["__end_ts"] = end_ts

        work = work.loc[work["__eligible_ts"].notna() & work["__start_ts"].notna()].copy()
        if work.empty:
            return PluginResult("ok", "No valid timestamps", {"groups": 0}, [], [], None)

        work["__eligible_wait_sec"] = (
            work["__start_ts"] - work["__eligible_ts"]
        ).dt.total_seconds().clip(lower=0).fillna(0)
        if end_ts is not None:
            work["__completion_sec"] = (
                work["__end_ts"] - work["__start_ts"]
            ).dt.total_seconds().clip(lower=0).fillna(0)

        group_cols: list[str] = []
        if process_col:
            group_cols.append(process_col)
        if module_col:
            group_cols.append(module_col)

        max_groups = int(ctx.settings.get("max_groups", 10))
        findings = []

        if group_cols:
            grouped = work.groupby(group_cols, dropna=False)
        else:
            grouped = [("overall", work)]

        for key, frame in grouped:
            if isinstance(key, tuple):
                label = {
                    "process": key[0] if process_col else None,
                    "module": key[1] if module_col and len(key) > 1 else None,
                }
            else:
                label = {"process": key if process_col else None, "module": None}

            eligible_quantiles = frame["__eligible_wait_sec"].quantile([0.5, 0.95, 0.99])
            completion_quantiles = None
            if "__completion_sec" in frame:
                completion_quantiles = frame["__completion_sec"].quantile([0.5, 0.95, 0.99])

            finding = {
                "kind": "percentile_stats",
                "process": label.get("process"),
                "module": label.get("module"),
                "eligible_wait_p50": float(eligible_quantiles.loc[0.5]),
                "eligible_wait_p95": float(eligible_quantiles.loc[0.95]),
                "eligible_wait_p99": float(eligible_quantiles.loc[0.99]),
                "eligible_basis": eligible_basis,
                "measurement_type": "measured",
                "columns": [
                    col
                    for col in [process_col, module_col, queue_col, eligible_col, start_col, end_col]
                    if col
                ],
            }
            if completion_quantiles is not None:
                finding.update(
                    {
                        "completion_p50": float(completion_quantiles.loc[0.5]),
                        "completion_p95": float(completion_quantiles.loc[0.95]),
                        "completion_p99": float(completion_quantiles.loc[0.99]),
                    }
                )
            findings.append(finding)
            if len(findings) >= max_groups:
                break

        artifacts_dir = ctx.artifacts_dir("analysis_percentile_analysis")
        out_path = artifacts_dir / "percentiles.json"
        write_json(
            out_path,
            {
                "summary": {
                    "process_column": process_col,
                    "module_column": module_col,
                    "queue_column": queue_col,
                    "eligible_column": eligible_col,
                    "start_column": start_col,
                    "end_column": end_col,
                    "eligible_basis": eligible_basis,
                },
                "groups": len(findings),
            },
        )
        artifacts = [
            PluginArtifact(
                path=str(out_path.relative_to(ctx.run_dir)),
                type="json",
                description="Percentile analysis summary",
            )
        ]

        return PluginResult(
            "ok",
            "Computed percentile statistics",
            {"groups": len(findings)},
            findings,
            artifacts,
            None,
        )
