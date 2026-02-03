from __future__ import annotations

from typing import Any

import pandas as pd

from statistic_harness.core.column_inference import choose_timestamp_column
from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json


INVALID_STRINGS = {"", "nan", "none", "null"}


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


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


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
        user_col = _pick_column(
            ctx.settings.get("user_column"),
            columns,
            role_by_name,
            {"user_id"},
            ["user", "owner", "operator"],
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

        if not start_col or (not queue_col and not eligible_col):
            return PluginResult(
                "ok",
                "Attribution not applicable",
                {"total_rows": 0},
                [
                    {
                        "kind": "attribution",
                        "measurement_type": "not_applicable",
                        "reason": "Missing queue/eligible and start columns.",
                        "columns": [
                            col
                            for col in [queue_col, eligible_col, start_col]
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
                user_col,
                queue_col,
                eligible_col,
                start_col,
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
        work["__eligible_ts"] = eligible_ts
        work["__start_ts"] = start_ts
        work = work.loc[work["__eligible_ts"].notna() & work["__start_ts"].notna()].copy()
        if work.empty:
            return PluginResult("ok", "No valid timestamps", {"total_rows": 0}, [], [], None)

        wait_sec = (work["__start_ts"] - work["__eligible_ts"]).dt.total_seconds()
        wait_sec = wait_sec.clip(lower=0).fillna(0)
        work["__eligible_wait_sec"] = wait_sec

        threshold = float(ctx.settings.get("wait_threshold_seconds", 60))
        work["__tail_wait_sec"] = work["__eligible_wait_sec"].where(
            work["__eligible_wait_sec"] > threshold, 0.0
        )

        max_groups = int(ctx.settings.get("max_groups", 5))
        max_examples = int(ctx.settings.get("max_examples", 25))

        findings = []

        def _emit_dimension(name: str, col: str | None) -> None:
            if not col or col not in work.columns:
                return
            temp = work.copy()
            temp["__key"] = temp[col].map(_normalize_text)
            temp = temp.loc[~temp["__key"].str.lower().isin(INVALID_STRINGS)].copy()
            if temp.empty:
                return
            grouped = (
                temp.groupby("__key")
                .agg(
                    runs=("__key", "size"),
                    wait_sec=("__eligible_wait_sec", "sum"),
                    tail_wait_sec=("__tail_wait_sec", "sum"),
                )
                .reset_index()
            )
            grouped = grouped.sort_values(
                ["wait_sec", "__key"], ascending=[False, True]
            )
            for _, row in grouped.head(max_groups).iterrows():
                key = row["__key"]
                row_ids = temp.loc[temp["__key"] == key].index.tolist()[:max_examples]
                findings.append(
                    {
                        "kind": "attribution",
                        "dimension": name,
                        "key": key,
                        "runs": int(row["runs"]),
                        "eligible_wait_hours": float(row["wait_sec"]) / 3600.0,
                        "tail_wait_hours": float(row["tail_wait_sec"]) / 3600.0,
                        "threshold_seconds": threshold,
                        "eligible_basis": eligible_basis,
                        "measurement_type": "measured",
                        "row_ids": [int(i) for i in row_ids],
                        "columns": [
                            col
                            for col in [process_col, module_col, user_col]
                            if col
                        ],
                    }
                )

        _emit_dimension("process", process_col)
        _emit_dimension("module", module_col)
        _emit_dimension("user", user_col)

        artifacts_dir = ctx.artifacts_dir("analysis_attribution")
        out_path = artifacts_dir / "attribution.json"
        write_json(
            out_path,
            {
                "summary": {
                    "process_column": process_col,
                    "module_column": module_col,
                    "user_column": user_col,
                    "queue_column": queue_col,
                    "eligible_column": eligible_col,
                    "start_column": start_col,
                    "eligible_basis": eligible_basis,
                    "threshold_seconds": threshold,
                },
                "total_rows": int(work.shape[0]),
            },
        )
        artifacts = [
            PluginArtifact(
                path=str(out_path.relative_to(ctx.run_dir)),
                type="json",
                description="Attribution summary",
            )
        ]

        return PluginResult(
            "ok",
            "Attributed eligible wait by dimension",
            {"total_rows": int(work.shape[0])},
            findings,
            artifacts,
            None,
        )
