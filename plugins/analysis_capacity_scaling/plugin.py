from __future__ import annotations

from typing import Any

import pandas as pd

from statistic_harness.core.column_inference import choose_timestamp_column
from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json


INVALID_STRINGS = {"", "nan", "none", "null"}


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _pick_column(
    preferred: str | None,
    columns: list[str],
    role_by_name: dict[str, str],
    roles: set[str],
    patterns: list[str],
    lower_names: dict[str, str],
    exclude: set[str],
) -> str | None:
    if preferred and preferred in columns:
        return preferred
    for col in columns:
        if col in exclude:
            continue
        if role_by_name.get(col) in roles:
            return col
    for col in columns:
        if col in exclude:
            continue
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
    exclude: set[str],
    df: pd.DataFrame,
) -> str | None:
    candidates: list[str] = []
    if preferred and preferred in columns and preferred not in exclude:
        candidates.append(preferred)
    for col in columns:
        if col in exclude:
            continue
        if role_by_name.get(col) in roles and col not in candidates:
            candidates.append(col)
    for col in columns:
        if col in exclude:
            continue
        name = lower_names[col]
        if any(pattern in name for pattern in patterns) and col not in candidates:
            candidates.append(col)
    if not candidates:
        return None
    return choose_timestamp_column(df, candidates)


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
        used: set[str] = set()

        process_col = _pick_column(
            ctx.settings.get("process_column"),
            columns,
            role_by_name,
            {"process_name", "process_id"},
            ["process", "activity", "event", "step", "task", "action", "job"],
            lower_names,
            used,
        )
        if process_col:
            used.add(process_col)

        module_col = _pick_column(
            ctx.settings.get("module_column"),
            columns,
            role_by_name,
            {"module_code"},
            ["module", "mod"],
            lower_names,
            used,
        )
        if module_col:
            used.add(module_col)

        host_col = _pick_column(
            ctx.settings.get("host_column"),
            columns,
            role_by_name,
            {"server", "host", "node", "instance"},
            ["server", "host", "node", "instance", "machine"],
            lower_names,
            used,
        )
        if host_col:
            used.add(host_col)

        queue_col = _pick_timestamp_column(
            ctx.settings.get("queue_column"),
            columns,
            role_by_name,
            {"queue", "queued", "enqueue"},
            ["queue", "queued", "enqueue"],
            lower_names,
            used,
            df,
        )
        if queue_col:
            used.add(queue_col)

        eligible_col = _pick_timestamp_column(
            ctx.settings.get("eligible_column"),
            columns,
            role_by_name,
            {"eligible", "ready", "available"},
            ["eligible", "ready", "available"],
            lower_names,
            used,
            df,
        )
        if eligible_col:
            used.add(eligible_col)

        start_col = _pick_timestamp_column(
            ctx.settings.get("start_column"),
            columns,
            role_by_name,
            {"start_time", "start"},
            ["start", "begin"],
            lower_names,
            used,
            df,
        )
        if start_col:
            used.add(start_col)

        if not start_col or (not queue_col and not eligible_col):
            return PluginResult(
                "ok",
                "Capacity scaling not applicable",
                {"rows": 0},
                [
                    {
                        "kind": "capacity_scaling",
                        "measurement_type": "not_applicable",
                        "reason": "Missing queue/eligible and start columns.",
                        "columns": [
                            col for col in [queue_col, eligible_col, start_col] if col
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
                host_col,
                queue_col,
                eligible_col,
                start_col,
            ]
            if col
        ]
        work = df.loc[:, selected].copy()

        eligible_ts = (
            pd.to_datetime(work[eligible_col], errors="coerce", utc=False)
            if eligible_col
            else None
        )
        if eligible_ts is None:
            eligible_ts = pd.to_datetime(work[queue_col], errors="coerce", utc=False)
            eligible_basis = "queue"
        else:
            eligible_basis = "eligible"
        start_ts = pd.to_datetime(work[start_col], errors="coerce", utc=False)

        work["__eligible_ts"] = eligible_ts
        work["__start_ts"] = start_ts
        work = work.loc[work["__eligible_ts"].notna() & work["__start_ts"].notna()].copy()
        if work.empty:
            return PluginResult(
                "ok",
                "No valid timestamps",
                {"rows": 0},
                [],
                [],
                None,
            )

        work["__eligible_wait_sec"] = (
            work["__start_ts"] - work["__eligible_ts"]
        ).dt.total_seconds().clip(lower=0).fillna(0)

        host_count = 0
        if host_col and host_col in work.columns:
            work["__host"] = work[host_col].map(_normalize_text)
            work["__host_norm"] = work["__host"].str.lower()
            work = work.loc[~work["__host_norm"].isin(INVALID_STRINGS)].copy()
            host_count = int(work["__host_norm"].nunique())

        raw_scale_factor = ctx.settings.get("scale_factor")
        scale_factor = raw_scale_factor
        try:
            scale_factor = float(scale_factor) if scale_factor is not None else None
        except (TypeError, ValueError):
            scale_factor = None
        derived_scale = scale_factor is None
        if scale_factor is None:
            if host_count >= 1:
                scale_factor = (host_count + 1) / float(host_count)
            else:
                scale_factor = 1.0
        if scale_factor <= 0:
            scale_factor = 1.0
        host_count_modeled = None
        if derived_scale and host_count > 0:
            host_count_modeled = host_count + 1

        work["__scaled_wait_sec"] = work["__eligible_wait_sec"] / float(scale_factor)
        work["__reduction_sec"] = work["__eligible_wait_sec"] - work["__scaled_wait_sec"]

        assumptions = [
            "Eligible wait scales inversely with capacity.",
            "Prerequisite wait unchanged.",
        ]
        scope = {"metric": "eligible_wait_hours", "eligible_basis": eligible_basis}

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
                    rows=("__key", "size"),
                    baseline_sec=("__eligible_wait_sec", "sum"),
                    modeled_sec=("__scaled_wait_sec", "sum"),
                    reduction_sec=("__reduction_sec", "sum"),
                )
                .reset_index()
            )
            grouped = grouped.sort_values(
                ["baseline_sec", "__key"], ascending=[False, True]
            )
            for _, row in grouped.head(max_groups).iterrows():
                key = row["__key"]
                row_ids = temp.loc[temp["__key"] == key].index.tolist()[:max_examples]
                baseline_hours = float(row["baseline_sec"]) / 3600.0
                modeled_hours = float(row["modeled_sec"]) / 3600.0
                findings.append(
                    {
                        "kind": "capacity_scaling",
                        "dimension": name,
                        "key": key,
                        "rows": int(row["rows"]),
                        "baseline_wait_hours": baseline_hours,
                        "modeled_wait_hours": modeled_hours,
                        "reduction_hours": float(row["reduction_sec"]) / 3600.0,
                        "scale_factor": float(scale_factor),
                        "scale_factor_standard": float(scale_factor),
                        "scale_factor_original": float(scale_factor),
                        "scale_factor_original_definition": "scale_factor_standard = new_host_count / baseline_host_count",
                        "host_count_baseline": host_count or None,
                        "host_count_modeled": host_count_modeled,
                        "baseline_host_count": host_count or None,
                        "modeled_host_count": host_count_modeled,
                        "eligible_basis": eligible_basis,
                        "assumptions": assumptions,
                        "scope": scope,
                        "modeled_assumptions": assumptions,
                        "modeled_scope": scope,
                        "baseline_value": baseline_hours,
                        "modeled_value": modeled_hours,
                        "delta_value": modeled_hours - baseline_hours,
                        "unit": "hours",
                        "measurement_type": "modeled",
                        "row_ids": [int(i) for i in row_ids],
                        "columns": [
                            col
                            for col in [process_col, module_col, host_col, queue_col, eligible_col, start_col]
                            if col
                        ],
                    }
                )

        _emit_dimension("process", process_col)
        _emit_dimension("module", module_col)

        total_baseline = float(work["__eligible_wait_sec"].sum())
        total_modeled = float(work["__scaled_wait_sec"].sum())
        total_reduction = float(work["__reduction_sec"].sum())

        baseline_hours = total_baseline / 3600.0
        modeled_hours = total_modeled / 3600.0
        findings.insert(
            0,
            {
                "kind": "capacity_scaling",
                "dimension": "overall",
                "key": "overall",
                "rows": int(work.shape[0]),
                "baseline_wait_hours": baseline_hours,
                "modeled_wait_hours": modeled_hours,
                "reduction_hours": total_reduction / 3600.0,
                "scale_factor": float(scale_factor),
                "scale_factor_standard": float(scale_factor),
                "scale_factor_original": float(scale_factor),
                "scale_factor_original_definition": "scale_factor_standard = new_host_count / baseline_host_count",
                "host_count_baseline": host_count or None,
                "host_count_modeled": host_count_modeled,
                "baseline_host_count": host_count or None,
                "modeled_host_count": host_count_modeled,
                "eligible_basis": eligible_basis,
                "assumptions": assumptions,
                "scope": scope,
                "modeled_assumptions": assumptions,
                "modeled_scope": scope,
                "baseline_value": baseline_hours,
                "modeled_value": modeled_hours,
                "delta_value": modeled_hours - baseline_hours,
                "unit": "hours",
                "measurement_type": "modeled",
                "row_ids": [int(i) for i in work.index.tolist()[:max_examples]],
                "columns": [
                    col
                    for col in [process_col, module_col, host_col, queue_col, eligible_col, start_col]
                    if col
                ],
            },
        )

        metrics = {
            "rows": int(work.shape[0]),
            "baseline_wait_hours": total_baseline / 3600.0,
            "modeled_wait_hours": total_modeled / 3600.0,
            "reduction_hours": total_reduction / 3600.0,
            "scale_factor": float(scale_factor),
            "host_count": host_count,
        }

        artifacts_dir = ctx.artifacts_dir("analysis_capacity_scaling")
        out_path = artifacts_dir / "capacity_scaling.json"
        write_json(
            out_path,
            {
                "summary": {
                    "process_column": process_col,
                    "module_column": module_col,
                    "host_column": host_col,
                    "queue_column": queue_col,
                    "eligible_column": eligible_col,
                    "start_column": start_col,
                "eligible_basis": eligible_basis,
                "scale_factor": float(scale_factor),
                "assumptions": assumptions,
            },
                "metrics": metrics,
            },
        )
        artifacts = [
            PluginArtifact(
                path=str(out_path.relative_to(ctx.run_dir)),
                type="json",
                description="Capacity scaling summary",
            )
        ]

        return PluginResult(
            "ok",
            "Modeled capacity scaling impact",
            metrics,
            findings,
            artifacts,
            None,
        )
