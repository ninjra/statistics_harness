from __future__ import annotations

from typing import Any

import pandas as pd

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

        dep_col = _pick_column(
            ctx.settings.get("dependency_column"),
            columns,
            role_by_name,
            {"dependency_id"},
            ["dep", "dependency", "parent", "prereq", "preced"],
            lower_names,
        )
        id_col = _pick_column(
            ctx.settings.get("id_column"),
            columns,
            role_by_name,
            {"process_id", "id"},
            ["process_id", "proc_id", "id"],
            lower_names,
        )
        start_col = _pick_column(
            ctx.settings.get("start_column"),
            columns,
            role_by_name,
            {"start_time", "start"},
            ["start", "begin"],
            lower_names,
        )
        end_col = _pick_column(
            ctx.settings.get("end_column"),
            columns,
            role_by_name,
            {"end_time", "end"},
            ["end", "finish", "complete", "stop"],
            lower_names,
        )

        if not dep_col or not id_col or not start_col or not end_col:
            return PluginResult(
                "ok",
                "Dependency join not applicable",
                {
                    "dependency_rows": 0,
                    "near_zero_ratio": 0.0,
                },
                [
                    {
                        "kind": "dependency_lag_summary",
                        "measurement_type": "not_applicable",
                        "reason": "Missing dependency/id/start/end columns.",
                        "columns": [
                            col
                            for col in [dep_col, id_col, start_col, end_col]
                            if col
                        ],
                    }
                ],
                [],
                None,
            )

        work = df.loc[:, [dep_col, id_col, start_col, end_col]].copy()
        work["__dep"] = work[dep_col].map(_normalize_text)
        work["__id"] = work[id_col].map(_normalize_text)
        work["__start_ts"] = pd.to_datetime(work[start_col], errors="coerce", utc=False)
        work["__end_ts"] = pd.to_datetime(work[end_col], errors="coerce", utc=False)

        work = work.loc[~work["__dep"].str.lower().isin(INVALID_STRINGS)].copy()
        if work.empty:
            return PluginResult(
                "ok",
                "No dependency rows found",
                {"dependency_rows": 0, "near_zero_ratio": 0.0},
                [
                    {
                        "kind": "dependency_lag_summary",
                        "measurement_type": "not_applicable",
                        "reason": "No dependency ids present.",
                        "columns": [dep_col, id_col, start_col, end_col],
                    }
                ],
                [],
                None,
            )

        end_lookup = (
            work.loc[work["__id"].astype(bool) & work["__end_ts"].notna()]
            .groupby("__id")["__end_ts"]
            .max()
        )
        work["__dep_end_ts"] = work["__dep"].map(end_lookup)
        work = work.loc[work["__dep_end_ts"].notna() & work["__start_ts"].notna()].copy()
        if work.empty:
            return PluginResult(
                "ok",
                "No dependency end timestamps found",
                {"dependency_rows": 0, "near_zero_ratio": 0.0},
                [
                    {
                        "kind": "dependency_lag_summary",
                        "measurement_type": "not_applicable",
                        "reason": "No matching dependency end times.",
                        "columns": [dep_col, id_col, start_col, end_col],
                    }
                ],
                [],
                None,
            )

        lag_sec = (work["__start_ts"] - work["__dep_end_ts"]).dt.total_seconds()
        lag_sec = lag_sec.clip(lower=0).fillna(0)
        work["__lag_sec"] = lag_sec

        threshold = float(ctx.settings.get("near_zero_threshold_seconds", 2))
        near_zero = work["__lag_sec"] <= threshold
        near_zero_ratio = float(near_zero.mean()) if len(work) else 0.0

        p50 = float(work["__lag_sec"].quantile(0.5))
        p95 = float(work["__lag_sec"].quantile(0.95))
        p99 = float(work["__lag_sec"].quantile(0.99))

        max_examples = int(ctx.settings.get("max_examples", 25))
        example_rows = work.loc[near_zero].index.tolist()[:max_examples]

        findings = [
            {
                "kind": "dependency_lag_summary",
                "dependency_rows": int(len(work)),
                "near_zero_ratio": near_zero_ratio,
                "near_zero_threshold_seconds": threshold,
                "lag_p50_seconds": p50,
                "lag_p95_seconds": p95,
                "lag_p99_seconds": p99,
                "measurement_type": "measured",
                "row_ids": [int(i) for i in example_rows],
                "columns": [dep_col, id_col, start_col, end_col],
            }
        ]

        artifacts_dir = ctx.artifacts_dir("analysis_dependency_resolution_join")
        out_path = artifacts_dir / "dependency_lag.json"
        write_json(
            out_path,
            {
                "summary": {
                    "dependency_column": dep_col,
                    "id_column": id_col,
                    "start_column": start_col,
                    "end_column": end_col,
                    "near_zero_threshold_seconds": threshold,
                },
                "stats": {
                    "dependency_rows": int(len(work)),
                    "near_zero_ratio": near_zero_ratio,
                    "lag_p50_seconds": p50,
                    "lag_p95_seconds": p95,
                    "lag_p99_seconds": p99,
                },
            },
        )
        artifacts = [
            PluginArtifact(
                path=str(out_path.relative_to(ctx.run_dir)),
                type="json",
                description="Dependency lag summary",
            )
        ]

        return PluginResult(
            "ok",
            "Computed dependency lag distribution",
            {
                "dependency_rows": int(len(work)),
                "near_zero_ratio": near_zero_ratio,
            },
            findings,
            artifacts,
            None,
        )
