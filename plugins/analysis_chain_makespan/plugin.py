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

        sequence_col = _pick_column(
            ctx.settings.get("sequence_column"),
            columns,
            role_by_name,
            {"master_id", "sequence_id", "master", "chain", "batch", "case_id"},
            ["master", "sequence", "chain", "batch", "workflow", "case", "group"],
            lower_names,
            used,
        )
        if sequence_col:
            used.add(sequence_col)

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

        end_col = _pick_timestamp_column(
            ctx.settings.get("end_column"),
            columns,
            role_by_name,
            {"end_time", "end"},
            ["end", "finish", "complete", "stop"],
            lower_names,
            used,
            df,
        )
        if end_col:
            used.add(end_col)

        duration_col = _pick_column(
            ctx.settings.get("duration_column"),
            columns,
            role_by_name,
            {"duration", "runtime", "elapsed"},
            ["duration", "runtime", "elapsed", "secs", "seconds", "ms"],
            lower_names,
            used,
        )

        if not sequence_col or not start_col or (not end_col and not duration_col):
            return PluginResult(
                "ok",
                "Chain makespan not applicable",
                {"chains": 0},
                [
                    {
                        "kind": "chain_makespan",
                        "measurement_type": "not_applicable",
                        "reason": "Missing sequence and start/end columns.",
                        "columns": [
                            col
                            for col in [sequence_col, start_col, end_col, duration_col]
                            if col
                        ],
                    }
                ],
                [],
                None,
            )

        selected: list[str] = []
        for col in [sequence_col, start_col, end_col, duration_col]:
            if col and col in columns and col not in selected:
                selected.append(col)
        work = df.loc[:, selected].copy()

        work["__sequence"] = work[sequence_col].map(_normalize_text)
        work["__sequence_norm"] = work["__sequence"].str.lower()
        work = work.loc[~work["__sequence_norm"].isin(INVALID_STRINGS)].copy()
        if work.empty:
            return PluginResult(
                "ok",
                "No sequence ids found",
                {"chains": 0},
                [
                    {
                        "kind": "chain_makespan",
                        "measurement_type": "not_applicable",
                        "reason": "No sequence values detected.",
                        "columns": [sequence_col],
                    }
                ],
                [],
                None,
            )

        work["__start_ts"] = pd.to_datetime(
            work[start_col], errors="coerce", utc=False
        )
        if end_col:
            work["__end_ts"] = pd.to_datetime(
                work[end_col], errors="coerce", utc=False
            )
        else:
            work["__end_ts"] = pd.NaT

        if duration_col:
            work["__duration_sec"] = pd.to_numeric(
                work[duration_col], errors="coerce"
            )
        else:
            work["__duration_sec"] = pd.NA

        missing_end = work["__end_ts"].isna() & work["__duration_sec"].notna()
        if missing_end.any():
            work.loc[missing_end, "__end_ts"] = work.loc[
                missing_end, "__start_ts"
            ] + pd.to_timedelta(work.loc[missing_end, "__duration_sec"], unit="s")

        computed_duration = (work["__end_ts"] - work["__start_ts"]).dt.total_seconds()
        if duration_col:
            work["__duration_sec"] = computed_duration.where(
                computed_duration.notna(), work["__duration_sec"]
            )
        else:
            work["__duration_sec"] = computed_duration

        work = work.loc[work["__start_ts"].notna() & work["__end_ts"].notna()].copy()
        if work.empty:
            return PluginResult(
                "ok",
                "No valid timestamps",
                {"chains": 0},
                [
                    {
                        "kind": "chain_makespan",
                        "measurement_type": "not_applicable",
                        "reason": "No valid start/end timestamps.",
                        "columns": [start_col, end_col or duration_col],
                    }
                ],
                [],
                None,
            )

        grouped = work.groupby("__sequence_norm", sort=False)
        rows: list[tuple[str, str, float, float, float, list[int]]] = []
        for seq_norm, frame in grouped:
            seq_label = frame["__sequence"].iloc[0]
            start_min = frame["__start_ts"].min()
            end_max = frame["__end_ts"].max()
            makespan = (end_max - start_min).total_seconds()
            runtime = (
                frame["__duration_sec"].clip(lower=0).fillna(0).sum()
                if "__duration_sec" in frame
                else 0.0
            )
            makespan = float(max(makespan, 0.0))
            runtime = float(runtime)
            idle_gap = max(makespan - runtime, 0.0)
            row_ids = [int(i) for i in frame.index.tolist()]
            rows.append((seq_norm, seq_label, makespan, runtime, idle_gap, row_ids))

        if not rows:
            return PluginResult(
                "ok",
                "No chain rows found",
                {"chains": 0},
                [],
                [],
                None,
            )

        rows.sort(key=lambda item: (-item[2], item[0]))
        max_sequences = int(ctx.settings.get("max_sequences", 10))
        max_examples = int(ctx.settings.get("max_examples", 25))

        findings = []
        for seq_norm, seq_label, makespan, runtime, idle_gap, row_ids in rows[:max_sequences]:
            runtime_ratio = runtime / makespan if makespan > 0 else 0.0
            idle_ratio = idle_gap / makespan if makespan > 0 else 0.0
            findings.append(
                {
                    "kind": "chain_makespan",
                    "sequence_id": seq_label,
                    "sequence_norm": seq_norm,
                    "makespan_seconds": float(makespan),
                    "runtime_seconds": float(runtime),
                    "idle_gap_seconds": float(idle_gap),
                    "runtime_ratio": float(runtime_ratio),
                    "idle_ratio": float(idle_ratio),
                    "measurement_type": "measured",
                    "row_ids": row_ids[:max_examples],
                    "columns": [
                        col
                        for col in [sequence_col, start_col, end_col, duration_col]
                        if col
                    ],
                }
            )

        metrics = {
            "chains": len(rows),
            "max_makespan_seconds": float(rows[0][2]),
        }

        artifacts_dir = ctx.artifacts_dir("analysis_chain_makespan")
        out_path = artifacts_dir / "chain_makespan.json"
        write_json(
            out_path,
            {
                "summary": {
                    "sequence_column": sequence_col,
                    "start_column": start_col,
                    "end_column": end_col,
                    "duration_column": duration_col,
                },
                "chains": len(rows),
                "max_makespan_seconds": float(rows[0][2]),
            },
        )
        artifacts = [
            PluginArtifact(
                path=str(out_path.relative_to(ctx.run_dir)),
                type="json",
                description="Chain makespan summary",
            )
        ]

        return PluginResult(
            "ok",
            "Computed chain makespan statistics",
            metrics,
            findings,
            artifacts,
            None,
        )
