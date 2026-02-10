from __future__ import annotations

from typing import Any

import pandas as pd

from statistic_harness.core.column_inference import infer_timestamp_series
from statistic_harness.core.dataset_io import DatasetAccessor
from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json


INVALID_STRINGS = {"", "nan", "none", "null"}
MAX_SAMPLE_ROWS = 50000
MIN_DATETIME_RATIO = 0.15


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _candidate_columns(
    preferred: str | None,
    columns: list[str],
    role_by_name: dict[str, str],
    roles: set[str],
    patterns: list[str],
    lower_names: dict[str, str],
    exclude: set[str],
) -> list[str]:
    seen: set[str] = set()
    candidates: list[str] = []
    if preferred and preferred in columns and preferred not in exclude and preferred not in seen:
        candidates.append(preferred)
        seen.add(preferred)
    for col in columns:
        if col in exclude or col in seen:
            continue
        if role_by_name.get(col) in roles:
            candidates.append(col)
            seen.add(col)
    for col in columns:
        if col in exclude or col in seen:
            continue
        name = lower_names[col]
        if any(pattern in name for pattern in patterns):
            candidates.append(col)
            seen.add(col)
    return candidates


def _rank_case_columns(
    candidates: list[str], df: pd.DataFrame, limit: int
) -> list[str]:
    scored: list[tuple[float, str]] = []
    for col in candidates:
        sample = df[col].dropna()
        if sample.empty:
            continue
        if sample.shape[0] > MAX_SAMPLE_ROWS:
            sample = sample.head(MAX_SAMPLE_ROWS)
        unique_ratio = float(sample.nunique() / max(1, sample.shape[0]))
        score = 0.0
        lname = str(col).lower()
        if "case" in lname or "session" in lname or "trace" in lname:
            score += 2.0
        if 0.01 <= unique_ratio <= 0.5:
            score += 2.0
        score -= abs(unique_ratio - 0.15)
        scored.append((score, col))
    scored.sort(reverse=True, key=lambda item: item[0])
    return [col for _, col in scored[:limit]]


def _rank_process_columns(
    candidates: list[str], df: pd.DataFrame, limit: int
) -> list[str]:
    scored: list[tuple[float, str]] = []
    for col in candidates:
        sample = df[col].dropna()
        if sample.empty:
            continue
        if sample.shape[0] > MAX_SAMPLE_ROWS:
            sample = sample.head(MAX_SAMPLE_ROWS)
        values = sample.astype(str).str.strip()
        values = values[~values.str.lower().isin(INVALID_STRINGS)]
        if values.empty:
            continue
        alpha_ratio = float(values.str.contains(r"[A-Za-z]", regex=True).mean())
        unique_ratio = float(values.nunique() / max(len(values), 1))
        score = alpha_ratio * 2.5 + (1.0 - unique_ratio) * 1.5
        lname = str(col).lower()
        if "process" in lname or "activity" in lname:
            score += 1.5
        scored.append((score, col))
    scored.sort(reverse=True, key=lambda item: item[0])
    return [col for _, col in scored[:limit]]


def _rank_datetime_columns(
    candidates: list[str], df: pd.DataFrame, limit: int
) -> list[str]:
    scored: list[tuple[float, str]] = []
    for col in candidates:
        info = infer_timestamp_series(df[col], name_hint=col, sample_size=MAX_SAMPLE_ROWS)
        if not info.valid or info.parse_ratio < MIN_DATETIME_RATIO:
            continue
        scored.append((float(info.score), col))
    scored.sort(reverse=True, key=lambda item: item[0])
    return [col for _, col in scored[:limit]]


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=False)


def _confidence_weight(count: int) -> float:
    if count >= 2000:
        return 0.9
    if count >= 500:
        return 0.75
    if count >= 200:
        return 0.6
    if count >= 50:
        return 0.5
    return 0.4


class Plugin:
    def run(self, ctx) -> PluginResult:
        if not ctx.dataset_version_id:
            return PluginResult("skipped", "Dataset version missing", {}, [], [], None)

        accessor = DatasetAccessor(ctx.storage, ctx.dataset_version_id)
        df = accessor.load()
        if df.empty:
            return PluginResult("skipped", "Empty dataset", {}, [], [], None)

        columns_meta = []
        role_by_name: dict[str, str] = {}
        dataset_template = ctx.storage.fetch_dataset_template(ctx.dataset_version_id)
        if dataset_template and dataset_template.get("status") == "ready":
            fields = ctx.storage.fetch_template_fields(int(dataset_template["template_id"]))
            columns_meta = fields
            role_by_name = {field["name"]: (field.get("role") or "") for field in fields}
        else:
            columns_meta = ctx.storage.fetch_dataset_columns(ctx.dataset_version_id)
            role_by_name = {
                col["original_name"]: (col.get("role") or "")
                for col in columns_meta
            }

        columns = list(df.columns)
        lower_names = {col: str(col).lower() for col in columns}
        used: set[str] = set()

        case_candidates = _candidate_columns(
            ctx.settings.get("case_column"),
            columns,
            role_by_name,
            {"id", "case", "trace", "session"},
            ["case", "session", "trace", "order", "request"],
            lower_names,
            used,
        )
        max_case_cols = int(ctx.settings.get("max_case_columns", 2))
        case_cols = _rank_case_columns(case_candidates, df, max_case_cols)
        if not case_cols and case_candidates:
            case_cols = case_candidates[:1]
        for col in case_cols:
            used.add(col)

        process_candidates = _candidate_columns(
            ctx.settings.get("process_column"),
            columns,
            role_by_name,
            {"process", "activity", "event", "step", "task", "action"},
            ["process", "activity", "event", "step", "task", "action"],
            lower_names,
            used,
        )
        max_process_cols = int(ctx.settings.get("max_process_columns", 2))
        process_cols = _rank_process_columns(process_candidates, df, max_process_cols)
        if not process_cols and process_candidates:
            process_cols = process_candidates[:1]
        for col in process_cols:
            used.add(col)

        start_candidates = _candidate_columns(
            ctx.settings.get("start_column"),
            columns,
            role_by_name,
            {"start", "timestamp"},
            ["start", "begin", "timestamp", "time"],
            lower_names,
            used,
        )
        end_candidates = _candidate_columns(
            ctx.settings.get("end_column"),
            columns,
            role_by_name,
            {"end", "finish", "complete"},
            ["end", "finish", "complete", "stop"],
            lower_names,
            used,
        )
        max_start_cols = int(ctx.settings.get("max_start_columns", 2))
        max_end_cols = int(ctx.settings.get("max_end_columns", 2))
        start_cols = _rank_datetime_columns(start_candidates, df, max_start_cols)
        end_cols = _rank_datetime_columns(end_candidates, df, max_end_cols)
        if not start_cols and start_candidates:
            start_cols = start_candidates[:1]
        if not end_cols:
            end_cols = start_cols[:1]

        if not case_cols or not process_cols or not start_cols:
            return PluginResult(
                "skipped",
                "Missing case/process/start columns",
                {},
                [],
                [],
                None,
            )

        threshold = float(ctx.settings.get("wait_threshold_seconds", 60))
        baseline_quantile = float(ctx.settings.get("baseline_quantile", 0.25))
        min_count = int(ctx.settings.get("min_transition_count", 50))
        max_pairs = int(ctx.settings.get("max_pairs", 20))

        findings: list[dict[str, Any]] = []
        summary_rows: list[dict[str, Any]] = []

        for case_col in case_cols:
            case_series = df[case_col].map(_normalize_text)
            for process_col in process_cols:
                process_series = df[process_col].map(_normalize_text).str.lower()
                for start_col in start_cols:
                    start_ts = _to_datetime(df[start_col])
                    for end_col in end_cols:
                        end_ts = _to_datetime(df[end_col])
                        work = pd.DataFrame(
                            {
                                "case_id": case_series,
                                "process": process_series,
                                "start": start_ts,
                                "end": end_ts,
                            }
                        )
                        work = work.dropna(subset=["case_id", "process", "start", "end"])
                        work = work[
                            (work["case_id"].astype(str).str.len() > 0)
                            & (work["process"].astype(str).str.len() > 0)
                        ]
                        if work.empty:
                            continue
                        work = work.sort_values(["case_id", "start"])
                        work["next_case"] = work["case_id"].shift(-1)
                        work["next_process"] = work["process"].shift(-1)
                        work["next_start"] = work["start"].shift(-1)
                        mask = work["case_id"] == work["next_case"]
                        if not mask.any():
                            continue
                        work = work[mask].copy()
                        gap_sec = (work["next_start"] - work["end"]).dt.total_seconds()
                        work["gap_sec"] = gap_sec.fillna(0).clip(lower=0)
                        work["over_sec"] = (work["gap_sec"] - threshold).clip(lower=0)
                        work = work[work["over_sec"] > 0]
                        if work.empty:
                            continue

                        grouped = work.groupby(["process", "next_process"], sort=False)
                        totals = grouped["over_sec"].sum()
                        counts = grouped.size()
                        median_gap = grouped["gap_sec"].median()
                        baseline_gap = grouped["gap_sec"].quantile(baseline_quantile)
                        baseline_over = (baseline_gap - threshold).clip(lower=0)
                        modeled_total = baseline_over * counts
                        delta = totals - modeled_total

                        examples = (
                            work.sort_values("gap_sec", ascending=False)
                            .groupby(["process", "next_process"])["case_id"]
                            .apply(lambda s: list(dict.fromkeys(s.astype(str).head(3))))
                        )

                        rows = []
                        for key, total_over in totals.items():
                            count = int(counts.get(key, 0))
                            if count < min_count:
                                continue
                            modeled = float(modeled_total.get(key, 0.0))
                            baseline_val = float(total_over)
                            delta_val = baseline_val - modeled
                            if delta_val <= 0:
                                continue
                            rows.append(
                                {
                                    "process": key[0],
                                    "next_process": key[1],
                                    "transition_count": count,
                                    "baseline_over_sec": baseline_val,
                                    "modeled_over_sec": modeled,
                                    "delta_over_sec": delta_val,
                                    "median_gap_sec": float(median_gap.get(key, 0.0)),
                                    "baseline_gap_sec": float(baseline_gap.get(key, 0.0)),
                                    "examples": examples.get(key, []),
                                }
                            )

                        rows = sorted(
                            rows, key=lambda row: row["delta_over_sec"], reverse=True
                        )[:max_pairs]
                        for row in rows:
                            delta_hours = row["delta_over_sec"] / 3600.0
                            baseline_hours = row["baseline_over_sec"] / 3600.0
                            modeled_hours = row["modeled_over_sec"] / 3600.0
                            scale_factor = (
                                modeled_hours / baseline_hours if baseline_hours > 0 else 1.0
                            )
                            transition = f"{row['process']} -> {row['next_process']}"
                            confidence_weight = _confidence_weight(row["transition_count"])
                            controllability_weight = 0.55
                            impact_hours = delta_hours
                            relevance_score = (
                                impact_hours * confidence_weight * controllability_weight
                            )
                            finding = {
                                "kind": "sequence_bottleneck",
                                "transition": transition,
                                "process_id": row["process"],
                                "next_process_id": row["next_process"],
                                "transition_count": row["transition_count"],
                                "median_gap_sec": row["median_gap_sec"],
                                "baseline_gap_sec": row["baseline_gap_sec"],
                                "baseline_over_threshold_hours": baseline_hours,
                                "modeled_over_threshold_hours": modeled_hours,
                                "delta_hours": delta_hours,
                                "measurement_type": "modeled",
                                "action_type": "reduce_transition_gap",
                                "target": transition,
                                "scenario_id": f"reduce_gap_{row['process']}_to_{row['next_process']}",
                                "recommendation": (
                                    f"Reduce handoff gap {transition}: "
                                    f"{baseline_hours:.2f}h -> {modeled_hours:.2f}h "
                                    f"(Δ {delta_hours:.2f}h) by aligning to p"
                                    f"{int(baseline_quantile * 100)} gaps."
                                ),
                                "observed_count": row["transition_count"],
                                "confidence_weight": confidence_weight,
                                "controllability_weight": controllability_weight,
                                "impact_hours": impact_hours,
                                "relevance_score": relevance_score,
                                "modeled_assumptions": (
                                    "Assumes handoff gap can be reduced to the p"
                                    f"{int(baseline_quantile * 100)} baseline for this transition "
                                    "without changing throughput."
                                ),
                                "modeled_scope": "transition",
                                "baseline_host_count": None,
                                "modeled_host_count": None,
                                "baseline_value": baseline_hours,
                                "modeled_value": modeled_hours,
                                "delta_value": delta_hours,
                                "unit": "hours",
                                "scale_factor": scale_factor,
                                "scale_factor_standard": scale_factor,
                                "scale_factor_original": scale_factor,
                                "scale_factor_original_definition": (
                                    "scale_factor_standard = modeled_over_threshold_hours "
                                    "/ baseline_over_threshold_hours"
                                ),
                                "columns": {
                                    "case_column": case_col,
                                    "process_column": process_col,
                                    "start_column": start_col,
                                    "end_column": end_col,
                                },
                                "evidence": {
                                    "example_cases": row["examples"],
                                },
                            }
                            findings.append(finding)
                            summary_rows.append(
                                {
                                    "transition": transition,
                                    "delta_hours": delta_hours,
                                    "baseline_over_threshold_hours": baseline_hours,
                                    "modeled_over_threshold_hours": modeled_hours,
                                    "case_column": case_col,
                                    "process_column": process_col,
                                    "start_column": start_col,
                                    "end_column": end_col,
                                }
                            )

        if not findings:
            return PluginResult(
                "skipped", "No sequence bottlenecks detected", {}, [], [], None
            )

        artifacts_dir = ctx.artifacts_dir("analysis_process_sequence_bottlenecks")
        out_path = artifacts_dir / "results.json"
        write_json(
            out_path,
            {
                "summary": summary_rows,
                "findings": findings,
                "threshold_seconds": threshold,
                "baseline_quantile": baseline_quantile,
            },
        )

        artifacts = [
            PluginArtifact(
                path=str(out_path.relative_to(ctx.run_dir)),
                type="json",
                description="Sequence bottleneck findings",
            )
        ]

        metrics = {"findings": len(findings)}

        return PluginResult(
            "ok",
            f"Computed {len(findings)} sequence bottlenecks",
            metrics,
            findings,
            artifacts,
            None,
        )
