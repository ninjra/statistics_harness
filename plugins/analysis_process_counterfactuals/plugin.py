from __future__ import annotations

from typing import Any

import pandas as pd

from statistic_harness.core.column_inference import infer_timestamp_series
from statistic_harness.core.dataset_io import DatasetAccessor
from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json


INVALID_STRINGS = {"", "nan", "none", "null"}
MAX_SAMPLE_ROWS = 50000
MIN_PROCESS_ALPHA_RATIO = 0.15
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


def _sample_series(series: pd.Series, max_rows: int = MAX_SAMPLE_ROWS) -> pd.Series:
    if len(series) > max_rows:
        return series.head(max_rows)
    return series


def _normalize_values(series: pd.Series) -> pd.Series:
    sample = _sample_series(series)
    values = sample.dropna().astype(str).str.strip()
    if values.empty:
        return values
    values = values[~values.str.lower().isin(INVALID_STRINGS)]
    return values


def _score_process_column(series: pd.Series) -> tuple[float, float]:
    values = _normalize_values(series)
    if values.empty:
        return -1.0, 0.0
    alpha_ratio = float(values.str.contains(r"[A-Za-z]", regex=True).mean())
    digit_ratio = float(values.str.match(r"^\\d+(\\.0+)?$").mean())
    unique_ratio = float(values.nunique() / max(len(values), 1))
    score = alpha_ratio * 3.0 - digit_ratio * 2.0 + min(unique_ratio, 1.0) * 0.2
    return score, alpha_ratio


def _rank_process_columns(
    candidates: list[str], df: pd.DataFrame, limit: int
) -> list[str]:
    scored: list[tuple[float, str]] = []
    for col in candidates:
        score, alpha = _score_process_column(df[col])
        if alpha < MIN_PROCESS_ALPHA_RATIO:
            continue
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


def _rank_host_columns(
    candidates: list[str], df: pd.DataFrame, limit: int
) -> list[str]:
    scored: list[tuple[float, str]] = []
    for col in candidates:
        values = _normalize_values(df[col])
        if values.empty:
            continue
        unique_ratio = float(values.nunique() / max(len(values), 1))
        alpha_ratio = float(values.str.contains(r"[A-Za-z]", regex=True).mean())
        score = unique_ratio + alpha_ratio * 0.2
        scored.append((score, col))
    scored.sort(reverse=True, key=lambda item: item[0])
    return [col for _, col in scored[:limit]]


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=False)


def _confidence_weight(runs: int) -> float:
    if runs <= 0:
        return 0.3
    if runs >= 2000:
        return 0.9
    if runs >= 500:
        return 0.7
    if runs >= 200:
        return 0.6
    return 0.45


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

        process_candidates = _candidate_columns(
            ctx.settings.get("process_column"),
            columns,
            role_by_name,
            {"process", "activity", "event", "step", "task", "action"},
            ["process", "activity", "event", "step", "task", "action", "job"],
            lower_names,
            used,
        )
        max_process_cols = int(ctx.settings.get("max_process_columns", 3))
        process_cols = _rank_process_columns(process_candidates, df, max_process_cols)
        for col in process_cols:
            used.add(col)

        host_candidates = _candidate_columns(
            ctx.settings.get("host_column"),
            columns,
            role_by_name,
            {"host", "machine", "server"},
            ["host", "machine", "server", "node", "instance"],
            lower_names,
            used,
        )
        host_cols = _rank_host_columns(host_candidates, df, 1)
        host_col = host_cols[0] if host_cols else None

        eligible_candidates = _candidate_columns(
            ctx.settings.get("eligible_column") or ctx.settings.get("queue_column"),
            columns,
            role_by_name,
            {"eligible", "ready", "available", "queue", "queued", "enqueue"},
            ["eligible", "ready", "available", "queue", "queued", "enqueue"],
            lower_names,
            used,
        )
        start_candidates = _candidate_columns(
            ctx.settings.get("start_column"),
            columns,
            role_by_name,
            {"start", "begin"},
            ["start", "begin"],
            lower_names,
            used,
        )
        max_start_cols = int(ctx.settings.get("max_start_columns", 2))
        max_eligible_cols = int(ctx.settings.get("max_eligible_columns", 2))
        eligible_cols = _rank_datetime_columns(eligible_candidates, df, max_eligible_cols)
        start_cols = _rank_datetime_columns(start_candidates, df, max_start_cols)

        if not process_cols or not eligible_cols or not start_cols:
            return PluginResult(
                "skipped",
                "Missing process/start/eligible columns",
                {},
                [],
                [],
                None,
            )

        threshold = float(ctx.settings.get("wait_threshold_seconds", 60))
        baseline_quantile = float(ctx.settings.get("baseline_quantile", 0.25))
        min_runs = int(ctx.settings.get("min_runs", 200))
        max_processes = int(ctx.settings.get("max_processes", 20))

        process_series_map = {
            col: df[col].map(_normalize_text).str.lower() for col in process_cols
        }
        host_series = df[host_col].map(_normalize_text) if host_col else None

        findings: list[dict[str, Any]] = []
        summary_rows: list[dict[str, Any]] = []

        for eligible_col in eligible_cols:
            eligible_ts = _to_datetime(df[eligible_col])
            for start_col in start_cols:
                start_ts = _to_datetime(df[start_col])
                wait_sec = (start_ts - eligible_ts).dt.total_seconds()
                valid_mask = wait_sec.notna() & (wait_sec >= 0)
                if valid_mask.sum() < min_runs:
                    continue
                over_sec = (wait_sec - threshold).clip(lower=0)

                for process_col in process_cols:
                    process_norm = process_series_map[process_col]
                    work = pd.DataFrame(
                        {
                            "process": process_norm,
                            "wait_sec": wait_sec,
                            "over_sec": over_sec,
                            "host": host_series if host_series is not None else "",
                        }
                    )
                    work = work[valid_mask]
                    work = work[work["process"].astype(str).str.len() > 0]
                    if work.empty:
                        continue

                    grouped = work.groupby("process", sort=False)
                    totals = grouped["over_sec"].sum()
                    counts = grouped.size()
                    baseline_wait = grouped["wait_sec"].quantile(baseline_quantile)
                    baseline_over = (baseline_wait - threshold).clip(lower=0)
                    modeled_total = baseline_over * counts
                    delta = totals - modeled_total
                    host_counts = None
                    if host_col:
                        host_counts = grouped["host"].nunique()

                    rows = []
                    for process_id, total_over in totals.items():
                        count = int(counts.get(process_id, 0))
                        if count < min_runs:
                            continue
                        modeled = float(modeled_total.get(process_id, 0.0))
                        baseline_val = float(total_over)
                        delta_val = baseline_val - modeled
                        if delta_val <= 0:
                            continue
                        rows.append(
                            {
                                "process_id": process_id,
                                "runs_count": count,
                                "baseline_over_sec": baseline_val,
                                "modeled_over_sec": modeled,
                                "delta_over_sec": delta_val,
                                "baseline_wait_sec": float(
                                    baseline_wait.get(process_id, 0.0)
                                ),
                                "baseline_over_per_run_sec": float(
                                    baseline_over.get(process_id, 0.0)
                                ),
                                "host_count": int(host_counts.get(process_id, 0))
                                if host_counts is not None
                                else None,
                            }
                        )

                    rows = sorted(
                        rows, key=lambda row: row["delta_over_sec"], reverse=True
                    )[:max_processes]
                    for row in rows:
                        delta_hours = row["delta_over_sec"] / 3600.0
                        baseline_hours = row["baseline_over_sec"] / 3600.0
                        modeled_hours = row["modeled_over_sec"] / 3600.0
                        scale_factor = (
                            modeled_hours / baseline_hours if baseline_hours > 0 else 1.0
                        )
                        runs_count = row["runs_count"]
                        confidence_weight = _confidence_weight(runs_count)
                        controllability_weight = 0.6
                        impact_hours = delta_hours
                        relevance_score = (
                            impact_hours * confidence_weight * controllability_weight
                        )
                        process_id = row["process_id"]
                        finding = {
                            "kind": "process_counterfactual",
                            "process_id": process_id,
                            "process_norm": process_id,
                            "runs_count": runs_count,
                            "baseline_over_threshold_hours": baseline_hours,
                            "modeled_over_threshold_hours": modeled_hours,
                            "delta_hours": delta_hours,
                            "measurement_type": "modeled",
                            "action_type": "reduce_process_wait",
                            "target": process_id,
                            "scenario_id": f"reduce_wait_{process_id}",
                            "recommendation": (
                                f"Reduce >threshold wait for {process_id}: "
                                f"{baseline_hours:.2f}h -> {modeled_hours:.2f}h "
                                f"(Δ {delta_hours:.2f}h) by matching the p"
                                f"{int(baseline_quantile * 100)} baseline."
                            ),
                            "observed_count": runs_count,
                            "confidence_weight": confidence_weight,
                            "controllability_weight": controllability_weight,
                            "impact_hours": impact_hours,
                            "relevance_score": relevance_score,
                            "modeled_assumptions": (
                                "Assumes wait-to-start for this process can be reduced "
                                f"to the p{int(baseline_quantile * 100)} baseline within "
                                "the same volume and host mix."
                            ),
                            "modeled_scope": "process",
                            "baseline_host_count": row.get("host_count"),
                            "modeled_host_count": row.get("host_count"),
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
                                "process_column": process_col,
                                "eligible_column": eligible_col,
                                "start_column": start_col,
                                "host_column": host_col,
                            },
                        }
                        findings.append(finding)
                        summary_rows.append(
                            {
                                "process_id": process_id,
                                "runs_count": runs_count,
                                "baseline_over_threshold_hours": baseline_hours,
                                "modeled_over_threshold_hours": modeled_hours,
                                "delta_hours": delta_hours,
                                "process_column": process_col,
                                "eligible_column": eligible_col,
                                "start_column": start_col,
                            }
                        )

        if not findings:
            return PluginResult(
                "skipped", "No counterfactual savings detected", {}, [], [], None
            )

        artifacts_dir = ctx.artifacts_dir("analysis_process_counterfactuals")
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
                description="Process counterfactual savings",
            )
        ]

        metrics = {"findings": len(findings)}

        return PluginResult(
            "ok",
            f"Computed {len(findings)} process counterfactuals",
            metrics,
            findings,
            artifacts,
            None,
        )
