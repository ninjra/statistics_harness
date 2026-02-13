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


def _rank_text_columns(
    candidates: list[str], df: pd.DataFrame, limit: int, bonus_tokens: list[str]
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
        score = alpha_ratio * 2.0 + (1.0 - unique_ratio) * 1.0
        lname = str(col).lower()
        if any(token in lname for token in bonus_tokens):
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

        user_candidates = _candidate_columns(
            ctx.settings.get("user_column"),
            columns,
            role_by_name,
            {"user", "owner", "assignee", "operator"},
            ["user", "owner", "assignee", "operator", "agent"],
            lower_names,
            used,
        )
        host_candidates = _candidate_columns(
            ctx.settings.get("host_column"),
            columns,
            role_by_name,
            {"host", "machine", "server"},
            ["host", "machine", "server", "node", "instance"],
            lower_names,
            used,
        )
        max_dim_cols = int(ctx.settings.get("max_dimension_columns", 2))
        user_cols = _rank_text_columns(user_candidates, df, max_dim_cols, ["user", "owner"])
        host_cols = _rank_text_columns(host_candidates, df, max_dim_cols, ["host", "server"])

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

        if not eligible_cols or not start_cols or (not user_cols and not host_cols):
            return PluginResult(
                "skipped",
                "Missing user/host or eligible/start columns",
                {},
                [],
                [],
                None,
            )

        threshold = float(ctx.settings.get("wait_threshold_seconds", 60))
        baseline_quantile = float(ctx.settings.get("baseline_quantile", 0.25))
        min_runs = int(ctx.settings.get("min_runs", 100))
        max_groups = int(ctx.settings.get("max_groups", 20))

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

                for dimension, cols in (("user", user_cols), ("host", host_cols)):
                    for dim_col in cols:
                        dim_series = df[dim_col].map(_normalize_text)
                        work = pd.DataFrame(
                            {
                                "dimension": dim_series,
                                "over_sec": over_sec,
                            }
                        )
                        work = work[valid_mask]
                        work = work[work["dimension"].astype(str).str.len() > 0]
                        if work.empty:
                            continue

                        grouped = work.groupby("dimension", sort=False)
                        totals = grouped["over_sec"].sum()
                        counts = grouped.size()
                        avg_over = totals / counts
                        baseline_per_run = float(
                            avg_over.quantile(baseline_quantile)
                            if not avg_over.empty
                            else 0.0
                        )

                        rows = []
                        for group_id, total_over in totals.items():
                            count = int(counts.get(group_id, 0))
                            if count < min_runs:
                                continue
                            baseline_total = baseline_per_run * count
                            modeled_total = min(float(total_over), float(baseline_total))
                            delta_val = float(total_over) - modeled_total
                            if delta_val <= 0:
                                continue
                            rows.append(
                                {
                                    "group_id": str(group_id),
                                    "runs_count": count,
                                    "baseline_over_sec": float(total_over),
                                    "modeled_over_sec": modeled_total,
                                    "delta_over_sec": delta_val,
                                }
                            )

                        rows = sorted(
                            rows, key=lambda row: row["delta_over_sec"], reverse=True
                        )[:max_groups]
                        for row in rows:
                            delta_hours = row["delta_over_sec"] / 3600.0
                            baseline_hours = row["baseline_over_sec"] / 3600.0
                            modeled_hours = row["modeled_over_sec"] / 3600.0
                            scale_factor = (
                                modeled_hours / baseline_hours if baseline_hours > 0 else 1.0
                            )
                            group_id = row["group_id"]
                            confidence_weight = _confidence_weight(row["runs_count"])
                            controllability_weight = 0.5
                            impact_hours = delta_hours
                            relevance_score = (
                                impact_hours * confidence_weight * controllability_weight
                            )
                            finding = {
                                "kind": "user_host_savings",
                                "dimension": dimension,
                                "group_id": group_id,
                                "runs_count": row["runs_count"],
                                "baseline_over_threshold_hours": baseline_hours,
                                "modeled_over_threshold_hours": modeled_hours,
                                "delta_hours": delta_hours,
                                "measurement_type": "modeled",
                                "action_type": "rebalance_assignment"
                                if dimension == "user"
                                else "rebalance_host_load",
                                "target": group_id,
                                "scenario_id": f"{dimension}_{group_id}_rebalance",
                                "recommendation": (
                                    f"Rebalance {dimension} {group_id} to match p"
                                    f"{int(baseline_quantile * 100)} baseline "
                                    f"(Δ {delta_hours:.2f}h)."
                                ),
                                "observed_count": row["runs_count"],
                                "confidence_weight": confidence_weight,
                                "controllability_weight": controllability_weight,
                                "impact_hours": impact_hours,
                                "relevance_score": relevance_score,
                                "modeled_assumptions": (
                                    "Assumes per-run over-threshold wait for this "
                                    f"{dimension} can be reduced to the p"
                                    f"{int(baseline_quantile * 100)} baseline across "
                                    f"{dimension} groups."
                                ),
                                "modeled_scope": dimension,
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
                                    "dimension_column": dim_col,
                                    "eligible_column": eligible_col,
                                    "start_column": start_col,
                                },
                            }
                            findings.append(finding)
                            summary_rows.append(
                                {
                                    "dimension": dimension,
                                    "group_id": group_id,
                                    "delta_hours": delta_hours,
                                    "baseline_over_threshold_hours": baseline_hours,
                                    "modeled_over_threshold_hours": modeled_hours,
                                    "dimension_column": dim_col,
                                    "eligible_column": eligible_col,
                                    "start_column": start_col,
                                }
                            )

        if not findings:
            return PluginResult(
                "skipped", "No user/host savings detected", {}, [], [], None
            )

        artifacts_dir = ctx.artifacts_dir("analysis_user_host_savings")
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
                description="User/host savings findings",
            )
        ]

        metrics = {"findings": len(findings)}

        return PluginResult(
            "ok",
            f"Computed {len(findings)} user/host savings findings",
            metrics,
            findings,
            artifacts,
            None,
        )
