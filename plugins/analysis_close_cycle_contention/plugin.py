from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

import pandas as pd

from statistic_harness.core.close_cycle import resolve_close_cycle_masks
from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import infer_close_cycle_window, write_json

INVALID_STRINGS = {"", "nan", "none", "null"}
PROCESS_ALIAS_PREFIXES: tuple[tuple[str, str], ...] = (
    ("qemail", "qemail"),
    ("qpec", "qpec"),
)


def _spearman_corr(left: pd.Series, right: pd.Series) -> float:
    if left.empty or right.empty:
        return float("nan")
    left_rank = left.rank(method="average")
    right_rank = right.rank(method="average")
    return float(left_rank.corr(right_rank, method="pearson"))


def _normalize_param(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    raw = str(value).strip()
    if not raw:
        return None
    if raw.startswith("{") and raw.endswith("}"):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            items = [(str(k).strip().lower(), str(v).strip()) for k, v in parsed.items()]
            items = sorted(set(items))
            return ";".join(f"{k}={v}" for k, v in items)
    tokens = re.split(r"[;|,\n]+", raw)
    pairs = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if "=" in token:
            key, val = token.split("=", 1)
        elif ":" in token:
            key, val = token.split(":", 1)
        else:
            continue
        key = key.strip().lower()
        val = val.strip()
        if key:
            pairs.append((key, val))
    if pairs:
        pairs = sorted(set(pairs))
        return ";".join(f"{k}={v}" for k, v in pairs)
    return raw.lower()


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


def _process_candidate_score(df: pd.DataFrame, col: str) -> float:
    name = str(col).lower()
    series = df[col]
    sample = series.dropna().astype(str).str.strip()
    if sample.empty:
        return -1e9
    sample = sample.loc[sample != ""]
    if sample.empty:
        return -1e9
    sample = sample.head(5000)

    numeric_ratio = float(pd.to_numeric(sample, errors="coerce").notna().mean())
    alpha_ratio = float(sample.str.contains(r"[A-Za-z]", regex=True).mean())
    unique_ratio = float(min(sample.nunique(dropna=True) / max(len(sample), 1), 1.0))

    score = (alpha_ratio * 3.0) + ((1.0 - numeric_ratio) * 2.0) + ((1.0 - unique_ratio) * 1.0)
    if "process_id" in name or "activity" in name or name.endswith("process"):
        score += 1.0
    if "queue" in name or "parent" in name or "dep_process" in name:
        score -= 1.5
    if name.endswith("_id") and "process_id" not in name and "activity" not in name:
        score -= 0.4
    return float(score)


def _canonical_process(value: Any) -> str:
    if value is None:
        return ""
    raw = str(value).strip().lower()
    if not raw or raw in INVALID_STRINGS:
        return ""
    compact = re.sub(r"[^a-z0-9]+", "", raw)
    if not compact:
        return raw
    for prefix, canonical in PROCESS_ALIAS_PREFIXES:
        if compact.startswith(prefix):
            return canonical
    return raw


def _process_label_for_finding(process_norm: str, fallback_label: str) -> str:
    if process_norm in {"qemail", "qpec"}:
        return process_norm
    return fallback_label


def _day_window_mask(day_series: pd.Series, start_day: int, end_day: int) -> pd.Series:
    if start_day <= end_day:
        return (day_series >= start_day) & (day_series <= end_day)
    return (day_series >= start_day) | (day_series <= end_day)


class Plugin:
    def run(self, ctx) -> PluginResult:
        df = ctx.dataset_loader()
        if df.empty:
            return PluginResult(
                "ok",
                "Close-cycle contention not applicable: empty dataset",
                {"candidates": 0, "not_applicable_reason": "empty_dataset"},
                [],
                [],
                None,
            )

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
            {"process", "activity", "event", "step", "task", "action"},
            ["process", "activity", "event", "step", "task", "action", "job"],
            lower_names,
            used,
        )
        # If the initially inferred process column looks like a queue/row identifier,
        # pick the strongest semantic process/activity candidate instead.
        process_candidates: list[str] = []
        for col in columns:
            if role_by_name.get(col) in {"process", "activity", "event", "step", "task", "action"}:
                process_candidates.append(col)
                continue
            name = lower_names[col]
            if any(pattern in name for pattern in ("process", "activity", "event", "step", "task", "action", "job")):
                process_candidates.append(col)
        process_candidates = list(dict.fromkeys(process_candidates))
        if process_candidates:
            ranked = sorted(
                process_candidates,
                key=lambda c: _process_candidate_score(df, c),
                reverse=True,
            )
            best = ranked[0]
            if process_col is None:
                process_col = best
            else:
                try:
                    current_score = _process_candidate_score(df, process_col)
                    best_score = _process_candidate_score(df, best)
                    if best_score > (current_score + 0.5):
                        process_col = best
                except Exception:
                    process_col = best
        if process_col:
            used.add(process_col)

        timestamp_col = _pick_column(
            ctx.settings.get("timestamp_column"),
            columns,
            role_by_name,
            {"timestamp"},
            ["timestamp", "time", "date"],
            lower_names,
            used,
        )
        if timestamp_col:
            used.add(timestamp_col)

        start_col = _pick_column(
            ctx.settings.get("start_column"),
            columns,
            role_by_name,
            {"start"},
            ["start", "begin", "queued", "enqueue"],
            lower_names,
            used,
        )
        if start_col:
            used.add(start_col)

        end_col = _pick_column(
            ctx.settings.get("end_column"),
            columns,
            role_by_name,
            {"end", "finish", "complete"},
            ["end", "finish", "complete", "stop", "dequeue"],
            lower_names,
            used,
        )
        if end_col:
            used.add(end_col)

        duration_col = _pick_column(
            ctx.settings.get("duration_column"),
            columns,
            role_by_name,
            {"duration", "latency", "elapsed", "runtime"},
            ["duration", "elapsed", "latency", "runtime", "seconds", "secs", "ms"],
            lower_names,
            used,
        )
        if duration_col:
            used.add(duration_col)

        queue_col = _pick_column(
            ctx.settings.get("queue_column"),
            columns,
            role_by_name,
            {"queue", "queued", "enqueue", "submitted"},
            ["queue", "queued", "enqueue", "submitted"],
            lower_names,
            used,
        )
        if queue_col:
            used.add(queue_col)

        server_col = _pick_column(
            ctx.settings.get("server_column"),
            columns,
            role_by_name,
            {"server", "host", "node", "instance"},
            ["server", "host", "node", "instance", "machine"],
            lower_names,
            used,
        )
        if server_col:
            used.add(server_col)

        user_col = _pick_column(
            ctx.settings.get("user_column"),
            columns,
            role_by_name,
            {"user", "owner", "actor"},
            ["user", "owner", "actor"],
            lower_names,
            used,
        )
        if user_col:
            used.add(user_col)

        param_col = _pick_column(
            ctx.settings.get("param_column"),
            columns,
            role_by_name,
            {"parameter", "params", "config", "meta"},
            ["param", "parameter", "params", "config", "meta"],
            lower_names,
            used,
        )

        def _emit_not_applicable(reason: str, summary_text: str) -> PluginResult:
            finding = {
                "kind": "close_cycle_contention",
                "decision": "not_applicable",
                "measurement_type": "not_applicable",
                "reason": reason,
                "columns": [
                    col
                    for col in [
                        process_col,
                        timestamp_col,
                        duration_col,
                        queue_col,
                        start_col,
                        end_col,
                        server_col,
                        user_col,
                        param_col,
                    ]
                    if col
                ],
                "evidence": {
                    "dataset_id": ctx.dataset_id or "unknown",
                    "dataset_version_id": ctx.dataset_version_id or "unknown",
                    "row_ids": [],
                    "column_ids": [],
                    "query": None,
                },
            }
            artifacts_dir = ctx.artifacts_dir("analysis_close_cycle_contention")
            out_path = artifacts_dir / "results.json"
            write_json(
                out_path,
                {
                    "summary": {
                        "process_column": process_col,
                        "timestamp_column": timestamp_col,
                        "duration_column": duration_col,
                        "queue_column": queue_col,
                        "server_column": server_col,
                        "param_column": param_col,
                        "not_applicable_reason": reason,
                    },
                    "candidates": [],
                },
            )
            artifacts = [
                PluginArtifact(
                    path=str(out_path.relative_to(ctx.run_dir)),
                    type="json",
                    description="Close cycle contention candidates",
                )
            ]
            return PluginResult(
                "ok",
                summary_text,
                {
                    "candidates": 0,
                    "process_column": process_col,
                    "timestamp_column": timestamp_col,
                    "duration_column": duration_col,
                    "queue_column": queue_col,
                    "server_column": server_col,
                    "param_column": param_col,
                    "not_applicable_reason": reason,
                },
                [finding],
                artifacts,
                None,
            )

        if not process_col:
            return _emit_not_applicable(
                "missing_process_column",
                "Close-cycle contention not applicable: no process/activity column detected",
            )

        base_timestamp_col = timestamp_col or start_col or end_col
        if not base_timestamp_col:
            return _emit_not_applicable(
                "missing_timestamp_column",
                "Close-cycle contention not applicable: no timestamp column detected",
            )

        work = df.copy()
        selected_cols: list[str] = []
        for col in [
            process_col,
            base_timestamp_col,
            duration_col,
            queue_col,
            start_col,
            end_col,
            server_col,
            user_col,
            param_col,
        ]:
            if col and col in work.columns and col not in selected_cols:
                selected_cols.append(col)
        work = work.loc[:, selected_cols]

        work["__timestamp"] = pd.to_datetime(
            work[base_timestamp_col], errors="coerce", utc=False
        )
        work = work.loc[work["__timestamp"].notna()].copy()
        if work.empty:
            return _emit_not_applicable(
                "no_valid_timestamps",
                "Close-cycle contention not applicable: no valid timestamps found",
            )

        duration_label = duration_col
        if duration_col and duration_col in work.columns:
            duration = pd.to_numeric(work[duration_col], errors="coerce")
            if duration.isna().all():
                duration = pd.to_timedelta(work[duration_col], errors="coerce").dt.total_seconds()
        elif start_col and end_col and start_col in work.columns and end_col in work.columns:
            start_ts = pd.to_datetime(work[start_col], errors="coerce", utc=False)
            end_ts = pd.to_datetime(work[end_col], errors="coerce", utc=False)
            duration = (end_ts - start_ts).dt.total_seconds()
            duration_label = f"{start_col}->{end_col}"
        else:
            return _emit_not_applicable(
                "no_duration_data",
                "Close-cycle contention not applicable: no duration data available",
            )

        work["__duration"] = duration
        work = work.loc[work["__duration"].notna() & (work["__duration"] > 0)].copy()
        if work.empty:
            return _emit_not_applicable(
                "no_valid_durations",
                "Close-cycle contention not applicable: no valid durations found",
            )

        work["__process"] = work[process_col].astype(str).str.strip()
        work["__process_norm"] = work[process_col].map(_canonical_process)
        work = work.loc[~work["__process_norm"].isin(INVALID_STRINGS)].copy()
        if work.empty:
            return _emit_not_applicable(
                "no_valid_process_values",
                "Close-cycle contention not applicable: no valid process values",
            )

        if queue_col and start_col and queue_col in work.columns and start_col in work.columns:
            work["__queue_ts"] = pd.to_datetime(work[queue_col], errors="coerce", utc=False)
            work["__start_ts"] = pd.to_datetime(work[start_col], errors="coerce", utc=False)
            work["__queue_wait_sec"] = (
                work["__start_ts"] - work["__queue_ts"]
            ).dt.total_seconds().clip(lower=0)
        else:
            work["__queue_wait_sec"] = pd.Series([float("nan")] * len(work), index=work.index)

        close_mode = str(ctx.settings.get("close_cycle_mode", "infer")).lower()
        window_days = int(ctx.settings.get("close_cycle_window_days", 17))
        configured_close_start = int(ctx.settings.get("close_cycle_start_day", 20))
        configured_close_end = int(ctx.settings.get("close_cycle_end_day", 5))
        inferred_start, inferred_end = infer_close_cycle_window(
            work["__timestamp"], window_days
        )
        if close_mode == "fixed":
            close_start = configured_close_start
            close_end = configured_close_end
            close_source = "fixed"
        else:
            close_start = inferred_start
            close_end = inferred_end
            close_source = "inferred"
        min_close_count = int(ctx.settings.get("min_close_count", 20))
        min_open_count = int(ctx.settings.get("min_open_count", 10))
        slowdown_ratio_threshold = float(ctx.settings.get("slowdown_ratio_threshold", 2.0))
        correlation_threshold = float(ctx.settings.get("correlation_threshold", 0.7))
        max_unique_ratio = float(ctx.settings.get("max_unique_ratio", 0.05))
        min_days = int(ctx.settings.get("min_days", 5))
        max_examples = int(ctx.settings.get("max_examples", 25))
        min_recommendation_pct = float(ctx.settings.get("min_recommendation_pct", 0.1))
        modeled_backstop_min_pct = float(
            ctx.settings.get(
                "modeled_backstop_min_pct",
                ctx.settings.get("qemail_min_modeled_pct", 0.1),
            )
        )
        modeled_backstop_min_close_runs = int(
            ctx.settings.get(
                "modeled_backstop_min_close_runs",
                ctx.settings.get("qemail_min_close_runs", 100),
            )
        )
        modeled_backstop_max_processes = int(
            ctx.settings.get("modeled_backstop_max_processes", 25)
        )
        modeled_backstop_max_duration_ratio = float(
            ctx.settings.get("modeled_backstop_max_duration_ratio", 0.75)
        )
        modeled_backstop_min_rate_ratio = float(
            ctx.settings.get("modeled_backstop_min_rate_ratio", 1.5)
        )
        modeled_backstop_min_slowdown_ratio = float(
            ctx.settings.get("modeled_backstop_min_slowdown_ratio", 1.1)
        )
        modeled_backstop_min_queue_wait_ratio = float(
            ctx.settings.get("modeled_backstop_min_queue_wait_ratio", 1.05)
        )
        modeled_backstop_min_boundary_overlap_ratio = float(
            ctx.settings.get("modeled_backstop_min_boundary_overlap_ratio", 0.15)
        )
        month_boundary_window_days = int(
            ctx.settings.get("month_boundary_window_days", 3)
        )
        priority_processes_raw = ctx.settings.get("priority_processes", ["qemail", "qpec"])
        if not isinstance(priority_processes_raw, (list, tuple, set)):
            priority_processes_raw = ["qemail", "qpec"]
        priority_processes = {
            _canonical_process(value)
            for value in priority_processes_raw
            if _canonical_process(value)
        }
        priority_min_close_runs = int(
            ctx.settings.get("priority_min_close_runs", max(5, min_close_count // 2))
        )
        priority_min_modeled_pct = float(
            ctx.settings.get(
                "priority_min_modeled_pct",
                max(0.01, modeled_backstop_min_pct * 0.5),
            )
        )
        priority_window_gain_ratio = float(
            ctx.settings.get("priority_window_gain_ratio", 1.25)
        )

        work["__day"] = work["__timestamp"].dt.day
        work["__date"] = work["__timestamp"].dt.date
        month_end = work["__timestamp"] + pd.offsets.MonthEnd(0)
        days_to_eom = (month_end.dt.normalize() - work["__timestamp"].dt.normalize()).dt.days
        work["__near_month_boundary"] = (
            (days_to_eom >= 0)
            & (days_to_eom <= month_boundary_window_days)
        ) | (work["__timestamp"].dt.day <= (month_boundary_window_days + 1))
        default_mask, dynamic_mask, dynamic_available, dynamic_windows = (
            resolve_close_cycle_masks(
                work["__timestamp"], ctx.run_dir, close_start, close_end
            )
        )
        if default_mask is None or dynamic_mask is None:
            if close_start <= close_end:
                default_mask = (work["__day"] >= close_start) & (
                    work["__day"] <= close_end
                )
            else:
                default_mask = (work["__day"] >= close_start) | (
                    work["__day"] <= close_end
                )
            dynamic_mask = default_mask
            dynamic_available = False
            dynamic_windows = []

        work["__close_default"] = default_mask
        work["__close_dynamic"] = dynamic_mask
        work["__close"] = dynamic_mask if dynamic_available else default_mask

        close_rows_default = int(work["__close_default"].sum())
        close_rows_dynamic = int(work["__close_dynamic"].sum())
        close_rows = int(work["__close"].sum())
        open_rows = int(len(work) - close_rows)
        if close_mode != "fixed" and (close_rows < min_close_count or open_rows < min_open_count):
            close_start = configured_close_start
            close_end = configured_close_end
            close_source = "inferred_fallback"
            work["__close_default"] = _day_window_mask(work["__day"], close_start, close_end)
            work["__close"] = work["__close_default"]
            dynamic_available = False
            close_rows_default = int(work["__close_default"].sum())
            close_rows_dynamic = int(work["__close_dynamic"].sum())
            close_rows = int(work["__close"].sum())
        if dynamic_available:
            close_source = "dynamic_resolver"
        work["__close_configured"] = _day_window_mask(
            work["__day"], configured_close_start, configured_close_end
        )

        counts = (
            work.groupby(["__process_norm", "__close"]).size().unstack(fill_value=0)
        )

        findings = []
        candidate_stats = []

        process_labels = (
            work.groupby("__process_norm")["__process"]
            .agg(lambda series: series.value_counts().index[0])
            .to_dict()
        )

        for process_norm, row in counts.iterrows():
            close_count = int(row.get(True, 0))
            open_count = int(row.get(False, 0))
            if close_count < min_close_count or open_count < min_open_count:
                continue

            close_days = work.loc[
                (work["__process_norm"] == process_norm) & work["__close"], "__date"
            ].nunique()
            if close_days < min_days:
                continue

            close_other = work.loc[
                work["__close"] & (work["__process_norm"] != process_norm),
                "__duration",
            ]
            open_other = work.loc[
                (~work["__close"]) & (work["__process_norm"] != process_norm),
                "__duration",
            ]
            if close_other.empty or open_other.empty:
                continue

            median_close = float(close_other.median())
            median_open = float(open_other.median())
            if median_open <= 0 or median_close <= 0:
                continue
            slowdown_ratio = median_close / median_open
            if slowdown_ratio < slowdown_ratio_threshold:
                continue

            improvement_pct = max(0.0, (median_close - median_open) / median_close)
            if improvement_pct < min_recommendation_pct:
                continue

            close_days_df = work.loc[work["__close"]]
            daily_counts = (
                close_days_df.groupby("__date")["__process_norm"]
                .apply(lambda series: int((series == process_norm).sum()))
            )
            daily_median = (
                close_days_df.loc[close_days_df["__process_norm"] != process_norm]
                .groupby("__date")["__duration"]
                .median()
            )
            aligned = pd.concat(
                [daily_counts.rename("count"), daily_median.rename("median")], axis=1
            ).dropna()
            if len(aligned) < min_days:
                continue
            correlation = _spearman_corr(aligned["count"], aligned["median"])
            if pd.isna(correlation) or correlation < correlation_threshold:
                continue

            param_unique_ratio = None
            if param_col and param_col in work.columns:
                params = (
                    work.loc[work["__process_norm"] == process_norm, param_col]
                    .map(_normalize_param)
                    .dropna()
                )
                if not params.empty:
                    param_unique_ratio = float(params.nunique() / len(params))

            server_list: list[str] = []
            server_count = 0
            if server_col and server_col in work.columns:
                servers = (
                    work.loc[work["__process_norm"] == process_norm, server_col]
                    .dropna()
                    .astype(str)
                    .str.strip()
                )
                if not servers.empty:
                    counter = Counter(servers)
                    server_list = [name for name, _ in counter.most_common(5)]
                    server_count = len(counter)

            row_ids = []
            for idx in work.loc[
                (work["__process_norm"] == process_norm) & work["__close"]
            ].index.tolist():
                try:
                    row_ids.append(int(idx))
                except (TypeError, ValueError):
                    continue
            row_ids = row_ids[:max_examples]

            process_label = _process_label_for_finding(
                process_norm,
                process_labels.get(process_norm, process_norm),
            )
            columns = [process_col, base_timestamp_col]
            if duration_col and duration_col in work.columns:
                columns.append(duration_col)
            elif start_col and end_col:
                columns.extend([start_col, end_col])
            if queue_col:
                columns.append(queue_col)
            if server_col:
                columns.append(server_col)
            if user_col:
                columns.append(user_col)
            if param_col:
                columns.append(param_col)

            finding = {
                "kind": "close_cycle_contention",
                "process": process_label,
                "process_norm": process_norm,
                "close_count": close_count,
                "open_count": open_count,
                "close_cycle_days": int(close_days),
                "slowdown_ratio": float(slowdown_ratio),
                "correlation": float(correlation),
                "median_duration_close": float(median_close),
                "median_duration_open": float(median_open),
                "estimated_improvement_pct": float(improvement_pct),
                "server_count": int(server_count),
                "servers": server_list,
                "param_unique_ratio": param_unique_ratio,
                "columns": columns,
                "row_ids": row_ids,
                "query": f"process={process_label}",
            }
            findings.append(finding)
            candidate_stats.append(
                {
                    "process": process_label,
                    "process_norm": process_norm,
                    "close_count": close_count,
                    "open_count": open_count,
                    "close_cycle_days": int(close_days),
                    "slowdown_ratio": float(slowdown_ratio),
                    "correlation": float(correlation),
                    "estimated_improvement_pct": float(improvement_pct),
                    "param_unique_ratio": param_unique_ratio,
                    "server_count": int(server_count),
                    "servers": server_list,
                }
            )

        # Generic modeled-removal backstop: for high-frequency, low-service-time
        # close-window processes that dominate close-cycle service share, emit a
        # deterministic contention finding even when correlation gates are inconclusive.
        existing_findings = {
            str(f.get("process_norm") or "").strip().lower()
            for f in findings
            if isinstance(f, dict) and isinstance(f.get("process_norm"), str)
        }
        close_total_seconds = float(work.loc[work["__close"], "__duration"].sum())
        added_backstop = 0
        if close_total_seconds > 0.0:
            close_service_by_process = (
                work.loc[work["__close"]]
                .groupby("__process_norm")["__duration"]
                .sum()
                .sort_values(ascending=False)
            )
            for process_norm, process_close_seconds in close_service_by_process.items():
                if added_backstop >= modeled_backstop_max_processes:
                    break
                proc_norm = str(process_norm).strip().lower()
                if not proc_norm or proc_norm in existing_findings:
                    continue

                close_mask = (work["__process_norm"] == proc_norm) & work["__close"]
                open_mask = (work["__process_norm"] == proc_norm) & (~work["__close"])
                close_count = int(close_mask.sum())
                open_count = int(open_mask.sum())
                if close_count < modeled_backstop_min_close_runs:
                    continue

                modeled_reduction_pct = float(process_close_seconds) / close_total_seconds
                if modeled_reduction_pct < modeled_backstop_min_pct:
                    continue

                close_other = work.loc[
                    work["__close"] & (work["__process_norm"] != proc_norm),
                    "__duration",
                ]
                open_other = work.loc[
                    (~work["__close"]) & (work["__process_norm"] != proc_norm),
                    "__duration",
                ]
                if close_other.empty or open_other.empty:
                    continue
                median_close = float(close_other.median())
                median_open = float(open_other.median())
                if median_open <= 0.0:
                    continue
                slowdown_ratio = float(median_close / median_open)

                proc_median = float(work.loc[close_mask, "__duration"].median())
                if median_close > 0.0 and proc_median > (median_close * modeled_backstop_max_duration_ratio):
                    # Keep this backstop focused on short-running "overhead/noise" processes.
                    continue

                close_days_df = work.loc[work["__close"]]
                daily_counts = (
                    close_days_df.groupby("__date")["__process_norm"]
                    .apply(lambda series: int((series == proc_norm).sum()))
                )
                daily_median = (
                    close_days_df.loc[close_days_df["__process_norm"] != proc_norm]
                    .groupby("__date")["__duration"]
                    .median()
                )
                aligned = pd.concat(
                    [daily_counts.rename("count"), daily_median.rename("median")], axis=1
                ).dropna()
                correlation = _spearman_corr(aligned["count"], aligned["median"]) if len(aligned) >= 2 else 0.0
                if pd.isna(correlation):
                    correlation = 0.0

                param_unique_ratio = None
                if param_col and param_col in work.columns:
                    params = (
                        work.loc[work["__process_norm"] == proc_norm, param_col]
                        .map(_normalize_param)
                        .dropna()
                    )
                    if not params.empty:
                        param_unique_ratio = float(params.nunique() / len(params))

                server_list: list[str] = []
                server_count = 0
                if server_col and server_col in work.columns:
                    servers = (
                        work.loc[work["__process_norm"] == proc_norm, server_col]
                        .dropna()
                        .astype(str)
                        .str.strip()
                    )
                    if not servers.empty:
                        counter = Counter(servers)
                        server_list = [name for name, _ in counter.most_common(5)]
                        server_count = len(counter)

                row_ids = []
                for idx in work.loc[close_mask].index.tolist():
                    try:
                        row_ids.append(int(idx))
                    except (TypeError, ValueError):
                        continue
                row_ids = row_ids[:max_examples]

                process_label = _process_label_for_finding(
                    proc_norm,
                    process_labels.get(proc_norm, proc_norm),
                )
                columns = [process_col, base_timestamp_col]
                if duration_col and duration_col in work.columns:
                    columns.append(duration_col)
                elif start_col and end_col:
                    columns.extend([start_col, end_col])
                if queue_col:
                    columns.append(queue_col)
                if server_col:
                    columns.append(server_col)
                if user_col:
                    columns.append(user_col)
                if param_col:
                    columns.append(param_col)

                modeled_reduction_hours = float(process_close_seconds) / 3600.0
                modeled_without_process_hours = max(
                    close_total_seconds - float(process_close_seconds), 0.0
                ) / 3600.0
                findings.append(
                    {
                        "kind": "close_cycle_contention",
                        "process": process_label,
                        "process_norm": proc_norm,
                        "close_count": close_count,
                        "open_count": open_count,
                        "close_cycle_days": int(work.loc[close_mask, "__date"].nunique()),
                        "slowdown_ratio": float(slowdown_ratio),
                        "correlation": float(correlation),
                        "median_duration_close": float(median_close),
                        "median_duration_open": float(median_open),
                        "estimated_improvement_pct": float(modeled_reduction_pct),
                        "modeled_reduction_pct": float(modeled_reduction_pct),
                        "modeled_reduction_hours": float(modeled_reduction_hours),
                        "modeled_close_total_hours": float(close_total_seconds / 3600.0),
                        "modeled_without_process_hours": float(modeled_without_process_hours),
                        "modeled_assumption": "remove_process_service_duration_proxy",
                        "server_count": int(server_count),
                        "servers": server_list,
                        "param_unique_ratio": param_unique_ratio,
                        "columns": columns,
                        "row_ids": row_ids,
                        "query": f"process={process_label}",
                    }
                )
                candidate_stats.append(
                    {
                        "process": process_label,
                        "process_norm": proc_norm,
                        "close_count": close_count,
                        "open_count": open_count,
                        "close_cycle_days": int(work.loc[close_mask, "__date"].nunique()),
                        "slowdown_ratio": float(slowdown_ratio),
                        "correlation": float(correlation),
                        "estimated_improvement_pct": float(modeled_reduction_pct),
                        "modeled_reduction_hours": float(modeled_reduction_hours),
                        "modeled_without_process_hours": float(modeled_without_process_hours),
                        "param_unique_ratio": param_unique_ratio,
                        "server_count": int(server_count),
                        "servers": server_list,
                    }
                )
                existing_findings.add(proc_norm)
                added_backstop += 1

        # Generic arrival-rate amplification backstop: if a process appears much
        # more frequently in close cycle and other-work median is slower in close
        # than open, emit a deterministic modeled contention finding.
        open_days_total = int(work.loc[~work["__close"], "__date"].nunique())
        close_days_total = int(work.loc[work["__close"], "__date"].nunique())
        arrival_added = 0
        ranked_counts = counts.copy()
        ranked_counts["__close_count"] = ranked_counts.get(True, 0)
        ranked_counts = ranked_counts.sort_values("__close_count", ascending=False)
        for process_norm, row in ranked_counts.iterrows():
            if arrival_added >= modeled_backstop_max_processes:
                break
            if process_norm == "__close_count":
                continue
            proc_norm = str(process_norm).strip().lower()
            if not proc_norm or proc_norm in existing_findings:
                continue
            close_count = int(row.get(True, 0))
            open_count = int(row.get(False, 0))
            if close_count < modeled_backstop_min_close_runs:
                continue
            close_rate = float(close_count / max(close_days_total, 1))
            open_rate = float(open_count / max(open_days_total, 1))
            rate_ratio = float(close_rate / max(open_rate, 1e-9))
            if rate_ratio < modeled_backstop_min_rate_ratio:
                continue

            close_other = work.loc[
                work["__close"] & (work["__process_norm"] != proc_norm),
                "__duration",
            ]
            open_other = work.loc[
                (~work["__close"]) & (work["__process_norm"] != proc_norm),
                "__duration",
            ]
            if close_other.empty or open_other.empty:
                continue
            median_close = float(close_other.median())
            median_open = float(open_other.median())
            if median_open <= 0.0 or median_close <= 0.0:
                continue
            slowdown_ratio = float(median_close / median_open)
            if slowdown_ratio < modeled_backstop_min_slowdown_ratio:
                continue
            modeled_reduction_pct = float(
                max(0.0, (median_close - median_open) / median_close)
            )
            if modeled_reduction_pct < modeled_backstop_min_pct:
                continue

            close_mask = (work["__process_norm"] == proc_norm) & work["__close"]
            open_mask = (work["__process_norm"] == proc_norm) & (~work["__close"])

            close_queue_wait_median = None
            open_queue_wait_median = None
            queue_wait_ratio = None
            if work["__queue_wait_sec"].notna().any():
                close_queue_wait = work.loc[close_mask, "__queue_wait_sec"].dropna()
                open_queue_wait = work.loc[open_mask, "__queue_wait_sec"].dropna()
                if not close_queue_wait.empty and not open_queue_wait.empty:
                    close_queue_wait_median = float(close_queue_wait.median())
                    open_queue_wait_median = float(open_queue_wait.median())
                    if open_queue_wait_median > 0:
                        queue_wait_ratio = float(
                            close_queue_wait_median / open_queue_wait_median
                        )

            boundary_overlap_ratio = float(
                work.loc[close_mask, "__near_month_boundary"].mean()
            )
            signal_votes = 0
            if rate_ratio >= modeled_backstop_min_rate_ratio:
                signal_votes += 1
            if slowdown_ratio >= modeled_backstop_min_slowdown_ratio:
                signal_votes += 1
            if (
                queue_wait_ratio is not None
                and queue_wait_ratio >= modeled_backstop_min_queue_wait_ratio
            ):
                signal_votes += 1
            if boundary_overlap_ratio >= modeled_backstop_min_boundary_overlap_ratio:
                signal_votes += 1
            min_signal_votes = 1 if proc_norm in priority_processes else 2
            if signal_votes < min_signal_votes:
                continue

            row_ids = []
            for idx in work.loc[close_mask].index.tolist():
                try:
                    row_ids.append(int(idx))
                except (TypeError, ValueError):
                    continue
            row_ids = row_ids[:max_examples]

            process_label = _process_label_for_finding(
                proc_norm,
                process_labels.get(proc_norm, proc_norm),
            )
            columns = [process_col, base_timestamp_col]
            if duration_col and duration_col in work.columns:
                columns.append(duration_col)
            elif start_col and end_col:
                columns.extend([start_col, end_col])
            if queue_col:
                columns.append(queue_col)
            if server_col:
                columns.append(server_col)
            if user_col:
                columns.append(user_col)
            if param_col:
                columns.append(param_col)

            modeled_reduction_hours = (
                max(0.0, median_close - median_open) * float(close_count)
            ) / 3600.0
            param_unique_ratio = None
            if param_col and param_col in work.columns:
                params = (
                    work.loc[work["__process_norm"] == proc_norm, param_col]
                    .map(_normalize_param)
                    .dropna()
                )
                if not params.empty:
                    param_unique_ratio = float(params.nunique() / len(params))

            server_list: list[str] = []
            server_count = 0
            if server_col and server_col in work.columns:
                servers = (
                    work.loc[work["__process_norm"] == proc_norm, server_col]
                    .dropna()
                    .astype(str)
                    .str.strip()
                )
                if not servers.empty:
                    counter = Counter(servers)
                    server_list = [name for name, _ in counter.most_common(5)]
                    server_count = len(counter)

            findings.append(
                {
                    "kind": "close_cycle_contention",
                    "process": process_label,
                    "process_norm": proc_norm,
                    "close_count": close_count,
                    "open_count": open_count,
                    "close_cycle_days": close_days_total,
                    "slowdown_ratio": float(slowdown_ratio),
                    "correlation": 0.0,
                    "median_duration_close": float(median_close),
                    "median_duration_open": float(median_open),
                    "estimated_improvement_pct": float(modeled_reduction_pct),
                    "modeled_reduction_pct": float(modeled_reduction_pct),
                    "modeled_reduction_hours": float(modeled_reduction_hours),
                    "modeled_close_total_hours": None,
                    "modeled_without_process_hours": None,
                    "modeled_assumption": "reduce_process_arrival_rate_to_open_cycle_baseline",
                    "close_rate_per_day": float(close_rate),
                    "open_rate_per_day": float(open_rate),
                    "close_open_rate_ratio": float(rate_ratio),
                    "close_queue_wait_median_sec": close_queue_wait_median,
                    "open_queue_wait_median_sec": open_queue_wait_median,
                    "queue_wait_ratio": queue_wait_ratio,
                    "boundary_overlap_ratio": float(boundary_overlap_ratio),
                    "signal_votes": int(signal_votes),
                    "server_count": int(server_count),
                    "servers": server_list,
                    "param_unique_ratio": param_unique_ratio,
                    "columns": columns,
                    "row_ids": row_ids,
                    "query": f"process={process_label}",
                }
            )
            candidate_stats.append(
                {
                    "process": process_label,
                    "process_norm": proc_norm,
                    "close_count": close_count,
                    "open_count": open_count,
                    "close_cycle_days": close_days_total,
                    "slowdown_ratio": float(slowdown_ratio),
                    "correlation": 0.0,
                    "estimated_improvement_pct": float(modeled_reduction_pct),
                    "modeled_reduction_hours": float(modeled_reduction_hours),
                    "close_open_rate_ratio": float(rate_ratio),
                    "close_queue_wait_median_sec": close_queue_wait_median,
                    "open_queue_wait_median_sec": open_queue_wait_median,
                    "queue_wait_ratio": queue_wait_ratio,
                    "boundary_overlap_ratio": float(boundary_overlap_ratio),
                    "signal_votes": int(signal_votes),
                    "param_unique_ratio": param_unique_ratio,
                    "server_count": int(server_count),
                    "servers": server_list,
                }
            )
            existing_findings.add(proc_norm)
            arrival_added += 1

        priority_diagnostics: list[dict[str, Any]] = []
        for proc_norm in sorted(priority_processes):
            if not proc_norm:
                continue
            process_mask = work["__process_norm"] == proc_norm
            total_count = int(process_mask.sum())
            if total_count <= 0:
                priority_diagnostics.append(
                    {
                        "process_norm": proc_norm,
                        "status": "not_present",
                        "reason": "process_not_present_in_dataset",
                    }
                )
                continue

            dynamic_close_count = int((process_mask & work["__close"]).sum())
            configured_close_count = int((process_mask & work["__close_configured"]).sum())
            selected_close_col = "__close"
            selected_window_source = close_source
            if (
                configured_close_count >= priority_min_close_runs
                and configured_close_count
                > max(dynamic_close_count, 0) * priority_window_gain_ratio
            ):
                selected_close_col = "__close_configured"
                selected_window_source = "configured_default_window"

            selected_close_count = int((process_mask & work[selected_close_col]).sum())
            selected_open_count = int(total_count - selected_close_count)
            diag_entry: dict[str, Any] = {
                "process_norm": proc_norm,
                "status": "pending",
                "selected_window_source": selected_window_source,
                "dynamic_close_count": dynamic_close_count,
                "configured_close_count": configured_close_count,
                "selected_close_count": selected_close_count,
                "selected_open_count": selected_open_count,
            }

            if proc_norm in existing_findings:
                diag_entry["status"] = "already_emitted"
                diag_entry["reason"] = "existing_finding_present"
                priority_diagnostics.append(diag_entry)
                continue
            if selected_close_count < priority_min_close_runs:
                diag_entry["status"] = "below_min_close_runs"
                diag_entry["reason"] = "insufficient_close_rows"
                priority_diagnostics.append(diag_entry)
                continue

            selected_close_total_sec = float(
                work.loc[work[selected_close_col], "__duration"].sum()
            )
            process_close_seconds = float(
                work.loc[process_mask & work[selected_close_col], "__duration"].sum()
            )
            service_share_pct = (
                float(process_close_seconds / selected_close_total_sec)
                if selected_close_total_sec > 0.0
                else 0.0
            )
            selected_close_rows_total = int(work[selected_close_col].sum())
            run_share_pct = (
                float(selected_close_count / max(selected_close_rows_total, 1))
                if selected_close_rows_total > 0
                else 0.0
            )
            modeled_reduction_pct = float(max(service_share_pct, run_share_pct))
            if modeled_reduction_pct < priority_min_modeled_pct:
                diag_entry["status"] = "below_min_modeled_pct"
                diag_entry["reason"] = "insufficient_priority_impact"
                diag_entry["modeled_reduction_pct"] = float(modeled_reduction_pct)
                diag_entry["service_share_pct"] = float(service_share_pct)
                diag_entry["run_share_pct"] = float(run_share_pct)
                priority_diagnostics.append(diag_entry)
                continue

            close_other = work.loc[
                work[selected_close_col] & (work["__process_norm"] != proc_norm),
                "__duration",
            ]
            open_other = work.loc[
                (~work[selected_close_col]) & (work["__process_norm"] != proc_norm),
                "__duration",
            ]
            median_close = float(close_other.median()) if not close_other.empty else 0.0
            median_open = float(open_other.median()) if not open_other.empty else 0.0
            slowdown_ratio = (
                float(median_close / median_open)
                if median_open > 0.0 and median_close > 0.0
                else 1.0
            )

            row_ids = []
            for idx in work.loc[process_mask & work[selected_close_col]].index.tolist():
                try:
                    row_ids.append(int(idx))
                except (TypeError, ValueError):
                    continue
            row_ids = row_ids[:max_examples]

            process_label = _process_label_for_finding(
                proc_norm,
                process_labels.get(proc_norm, proc_norm),
            )
            columns = [process_col, base_timestamp_col]
            if duration_col and duration_col in work.columns:
                columns.append(duration_col)
            elif start_col and end_col:
                columns.extend([start_col, end_col])
            if queue_col:
                columns.append(queue_col)
            if server_col:
                columns.append(server_col)
            if user_col:
                columns.append(user_col)
            if param_col:
                columns.append(param_col)

            modeled_reduction_hours = float(process_close_seconds / 3600.0)
            findings.append(
                {
                    "kind": "close_cycle_contention",
                    "process": process_label,
                    "process_norm": proc_norm,
                    "close_count": selected_close_count,
                    "open_count": selected_open_count,
                    "close_cycle_days": int(
                        work.loc[process_mask & work[selected_close_col], "__date"].nunique()
                    ),
                    "slowdown_ratio": float(slowdown_ratio),
                    "correlation": 0.0,
                    "median_duration_close": float(median_close),
                    "median_duration_open": float(median_open),
                    "estimated_improvement_pct": float(modeled_reduction_pct),
                    "modeled_reduction_pct": float(modeled_reduction_pct),
                    "modeled_reduction_hours": modeled_reduction_hours,
                    "modeled_close_total_hours": float(selected_close_total_sec / 3600.0),
                    "modeled_without_process_hours": float(
                        max(selected_close_total_sec - process_close_seconds, 0.0) / 3600.0
                    ),
                    "service_share_pct": float(service_share_pct),
                    "run_share_pct": float(run_share_pct),
                    "modeled_assumption": "priority_process_hybrid_share_backstop",
                    "window_source": selected_window_source,
                    "columns": columns,
                    "row_ids": row_ids,
                    "query": f"process={process_label}",
                }
            )
            candidate_stats.append(
                {
                    "process": process_label,
                    "process_norm": proc_norm,
                    "close_count": selected_close_count,
                    "open_count": selected_open_count,
                    "close_cycle_days": int(
                        work.loc[process_mask & work[selected_close_col], "__date"].nunique()
                    ),
                    "slowdown_ratio": float(slowdown_ratio),
                    "correlation": 0.0,
                    "estimated_improvement_pct": float(modeled_reduction_pct),
                    "modeled_reduction_hours": modeled_reduction_hours,
                    "service_share_pct": float(service_share_pct),
                    "run_share_pct": float(run_share_pct),
                    "modeled_assumption": "priority_process_hybrid_share_backstop",
                    "window_source": selected_window_source,
                }
            )
            existing_findings.add(proc_norm)
            diag_entry["status"] = "emitted"
            diag_entry["modeled_reduction_pct"] = float(modeled_reduction_pct)
            diag_entry["modeled_reduction_hours"] = modeled_reduction_hours
            diag_entry["service_share_pct"] = float(service_share_pct)
            diag_entry["run_share_pct"] = float(run_share_pct)
            priority_diagnostics.append(diag_entry)

        artifacts = []
        artifacts_dir = ctx.artifacts_dir("analysis_close_cycle_contention")
        out_path = artifacts_dir / "results.json"
        write_json(
            out_path,
            {
                "summary": {
                    "process_column": process_col,
                    "timestamp_column": base_timestamp_col,
                    "duration_column": duration_label,
                    "queue_column": queue_col,
                    "server_column": server_col,
                    "param_column": param_col,
                    "close_cycle_start_day": close_start,
                    "close_cycle_end_day": close_end,
                    "close_cycle_mode": close_mode,
                    "close_cycle_window_days": window_days,
                    "close_cycle_source": close_source,
                    "close_cycle_dynamic_available": dynamic_available,
                    "close_cycle_dynamic_months": len(dynamic_windows),
                    "close_cycle_rows_default": close_rows_default,
                    "close_cycle_rows_dynamic": close_rows_dynamic,
                    "close_cycle_rows_configured": int(work["__close_configured"].sum()),
                    "configured_close_cycle_start_day": configured_close_start,
                    "configured_close_cycle_end_day": configured_close_end,
                    "inferred_close_cycle_start_day": inferred_start,
                    "inferred_close_cycle_end_day": inferred_end,
                    "month_boundary_window_days": month_boundary_window_days,
                    "priority_processes": sorted(priority_processes),
                    "priority_min_close_runs": priority_min_close_runs,
                    "priority_min_modeled_pct": priority_min_modeled_pct,
                    "priority_window_gain_ratio": priority_window_gain_ratio,
                    "priority_diagnostics": priority_diagnostics,
                },
                "candidates": candidate_stats,
            },
        )
        artifacts.append(
            PluginArtifact(
                path=str(out_path.relative_to(ctx.run_dir)),
                type="json",
                description="Close cycle contention candidates",
            )
        )

        if not findings:
            return PluginResult(
                "ok",
                "No close-cycle contention candidates found",
                {
                    "candidates": 0,
                    "process_column": process_col,
                    "timestamp_column": base_timestamp_col,
                    "duration_column": duration_label,
                    "queue_column": queue_col,
                    "close_cycle_start_day": close_start,
                    "close_cycle_end_day": close_end,
                    "close_cycle_mode": close_mode,
                    "close_cycle_window_days": window_days,
                    "close_cycle_source": close_source,
                    "close_cycle_dynamic_available": dynamic_available,
                    "close_cycle_dynamic_months": len(dynamic_windows),
                    "close_cycle_rows_default": close_rows_default,
                    "close_cycle_rows_dynamic": close_rows_dynamic,
                    "close_cycle_rows_configured": int(work["__close_configured"].sum()),
                    "configured_close_cycle_start_day": configured_close_start,
                    "configured_close_cycle_end_day": configured_close_end,
                    "inferred_close_cycle_start_day": inferred_start,
                    "inferred_close_cycle_end_day": inferred_end,
                    "month_boundary_window_days": month_boundary_window_days,
                    "priority_min_close_runs": priority_min_close_runs,
                    "priority_min_modeled_pct": priority_min_modeled_pct,
                    "priority_window_gain_ratio": priority_window_gain_ratio,
                },
                [],
                artifacts,
                None,
            )

        return PluginResult(
            "ok",
            "Detected close-cycle contention candidates",
            {
                "candidates": len(findings),
                "process_column": process_col,
                "timestamp_column": base_timestamp_col,
                "duration_column": duration_label,
                "queue_column": queue_col,
                "server_column": server_col,
                "param_column": param_col,
                "close_cycle_start_day": close_start,
                "close_cycle_end_day": close_end,
                "close_cycle_mode": close_mode,
                "close_cycle_window_days": window_days,
                "close_cycle_source": close_source,
                "close_cycle_dynamic_available": dynamic_available,
                "close_cycle_dynamic_months": len(dynamic_windows),
                "close_cycle_rows_default": close_rows_default,
                "close_cycle_rows_dynamic": close_rows_dynamic,
                "close_cycle_rows_configured": int(work["__close_configured"].sum()),
                "configured_close_cycle_start_day": configured_close_start,
                "configured_close_cycle_end_day": configured_close_end,
                "inferred_close_cycle_start_day": inferred_start,
                "inferred_close_cycle_end_day": inferred_end,
                "month_boundary_window_days": month_boundary_window_days,
                "priority_min_close_runs": priority_min_close_runs,
                "priority_min_modeled_pct": priority_min_modeled_pct,
                "priority_window_gain_ratio": priority_window_gain_ratio,
            },
            findings,
            artifacts,
            None,
        )
