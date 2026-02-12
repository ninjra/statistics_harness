from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

import pandas as pd

from statistic_harness.core.close_cycle import resolve_close_cycle_masks
from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import infer_close_cycle_window, write_json


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

        if not process_col:
            return PluginResult(
                "skipped", "No process/activity column detected", {}, [], [], None
            )

        base_timestamp_col = timestamp_col or start_col or end_col
        if not base_timestamp_col:
            return PluginResult(
                "skipped", "No timestamp column detected", {}, [], [], None
            )

        work = df.copy()
        selected_cols: list[str] = []
        for col in [
            process_col,
            base_timestamp_col,
            duration_col,
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
            return PluginResult(
                "skipped", "No valid timestamps found", {}, [], [], None
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
            return PluginResult(
                "skipped",
                "No duration data available",
                {},
                [],
                [],
                None,
            )

        work["__duration"] = duration
        work = work.loc[work["__duration"].notna() & (work["__duration"] > 0)].copy()
        if work.empty:
            return PluginResult(
                "skipped", "No valid durations found", {}, [], [], None
            )

        work["__process"] = work[process_col].astype(str).str.strip()
        work["__process_norm"] = work["__process"].str.lower()
        invalid = {"", "nan", "none", "null"}
        work = work.loc[~work["__process_norm"].isin(invalid)].copy()
        if work.empty:
            return PluginResult(
                "skipped", "No valid process values", {}, [], [], None
            )

        close_mode = str(ctx.settings.get("close_cycle_mode", "infer")).lower()
        window_days = int(ctx.settings.get("close_cycle_window_days", 17))
        inferred_start, inferred_end = infer_close_cycle_window(
            work["__timestamp"], window_days
        )
        if close_mode == "fixed":
            close_start = int(ctx.settings.get("close_cycle_start_day", inferred_start))
            close_end = int(ctx.settings.get("close_cycle_end_day", inferred_end))
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
            ctx.settings.get("modeled_backstop_max_processes", 3)
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

        work["__day"] = work["__timestamp"].dt.day
        work["__date"] = work["__timestamp"].dt.date
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
            close_start = int(ctx.settings.get("close_cycle_start_day", close_start))
            close_end = int(ctx.settings.get("close_cycle_end_day", close_end))
            close_source = "inferred_fallback"
            if close_start <= close_end:
                work["__close_default"] = (work["__day"] >= close_start) & (
                    work["__day"] <= close_end
                )
            else:
                work["__close_default"] = (work["__day"] >= close_start) | (
                    work["__day"] <= close_end
                )
            work["__close"] = work["__close_default"]
            dynamic_available = False
            close_rows_default = int(work["__close_default"].sum())
            close_rows_dynamic = int(work["__close_dynamic"].sum())
            close_rows = int(work["__close"].sum())
        if dynamic_available:
            close_source = "dynamic_resolver"

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

            process_label = process_labels.get(process_norm, process_norm)
            columns = [process_col, base_timestamp_col]
            if duration_col and duration_col in work.columns:
                columns.append(duration_col)
            elif start_col and end_col:
                columns.extend([start_col, end_col])
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

                process_label = process_labels.get(proc_norm, proc_norm)
                columns = [process_col, base_timestamp_col]
                if duration_col and duration_col in work.columns:
                    columns.append(duration_col)
                elif start_col and end_col:
                    columns.extend([start_col, end_col])
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
        for process_norm, row in counts.iterrows():
            if arrival_added >= modeled_backstop_max_processes:
                break
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
            row_ids = []
            for idx in work.loc[close_mask].index.tolist():
                try:
                    row_ids.append(int(idx))
                except (TypeError, ValueError):
                    continue
            row_ids = row_ids[:max_examples]

            process_label = process_labels.get(proc_norm, proc_norm)
            columns = [process_col, base_timestamp_col]
            if duration_col and duration_col in work.columns:
                columns.append(duration_col)
            elif start_col and end_col:
                columns.extend([start_col, end_col])
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
                    "param_unique_ratio": param_unique_ratio,
                    "server_count": int(server_count),
                    "servers": server_list,
                }
            )
            existing_findings.add(proc_norm)
            arrival_added += 1

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
                    "inferred_close_cycle_start_day": inferred_start,
                    "inferred_close_cycle_end_day": inferred_end,
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
                    "close_cycle_start_day": close_start,
                    "close_cycle_end_day": close_end,
                    "close_cycle_mode": close_mode,
                    "close_cycle_window_days": window_days,
                    "close_cycle_source": close_source,
                    "close_cycle_dynamic_available": dynamic_available,
                    "close_cycle_dynamic_months": len(dynamic_windows),
                    "close_cycle_rows_default": close_rows_default,
                    "close_cycle_rows_dynamic": close_rows_dynamic,
                    "inferred_close_cycle_start_day": inferred_start,
                    "inferred_close_cycle_end_day": inferred_end,
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
                "inferred_close_cycle_start_day": inferred_start,
                "inferred_close_cycle_end_day": inferred_end,
            },
            findings,
            artifacts,
            None,
        )
