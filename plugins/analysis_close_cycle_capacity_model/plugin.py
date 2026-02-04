from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Iterable

import numpy as np
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


def _candidate_columns(
    columns: list[str],
    role_by_name: dict[str, str],
    roles: set[str],
    patterns: list[str],
    lower_names: dict[str, str],
    exclude: set[str],
) -> list[str]:
    candidates: list[str] = []
    for col in columns:
        if col in exclude:
            continue
        if role_by_name.get(col) in roles:
            candidates.append(col)
    for col in columns:
        if col in exclude or col in candidates:
            continue
        name = lower_names[col]
        if any(pattern in name for pattern in patterns):
            candidates.append(col)
    return candidates


def _score_process_column(name: str, series: pd.Series) -> float:
    score = 0.0
    lower_name = name.lower()
    if lower_name in {"process", "process_id"}:
        score += 3.0
    if lower_name.endswith("_id") or lower_name.endswith("id"):
        score += 1.5
    for token in (
        "queue",
        "status",
        "step",
        "parent",
        "child",
        "hold",
        "lock",
        "schedule",
        "master",
        "dep",
        "ext",
        "attempt",
        "priority",
    ):
        if token in lower_name:
            score -= 2.0

    sample = series.dropna()
    if sample.empty:
        return score - 5.0
    if sample.shape[0] > 5000:
        sample = sample.sample(5000, random_state=0)

    if pd.api.types.is_numeric_dtype(sample):
        score -= 1.5
    else:
        score += 1.5

    sample_str = sample.astype(str).str.strip()
    if not pd.api.types.is_numeric_dtype(sample):
        numeric_like = sample_str.str.match(r"^\d+(\.\d+)?$").mean()
        if numeric_like > 0.8:
            score -= 2.0

    unique_ratio = sample.nunique(dropna=True) / max(1, sample.shape[0])
    score += (1.0 - unique_ratio) * 4.0
    if unique_ratio > 0.9:
        score -= 2.0

    lengths = sample_str.str.len()
    median_len = float(lengths.median()) if not lengths.empty else 0.0
    if 3 <= median_len <= 20:
        score += 0.5
    elif median_len > 40:
        score -= 0.5

    return score


def _choose_best_process_column(
    candidates: Iterable[str], df: pd.DataFrame
) -> str | None:
    candidates = list(candidates)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    scored = []
    for col in candidates:
        scored.append((_score_process_column(str(col), df[col]), col))
    scored.sort(reverse=True, key=lambda item: item[0])
    return scored[0][1]


def _score_host_column(name: str, series: pd.Series) -> float:
    score = 0.0
    lower_name = name.lower()
    for token in ("host", "server", "node", "instance", "machine"):
        if token in lower_name:
            score += 2.0
    if "process" in lower_name:
        score -= 3.0
    sample = series.dropna()
    if sample.empty:
        return score - 5.0
    if sample.shape[0] > 5000:
        sample = sample.sample(5000, random_state=0)

    if pd.api.types.is_numeric_dtype(sample):
        score -= 2.0
    else:
        score += 1.5

    sample_str = sample.astype(str).str.strip()
    numeric_like = sample_str.str.match(r"^\d+(\.\d+)?$").mean()
    if numeric_like > 0.8:
        score -= 1.5

    unique_ratio = sample.nunique(dropna=True) / max(1, sample.shape[0])
    if unique_ratio < 0.01:
        score += 3.0
    elif unique_ratio < 0.1:
        score += 1.0
    elif unique_ratio > 0.5:
        score -= 1.5

    return score


def _choose_best_host_column(
    candidates: Iterable[str], df: pd.DataFrame
) -> str | None:
    candidates = list(candidates)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    scored = []
    for col in candidates:
        scored.append((_score_host_column(str(col), df[col]), col))
    scored.sort(reverse=True, key=lambda item: item[0])
    return scored[0][1]


def _score_datetime_column(name: str, series: pd.Series, tokens: Iterable[str]) -> float:
    score = 0.0
    lower_name = name.lower()
    if any(token in lower_name for token in tokens):
        score += 2.0

    sample = series.dropna()
    if sample.empty:
        return score - 5.0
    if sample.shape[0] > 2000:
        sample = sample.sample(2000, random_state=0)

    if pd.api.types.is_numeric_dtype(sample):
        max_val = float(sample.max()) if not sample.empty else 0.0
        if max_val < 1e8:
            score -= 5.0

    parsed = pd.to_datetime(sample, errors="coerce", utc=False)
    success = float(parsed.notna().mean())
    score += success * 5.0
    return score


def _choose_best_datetime_column(
    candidates: Iterable[str], df: pd.DataFrame, tokens: Iterable[str]
) -> str | None:
    candidates = list(candidates)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    scored = []
    for col in candidates:
        scored.append((_score_datetime_column(str(col), df[col], tokens), col))
    scored.sort(reverse=True, key=lambda item: item[0])
    best_score, best_col = scored[0]
    if best_score <= 0:
        return None
    return best_col


def _series_for_column(frame: pd.DataFrame, column: str) -> pd.Series:
    data = frame[column]
    if isinstance(data, pd.DataFrame):
        return data.iloc[:, 0]
    return data


def _infer_baseline_host_count(values: pd.Series) -> int | None:
    cleaned = values.dropna()
    if cleaned.empty:
        return None
    try:
        cleaned = cleaned.astype(int)
    except (TypeError, ValueError):
        cleaned = cleaned.astype(float).round().astype(int)
    counts = cleaned.value_counts()
    if counts.empty:
        return None
    max_count = int(counts.max())
    candidates = sorted(int(v) for v in counts[counts == max_count].index.tolist())
    return candidates[0] if candidates else None


def _bucket_floor(ts: pd.Series, bucket_size: str) -> pd.Series:
    if bucket_size == "hour":
        return ts.dt.floor("h")
    return ts.dt.floor("D")


def _max_concurrent_hosts(
    frame: pd.DataFrame,
    bucket_start: pd.Timestamp,
    bucket_end: pd.Timestamp,
) -> int:
    events: list[tuple[pd.Timestamp, int]] = []
    grouped = frame.groupby("__host_norm", sort=False)
    for _, host_frame in grouped:
        intervals = host_frame[["__start_ts", "__end_ts"]].sort_values("__start_ts")
        current_start: pd.Timestamp | None = None
        current_end: pd.Timestamp | None = None
        for start_ts, end_ts in intervals.itertuples(index=False, name=None):
            if start_ts is None or end_ts is None:
                continue
            start_ts = max(start_ts, bucket_start)
            end_ts = min(end_ts, bucket_end)
            if end_ts < start_ts:
                continue
            if current_start is None:
                current_start = start_ts
                current_end = end_ts
                continue
            if start_ts <= current_end:
                if end_ts > current_end:
                    current_end = end_ts
            else:
                events.append((current_start, 1))
                events.append((current_end, -1))
                current_start = start_ts
                current_end = end_ts
        if current_start is not None and current_end is not None:
            events.append((current_start, 1))
            events.append((current_end, -1))

    if not events:
        return 0

    events.sort(key=lambda item: (item[0], -item[1]))
    current = 0
    peak = 0
    for _, delta in events:
        current += delta
        peak = max(peak, current)
    return peak


def _month_key(day: date) -> str:
    return f"{day.year:04d}-{day.month:02d}"


def _build_calendar_days(start_day: date, end_day: date) -> list[date]:
    days: list[date] = []
    cursor = start_day
    while cursor <= end_day:
        days.append(cursor)
        cursor = cursor + timedelta(days=1)
    return days


def _calendar_close_window(
    daily: pd.DataFrame, close_start_day: int, close_end_day: int, mode: str
) -> tuple[set[date], list[dict[str, Any]], set[str]]:
    close_dates: set[date] = set()
    for day in daily["date"].tolist():
        if close_start_day <= close_end_day:
            is_close = close_start_day <= day.day <= close_end_day
        else:
            is_close = day.day >= close_start_day or day.day <= close_end_day
        if is_close:
            close_dates.add(day)
    close_windows = [
        {
            "mode": mode,
            "start_day": close_start_day,
            "end_day": close_end_day,
        }
    ]
    confident_months = {day.strftime("%Y-%m") for day in close_dates}
    return close_dates, close_windows, confident_months


def _infer_close_windows(
    daily: pd.DataFrame,
    min_days: int,
    max_days: int,
    lookahead_days: int,
    min_confidence: float,
    min_data_ratio: float,
) -> tuple[set[date], list[dict[str, Any]], set[str]]:
    if daily.empty:
        return set(), [], set()

    daily = daily.copy()
    daily["month"] = daily["date"].apply(_month_key)
    daily_map = {
        row["date"]: {
            "count": int(row["count"]),
            "median": float(row["median_ttc"]) if row["median_ttc"] is not None else None,
        }
        for row in daily.to_dict("records")
    }

    months = sorted(daily["month"].unique())
    close_windows: list[dict[str, Any]] = []
    close_dates: dict[date, str] = {}
    confident_months: set[str] = set()

    for month in months:
        year, month_num = [int(part) for part in month.split("-")]
        month_start = date(year, month_num, 1)
        if month_num == 12:
            next_month_start = date(year + 1, 1, 1)
        else:
            next_month_start = date(year, month_num + 1, 1)
        month_end = next_month_start - timedelta(days=1)

        month_days = _build_calendar_days(month_start, month_end)
        next_days = _build_calendar_days(
            next_month_start, next_month_start + timedelta(days=max(0, lookahead_days - 1))
        )
        extended_days = month_days + next_days

        counts = [daily_map.get(day, {}).get("count", 0) for day in month_days]
        medians = [daily_map.get(day, {}).get("median") for day in month_days]
        medians_clean = [val for val in medians if val is not None]
        count_mean = float(np.mean(counts)) if counts else 0.0
        count_std = float(np.std(counts)) if counts else 0.0
        median_mean = float(np.mean(medians_clean)) if medians_clean else 0.0
        median_std = float(np.std(medians_clean)) if medians_clean else 0.0

        def _z(val: float, mean: float, std: float) -> float:
            if std <= 0:
                return 0.0
            return (val - mean) / std

        pressures = []
        for day in extended_days:
            record = daily_map.get(day)
            count_val = record.get("count", 0) if record else 0
            median_val = record.get("median") if record else None
            z_count = _z(float(count_val), count_mean, count_std)
            z_median = 0.0
            if median_val is not None:
                z_median = _z(float(median_val), median_mean, median_std)
            pressures.append(z_count + z_median)

        best_score = None
        best_window = None
        second_score = None

        for length in range(min_days, max_days + 1):
            if length <= 0 or length > len(extended_days):
                continue
            for start_idx in range(len(month_days)):
                end_idx = start_idx + length
                if end_idx > len(extended_days):
                    continue
                score = float(sum(pressures[start_idx:end_idx]))
                if best_score is None or score > best_score:
                    second_score = best_score
                    best_score = score
                    best_window = (start_idx, end_idx)
                elif second_score is None or score > second_score:
                    second_score = score

        confidence = 0.0
        if best_score is not None and best_score > 0:
            if second_score is None:
                confidence = 1.0
            else:
                confidence = (best_score - second_score) / abs(best_score)

        if best_window is None or confidence < min_confidence:
            close_windows.append(
                {
                    "month": month,
                    "start": None,
                    "end": None,
                    "length": 0,
                    "confidence": float(confidence),
                    "mode": "infer",
                }
            )
            continue

        start_idx, end_idx = best_window
        window_days = extended_days[start_idx:end_idx]
        data_days = [day for day in window_days if daily_map.get(day, {}).get("count", 0) > 0]
        data_ratio = len(data_days) / max(1, len(window_days))
        if data_ratio < min_data_ratio:
            close_windows.append(
                {
                    "month": month,
                    "start": None,
                    "end": None,
                    "length": len(window_days),
                    "confidence": float(confidence),
                    "mode": "infer",
                    "reason": "insufficient_data_days",
                }
            )
            continue

        start_day = window_days[0]
        end_day = window_days[-1]
        close_windows.append(
            {
                "month": month,
                "start": start_day.isoformat(),
                "end": end_day.isoformat(),
                "length": len(window_days),
                "confidence": float(confidence),
                "mode": "infer",
            }
        )
        confident_months.add(month)

        for day in window_days:
            close_dates.setdefault(day, month)

    close_date_set = set(close_dates.keys())
    return close_date_set, close_windows, confident_months


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

        process_candidates = _candidate_columns(
            columns,
            role_by_name,
            {"process", "activity", "event", "step", "task", "action"},
            ["process", "activity", "event", "step", "task", "action", "job"],
            lower_names,
            used,
        )
        process_col = _choose_best_process_column(process_candidates, df)
        if process_col:
            used.add(process_col)

        host_candidates = _candidate_columns(
            columns,
            role_by_name,
            {"server", "host", "node", "instance"},
            ["server", "host", "node", "instance", "machine"],
            lower_names,
            used,
        )
        preferred_host = ctx.settings.get("host_column")
        if preferred_host and preferred_host in columns:
            host_col = preferred_host
        else:
            host_col = _choose_best_host_column(host_candidates, df)
        if host_col:
            used.add(host_col)

        start_candidates = _candidate_columns(
            columns,
            role_by_name,
            {"start_time", "start"},
            ["start", "begin"],
            lower_names,
            used,
        )
        preferred_start = ctx.settings.get("start_column")
        if preferred_start and preferred_start in columns:
            start_col = preferred_start
        else:
            start_col = _choose_best_datetime_column(
                start_candidates, df, ("start", "begin")
            )
        if start_col:
            used.add(start_col)

        end_candidates = _candidate_columns(
            columns,
            role_by_name,
            {"end_time", "end", "finish", "complete"},
            ["end", "finish", "complete", "stop"],
            lower_names,
            used,
        )
        preferred_end = ctx.settings.get("end_column")
        if preferred_end and preferred_end in columns:
            end_col = preferred_end
        else:
            end_col = _choose_best_datetime_column(
                end_candidates, df, ("end", "finish", "complete", "stop")
            )
        if end_col:
            used.add(end_col)

        queue_col = _pick_column(
            ctx.settings.get("queue_column"),
            columns,
            role_by_name,
            {"queue", "queued", "enqueue"},
            ["queue", "queued", "enqueue"],
            lower_names,
            used,
        )
        if queue_col:
            used.add(queue_col)

        eligible_col = _pick_column(
            ctx.settings.get("eligible_column"),
            columns,
            role_by_name,
            {"eligible", "ready", "available"},
            ["eligible", "ready", "available"],
            lower_names,
            used,
        )
        eligible_fallback = None
        if eligible_col:
            used.add(eligible_col)
        elif queue_col:
            # If no explicit eligible column is detected, reuse queue timestamp as eligible.
            eligible_col = queue_col
            eligible_fallback = "queue_column"

        baseline_host_setting = ctx.settings.get("baseline_host_count", None)
        try:
            baseline_host_setting_val = (
                int(baseline_host_setting) if baseline_host_setting is not None else None
            )
        except (TypeError, ValueError):
            baseline_host_setting_val = None
        if baseline_host_setting_val is not None and baseline_host_setting_val <= 0:
            baseline_host_setting_val = None
        added_hosts = int(ctx.settings.get("added_hosts", 1))

        summary: dict[str, Any] = {
            "process_column": process_col,
            "host_column": host_col,
            "start_column": start_col,
            "end_column": end_col,
            "queue_column": queue_col,
            "eligible_column": eligible_col,
            "eligible_column_fallback": eligible_fallback,
            "baseline_host_setting": baseline_host_setting_val,
            "added_hosts": added_hosts,
        }

        metric_types = ["ttc", "queue_to_end", "eligible_to_end"]
        host_metrics = ["concurrent", "unique"]

        def _emit_not_applicable(reason: str) -> PluginResult:
            findings = []
            for metric in host_metrics:
                for metric_type in metric_types:
                    findings.append(
                        {
                            "kind": "close_cycle_capacity_model",
                            "host_metric": metric,
                            "metric_type": metric_type,
                            "decision": "not_applicable",
                            "measurement_type": "not_applicable",
                            "reason": reason,
                            "columns": [
                                col
                                for col in [
                                    process_col,
                                    host_col,
                                    start_col,
                                    end_col,
                                    queue_col,
                                    eligible_col,
                                ]
                                if col
                            ],
                        }
                    )
            artifacts_dir = ctx.artifacts_dir("analysis_close_cycle_capacity_model")
            out_path = artifacts_dir / "results.json"
            write_json(out_path, {"summary": summary, "findings": findings})
            artifacts = [
                PluginArtifact(
                    path=str(out_path.relative_to(ctx.run_dir)),
                    type="json",
                    description="Close-cycle capacity model summary",
                )
            ]
            csv_path = artifacts_dir / "results.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                header = [
                    "host_metric",
                    "metric_type",
                    "decision",
                    "baseline_median_hours",
                    "modeled_median_hours",
                    "effect",
                    "target_reduction",
                    "tolerance",
                    "target_met",
                    "bucket_count",
                    "months",
                    "reason",
                ]
                handle.write(",".join(header) + "\n")
                for item in findings:
                    handle.write(
                        ",".join(
                            [
                                str(item.get("host_metric")),
                                str(item.get("metric_type")),
                                str(item.get("decision")),
                                str(item.get("baseline_median_hours")),
                                str(item.get("modeled_median_hours")),
                                str(item.get("effect")),
                                str(item.get("target_reduction")),
                                str(item.get("tolerance")),
                                str(item.get("target_met")),
                                str(item.get("bucket_count")),
                                str(item.get("months")),
                                str(item.get("reason")),
                            ]
                        )
                        + "\n"
                    )
            artifacts.append(
                PluginArtifact(
                    path=str(csv_path.relative_to(ctx.run_dir)),
                    type="csv",
                    description="Capacity model detail table",
                )
            )

            md_path = artifacts_dir / "results.md"
            lines = [
                "# Close-cycle capacity model",
                "",
                "Summary:",
                f"- close_window_mode: {summary.get('close_window_mode')}",
                f"- close_cycle_start_day: {summary.get('close_cycle_start_day')}",
                f"- close_cycle_end_day: {summary.get('close_cycle_end_day')}",
                "",
                "Modeled findings:",
                "_None_",
                "",
                "Not applicable:",
            ]
            for item in findings:
                lines.append(
                    f"- {item.get('host_metric')} {item.get('metric_type')}: {item.get('reason')}"
                )
            md_path.write_text("\n".join(lines), encoding="utf-8")
            artifacts.append(
                PluginArtifact(
                    path=str(md_path.relative_to(ctx.run_dir)),
                    type="markdown",
                    description="Capacity model summary",
                )
            )

            return PluginResult(
                "ok",
                reason,
                {"findings": len(findings)},
                findings,
                artifacts,
                None,
            )

        if not start_col or not end_col:
            return _emit_not_applicable("missing_start_end")
        if not host_col:
            return _emit_not_applicable("missing_host_column")

        selected_cols: list[str] = []
        for col in [process_col, host_col, start_col, end_col, queue_col, eligible_col]:
            if col and col not in selected_cols:
                selected_cols.append(col)
        work = df.loc[:, selected_cols].copy()

        work["__start_ts"] = pd.to_datetime(
            _series_for_column(work, start_col), errors="coerce", utc=False
        )
        work["__end_ts"] = pd.to_datetime(
            _series_for_column(work, end_col), errors="coerce", utc=False
        )
        work = work.loc[work["__start_ts"].notna() & work["__end_ts"].notna()].copy()
        if work.empty:
            return _emit_not_applicable("no_valid_timestamps")

        work = work.loc[work["__end_ts"] >= work["__start_ts"]].copy()
        if work.empty:
            return _emit_not_applicable("no_non_negative_durations")

        work["__service_sec"] = (
            work["__end_ts"] - work["__start_ts"]
        ).dt.total_seconds()
        work = work.loc[work["__service_sec"] > 0].copy()
        if work.empty:
            return _emit_not_applicable("no_positive_ttc")

        work["__host"] = _series_for_column(work, host_col).map(_normalize_text)
        work["__host_norm"] = work["__host"].str.lower()
        work = work.loc[~work["__host_norm"].isin(INVALID_STRINGS)].copy()
        if work.empty:
            return _emit_not_applicable("no_valid_host_values")

        if queue_col and queue_col in work.columns:
            work["__queue_ts"] = pd.to_datetime(
                _series_for_column(work, queue_col), errors="coerce", utc=False
            )
            work["__queue_wait_sec"] = (
                work["__start_ts"] - work["__queue_ts"]
            ).dt.total_seconds().clip(lower=0)
            work["__queue_to_end_sec"] = (
                work["__end_ts"] - work["__queue_ts"]
            ).dt.total_seconds().clip(lower=0)
        else:
            work["__queue_wait_sec"] = np.nan
            work["__queue_to_end_sec"] = np.nan

        if eligible_col and eligible_col in work.columns:
            work["__eligible_ts"] = pd.to_datetime(
                _series_for_column(work, eligible_col), errors="coerce", utc=False
            )
            work["__eligible_wait_sec"] = (
                work["__start_ts"] - work["__eligible_ts"]
            ).dt.total_seconds().clip(lower=0)
            work["__eligible_to_end_sec"] = (
                work["__end_ts"] - work["__eligible_ts"]
            ).dt.total_seconds().clip(lower=0)
        else:
            work["__eligible_wait_sec"] = np.nan
            work["__eligible_to_end_sec"] = np.nan

        close_mode = str(ctx.settings.get("close_window_mode", "infer") or "infer").lower()
        if close_mode == "calendar":
            close_mode = "override"
        close_start_day = int(ctx.settings.get("close_cycle_start_day", 20))
        close_end_day = int(ctx.settings.get("close_cycle_end_day", 5))
        min_close_days = int(ctx.settings.get("min_close_days", 5))
        max_close_days = int(ctx.settings.get("max_close_days", 20))
        lookahead_days = int(ctx.settings.get("lookahead_days", 7))
        min_close_confidence = float(ctx.settings.get("min_close_confidence", 0.1))
        min_close_data_ratio = float(ctx.settings.get("min_close_data_ratio", 0.5))

        work["__date"] = work["__start_ts"].dt.date
        daily = (
            work.groupby("__date")
            .agg(count=("__date", "size"), median_ttc=("__service_sec", "median"))
            .reset_index()
            .rename(columns={"__date": "date"})
        )

        close_dates: set[date] = set()
        close_windows: list[dict[str, Any]] = []
        confident_months: set[str] = set()
        fallback_used = False
        fallback_reason: str | None = None

        if close_mode == "override":
            close_dates, close_windows, confident_months = _calendar_close_window(
                daily, close_start_day, close_end_day, "override"
            )
        elif close_mode == "infer_or_default":
            close_dates, close_windows, confident_months = _infer_close_windows(
                daily,
                min_close_days,
                max_close_days,
                lookahead_days,
                min_close_confidence,
                min_close_data_ratio,
            )
            if not close_dates:
                fallback_used = True
                fallback_reason = "calendar_default"
                close_dates, close_windows, confident_months = _calendar_close_window(
                    daily, close_start_day, close_end_day, "fallback"
                )
        else:
            close_dates, close_windows, confident_months = _infer_close_windows(
                daily,
                min_close_days,
                max_close_days,
                lookahead_days,
                min_close_confidence,
                min_close_data_ratio,
            )

        summary.update(
            {
                "close_window_mode": close_mode,
                "close_cycle_start_day": close_start_day,
                "close_cycle_end_day": close_end_day,
                "min_close_days": min_close_days,
                "max_close_days": max_close_days,
                "lookahead_days": lookahead_days,
                "min_close_confidence": min_close_confidence,
                "min_close_data_ratio": min_close_data_ratio,
                "close_windows": close_windows,
                "close_window_fallback": fallback_used,
                "close_window_fallback_reason": fallback_reason,
            }
        )

        if not close_dates:
            return _emit_not_applicable("close_window_not_inferred")

        close_window_source = (
            "fallback"
            if fallback_used
            else ("override" if close_mode == "override" else "infer")
        )

        bucket_size = str(ctx.settings.get("bucket_size", "day") or "day").lower()
        min_bucket_rows = int(ctx.settings.get("min_bucket_rows", 50))

        work["__bucket_start"] = _bucket_floor(work["__start_ts"], bucket_size)
        if bucket_size == "hour":
            bucket_delta = timedelta(hours=1)
        else:
            bucket_delta = timedelta(days=1)

        buckets = []
        grouped = work.groupby("__bucket_start", sort=True)
        for bucket_start, frame in grouped:
            if pd.isna(bucket_start):
                continue
            bucket_start = pd.Timestamp(bucket_start)
            bucket_end = bucket_start + bucket_delta
            bucket_date = bucket_start.date()
            close_flag = bucket_date in close_dates
            row_count = int(frame.shape[0])
            if row_count < min_bucket_rows:
                continue
            median_service = float(frame["__service_sec"].median())
            median_queue_wait = (
                float(frame["__queue_wait_sec"].median())
                if frame["__queue_wait_sec"].notna().any()
                else None
            )
            median_eligible_wait = (
                float(frame["__eligible_wait_sec"].median())
                if frame["__eligible_wait_sec"].notna().any()
                else None
            )
            median_queue_to_end = (
                float(frame["__queue_to_end_sec"].median())
                if frame["__queue_to_end_sec"].notna().any()
                else None
            )
            median_eligible_to_end = (
                float(frame["__eligible_to_end_sec"].median())
                if frame["__eligible_to_end_sec"].notna().any()
                else None
            )
            host_unique = int(frame["__host_norm"].nunique())
            host_concurrent = _max_concurrent_hosts(frame, bucket_start, bucket_end)
            buckets.append(
                {
                    "bucket_start": bucket_start,
                    "bucket_end": bucket_end,
                    "bucket_date": bucket_date,
                    "month": _month_key(bucket_date),
                    "close": close_flag,
                    "rows": row_count,
                    "median_service": median_service,
                    "median_queue_wait": median_queue_wait,
                    "median_eligible_wait": median_eligible_wait,
                    "median_queue_to_end": median_queue_to_end,
                    "median_eligible_to_end": median_eligible_to_end,
                    "host_unique": host_unique,
                    "host_concurrent": host_concurrent,
                }
            )

        if not buckets:
            return _emit_not_applicable("no_buckets_after_filtering")

        bucket_df = pd.DataFrame(buckets)
        close_buckets = bucket_df.loc[bucket_df["close"]].copy()
        close_buckets = close_buckets.loc[
            close_buckets["month"].isin(confident_months)
        ].copy()
        if close_buckets.empty:
            return _emit_not_applicable("no_close_buckets")

        baseline_match_mode = str(
            ctx.settings.get("baseline_match_mode", "exact") or "exact"
        ).lower()
        min_buckets_per_group = int(ctx.settings.get("min_buckets_per_group", 5))
        min_months = int(ctx.settings.get("min_months", 1))
        target_reduction = float(ctx.settings.get("target_reduction", 0.30))
        tolerance = float(ctx.settings.get("tolerance", 0.05))
        max_examples = int(ctx.settings.get("max_examples", 25))

        if added_hosts <= 0:
            return _emit_not_applicable("invalid_added_hosts")

        findings = []

        def _within_tolerance(effect: float | None) -> bool | None:
            if effect is None:
                return None
            band_low = -(target_reduction + tolerance)
            band_high = -(target_reduction - tolerance)
            return band_low <= effect <= band_high

        for metric in host_metrics:
            metric_col = "host_concurrent" if metric == "concurrent" else "host_unique"
            baseline_host_count = baseline_host_setting_val
            baseline_host_source = "config"
            if baseline_host_count is None:
                baseline_host_count = _infer_baseline_host_count(close_buckets[metric_col])
                baseline_host_source = "inferred_mode"
            if not baseline_host_count or baseline_host_count <= 0:
                for metric_type in metric_types:
                    findings.append(
                        {
                            "kind": "close_cycle_capacity_model",
                            "host_metric": metric,
                            "metric_type": metric_type,
                            "decision": "not_applicable",
                            "measurement_type": "not_applicable",
                            "reason": "invalid_baseline_host_count",
                            "close_window_mode": close_mode,
                            "close_window_fallback": fallback_used,
                            "close_window_source": close_window_source,
                            "close_window_reason": fallback_reason,
                            "baseline_host_count": baseline_host_count,
                            "baseline_host_source": baseline_host_source,
                            "added_hosts": added_hosts,
                            "scale_factor": None,
                            "target_reduction": target_reduction,
                            "tolerance": tolerance,
                            "columns": [
                                col
                                for col in [
                                    process_col,
                                    host_col,
                                    start_col,
                                    end_col,
                                    queue_col,
                                    eligible_col,
                                ]
                                if col
                            ],
                        }
                    )
                continue

            scale_factor = (baseline_host_count + added_hosts) / float(baseline_host_count)
            assumption = (
                "Queue/eligible waits scale inversely with capacity; service time unchanged; "
                f"modeled factor {scale_factor:.3f}."
            )
            baseline_mode_effective = baseline_match_mode
            baseline_fallback = None
            if baseline_match_mode == "at_most":
                baseline_mask = close_buckets[metric_col] <= baseline_host_count
            else:
                baseline_mask = close_buckets[metric_col] == baseline_host_count

            baseline = close_buckets.loc[baseline_mask].copy()
            reasons = []
            if baseline.shape[0] < min_buckets_per_group:
                reasons.append("insufficient_buckets")

            months = sorted(baseline["month"].unique().tolist())
            if len(months) < min_months:
                reasons.append("insufficient_months")

            if reasons and baseline_match_mode == "exact":
                alt_mask = close_buckets[metric_col] <= baseline_host_count
                alt_baseline = close_buckets.loc[alt_mask].copy()
                alt_reasons = []
                if alt_baseline.shape[0] < min_buckets_per_group:
                    alt_reasons.append("insufficient_buckets")
                alt_months = sorted(alt_baseline["month"].unique().tolist())
                if len(alt_months) < min_months:
                    alt_reasons.append("insufficient_months")
                if not alt_reasons:
                    baseline = alt_baseline
                    reasons = []
                    baseline_mode_effective = "at_most"
                    baseline_fallback = "at_most"
                    months = alt_months
            if reasons and not close_buckets.empty:
                # Final fallback: use all close-cycle buckets when baseline host count not observed.
                baseline = close_buckets.copy()
                months = sorted(baseline["month"].unique().tolist())
                if len(months) >= min_months:
                    reasons = []
                    baseline_mode_effective = "all"
                    baseline_fallback = "all_close_buckets"

            if reasons:
                for metric_type in metric_types:
                    findings.append(
                        {
                            "kind": "close_cycle_capacity_model",
                            "host_metric": metric,
                            "metric_type": metric_type,
                            "decision": "not_applicable",
                            "measurement_type": "not_applicable",
                            "reason": ",".join(reasons),
                            "close_window_mode": close_mode,
                            "close_window_fallback": fallback_used,
                            "close_window_source": close_window_source,
                            "close_window_reason": fallback_reason,
                            "baseline_host_count": baseline_host_count,
                            "baseline_host_source": baseline_host_source,
                            "added_hosts": added_hosts,
                            "scale_factor": scale_factor,
                            "target_reduction": target_reduction,
                            "tolerance": tolerance,
                            "baseline_match_mode": baseline_mode_effective,
                            "baseline_match_fallback": baseline_fallback,
                            "columns": [
                                col
                                for col in [
                                    process_col,
                                    host_col,
                                    start_col,
                                    end_col,
                                    queue_col,
                                    eligible_col,
                                ]
                                if col
                            ],
                        }
                    )
                continue

            baseline_service = float(baseline["median_service"].median())
            queue_waits = baseline["median_queue_wait"].dropna()
            eligible_waits = baseline["median_eligible_wait"].dropna()

            baseline_queue_wait = (
                float(queue_waits.median()) if not queue_waits.empty else None
            )
            baseline_eligible_wait = (
                float(eligible_waits.median()) if not eligible_waits.empty else None
            )

            baseline_queue_to_end = None
            if baseline_queue_wait is not None:
                baseline_queue_to_end = baseline_queue_wait + baseline_service
            else:
                direct = baseline["median_queue_to_end"].dropna()
                if not direct.empty:
                    baseline_queue_to_end = float(direct.median())

            baseline_eligible_to_end = None
            if baseline_eligible_wait is not None:
                baseline_eligible_to_end = baseline_eligible_wait + baseline_service
            else:
                direct = baseline["median_eligible_to_end"].dropna()
                if not direct.empty:
                    baseline_eligible_to_end = float(direct.median())

            modeled_service = baseline_service
            modeled_queue_to_end = (
                baseline_queue_wait / scale_factor + baseline_service
                if baseline_queue_wait is not None
                else None
            )
            modeled_eligible_to_end = (
                baseline_eligible_wait / scale_factor + baseline_service
                if baseline_eligible_wait is not None
                else None
            )

            row_ids = work.index.tolist()[:max_examples]

            def _emit_metric(metric_type: str, baseline_val: float | None, modeled_val: float | None, reason_hint: str | None) -> None:
                if baseline_val is None or baseline_val <= 0:
                    findings.append(
                        {
                            "kind": "close_cycle_capacity_model",
                            "host_metric": metric,
                            "metric_type": metric_type,
                            "decision": "not_applicable",
                            "measurement_type": "not_applicable",
                            "reason": reason_hint or "missing_metric",
                            "close_window_mode": close_mode,
                            "close_window_fallback": fallback_used,
                            "close_window_source": close_window_source,
                            "close_window_reason": fallback_reason,
                            "baseline_host_count": baseline_host_count,
                            "added_hosts": added_hosts,
                            "scale_factor": scale_factor,
                            "target_reduction": target_reduction,
                            "tolerance": tolerance,
                            "bucket_count": int(baseline.shape[0]),
                            "months": months,
                            "columns": [
                                col
                                for col in [
                                    process_col,
                                    host_col,
                                    start_col,
                                    end_col,
                                    queue_col,
                                    eligible_col,
                                ]
                                if col
                            ],
                        }
                    )
                    return

                effect = None
                if modeled_val is not None and baseline_val > 0:
                    effect = (modeled_val / baseline_val) - 1.0

                findings.append(
                    {
                        "kind": "close_cycle_capacity_model",
                        "host_metric": metric,
                        "metric_type": metric_type,
                        "decision": "modeled",
                        "measurement_type": "modeled",
                        "reason": "fallback_calendar_default" if fallback_used else "ok",
                        "close_window_mode": close_mode,
                        "close_window_fallback": fallback_used,
                        "close_window_source": close_window_source,
                        "close_window_reason": fallback_reason,
                        "assumptions": [assumption],
                        "scope": {
                            "host_metric": metric,
                            "metric_type": metric_type,
                            "close_window_mode": close_mode,
                            "close_window_source": close_window_source,
                        },
                        "baseline_median_sec": baseline_val,
                        "modeled_median_sec": modeled_val,
                        "effect": effect,
                        "target_reduction": target_reduction,
                        "tolerance": tolerance,
                        "baseline_match_mode": baseline_mode_effective,
                        "baseline_match_fallback": baseline_fallback,
                        "target_met": _within_tolerance(effect),
                        "baseline_host_count": baseline_host_count,
                        "baseline_host_source": baseline_host_source,
                        "added_hosts": added_hosts,
                        "scale_factor": scale_factor,
                        "host_count_baseline": baseline_host_count,
                        "host_count_modeled": (
                            baseline_host_count + added_hosts
                            if baseline_host_count is not None
                            else None
                        ),
                        "bucket_count": int(baseline.shape[0]),
                        "months": months,
                        "row_ids": [int(i) for i in row_ids],
                        "columns": [
                            col
                            for col in [
                                process_col,
                                host_col,
                                start_col,
                                end_col,
                                queue_col,
                                eligible_col,
                            ]
                            if col
                        ],
                    }
                )

            _emit_metric("ttc", baseline_service, modeled_service, None)
            _emit_metric(
                "queue_to_end",
                baseline_queue_to_end,
                modeled_queue_to_end,
                "missing_queue_column",
            )
            _emit_metric(
                "eligible_to_end",
                baseline_eligible_to_end,
                modeled_eligible_to_end,
                "missing_eligible_column",
            )

        artifacts_dir = ctx.artifacts_dir("analysis_close_cycle_capacity_model")
        out_path = artifacts_dir / "results.json"
        write_json(out_path, {"summary": summary, "findings": findings})
        artifacts = [
            PluginArtifact(
                path=str(out_path.relative_to(ctx.run_dir)),
                type="json",
                description="Close-cycle capacity model summary",
            )
        ]

        csv_path = artifacts_dir / "results.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            header = [
                "host_metric",
                "metric_type",
                "decision",
                "baseline_median_hours",
                "modeled_median_hours",
                "effect",
                "target_reduction",
                "tolerance",
                "target_met",
                "bucket_count",
                "months",
                "reason",
            ]
            handle.write(",".join(header) + "\n")
            for item in findings:
                baseline_hours = (
                    float(item.get("baseline_median_sec")) / 3600.0
                    if item.get("baseline_median_sec") is not None
                    else None
                )
                modeled_hours = (
                    float(item.get("modeled_median_sec")) / 3600.0
                    if item.get("modeled_median_sec") is not None
                    else None
                )
                handle.write(
                    ",".join(
                        [
                            str(item.get("host_metric")),
                            str(item.get("metric_type")),
                            str(item.get("decision")),
                            str(baseline_hours),
                            str(modeled_hours),
                            str(item.get("effect")),
                            str(item.get("target_reduction")),
                            str(item.get("tolerance")),
                            str(item.get("target_met")),
                            str(item.get("bucket_count")),
                            str(item.get("months")),
                            str(item.get("reason")),
                        ]
                    )
                    + "\n"
                )
        artifacts.append(
            PluginArtifact(
                path=str(csv_path.relative_to(ctx.run_dir)),
                type="csv",
                description="Capacity model detail table",
            )
        )

        md_path = artifacts_dir / "results.md"
        lines = [
            "# Close-cycle capacity model",
            "",
            "Summary:",
            f"- close_window_mode: {summary.get('close_window_mode')}",
            f"- close_cycle_start_day: {summary.get('close_cycle_start_day')}",
            f"- close_cycle_end_day: {summary.get('close_cycle_end_day')}",
            f"- baseline_host_count: {baseline_host_count}",
            f"- added_hosts: {added_hosts}",
            f"- scale_factor: {scale_factor:.3f}",
            f"- target_reduction: {target_reduction}",
            f"- tolerance: {tolerance}",
            "",
            "Modeled findings:",
        ]

        modeled_rows = [
            item for item in findings if item.get("decision") == "modeled"
        ]
        if modeled_rows:
            lines.append(
                "| host_metric | metric_type | baseline_hours | modeled_hours | effect | target_met | buckets |"
            )
            lines.append("| --- | --- | --- | --- | --- | --- | --- |")
            for item in modeled_rows:
                baseline_hours = (
                    float(item.get("baseline_median_sec")) / 3600.0
                    if item.get("baseline_median_sec") is not None
                    else 0.0
                )
                modeled_hours = (
                    float(item.get("modeled_median_sec")) / 3600.0
                    if item.get("modeled_median_sec") is not None
                    else 0.0
                )
                effect = item.get("effect")
                effect_str = f"{effect:.4f}" if effect is not None else ""
                lines.append(
                    "| {host_metric} | {metric_type} | {baseline_hours:.2f} | {modeled_hours:.2f} | {effect} | {target_met} | {bucket_count} |".format(
                        host_metric=item.get("host_metric"),
                        metric_type=item.get("metric_type"),
                        baseline_hours=baseline_hours,
                        modeled_hours=modeled_hours,
                        effect=effect_str,
                        target_met=item.get("target_met"),
                        bucket_count=item.get("bucket_count"),
                    )
                )
        else:
            lines.append("_None_")

        lines.extend(["", "Not applicable:"])
        for item in findings:
            if item.get("decision") != "modeled":
                lines.append(
                    f"- {item.get('host_metric')} {item.get('metric_type')}: {item.get('reason')}"
                )

        md_path.write_text("\n".join(lines), encoding="utf-8")
        artifacts.append(
            PluginArtifact(
                path=str(md_path.relative_to(ctx.run_dir)),
                type="markdown",
                description="Capacity model summary",
            )
        )

        metrics = {
            "findings": len(findings),
            "modeled_findings": len(modeled_rows),
        }
        return PluginResult(
            "ok",
            "Computed close-cycle capacity model",
            metrics,
            findings,
            artifacts,
            None,
        )
