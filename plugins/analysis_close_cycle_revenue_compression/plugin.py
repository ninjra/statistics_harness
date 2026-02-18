from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Iterable
import math
import re

import numpy as np
import pandas as pd

from statistic_harness.core.close_cycle import load_preferred_close_cycle_windows
from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json


def _dates_from_windows(
    windows: list[Any], start_attr: str, end_attr: str
) -> set[date]:
    dates: set[date] = set()
    if not windows:
        return dates
    for window in windows:
        start = getattr(window, start_attr, None)
        end = getattr(window, end_attr, None)
        if start is None or end is None:
            continue
        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
        for day in pd.date_range(start_ts, end_ts, freq="D"):
            dates.add(day.date())
    return dates


INVALID_STRINGS = {"", "nan", "none", "null"}


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


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


def _series_is_datetime(series: pd.Series, min_success: float = 0.8) -> bool:
    sample = series.dropna()
    if sample.empty:
        return False
    if sample.shape[0] > 2000:
        sample = sample.sample(2000, random_state=0)
    sample_str = sample.astype(str)
    digit_ratio = float(sample_str.str.contains(r"\d").mean())
    if digit_ratio < 0.5:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", utc=False)
    return float(parsed.notna().mean()) >= min_success


def _score_selector_column(series: pd.Series) -> float:
    sample = series.dropna()
    if sample.empty:
        return -10.0
    if sample.shape[0] > 5000:
        sample = sample.sample(5000, random_state=0)
    non_null_ratio = float(series.notna().mean())
    nunique = int(sample.nunique(dropna=True))
    if nunique <= 0:
        return -10.0
    sample_str = sample.astype(str).str.strip()
    code_like = float(sample_str.str.match(r"^[A-Za-z0-9_-]{2,12}$").mean())
    cardinality_score = 1.0 / (1.0 + math.log1p(nunique))
    return non_null_ratio * 0.5 + cardinality_score * 0.4 + code_like * 0.3


def _build_close_position(close_dates: set[date]) -> dict[date, float]:
    close_by_month: dict[str, list[date]] = {}
    for day in close_dates:
        close_by_month.setdefault(day.strftime("%Y-%m"), []).append(day)
    position: dict[date, float] = {}
    for month, days in close_by_month.items():
        days_sorted = sorted(days)
        if len(days_sorted) == 1:
            position[days_sorted[0]] = 1.0
            continue
        denom = float(len(days_sorted) - 1)
        for idx, day in enumerate(days_sorted):
            position[day] = idx / denom
    return position


def _select_composite_revenue(
    work: pd.DataFrame,
    close_dates: set[date],
    exclude: set[str],
    settings: dict[str, Any],
) -> tuple[pd.Series, dict[str, Any], pd.Series | None]:
    selector_min_coverage = float(settings.get("selector_min_coverage", 0.2))
    selector_min_cardinality = int(settings.get("selector_min_cardinality", 2))
    selector_max_cardinality = int(settings.get("selector_max_cardinality", 200))
    selector_top_columns = int(settings.get("selector_top_columns", 3))
    selector_max_composite = int(settings.get("selector_max_composite", 5000))
    selector_min_close_rows = int(settings.get("selector_min_close_rows", 100))
    selector_lift_threshold = float(settings.get("selector_lift_threshold", 1.2))
    selector_late_threshold = float(settings.get("selector_late_threshold", 0.7))
    selector_min_late_fraction = float(settings.get("selector_min_late_fraction", 0.2))
    selector_min_close_share = float(settings.get("selector_min_close_share", 0.8))
    selector_top_keys = int(settings.get("selector_top_keys", 10))

    candidates: list[tuple[str, float]] = []
    for col in work.columns:
        if col in exclude:
            continue
        series = work[col]
        if series.isna().all():
            continue
        if _series_is_datetime(series):
            continue
        non_null_ratio = float(series.notna().mean())
        if non_null_ratio < selector_min_coverage:
            continue
        nunique = int(series.nunique(dropna=True))
        if nunique < selector_min_cardinality or nunique > selector_max_cardinality:
            continue
        if pd.api.types.is_numeric_dtype(series):
            sample = series.dropna()
            if sample.shape[0] > 2000:
                sample = sample.sample(2000, random_state=0)
            if (sample % 1 != 0).any():
                continue
        candidates.append((col, _score_selector_column(series)))

    relaxed = False
    if not candidates:
        relaxed = True
        relaxed_max_cardinality = selector_max_cardinality * 5
        for col in work.columns:
            if col in exclude:
                continue
            if str(col).startswith("__"):
                continue
            series = work[col]
            if series.isna().all():
                continue
            non_null_ratio = float(series.notna().mean())
            if non_null_ratio <= 0:
                continue
            nunique = int(series.nunique(dropna=True))
            if nunique < selector_min_cardinality or nunique > relaxed_max_cardinality:
                continue
            candidates.append((col, _score_selector_column(series)))

    candidates.sort(key=lambda item: item[1], reverse=True)
    selected_cols = [col for col, _score in candidates[:selector_top_columns]]

    selector_info: dict[str, Any] = {
        "strategy": "composite",
        "selector_columns": selected_cols,
        "candidate_columns": [
            {"column": col, "score": round(score, 4)} for col, score in candidates[:20]
        ],
        "relaxed": relaxed,
    }

    if not selected_cols:
        return work["__start_ts"].isna(), selector_info, None

    def _make_key(cols: list[str]) -> pd.Series:
        normalized = work[cols].copy()
        for col in cols:
            normalized[col] = normalized[col].map(_normalize_text)
        key = normalized.agg("|".join, axis=1)
        empty_mask = key.str.replace("|", "", regex=False).str.strip().eq("")
        key = key.mask(empty_mask)
        return key

    key_series = _make_key(selected_cols)
    while key_series.nunique(dropna=True) > selector_max_composite and len(selected_cols) > 1:
        selected_cols = selected_cols[:-1]
        key_series = _make_key(selected_cols)

    selector_info["selector_columns"] = selected_cols
    selector_info["selector_key_count"] = int(key_series.nunique(dropna=True))

    close_mask = work["__start_ts"].dt.date.isin(close_dates)
    if not close_mask.any():
        return work["__start_ts"].isna(), selector_info, key_series

    overall_close_ratio = float(close_mask.mean())
    position = _build_close_position(close_dates)
    close_pos = work["__start_ts"].dt.date.map(position).fillna(0.0)
    late_mask = close_mask & (close_pos >= selector_late_threshold)

    total_counts = key_series.value_counts()
    close_counts = key_series[close_mask].value_counts()
    late_counts = key_series[late_mask].value_counts()

    key_rows = []
    for key, close_cnt in close_counts.items():
        total_cnt = int(total_counts.get(key, 0))
        if total_cnt == 0:
            continue
        close_share = close_cnt / total_cnt
        lift = close_share / overall_close_ratio if overall_close_ratio > 0 else 0.0
        late_frac = (
            float(late_counts.get(key, 0)) / float(close_cnt)
            if close_cnt > 0
            else 0.0
        )
        score = math.log1p(close_cnt) * lift * (1.0 + late_frac)
        key_rows.append(
            {
                "key": key,
                "close_rows": int(close_cnt),
                "total_rows": total_cnt,
                "close_share": float(close_share),
                "close_lift": float(lift),
                "late_fraction": float(late_frac),
                "score": float(score),
            }
        )

    key_rows.sort(key=lambda item: item["score"], reverse=True)
    selector_info["selector_keys"] = key_rows[:selector_top_keys]

    selected_keys = [
        row["key"]
        for row in key_rows
        if row["close_rows"] >= selector_min_close_rows
        and row["close_share"] >= selector_min_close_share
        and row["close_lift"] >= selector_lift_threshold
        and row["late_fraction"] >= selector_min_late_fraction
    ]

    selector_info["selector_thresholds"] = {
        "min_close_rows": selector_min_close_rows,
        "min_close_share": selector_min_close_share,
        "min_close_lift": selector_lift_threshold,
        "min_late_fraction": selector_min_late_fraction,
        "late_threshold": selector_late_threshold,
        "max_composite": selector_max_composite,
    }

    if not selected_keys:
        return work["__start_ts"].isna(), selector_info, key_series

    mask = key_series.isin(selected_keys)
    return mask, selector_info, key_series


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


def _match_revenue(process_values: pd.Series, names: list[str], patterns: list[str]) -> pd.Series:
    if process_values.empty:
        return pd.Series([], dtype=bool)
    series = process_values.fillna("").astype(str)
    normalized = series.str.strip().str.lower()

    if names:
        name_set = {name.lower() for name in names}
        return normalized.isin(name_set)

    if not patterns:
        return pd.Series([False] * len(series), index=series.index)

    mask = pd.Series([False] * len(series), index=series.index)
    for raw in patterns:
        if not raw:
            continue
        pattern = raw.strip()
        if pattern.lower().startswith("re:"):
            regex = pattern[3:]
            try:
                mask = mask | normalized.str.contains(regex, regex=True, na=False)
            except re.error:
                continue
        else:
            mask = mask | normalized.str.contains(re.escape(pattern.lower()), regex=True, na=False)
    return mask


def _modeled_span(end_sec: np.ndarray, wait_sec: np.ndarray, scale: float) -> float:
    if scale <= 0:
        return float("inf")
    adj_end = end_sec - wait_sec * (1.0 - 1.0 / scale)
    return float(adj_end.max())


def _required_scale(
    end_sec: np.ndarray,
    wait_sec: np.ndarray,
    target_sec: float,
    max_scale: float,
) -> tuple[float | None, float, bool]:
    baseline_span = _modeled_span(end_sec, wait_sec, 1.0)
    if baseline_span <= target_sec:
        return 1.0, baseline_span, True

    high = 1.0
    while high < max_scale and _modeled_span(end_sec, wait_sec, high) > target_sec:
        high *= 2.0
    if high > max_scale:
        high = max_scale

    if _modeled_span(end_sec, wait_sec, high) > target_sec:
        return None, baseline_span, False

    low = 1.0
    for _ in range(40):
        mid = (low + high) / 2.0
        if _modeled_span(end_sec, wait_sec, mid) <= target_sec:
            high = mid
        else:
            low = mid
    return high, baseline_span, True


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
        process_col = ctx.settings.get("process_column")
        if not process_col or process_col not in columns:
            process_col = _choose_best_process_column(process_candidates, df)
        if process_col:
            used.add(process_col)

        start_candidates = _candidate_columns(
            columns,
            role_by_name,
            {"start_time", "start"},
            ["start", "begin"],
            lower_names,
            used,
        )
        start_col = ctx.settings.get("start_column")
        if not start_col or start_col not in columns:
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
        end_col = ctx.settings.get("end_column")
        if not end_col or end_col not in columns:
            end_col = _choose_best_datetime_column(
                end_candidates, df, ("end", "finish", "complete", "stop")
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
        if eligible_col:
            used.add(eligible_col)

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

        summary: dict[str, Any] = {
            "process_column": process_col,
            "start_column": start_col,
            "end_column": end_col,
            "duration_column": duration_col,
            "queue_column": queue_col,
            "eligible_column": eligible_col,
            "host_column": host_col,
        }

        def _emit_not_applicable(reason: str) -> PluginResult:
            findings = [
                {
                    "kind": "close_cycle_revenue_compression",
                    "decision": "not_applicable",
                    "measurement_type": "not_applicable",
                    "reason": reason,
                    "columns": [
                        col
                        for col in [
                            process_col,
                            start_col,
                            end_col,
                            duration_col,
                            queue_col,
                            eligible_col,
                            host_col,
                        ]
                        if col
                    ],
                }
            ]
            artifacts_dir = ctx.artifacts_dir("analysis_close_cycle_revenue_compression")
            out_path = artifacts_dir / "results.json"
            write_json(out_path, {"summary": summary, "findings": findings})
            artifacts = [
                PluginArtifact(
                    path=str(out_path.relative_to(ctx.run_dir)),
                    type="json",
                    description="Revenue compression summary",
                )
            ]
            csv_path = artifacts_dir / "results.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                header = [
                    "decision",
                    "reason",
                ]
                handle.write(",".join(header) + "\n")
                for item in findings:
                    handle.write(",".join([str(item.get("decision")), str(item.get("reason"))]) + "\n")
            artifacts.append(
                PluginArtifact(
                    path=str(csv_path.relative_to(ctx.run_dir)),
                    type="csv",
                    description="Revenue compression detail table",
                )
            )

            md_path = artifacts_dir / "results.md"
            lines = [
                "# Close-cycle revenue compression",
                "",
                f"Not applicable: {reason}",
            ]
            selector = summary.get("revenue_selector")
            if isinstance(selector, dict) and selector:
                lines.extend(["", "Selector:", f"- strategy: {summary.get('revenue_selector_strategy')}"])
                selector_cols = selector.get("selector_columns") or []
                if selector_cols:
                    lines.append(f"- columns: {', '.join(selector_cols)}")
                key_count = selector.get("selector_key_count")
                if key_count is not None:
                    lines.append(f"- key_count: {key_count}")
            md_path.write_text("\n".join(lines), encoding="utf-8")
            artifacts.append(
                PluginArtifact(
                    path=str(md_path.relative_to(ctx.run_dir)),
                    type="markdown",
                    description="Revenue compression summary",
                )
            )

            return PluginResult("ok", reason, {"findings": 1}, findings, artifacts, None)

        if not process_col:
            return _emit_not_applicable("missing_process_column")
        if not start_col:
            return _emit_not_applicable("missing_start_column")
        if not end_col and not duration_col:
            return _emit_not_applicable("missing_end_or_duration")

        selected_cols = [
            col
            for col in [
                process_col,
                start_col,
                end_col,
                duration_col,
                queue_col,
                eligible_col,
                host_col,
            ]
            if col
        ]
        work = df.loc[:, selected_cols].copy()

        work["__start_ts"] = pd.to_datetime(work[start_col], errors="coerce", utc=False)
        if end_col and end_col in work.columns:
            work["__end_ts"] = pd.to_datetime(work[end_col], errors="coerce", utc=False)
        else:
            work["__end_ts"] = pd.NaT

        if duration_col and duration_col in work.columns:
            duration = pd.to_numeric(work[duration_col], errors="coerce")
            if duration.isna().all():
                duration = pd.to_timedelta(work[duration_col], errors="coerce").dt.total_seconds()
            work["__duration_sec"] = duration
        else:
            work["__duration_sec"] = np.nan

        if end_col and work["__end_ts"].notna().any():
            pass
        elif work["__duration_sec"].notna().any():
            work["__end_ts"] = work["__start_ts"] + pd.to_timedelta(
                work["__duration_sec"], unit="s"
            )
        else:
            return _emit_not_applicable("missing_end_timestamp")

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

        close_mode = str(ctx.settings.get("close_window_mode", "infer_or_default") or "infer_or_default").lower()
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

        close_dates_default = close_dates
        dynamic_windows, dynamic_source_plugin = load_preferred_close_cycle_windows(
            ctx.run_dir
        )
        close_dates_dynamic = _dates_from_windows(
            dynamic_windows, "dynamic_start", "dynamic_end"
        )
        dynamic_available = bool(close_dates_dynamic)
        close_rows_default = int(work["__date"].isin(close_dates_default).sum())
        close_rows_dynamic = int(work["__date"].isin(close_dates_dynamic).sum())
        if dynamic_available:
            close_dates = close_dates_dynamic

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
                "close_cycle_dynamic_available": dynamic_available,
                "close_cycle_dynamic_source_plugin": dynamic_source_plugin,
                "close_cycle_dynamic_months": len(dynamic_windows),
                "close_cycle_rows_default": close_rows_default,
                "close_cycle_rows_dynamic": close_rows_dynamic,
            }
        )

        if not close_dates:
            return _emit_not_applicable("close_window_not_inferred")

        close_window_source = (
            "fallback" if fallback_used else ("override" if close_mode == "override" else "infer")
        )
        if dynamic_available:
            close_window_source = (
                f"dynamic_{dynamic_source_plugin}"
                if isinstance(dynamic_source_plugin, str) and dynamic_source_plugin
                else "dynamic_resolver"
            )

        selector_strategy = str(
            ctx.settings.get("revenue_selector_strategy", "auto") or "auto"
        ).lower()
        if selector_strategy not in {"auto", "value_hints", "composite"}:
            selector_strategy = "auto"

        revenue_names = _normalize_list(ctx.settings.get("revenue_process_names"))
        revenue_patterns = _normalize_list(ctx.settings.get("revenue_process_patterns"))
        selector_info: dict[str, Any] = {}
        revenue_mask: pd.Series | None = None
        strategy_used: str | None = None

        if selector_strategy in {"auto", "value_hints"} and (revenue_names or revenue_patterns):
            if process_col and process_col in work.columns:
                work["__process"] = work[process_col].map(_normalize_text)
                work["__process_norm"] = work["__process"].str.lower()
                work = work.loc[~work["__process_norm"].isin(INVALID_STRINGS)].copy()
                if work.empty:
                    return _emit_not_applicable("no_valid_process_values")

                revenue_mask = _match_revenue(
                    work["__process_norm"], revenue_names, revenue_patterns
                )
                selector_info = {
                    "strategy": "value_hints",
                    "process_column": process_col,
                    "revenue_process_names": revenue_names,
                    "revenue_process_patterns": revenue_patterns,
                }
                strategy_used = "value_hints"
            elif selector_strategy == "value_hints":
                return _emit_not_applicable("missing_process_column_for_hints")

        if (revenue_mask is None or not revenue_mask.any()) and selector_strategy in {
            "auto",
            "composite",
        }:
            exclude_cols = {
                col
                for col in [
                    start_col,
                    end_col,
                    duration_col,
                    queue_col,
                    eligible_col,
                    host_col,
                ]
                if col
            }
            composite_mask, composite_info, key_series = _select_composite_revenue(
                work, close_dates, exclude_cols, ctx.settings
            )
            selector_info = composite_info
            strategy_used = "composite"
            if key_series is not None:
                work["__process"] = key_series
                work["__process_norm"] = work["__process"].astype(str).str.lower()
            revenue_mask = composite_mask

        if revenue_mask is None or not revenue_mask.any():
            summary.update(
                {
                    "revenue_selector_strategy": strategy_used,
                    "revenue_selector": selector_info,
                }
            )
            return _emit_not_applicable("no_revenue_process_match")

        revenue_rows = work.loc[revenue_mask].copy()
        summary.update(
            {
                "revenue_selector_strategy": strategy_used,
                "revenue_selector": selector_info,
            }
        )

        revenue_rows = revenue_rows.loc[
            revenue_rows["__start_ts"].dt.date.isin(close_dates)
        ].copy()
        if revenue_rows.empty:
            return _emit_not_applicable("no_revenue_rows_in_close_window")

        if queue_col and queue_col in revenue_rows.columns:
            revenue_rows["__queue_ts"] = pd.to_datetime(
                revenue_rows[queue_col], errors="coerce", utc=False
            )
            revenue_rows["__queue_wait_sec"] = (
                revenue_rows["__start_ts"] - revenue_rows["__queue_ts"]
            ).dt.total_seconds().clip(lower=0)
        else:
            revenue_rows["__queue_ts"] = pd.NaT
            revenue_rows["__queue_wait_sec"] = np.nan

        if eligible_col and eligible_col in revenue_rows.columns:
            revenue_rows["__eligible_ts"] = pd.to_datetime(
                revenue_rows[eligible_col], errors="coerce", utc=False
            )
            revenue_rows["__eligible_wait_sec"] = (
                revenue_rows["__start_ts"] - revenue_rows["__eligible_ts"]
            ).dt.total_seconds().clip(lower=0)
        else:
            revenue_rows["__eligible_ts"] = pd.NaT
            revenue_rows["__eligible_wait_sec"] = np.nan

        basis = "duration"
        anchor_basis = "start"
        wait_series = revenue_rows["__service_sec"]
        anchor_series = revenue_rows["__start_ts"]
        assumption = "Full end-to-end duration scales inversely with capacity."

        max_wait_days = float(ctx.settings.get("max_wait_days", 30.0))
        max_wait_sec = max_wait_days * 86400.0

        eligible_wait = revenue_rows["__eligible_wait_sec"]
        queue_wait = revenue_rows["__queue_wait_sec"]
        eligible_median = float(eligible_wait.median()) if eligible_wait.notna().any() else None
        queue_median = float(queue_wait.median()) if queue_wait.notna().any() else None

        if eligible_median is not None and eligible_median <= max_wait_sec:
            basis = "eligible_wait"
            anchor_basis = "eligible"
            wait_series = eligible_wait
            anchor_series = revenue_rows["__eligible_ts"]
            assumption = "Eligible wait scales inversely with capacity; service time unchanged."
        elif queue_median is not None and queue_median <= max_wait_sec:
            basis = "queue_wait"
            anchor_basis = "queue"
            wait_series = queue_wait
            anchor_series = revenue_rows["__queue_ts"]
            assumption = "Queue wait scales inversely with capacity; service time unchanged."

        revenue_rows = revenue_rows.loc[anchor_series.notna()].copy()
        if revenue_rows.empty:
            return _emit_not_applicable("missing_anchor_timestamps")

        target_days = float(ctx.settings.get("target_days", 7.0))
        target_sec = target_days * 86400.0
        max_scale = float(ctx.settings.get("max_scale", 10.0))
        min_month_rows = int(ctx.settings.get("min_month_rows", 10))
        max_months_output = int(ctx.settings.get("max_months_output", 24))

        host_count = None
        if host_col and host_col in revenue_rows.columns:
            host_series = revenue_rows[host_col].map(_normalize_text).str.lower()
            host_series = host_series.loc[~host_series.isin(INVALID_STRINGS)]
            if not host_series.empty:
                host_count = int(host_series.nunique())

        if close_start_day > close_end_day:
            def _close_cycle_month(ts: pd.Timestamp) -> str:
                if ts.day >= close_start_day:
                    return f"{ts.year:04d}-{ts.month:02d}"
                prev = (ts.replace(day=1) - pd.Timedelta(days=1))
                return f"{prev.year:04d}-{prev.month:02d}"

            revenue_rows["__month"] = revenue_rows["__start_ts"].apply(_close_cycle_month)
        else:
            revenue_rows["__month"] = revenue_rows["__start_ts"].dt.to_period("M").astype(str)
        month_stats: list[dict[str, Any]] = []
        for month, frame in revenue_rows.groupby("__month"):
            if frame.empty:
                continue
            month_rows = int(frame.shape[0])
            reason = "ok"
            decision = "modeled"
            required_scale = None
            modeled_span = None
            baseline_span = None
            achievable = True
            if month_rows < min_month_rows:
                decision = "not_applicable"
                reason = "insufficient_rows"
            else:
                anchor_min = anchor_series.loc[frame.index].min()
                end_sec = (
                    frame["__end_ts"] - anchor_min
                ).dt.total_seconds().to_numpy(dtype=float)
                wait_sec = wait_series.loc[frame.index].fillna(0.0).to_numpy(dtype=float)
                required_scale, baseline_span, achievable = _required_scale(
                    end_sec, wait_sec, target_sec, max_scale
                )
                if required_scale is None and not achievable:
                    decision = "not_applicable"
                    reason = "exceeds_max_scale"
                elif required_scale == 1.0:
                    reason = "already_within_target"
                if required_scale is not None:
                    modeled_span = _modeled_span(end_sec, wait_sec, required_scale)
            month_stats.append(
                {
                    "month": month,
                    "rows": month_rows,
                    "decision": decision,
                    "reason": reason,
                    "baseline_span_days": baseline_span / 86400.0 if baseline_span is not None else None,
                    "modeled_span_days": modeled_span / 86400.0 if modeled_span is not None else None,
                    "required_scale_factor": required_scale,
                }
            )

        valid_scales = [m for m in month_stats if m.get("required_scale_factor") is not None]
        if valid_scales:
            worst_month = max(valid_scales, key=lambda m: m.get("required_scale_factor") or 0.0)
            scales = [m["required_scale_factor"] for m in valid_scales if m.get("required_scale_factor") is not None]
            scales_sorted = sorted(scales)
            scale_median = float(np.median(scales_sorted))
            scale_p90 = float(np.quantile(scales_sorted, 0.9))
            baseline_spans = [m["baseline_span_days"] for m in valid_scales if m.get("baseline_span_days") is not None]
            baseline_span_median = float(np.median(baseline_spans)) if baseline_spans else None
            decision = "modeled"
            reason = "fallback_calendar_default" if fallback_used else "ok"
        else:
            worst_month = None
            scale_median = None
            scale_p90 = None
            baseline_span_median = None
            decision = "not_applicable"
            reason = "no_eligible_months"

        selector_columns = []
        selector_info = summary.get("revenue_selector")
        if isinstance(selector_info, dict):
            selector_columns = selector_info.get("selector_columns") or []

        columns_used: list[str] = []
        for col in [
            process_col,
            start_col,
            end_col,
            duration_col,
            queue_col,
            eligible_col,
            host_col,
            *selector_columns,
        ]:
            if col and col not in columns_used:
                columns_used.append(col)

        baseline_value = baseline_span_median
        modeled_value = target_days if decision == "modeled" else None
        delta_value = (
            (modeled_value - baseline_value)
            if isinstance(modeled_value, (int, float))
            and isinstance(baseline_value, (int, float))
            else None
        )
        modeled_host_count = (
            int(round(host_count * scale_median))
            if host_count is not None and scale_median
            else None
        )
        finding = {
            "kind": "close_cycle_revenue_compression",
            "decision": decision,
            "measurement_type": "modeled" if decision == "modeled" else "not_applicable",
            "reason": reason,
            "process_label": revenue_rows["__process"].value_counts().index[0],
            "process_matches": sorted(revenue_rows["__process_norm"].unique().tolist())[:10],
            "basis": basis,
            "anchor_basis": anchor_basis,
            "assumptions": [assumption],
            "scope": {
                "basis": basis,
                "anchor_basis": anchor_basis,
                "close_window_mode": close_mode,
                "close_window_source": close_window_source,
            },
            "modeled_assumptions": [assumption],
            "modeled_scope": {
                "basis": basis,
                "anchor_basis": anchor_basis,
                "close_window_mode": close_mode,
                "close_window_source": close_window_source,
            },
            "target_days": target_days,
            "baseline_span_days_median": baseline_span_median,
            "baseline_value": baseline_value,
            "modeled_value": modeled_value,
            "delta_value": delta_value,
            "unit": "days",
            "required_scale_factor_median": scale_median,
            "required_scale_factor_p90": scale_p90,
            "worst_month": worst_month.get("month") if worst_month else None,
            "worst_month_required_scale": worst_month.get("required_scale_factor") if worst_month else None,
            "worst_month_baseline_days": worst_month.get("baseline_span_days") if worst_month else None,
            "host_count": host_count,
            "scale_factor": scale_median,
            "scale_factor_standard": scale_median,
            "scale_factor_original": scale_median,
            "scale_factor_original_definition": "scale_factor_standard = modeled_host_count / baseline_host_count",
            "host_count_baseline": host_count,
            "host_count_modeled": modeled_host_count,
            "baseline_host_count": host_count,
            "modeled_host_count": modeled_host_count,
            "close_window_mode": close_mode,
            "close_window_source": close_window_source,
            "close_window_fallback": fallback_used,
            "close_window_reason": fallback_reason,
            "revenue_selector_strategy": summary.get("revenue_selector_strategy"),
            "revenue_selector": summary.get("revenue_selector"),
            "columns": columns_used,
            "row_ids": [int(i) for i in revenue_rows.index.tolist()[:50]],
        }

        findings = [finding]

        artifacts_dir = ctx.artifacts_dir("analysis_close_cycle_revenue_compression")
        out_path = artifacts_dir / "results.json"
        write_json(out_path, {"summary": summary, "findings": findings})
        artifacts = [
            PluginArtifact(
                path=str(out_path.relative_to(ctx.run_dir)),
                type="json",
                description="Revenue compression summary",
            )
        ]

        csv_path = artifacts_dir / "results.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            header = [
                "decision",
                "reason",
                "month",
                "rows",
                "baseline_span_days",
                "modeled_span_days",
                "required_scale_factor",
            ]
            handle.write(",".join(header) + "\n")
            for entry in month_stats:
                row = [
                    str(entry.get("decision")),
                    str(entry.get("reason")),
                    str(entry.get("month")),
                    str(entry.get("rows")),
                    str(entry.get("baseline_span_days")),
                    str(entry.get("modeled_span_days")),
                    str(entry.get("required_scale_factor")),
                ]
                handle.write(",".join(row) + "\n")
        artifacts.append(
            PluginArtifact(
                path=str(csv_path.relative_to(ctx.run_dir)),
                type="csv",
                description="Revenue compression detail table",
            )
        )

        md_path = artifacts_dir / "results.md"
        lines = [
            "# Close-cycle revenue compression",
            "",
            "Summary:",
            f"- close_window_mode: {close_mode}",
            f"- close_window_source: {close_window_source}",
            f"- target_days: {target_days}",
            f"- basis: {basis}",
            f"- anchor_basis: {anchor_basis}",
            f"- months_evaluated: {len(month_stats)}",
        ]
        selector = finding.get("revenue_selector") if isinstance(finding, dict) else None
        if isinstance(selector, dict) and selector:
            lines.extend(
                [
                    "",
                    "Selector:",
                    f"- strategy: {finding.get('revenue_selector_strategy')}",
                ]
            )
            selector_cols = selector.get("selector_columns") or []
            if selector_cols:
                lines.append(f"- columns: {', '.join(selector_cols)}")
            key_count = selector.get("selector_key_count")
            if key_count is not None:
                lines.append(f"- key_count: {key_count}")
            key_rows = selector.get("selector_keys") or []
            if key_rows:
                lines.extend(
                    [
                        "",
                        "Selector keys (top by score):",
                        "| key | close_rows | total_rows | close_share | close_lift | late_fraction |",
                        "| --- | --- | --- | --- | --- | --- |",
                    ]
                )
                for row in key_rows[:10]:
                    lines.append(
                        "| {key} | {close_rows} | {total_rows} | {close_share:.2f} | {close_lift:.2f} | {late_fraction:.2f} |".format(
                            key=row.get("key"),
                            close_rows=row.get("close_rows"),
                            total_rows=row.get("total_rows"),
                            close_share=float(row.get("close_share", 0.0)),
                            close_lift=float(row.get("close_lift", 0.0)),
                            late_fraction=float(row.get("late_fraction", 0.0)),
                        )
                    )
        lines.extend(
            [
                "",
                "Result:",
                "| process | scale_median | scale_p90 | worst_month | worst_month_scale |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        lines.append(
            "| {process_label} | {scale_median} | {scale_p90} | {worst_month} | {worst_month_scale} |".format(
                process_label=finding.get("process_label"),
                scale_median=(
                    f"{finding.get('required_scale_factor_median'):.3f}"
                    if finding.get("required_scale_factor_median") is not None
                    else ""
                ),
                scale_p90=(
                    f"{finding.get('required_scale_factor_p90'):.3f}"
                    if finding.get("required_scale_factor_p90") is not None
                    else ""
                ),
                worst_month=finding.get("worst_month") or "",
                worst_month_scale=(
                    f"{finding.get('worst_month_required_scale'):.3f}"
                    if finding.get("worst_month_required_scale") is not None
                    else ""
                ),
            )
        )
        if month_stats:
            lines.extend(["", "Month details:"])
            lines.append("| month | rows | baseline_days | modeled_days | required_scale | reason |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            for entry in month_stats[:max_months_output]:
                lines.append(
                    "| {month} | {rows} | {baseline} | {modeled} | {scale} | {reason} |".format(
                        month=entry.get("month"),
                        rows=entry.get("rows"),
                        baseline=(
                            f"{entry.get('baseline_span_days'):.2f}"
                            if entry.get("baseline_span_days") is not None
                            else ""
                        ),
                        modeled=(
                            f"{entry.get('modeled_span_days'):.2f}"
                            if entry.get("modeled_span_days") is not None
                            else ""
                        ),
                        scale=(
                            f"{entry.get('required_scale_factor'):.3f}"
                            if entry.get("required_scale_factor") is not None
                            else ""
                        ),
                        reason=entry.get("reason"),
                    )
                )
        md_path.write_text("\n".join(lines), encoding="utf-8")
        artifacts.append(
            PluginArtifact(
                path=str(md_path.relative_to(ctx.run_dir)),
                type="markdown",
                description="Revenue compression summary",
            )
        )

        metrics = {
            "findings": len(findings),
            "decision": decision,
            "required_scale_factor_median": scale_median,
            "required_scale_factor_p90": scale_p90,
            "baseline_span_days_median": baseline_span_median,
            "months_evaluated": len(month_stats),
            "close_window_mode": close_mode,
            "close_window_source": close_window_source,
            "close_cycle_start_day": close_start_day,
            "close_cycle_end_day": close_end_day,
            "close_cycle_dynamic_available": dynamic_available,
            "close_cycle_dynamic_months": len(dynamic_windows),
            "close_cycle_rows_default": close_rows_default,
            "close_cycle_rows_dynamic": close_rows_dynamic,
        }

        return PluginResult(
            "ok",
            "Computed revenue close-cycle compression model",
            metrics,
            findings,
            artifacts,
            None,
        )
