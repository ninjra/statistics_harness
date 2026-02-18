from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from typing import Any, Iterable

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
                current_end = max(current_end, end_ts)
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
    events.sort(key=lambda item: (item[0], item[1]))
    current = 0
    max_conc = 0
    for _, delta in events:
        current += delta
        if current > max_conc:
            max_conc = current
    return max_conc


def _js_divergence(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    eps = 1e-12
    p = []
    q = []
    for key in keys:
        p.append(left.get(key, 0.0) + eps)
        q.append(right.get(key, 0.0) + eps)
    p_sum = sum(p)
    q_sum = sum(q)
    p = [val / p_sum for val in p]
    q = [val / q_sum for val in q]
    m = [(a + b) * 0.5 for a, b in zip(p, q)]

    def _kl(a: list[float], b: list[float]) -> float:
        total = 0.0
        for ai, bi in zip(a, b):
            if ai <= 0 or bi <= 0:
                continue
            total += ai * math.log(ai / bi)
        return total

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _bootstrap_ci(
    rng: np.random.Generator,
    hi_values: np.ndarray,
    lo_values: np.ndarray,
    samples: int,
    alpha: float,
) -> tuple[float | None, float | None]:
    if hi_values.size == 0 or lo_values.size == 0:
        return None, None
    effects = []
    for _ in range(samples):
        hi_sample = rng.choice(hi_values, size=hi_values.size, replace=True)
        lo_sample = rng.choice(lo_values, size=lo_values.size, replace=True)
        lo_med = float(np.median(lo_sample))
        hi_med = float(np.median(hi_sample))
        if lo_med <= 0:
            continue
        effects.append((hi_med / lo_med) - 1.0)
    if not effects:
        return None, None
    effects = np.asarray(effects)
    low = float(np.quantile(effects, alpha / 2.0))
    high = float(np.quantile(effects, 1.0 - alpha / 2.0))
    return low, high


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

    for idx, month in enumerate(months):
        year, month_num = [int(part) for part in month.split("-")]
        month_start = date(year, month_num, 1)
        if month_num == 12:
            next_month_start = date(year + 1, 1, 1)
        else:
            next_month_start = date(year, month_num + 1, 1)
        month_end = next_month_start - timedelta(days=1)

        month_days = _build_calendar_days(month_start, month_end)
        next_days = _build_calendar_days(next_month_start, next_month_start + timedelta(days=max(0, lookahead_days - 1)))
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
        if eligible_col:
            used.add(eligible_col)

        summary = {
            "process_column": process_col,
            "host_column": host_col,
            "start_column": start_col,
            "end_column": end_col,
            "queue_column": queue_col,
            "eligible_column": eligible_col,
        }

        def _emit_not_applicable(reason: str) -> PluginResult:
            findings = []
            for metric in ("concurrent", "unique"):
                findings.append(
                    {
                        "kind": "close_cycle_capacity_impact",
                        "host_metric": metric,
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
            artifacts_dir = ctx.artifacts_dir("analysis_close_cycle_capacity_impact")
            out_path = artifacts_dir / "results.json"
            write_json(out_path, {"summary": summary, "findings": findings})
            artifacts = [
                PluginArtifact(
                    path=str(out_path.relative_to(ctx.run_dir)),
                    type="json",
                    description="Close-cycle capacity impact summary",
                )
            ]
            csv_path = artifacts_dir / "results.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                header = [
                    "host_metric",
                    "decision",
                    "effect",
                    "ci_low",
                    "ci_high",
                    "target_reduction",
                    "tolerance",
                    "hi_bucket_count",
                    "lo_bucket_count",
                    "median_rows_hi",
                    "median_rows_lo",
                    "volume_ratio",
                    "process_mix_js",
                    "reason",
                ]
                handle.write(",".join(header) + "\n")
                for item in findings:
                    handle.write(
                        ",".join(
                            [
                                str(item.get("host_metric")),
                                str(item.get("decision")),
                                str(item.get("effect")),
                                str(item.get("ci_low")),
                                str(item.get("ci_high")),
                                str(item.get("target_reduction")),
                                str(item.get("tolerance")),
                                str(item.get("hi_bucket_count")),
                                str(item.get("lo_bucket_count")),
                                str(item.get("median_rows_hi")),
                                str(item.get("median_rows_lo")),
                                str(item.get("volume_ratio")),
                                str(item.get("process_mix_js")),
                                str(item.get("reason")),
                            ]
                        )
                        + "\n"
                    )
            artifacts.append(
                PluginArtifact(
                    path=str(csv_path.relative_to(ctx.run_dir)),
                    type="csv",
                    description="Capacity impact detail table",
                )
            )

            md_path = artifacts_dir / "results.md"
            lines = [
                "# Close-cycle capacity impact",
                "",
                "Summary:",
                f"- close_window_mode: {summary.get('close_window_mode')}",
                f"- close_cycle_start_day: {summary.get('close_cycle_start_day')}",
                f"- close_cycle_end_day: {summary.get('close_cycle_end_day')}",
                "",
                "Detected:",
                "_None_",
                "",
                "Not applicable:",
            ]
            for item in findings:
                lines.append(f"- {item.get('host_metric')}: {item.get('reason')}")
            md_path.write_text("\n".join(lines), encoding="utf-8")
            artifacts.append(
                PluginArtifact(
                    path=str(md_path.relative_to(ctx.run_dir)),
                    type="markdown",
                    description="Capacity impact summary",
                )
            )
            return PluginResult(
                "ok",
                reason,
                summary,
                findings,
                artifacts,
                None,
            )

        if not start_col or not end_col:
            return _emit_not_applicable("missing_start_end")
        if not host_col:
            return _emit_not_applicable("missing_host_column")

        selected_cols = [
            col
            for col in [process_col, host_col, start_col, end_col, queue_col, eligible_col]
            if col
        ]
        work = df.loc[:, selected_cols].copy()

        work["__start_ts"] = pd.to_datetime(work[start_col], errors="coerce", utc=False)
        work["__end_ts"] = pd.to_datetime(work[end_col], errors="coerce", utc=False)
        work = work.loc[work["__start_ts"].notna() & work["__end_ts"].notna()].copy()
        if work.empty:
            return _emit_not_applicable("no_valid_timestamps")

        work = work.loc[work["__end_ts"] >= work["__start_ts"]].copy()
        if work.empty:
            return _emit_not_applicable("no_non_negative_durations")

        work["__ttc_sec"] = (work["__end_ts"] - work["__start_ts"]).dt.total_seconds()
        work = work.loc[work["__ttc_sec"] > 0].copy()
        if work.empty:
            return _emit_not_applicable("no_positive_ttc")

        work["__host"] = work[host_col].map(_normalize_text)
        work["__host_norm"] = work["__host"].str.lower()
        work = work.loc[~work["__host_norm"].isin(INVALID_STRINGS)].copy()
        if work.empty:
            return _emit_not_applicable("no_valid_host_values")

        if process_col:
            work["__process"] = work[process_col].map(_normalize_text)
            work["__process_norm"] = work["__process"].str.lower()

        if queue_col and queue_col in work.columns:
            work["__queue_ts"] = pd.to_datetime(work[queue_col], errors="coerce", utc=False)
            work["__queue_to_end_sec"] = (
                work["__end_ts"] - work["__queue_ts"]
            ).dt.total_seconds().clip(lower=0)
        else:
            work["__queue_to_end_sec"] = np.nan

        if eligible_col and eligible_col in work.columns:
            work["__eligible_ts"] = pd.to_datetime(work[eligible_col], errors="coerce", utc=False)
            work["__eligible_to_end_sec"] = (
                work["__end_ts"] - work["__eligible_ts"]
            ).dt.total_seconds().clip(lower=0)
        else:
            work["__eligible_to_end_sec"] = np.nan

        close_mode = str(ctx.settings.get("close_window_mode", "infer") or "infer").lower()
        if close_mode == "calendar":
            close_mode = "override"
        close_start_day = int(ctx.settings.get("close_cycle_start_day", 20))
        close_end_day = int(ctx.settings.get("close_cycle_end_day", 5))
        min_close_days = int(ctx.settings.get("min_close_days", 5))
        max_close_days = int(ctx.settings.get("max_close_days", 20))
        lookahead_days = int(ctx.settings.get("lookahead_days", 7))
        min_close_confidence = float(ctx.settings.get("min_close_confidence", 0.15))
        min_close_data_ratio = float(ctx.settings.get("min_close_data_ratio", 0.5))

        work["__date"] = work["__start_ts"].dt.date
        daily = (
            work.groupby("__date")
            .agg(count=("__date", "size"), median_ttc=("__ttc_sec", "median"))
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
            "fallback"
            if fallback_used
            else ("override" if close_mode == "override" else "infer")
        )
        if dynamic_available:
            close_window_source = (
                f"dynamic_{dynamic_source_plugin}"
                if isinstance(dynamic_source_plugin, str) and dynamic_source_plugin
                else "dynamic_resolver"
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
            if bucket_date not in close_dates:
                close_flag = False
            else:
                close_flag = True
            row_count = int(frame.shape[0])
            if row_count < min_bucket_rows:
                continue
            median_ttc = float(frame["__ttc_sec"].median())
            median_qe = float(frame["__queue_to_end_sec"].median()) if frame["__queue_to_end_sec"].notna().any() else None
            median_ee = float(frame["__eligible_to_end_sec"].median()) if frame["__eligible_to_end_sec"].notna().any() else None
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
                    "median_ttc": median_ttc,
                    "median_qe": median_qe,
                    "median_ee": median_ee,
                    "host_unique": host_unique,
                    "host_concurrent": host_concurrent,
                }
            )

        if not buckets:
            return _emit_not_applicable("no_buckets_after_filtering")

        bucket_df = pd.DataFrame(buckets)
        close_buckets = bucket_df.loc[bucket_df["close"]].copy()
        close_buckets = close_buckets.loc[close_buckets["month"].isin(confident_months)].copy()
        if close_buckets.empty:
            return _emit_not_applicable("no_close_buckets")

        min_buckets_per_group = int(ctx.settings.get("min_buckets_per_group", 5))
        min_buckets_per_month = int(ctx.settings.get("min_buckets_per_month", 3))
        min_months = int(ctx.settings.get("min_months", 2))
        target_reduction = float(ctx.settings.get("target_reduction", 0.30))
        tolerance = float(ctx.settings.get("tolerance", 0.05))
        alpha = float(ctx.settings.get("alpha", 0.01))
        bootstrap_samples = int(ctx.settings.get("bootstrap_samples", 1000))
        max_js = float(ctx.settings.get("max_js_divergence", 0.2))
        min_volume_ratio = float(ctx.settings.get("min_volume_ratio", 0.5))
        max_volume_ratio = float(ctx.settings.get("max_volume_ratio", 2.0))
        max_examples = int(ctx.settings.get("max_examples", 25))

        findings = []

        for metric in ("concurrent", "unique"):
            if metric == "concurrent":
                metric_col = "host_concurrent"
            else:
                metric_col = "host_unique"

            hi = close_buckets.loc[close_buckets[metric_col] >= 3].copy()
            lo = close_buckets.loc[close_buckets[metric_col] <= 2].copy()

            reasons: list[str] = []
            decision = "not_applicable"

            if hi.shape[0] < min_buckets_per_group or lo.shape[0] < min_buckets_per_group:
                reasons.append("insufficient_buckets")

            if process_col is None:
                reasons.append("missing_process_column")

            # Process mix divergence
            js_div = None
            if process_col and hi.shape[0] >= 1 and lo.shape[0] >= 1:
                hi_bucket_keys = set(hi["bucket_start"].tolist())
                lo_bucket_keys = set(lo["bucket_start"].tolist())
                hi_rows = work.loc[work["__bucket_start"].isin(hi_bucket_keys)]
                lo_rows = work.loc[work["__bucket_start"].isin(lo_bucket_keys)]
                if hi_rows.empty or lo_rows.empty:
                    reasons.append("insufficient_rows_for_mix")
                else:
                    hi_counts = (
                        hi_rows["__process_norm"].value_counts(normalize=True).to_dict()
                        if "__process_norm" in hi_rows
                        else {}
                    )
                    lo_counts = (
                        lo_rows["__process_norm"].value_counts(normalize=True).to_dict()
                        if "__process_norm" in lo_rows
                        else {}
                    )
                    js_div = _js_divergence(hi_counts, lo_counts)
                    if js_div > max_js:
                        reasons.append("process_mix_divergence")

            # Volume parity guard
            median_rows_hi = float(hi["rows"].median()) if not hi.empty else 0.0
            median_rows_lo = float(lo["rows"].median()) if not lo.empty else 0.0
            volume_ratio = None
            if median_rows_lo > 0:
                volume_ratio = median_rows_hi / median_rows_lo
                if volume_ratio < min_volume_ratio or volume_ratio > max_volume_ratio:
                    reasons.append("volume_parity")
            else:
                reasons.append("volume_parity")

            # Time trend guard (per month)
            month_effects = []
            for month, month_frame in close_buckets.groupby("month"):
                hi_month = month_frame.loc[month_frame[metric_col] >= 3]
                lo_month = month_frame.loc[month_frame[metric_col] <= 2]
                if (
                    hi_month.shape[0] < min_buckets_per_month
                    or lo_month.shape[0] < min_buckets_per_month
                ):
                    continue
                hi_med = float(hi_month["median_ttc"].median())
                lo_med = float(lo_month["median_ttc"].median())
                if lo_med <= 0:
                    continue
                month_effects.append((hi_med / lo_med) - 1.0)

            if len(month_effects) < min_months:
                reasons.append("insufficient_months")
            else:
                if any(effect >= 0 for effect in month_effects):
                    reasons.append("month_effect_sign")

            # Effect + CI
            effect = None
            ci_low = None
            ci_high = None
            if hi.shape[0] > 0 and lo.shape[0] > 0:
                hi_vals = hi["median_ttc"].to_numpy(dtype=float)
                lo_vals = lo["median_ttc"].to_numpy(dtype=float)
                lo_med = float(np.median(lo_vals))
                hi_med = float(np.median(hi_vals))
                if lo_med > 0:
                    effect = (hi_med / lo_med) - 1.0
                rng = np.random.default_rng(ctx.run_seed ^ (hash(metric_col) & 0xFFFFFFFF))
                ci_low, ci_high = _bootstrap_ci(
                    rng,
                    hi_vals,
                    lo_vals,
                    bootstrap_samples,
                    alpha,
                )

            if effect is None or ci_low is None or ci_high is None:
                reasons.append("effect_unavailable")
            else:
                band_low = -(target_reduction + tolerance)
                band_high = -(target_reduction - tolerance)
                if not (ci_low >= band_low and ci_high <= band_high):
                    reasons.append("effect_not_within_tolerance")

            if not reasons:
                decision = "detected"

            # Secondary diagnostics (queue/eligible to end)
            qe_effect = None
            ee_effect = None
            qe_ci_low = None
            qe_ci_high = None
            ee_ci_low = None
            ee_ci_high = None

            if hi.shape[0] > 0 and lo.shape[0] > 0:
                if hi["median_qe"].notna().any() and lo["median_qe"].notna().any():
                    hi_qe = hi["median_qe"].dropna().to_numpy(dtype=float)
                    lo_qe = lo["median_qe"].dropna().to_numpy(dtype=float)
                    if hi_qe.size and lo_qe.size:
                        lo_med = float(np.median(lo_qe))
                        if lo_med > 0:
                            qe_effect = (float(np.median(hi_qe)) / lo_med) - 1.0
                        rng = np.random.default_rng(ctx.run_seed ^ (hash(metric_col + "qe") & 0xFFFFFFFF))
                        qe_ci_low, qe_ci_high = _bootstrap_ci(rng, hi_qe, lo_qe, bootstrap_samples, alpha)
                if hi["median_ee"].notna().any() and lo["median_ee"].notna().any():
                    hi_ee = hi["median_ee"].dropna().to_numpy(dtype=float)
                    lo_ee = lo["median_ee"].dropna().to_numpy(dtype=float)
                    if hi_ee.size and lo_ee.size:
                        lo_med = float(np.median(lo_ee))
                        if lo_med > 0:
                            ee_effect = (float(np.median(hi_ee)) / lo_med) - 1.0
                        rng = np.random.default_rng(ctx.run_seed ^ (hash(metric_col + "ee") & 0xFFFFFFFF))
                        ee_ci_low, ee_ci_high = _bootstrap_ci(rng, hi_ee, lo_ee, bootstrap_samples, alpha)

            row_ids = work.index.tolist()[:max_examples]

            findings.append(
                {
                    "kind": "close_cycle_capacity_impact",
                    "host_metric": metric,
                    "decision": decision,
                    "measurement_type": "measured" if decision == "detected" else "not_applicable",
                    "reason": ",".join(reasons) if reasons else "ok",
                    "close_window_mode": close_mode,
                    "close_window_fallback": fallback_used,
                    "close_window_source": close_window_source,
                    "close_window_reason": fallback_reason,
                    "effect": effect,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "target_reduction": target_reduction,
                    "tolerance": tolerance,
                    "alpha": alpha,
                    "hi_bucket_count": int(hi.shape[0]),
                    "lo_bucket_count": int(lo.shape[0]),
                    "median_rows_hi": median_rows_hi,
                    "median_rows_lo": median_rows_lo,
                    "volume_ratio": volume_ratio,
                    "process_mix_js": js_div,
                    "month_effects": month_effects,
                    "qe_effect": qe_effect,
                    "qe_ci_low": qe_ci_low,
                    "qe_ci_high": qe_ci_high,
                    "ee_effect": ee_effect,
                    "ee_ci_low": ee_ci_low,
                    "ee_ci_high": ee_ci_high,
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
                    "row_ids": [int(i) for i in row_ids],
                }
            )

        artifacts_dir = ctx.artifacts_dir("analysis_close_cycle_capacity_impact")
        out_path = artifacts_dir / "results.json"
        write_json(out_path, {"summary": summary, "findings": findings})
        artifacts = [
            PluginArtifact(
                path=str(out_path.relative_to(ctx.run_dir)),
                type="json",
                description="Close-cycle capacity impact summary",
            )
        ]

        # Human-readable outputs
        csv_path = artifacts_dir / "results.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            header = [
                "host_metric",
                "decision",
                "effect",
                "ci_low",
                "ci_high",
                "target_reduction",
                "tolerance",
                "hi_bucket_count",
                "lo_bucket_count",
                "median_rows_hi",
                "median_rows_lo",
                "volume_ratio",
                "process_mix_js",
                "reason",
            ]
            handle.write(",".join(header) + "\n")
            for item in findings:
                handle.write(",".join(
                    [
                        str(item.get("host_metric")),
                        str(item.get("decision")),
                        str(item.get("effect")),
                        str(item.get("ci_low")),
                        str(item.get("ci_high")),
                        str(item.get("target_reduction")),
                        str(item.get("tolerance")),
                        str(item.get("hi_bucket_count")),
                        str(item.get("lo_bucket_count")),
                        str(item.get("median_rows_hi")),
                        str(item.get("median_rows_lo")),
                        str(item.get("volume_ratio")),
                        str(item.get("process_mix_js")),
                        str(item.get("reason")),
                    ]
                ) + "\n")
        artifacts.append(
            PluginArtifact(
                path=str(csv_path.relative_to(ctx.run_dir)),
                type="csv",
                description="Capacity impact detail table",
            )
        )

        md_path = artifacts_dir / "results.md"
        lines = [
            "# Close-cycle capacity impact",
            "",
            "Summary:",
            f"- close_window_mode: {summary.get('close_window_mode')}",
            f"- close_cycle_start_day: {summary.get('close_cycle_start_day')}",
            f"- close_cycle_end_day: {summary.get('close_cycle_end_day')}",
            f"- min_close_days: {summary.get('min_close_days')}",
            f"- max_close_days: {summary.get('max_close_days')}",
            f"- min_close_confidence: {summary.get('min_close_confidence')}",
            f"- target_reduction: {target_reduction}",
            f"- tolerance: {tolerance}",
            f"- alpha: {alpha}",
            "",
            "Detected:",
        ]

        detected_rows = [
            item for item in findings if item.get("decision") == "detected"
        ]
        if detected_rows:
            lines.append(
                "| host_metric | effect | ci_low | ci_high | hi_buckets | lo_buckets |"
            )
            lines.append("| --- | --- | --- | --- | --- | --- |")
            for item in detected_rows:
                lines.append(
                    "| {host_metric} | {effect:.4f} | {ci_low:.4f} | {ci_high:.4f} | {hi_bucket_count} | {lo_bucket_count} |".format(
                        host_metric=item.get("host_metric"),
                        effect=item.get("effect") or 0.0,
                        ci_low=item.get("ci_low") or 0.0,
                        ci_high=item.get("ci_high") or 0.0,
                        hi_bucket_count=item.get("hi_bucket_count"),
                        lo_bucket_count=item.get("lo_bucket_count"),
                    )
                )
        else:
            lines.append("_None_")

        lines.extend(["", "Not applicable:"])
        for item in findings:
            if item.get("decision") != "detected":
                lines.append(
                    f"- {item.get('host_metric')}: {item.get('reason')}"
                )

        md_path.write_text("\n".join(lines), encoding="utf-8")
        artifacts.append(
            PluginArtifact(
                path=str(md_path.relative_to(ctx.run_dir)),
                type="markdown",
                description="Capacity impact summary",
            )
        )

        metrics = {
            "findings": len(findings),
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
            "Computed close-cycle capacity impact",
            metrics,
            findings,
            artifacts,
            None,
        )
