from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import math
import hashlib

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IdeaspaceColumns:
    time_col: str | None
    eligible_col: str | None
    start_col: str | None
    end_col: str | None
    duration_col: str | None
    batch_col: str | None
    host_col: str | None
    case_col: str | None
    activity_col: str | None
    text_col: str | None
    group_cols: list[str]


def _pick_first_by_name(columns: list[str], hints: tuple[str, ...]) -> str | None:
    for hint in hints:
        for col in columns:
            if hint in str(col).lower():
                return col
    return None


def pick_columns(df: pd.DataFrame, inferred: dict[str, Any], config: dict[str, Any]) -> IdeaspaceColumns:
    ts_cols = [str(c) for c in (inferred.get("timestamp_columns") or []) if str(c) in df.columns]
    num_cols = [str(c) for c in (inferred.get("numeric_columns") or []) if str(c) in df.columns]
    cat_cols = [str(c) for c in (inferred.get("categorical_columns") or []) if str(c) in df.columns]
    text_cols = [str(c) for c in (inferred.get("text_columns") or []) if str(c) in df.columns]
    id_like = set(str(c) for c in (inferred.get("id_like_columns") or []))

    time_col = inferred.get("time_column")
    if not isinstance(time_col, str) or time_col not in df.columns:
        time_col = _pick_first_by_name(ts_cols, ("time", "timestamp", "ts", "date", "created", "updated"))

    eligible_col = _pick_first_by_name(ts_cols, ("eligible", "queued", "queue"))
    start_col = _pick_first_by_name(ts_cols, ("start", "begin", "started"))
    end_col = _pick_first_by_name(ts_cols, ("end", "finish", "completed", "done"))

    duration_col = _pick_first_by_name(
        num_cols, ("duration", "latency", "elapsed", "runtime", "run_time", "seconds", "sec", "wait")
    )
    batch_col = _pick_first_by_name(num_cols, ("batch", "items", "lines", "count", "size"))
    host_col = _pick_first_by_name(cat_cols, ("host", "server", "node", "machine", "worker"))
    case_col = _pick_first_by_name(cat_cols, ("case", "trace", "session", "ticket", "order", "request"))
    activity_col = _pick_first_by_name(cat_cols, ("activity", "step", "task", "event", "action", "stage"))

    text_col = _pick_first_by_name(text_cols, ("message", "error", "exception", "stack", "trace", "log"))
    if text_col is None and text_cols:
        text_col = text_cols[0]

    group_cols = list(inferred.get("group_by") or [])
    group_cols = [str(c) for c in group_cols if str(c) in df.columns and str(c) not in id_like]

    max_group_cols = int(config.get("ideaspace_max_group_cols", 3))
    group_cols = group_cols[:max_group_cols]

    return IdeaspaceColumns(
        time_col=time_col,
        eligible_col=eligible_col,
        start_col=start_col,
        end_col=end_col,
        duration_col=duration_col,
        batch_col=batch_col,
        host_col=host_col,
        case_col=case_col,
        activity_col=activity_col,
        text_col=text_col,
        group_cols=group_cols,
    )


def coerce_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, errors="coerce")


def duration_seconds(df: pd.DataFrame, duration_col: str | None, start_col: str | None, end_col: str | None) -> pd.Series:
    if duration_col and duration_col in df.columns:
        out = pd.to_numeric(df[duration_col], errors="coerce")
        return out
    if start_col and end_col and start_col in df.columns and end_col in df.columns:
        start = coerce_datetime(df[start_col])
        end = coerce_datetime(df[end_col])
        delta = (end - start).dt.total_seconds()
        return pd.to_numeric(delta, errors="coerce")
    return pd.Series(dtype=float)


def queue_delay_seconds(df: pd.DataFrame, eligible_col: str | None, start_col: str | None) -> pd.Series:
    if not eligible_col or not start_col:
        return pd.Series(dtype=float)
    if eligible_col not in df.columns or start_col not in df.columns:
        return pd.Series(dtype=float)
    eligible = coerce_datetime(df[eligible_col])
    start = coerce_datetime(df[start_col])
    delta = (start - eligible).dt.total_seconds()
    return pd.to_numeric(delta, errors="coerce")


def error_rate(df: pd.DataFrame, candidate_cols: list[str], redactor: Callable[[str], str]) -> float | None:
    """Heuristic error-rate detector (secure-by-default).

    Returns a fraction in [0, 1], or None if no suitable signal exists.
    """

    status_like = None
    for col in candidate_cols:
        name = str(col).lower()
        if any(tok in name for tok in ("status", "result", "outcome", "state", "success", "error")):
            status_like = col
            break
    if status_like is None:
        return None
    series = df[status_like].dropna()
    if series.empty:
        return None
    values = series.astype(str).str.lower()
    # Keep this very conservative; do not emit raw exemplars.
    is_error = values.str.contains("fail|error|exception|timeout|cancel", regex=True)
    return float(is_error.mean())


def entity_slices(
    df: pd.DataFrame,
    group_cols: list[str],
    max_groups_per_col: int,
    max_entities: int,
    redactor: Callable[[str], str],
) -> list[tuple[str, pd.DataFrame, dict[str, Any]]]:
    """Deterministically slice into entity cohorts.

    Returns list of (entity_key, slice_df, meta) where meta includes redacted label.
    """

    slices: list[tuple[str, pd.DataFrame, dict[str, Any]]] = [("ALL", df, {"label": "ALL", "group": {}})]

    def _safe_group_value(col: str, value: object) -> str:
        # Security: avoid emitting raw identity-ish tokens even if they don't match
        # regex-based PII patterns (e.g. usernames, hostnames).
        lowered = str(col).lower()
        # Note: avoid matching the generic token "name" (too broad); rely on more
        # specific identity hints.
        if any(tok in lowered for tok in ("user", "email", "account", "id", "host", "server", "ip")):
            digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:10]
            return f"hash:{digest}"
        return redactor(str(value))
    for col in group_cols:
        if col not in df.columns:
            continue
        counts = df[col].value_counts(dropna=False)
        for value in counts.index[:max_groups_per_col]:
            safe_value = _safe_group_value(col, value)
            key = f"{col}={safe_value}"
            label = f"{col}={safe_value}"
            slice_df = df.loc[df[col] == value]
            slices.append((key, slice_df, {"label": label, "group": {col: safe_value}}))
            if len(slices) >= max_entities:
                return slices[:max_entities]
    return slices[:max_entities]


def kpi_summary(series: pd.Series) -> dict[str, float] | None:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 5:
        return None
    return {
        "p50": float(np.nanpercentile(values, 50)),
        "p95": float(np.nanpercentile(values, 95)),
    }


def time_span_seconds(df: pd.DataFrame, time_col: str | None) -> float | None:
    if not time_col or time_col not in df.columns:
        return None
    ts = coerce_datetime(df[time_col])
    ts = ts.dropna()
    if ts.empty:
        return None
    span = (ts.max() - ts.min()).total_seconds()
    if not math.isfinite(span) or span <= 0:
        return None
    return float(span)


def buckets_from_time(df: pd.DataFrame, time_col: str, bucket_seconds: int = 60) -> pd.Series:
    ts = coerce_datetime(df[time_col])
    # Use integer bucket indices for determinism.
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    ts_utc = ts.dt.tz_convert("UTC") if ts.dt.tz is not None else ts.dt.tz_localize("UTC", nonexistent="NaT", ambiguous="NaT")
    seconds = (ts_utc - epoch).dt.total_seconds()
    return (seconds // float(bucket_seconds)).astype("Int64")


def concurrency_by_bucket(
    df: pd.DataFrame,
    start_col: str,
    end_col: str,
    bucket_seconds: int = 60,
    max_buckets: int = 2000,
) -> pd.DataFrame:
    """Approximate active concurrency in fixed time buckets.

    Returns a frame with columns: bucket, active, started, ended.
    """

    start = coerce_datetime(df[start_col])
    end = coerce_datetime(df[end_col])
    ok = start.notna() & end.notna()
    if not ok.any():
        return pd.DataFrame(columns=["bucket", "active", "started", "ended"])
    start = start[ok]
    end = end[ok]

    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    s_utc = start.dt.tz_convert("UTC") if start.dt.tz is not None else start.dt.tz_localize("UTC", nonexistent="NaT", ambiguous="NaT")
    e_utc = end.dt.tz_convert("UTC") if end.dt.tz is not None else end.dt.tz_localize("UTC", nonexistent="NaT", ambiguous="NaT")
    s_bucket = ((s_utc - epoch).dt.total_seconds() // float(bucket_seconds)).astype(int)
    e_bucket = ((e_utc - epoch).dt.total_seconds() // float(bucket_seconds)).astype(int)

    lo = int(min(s_bucket.min(), e_bucket.min()))
    hi = int(max(s_bucket.max(), e_bucket.max()))
    if hi - lo > max_buckets:
        # Reduce resolution deterministically to bound memory.
        stride = int(math.ceil((hi - lo) / float(max_buckets)))
        s_bucket = (s_bucket // stride).astype(int)
        e_bucket = (e_bucket // stride).astype(int)
        lo = int(min(s_bucket.min(), e_bucket.min()))
        hi = int(max(s_bucket.max(), e_bucket.max()))

    size = (hi - lo) + 2
    delta = np.zeros(size, dtype=int)
    started = np.zeros(size, dtype=int)
    ended = np.zeros(size, dtype=int)
    for s, e in zip(s_bucket.to_numpy(), e_bucket.to_numpy(), strict=False):
        si = int(s - lo)
        ei = int(e - lo)
        if 0 <= si < size:
            delta[si] += 1
            started[si] += 1
        if 0 <= (ei + 1) < size:
            delta[ei + 1] -= 1
        if 0 <= ei < size:
            ended[ei] += 1
    active = np.cumsum(delta)[:-1]
    buckets = np.arange(lo, hi + 1, dtype=int)
    frame = pd.DataFrame(
        {
            "bucket": buckets,
            "active": active.astype(int),
            "started": started[:-1].astype(int),
            "ended": ended[:-1].astype(int),
        }
    )
    return frame
