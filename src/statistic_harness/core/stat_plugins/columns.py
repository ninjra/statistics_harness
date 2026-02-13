from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import math

TIME_HINTS = ("time", "ts", "date", "start", "end", "queued", "eligible", "created", "updated")
TEXT_HINTS = ("message", "log", "exception", "error", "stack", "trace")


def infer_column_types(df: pd.DataFrame, sample_size: int = 2000) -> dict[str, list[str]]:
    numeric: list[str] = []
    categorical: list[str] = []
    text: list[str] = []
    timestamp: list[str] = []
    id_like: list[str] = []
    for col in df.columns:
        series = df[col]
        sample = series.dropna().head(sample_size)
        if sample.empty:
            continue
        name = str(col).lower()
        unique_ratio = sample.nunique() / max(len(sample), 1)
        # ID-like columns are frequently coerced to TEXT by normalization and can
        # accidentally parse as epoch timestamps (e.g. "1000" -> 1970-...).
        if _is_id_like(sample, unique_ratio, name):
            id_like.append(col)
            continue
        if _is_timestamp(sample, name):
            timestamp.append(col)
            continue
        if _is_numeric(sample):
            numeric.append(col)
            continue
        if _is_text(sample, unique_ratio, name):
            text.append(col)
            continue
        if unique_ratio <= 0.2 or sample.nunique() <= 500:
            categorical.append(col)
        else:
            text.append(col)
    return {
        "numeric": numeric,
        "categorical": categorical,
        "text": text,
        "timestamp": timestamp,
        "id_like": id_like,
    }


def infer_columns(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    types = infer_column_types(df)
    time_col = _resolve_time_column(df, config.get("time_column", "auto"), types["timestamp"])
    group_by = _resolve_group_by(config.get("group_by", "auto"), types["categorical"], config)
    value_columns = _resolve_values(config.get("value_columns", "auto"), types["numeric"], config)
    return {
        "time_column": time_col,
        "group_by": group_by,
        "value_columns": value_columns,
        "numeric_columns": types["numeric"],
        "categorical_columns": types["categorical"],
        "text_columns": types["text"],
        "timestamp_columns": types["timestamp"],
        "id_like_columns": types["id_like"],
    }


def _resolve_time_column(df: pd.DataFrame, setting: Any, candidates: list[str]) -> str | None:
    if isinstance(setting, str) and setting.lower() != "auto":
        return setting if setting in df.columns else None
    best = None
    best_score = -1.0
    for col in candidates:
        series = pd.to_datetime(df[col], errors="coerce")
        non_null = series.notna().mean()
        span = (series.max() - series.min()).total_seconds() if series.notna().any() else 0.0
        lowered = str(col).lower()
        hint_score = 0.0
        if "start" in lowered:
            hint_score += 0.35
        if "end" in lowered:
            hint_score += 0.30
        if "queue" in lowered or "queued" in lowered:
            hint_score += 0.25
        if "created" in lowered or "updated" in lowered:
            hint_score += 0.10
        if any(token in lowered for token in ("time", "date", "timestamp", "dt", "ts")):
            hint_score += 0.15

        # Prefer columns that parse well, span a meaningful range, and look like time.
        score = float(non_null) * 10.0 + (math.log1p(max(span, 0.0)) / 10.0) + hint_score
        if score > best_score:
            best_score = score
            best = col
    return best


def _resolve_group_by(setting: Any, candidates: list[str], config: dict[str, Any]) -> list[str]:
    if isinstance(setting, list):
        return [col for col in setting if col in candidates]
    if isinstance(setting, str) and setting.lower() != "auto":
        return [setting]
    max_groups = int(config.get("max_groups", 30))
    return list(candidates[:max_groups])


def _resolve_values(setting: Any, candidates: list[str], config: dict[str, Any]) -> list[str]:
    if isinstance(setting, list):
        return [col for col in setting if col in candidates]
    if isinstance(setting, str) and setting.lower() != "auto":
        return [setting]
    max_cols = int(config.get("max_cols", 80))
    return list(candidates[:max_cols])


def _is_numeric(sample: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(sample)


def _is_timestamp(sample: pd.Series, name: str) -> bool:
    has_hint = any(token in name for token in TIME_HINTS)
    # If the column name suggests an ID, do not treat it as a timestamp unless it
    # very explicitly looks like one (hinted by TIME_HINTS).
    if ("id" in name or name.endswith("_id") or name.endswith("id")) and not has_hint:
        return False
    if pd.api.types.is_numeric_dtype(sample) and not has_hint:
        return False
    if not has_hint:
        # Prevent pure-digit strings from being interpreted as epoch nanos.
        sample_str = sample.astype(str).str.strip()
        digit_ratio = float(sample_str.str.fullmatch(r"[+-]?\d+", na=False).mean())
        if digit_ratio >= 0.9:
            return False
    if has_hint:
        parsed = pd.to_datetime(sample, errors="coerce")
        return parsed.notna().mean() >= 0.7
    parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().mean() >= 0.9


def _is_id_like(sample: pd.Series, unique_ratio: float, name: str) -> bool:
    if "id" in name or name.endswith("_id") or name.endswith("id"):
        return unique_ratio >= 0.9
    if not pd.api.types.is_numeric_dtype(sample):
        return False
    if unique_ratio >= 0.98 and sample.nunique() >= 50:
        # Only mark as ID-like if it's effectively an integer key and monotonic.
        values = pd.to_numeric(sample, errors="coerce").dropna()
        if values.empty:
            return False
        int_ratio = float(((values % 1) == 0).mean())
        if int_ratio >= 0.98 and values.is_monotonic_increasing:
            return True
    return False


def _is_text(sample: pd.Series, unique_ratio: float, name: str) -> bool:
    if any(token in name for token in TEXT_HINTS):
        return True
    lengths = sample.astype(str).map(len)
    return unique_ratio >= 0.5 and float(np.median(lengths)) >= 30
