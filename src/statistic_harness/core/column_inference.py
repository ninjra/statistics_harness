from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class TimestampInference:
    parse_ratio: float
    min_year: int | None
    max_year: int | None
    span_days: float | None
    unit: str | None
    score: float
    valid: bool
    notes: tuple[str, ...] = ()


def _sample_series(series: pd.Series, sample_size: int) -> pd.Series:
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    values = series.dropna()
    if values.empty:
        return values
    if len(values) > sample_size:
        return values.head(sample_size)
    return values


def _score_parsed(
    parsed: pd.Series,
    parse_ratio: float,
    numeric_ratio: float,
    sep_ratio: float,
    name_hint: str | None,
    unit: str | None,
) -> TimestampInference:
    if parsed.empty or parse_ratio < 0.6:
        return TimestampInference(0.0, None, None, None, unit, 0.0, False, ("low_parse",))

    years = parsed.dt.year.dropna()
    if years.empty:
        return TimestampInference(parse_ratio, None, None, None, unit, 0.0, False, ("no_years",))

    min_year = int(years.min())
    max_year = int(years.max())
    out_of_range = ((years < 1970) | (years > 2100)).mean()
    span_seconds = float((parsed.max() - parsed.min()).total_seconds())
    span_days = span_seconds / 86400.0
    unique_ts = parsed.nunique()

    score = parse_ratio * 2.0
    notes: list[str] = []

    if 1990 <= min_year <= 2100 and 1990 <= max_year <= 2100:
        score += 0.5
        notes.append("year_range_ok")
    elif min_year < 1970 or max_year > 2100:
        score -= 0.8
        notes.append("year_range_bad")

    if out_of_range > 0.1:
        score -= 0.8
        notes.append("out_of_range")
    if span_seconds < 60.0:
        score -= 0.6
        notes.append("span_too_small")
    elif span_days >= 30.0:
        score += 0.4
        notes.append("span_large")
    if unique_ts < 2:
        score -= 0.4
        notes.append("unique_ts_low")

    if sep_ratio >= 0.3:
        score += 0.3
        notes.append("separator_strings")

    if numeric_ratio >= 0.9 and sep_ratio < 0.1:
        score -= 0.6
        notes.append("numeric_no_sep")
        if unit:
            score += 0.2
            notes.append("numeric_unit")

    if name_hint:
        hint = name_hint.lower()
        if any(token in hint for token in ("time", "date", "timestamp", "dt")):
            score += 0.2
            notes.append("name_hint")

    valid = (
        score >= 1.2
        and parse_ratio >= 0.6
        and out_of_range <= 0.1
        and (span_seconds >= 60.0 or unique_ts >= 3)
    )

    return TimestampInference(
        parse_ratio=parse_ratio,
        min_year=min_year,
        max_year=max_year,
        span_days=span_days,
        unit=unit,
        score=score,
        valid=valid,
        notes=tuple(notes),
    )


def _parse_numeric_units(values: pd.Series, units: Iterable[str]) -> list[tuple[str, pd.Series]]:
    parsed: list[tuple[str, pd.Series]] = []
    for unit in units:
        try:
            series = pd.to_datetime(values, errors="coerce", unit=unit, utc=False)
        except (ValueError, OverflowError):
            continue
        parsed.append((unit, series))
    return parsed


def _parse_digit_format(values: pd.Series, fmt: str) -> pd.Series | None:
    try:
        return pd.to_datetime(values, errors="coerce", format=fmt, utc=False)
    except (ValueError, TypeError):
        return None


def infer_timestamp_series(
    series: pd.Series, name_hint: str | None = None, sample_size: int = 500
) -> TimestampInference:
    sample = _sample_series(series, sample_size)
    if sample.empty:
        return TimestampInference(0.0, None, None, None, None, 0.0, False, ("empty",))

    sample_str = sample.astype(str)
    numeric = pd.to_numeric(sample, errors="coerce")
    numeric_ratio = float(numeric.notna().mean())
    sep_ratio = float(sample_str.str.contains(r"[-/:T]", na=False).mean())

    best: TimestampInference | None = None

    if numeric_ratio >= 0.9:
        numeric_values = numeric.dropna()
        for unit, parsed in _parse_numeric_units(numeric_values, ("s", "ms", "us", "ns")):
            parse_ratio = float(parsed.notna().mean())
            candidate = _score_parsed(
                parsed, parse_ratio, numeric_ratio, sep_ratio, name_hint, unit
            )
            if best is None or candidate.score > best.score:
                best = candidate

        digit_8 = sample_str.str.fullmatch(r"\d{8}", na=False)
        if digit_8.mean() >= 0.8:
            parsed = _parse_digit_format(sample_str[digit_8], "%Y%m%d")
            if parsed is not None:
                candidate = _score_parsed(
                    parsed, float(parsed.notna().mean()), numeric_ratio, sep_ratio, name_hint, "ymd"
                )
                if best is None or candidate.score > best.score:
                    best = candidate

        digit_14 = sample_str.str.fullmatch(r"\d{14}", na=False)
        if digit_14.mean() >= 0.8:
            parsed = _parse_digit_format(sample_str[digit_14], "%Y%m%d%H%M%S")
            if parsed is not None:
                candidate = _score_parsed(
                    parsed, float(parsed.notna().mean()), numeric_ratio, sep_ratio, name_hint, "ymdhms"
                )
                if best is None or candidate.score > best.score:
                    best = candidate
    else:
        parsed = pd.to_datetime(sample, errors="coerce", utc=False)
        candidate = _score_parsed(
            parsed,
            float(parsed.notna().mean()),
            numeric_ratio,
            sep_ratio,
            name_hint,
            None,
        )
        best = candidate

    return best or TimestampInference(0.0, None, None, None, None, 0.0, False, ("no_candidate",))


def choose_timestamp_column(
    df: pd.DataFrame, candidates: Iterable[str], sample_size: int = 500
) -> str | None:
    best_col: str | None = None
    best_score = -1.0
    for col in candidates:
        if col not in df.columns:
            continue
        info = infer_timestamp_series(df[col], name_hint=str(col), sample_size=sample_size)
        if info.valid and info.score > best_score:
            best_col = col
            best_score = info.score
    return best_col


def is_valid_timestamp_series(
    series: pd.Series, name_hint: str | None = None, sample_size: int = 500
) -> bool:
    info = infer_timestamp_series(series, name_hint=name_hint, sample_size=sample_size)
    return info.valid
