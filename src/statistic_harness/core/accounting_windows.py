from __future__ import annotations

"""Canonical accounting-window models and helpers.

This module is the shared home for accounting-month window contracts.
"""

import calendar
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import pandas as pd

@dataclass(frozen=True)
class AccountingWindow:
    accounting_month: str
    accounting_month_start_ts: datetime | None
    accounting_month_end_ts: datetime | None
    close_static_start_ts: datetime | None
    close_static_end_ts: datetime | None
    close_dynamic_start_ts: datetime | None
    close_dynamic_end_ts: datetime | None
    source_plugin: str | None
    confidence: float | None
    fallback_reason: str | None


DEFAULT_CLOSE_WINDOW_PLUGIN_IDS: tuple[str, ...] = (
    "analysis_close_cycle_window_resolver",
    "analysis_close_cycle_start_backtrack_v1",
    "analysis_dynamic_close_detection",
)

_MONTH_KEY_HINTS: tuple[str, ...] = (
    "accounting month",
    "acct month",
    "accounting_month",
    "acct_month",
    "accounting period",
    "acct period",
    "fiscal month",
    "fiscal period",
    "period",
)

_MONTH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(20\d{2})(0[1-9]|1[0-2])\s*$"),
    re.compile(r"^\s*(20\d{2})[-_/](0[1-9]|1[0-2])\s*$"),
    re.compile(r"^\s*(20\d{2})(0[1-9]|1[0-2])([0-3]\d)\s*$"),
    re.compile(r"^\s*(0[1-9]|1[0-2])[-_/](20\d{2})\s*$"),
)


def parse_accounting_month_value(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None

    for pat in _MONTH_PATTERNS:
        m = pat.match(text)
        if not m:
            continue
        if pat.pattern.startswith("^\\s*(0[1-9]|1[0-2])"):
            month = int(m.group(1))
            year = int(m.group(2))
        else:
            year = int(m.group(1))
            month = int(m.group(2))
        if 1 <= month <= 12 and 1900 <= year <= 2100:
            return datetime(year, month, 1)

    for fmt in ("%b %Y", "%B %Y", "%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            dt = datetime.strptime(text, fmt)
            return datetime(dt.year, dt.month, 1)
        except ValueError:
            continue

    try:
        parsed = pd.to_datetime(text, errors="coerce")
    except Exception:
        parsed = None
    if parsed is None or pd.isna(parsed):
        return None
    year = int(parsed.year)
    month = int(parsed.month)
    if year < 1900 or year > 2100 or month < 1 or month > 12:
        return None
    try:
        return datetime(year, month, 1)
    except ValueError:
        return None


def parse_accounting_month_from_params(
    params: Mapping[str, str],
    *,
    raw_params_text: str | None = None,
    reference_ts: datetime | None = None,
) -> datetime | None:
    best_month: datetime | None = None
    best_score = float("-inf")
    ref = reference_ts if isinstance(reference_ts, datetime) else None
    ref_month = datetime(ref.year, ref.month, 1) if ref is not None else None

    def _score(key: str, month: datetime) -> float:
        key_l = key.strip().lower()
        score = 0.0
        if key_l in _MONTH_KEY_HINTS:
            score += 6.0
        elif any(tok in key_l for tok in ("account", "acct", "fiscal", "period", "month")):
            score += 3.0
        else:
            score += 1.0
        if ref_month is not None:
            delta = abs((month.year - ref_month.year) * 12 + (month.month - ref_month.month))
            score += max(0.0, 3.0 - float(delta))
        return score

    for key, value in params.items():
        month = parse_accounting_month_value(value)
        if month is None:
            continue
        s = _score(str(key), month)
        if s > best_score:
            best_score = s
            best_month = month

    if best_month is not None:
        return best_month

    raw = str(raw_params_text or "").strip()
    if raw:
        token_matches = re.findall(r"(20\d{2}[01]\d|20\d{2}[-_/][01]\d|[01]\d[-_/]20\d{2})", raw)
        for token in token_matches:
            month = parse_accounting_month_value(token)
            if month is not None:
                return month
    return None


def assign_accounting_month(
    timestamps: Any,
    *,
    roll_day: int,
) -> pd.Series:
    ts = pd.to_datetime(timestamps, errors="coerce", utc=False)
    if not isinstance(ts, pd.Series):
        ts = pd.Series(ts)
    days = ts.dt.day
    shifted = ts - pd.DateOffset(months=1)
    month = ts.dt.to_period("M").astype(str)
    prev = shifted.dt.to_period("M").astype(str)
    return month.where(~(days <= int(roll_day)), prev)


def infer_roll_day_from_timestamps(
    timestamps: Any,
    *,
    min_day: int = 2,
    max_day: int = 10,
    default_day: int = 5,
) -> int:
    ts = pd.to_datetime(timestamps, errors="coerce", utc=False)
    if not isinstance(ts, pd.Series):
        ts = pd.Series(ts)
    ts = ts.dropna()
    if ts.empty:
        return int(default_day)

    best_day = int(default_day)
    best_cv = float("inf")
    for day in range(int(min_day), int(max_day) + 1):
        cohort = assign_accounting_month(ts, roll_day=day)
        counts = cohort.value_counts(dropna=True)
        if counts.empty:
            continue
        mean = float(counts.mean())
        if mean <= 0.0:
            continue
        std = float(counts.std(ddof=0))
        cv = std / mean
        if cv < best_cv:
            best_cv = cv
            best_day = day
    return int(best_day)


def _month_bounds(month_key: str) -> tuple[datetime | None, datetime | None]:
    try:
        start = datetime.strptime(month_key, "%Y-%m")
    except ValueError:
        return None, None
    _, last_day = calendar.monthrange(start.year, start.month)
    end = datetime(start.year, start.month, last_day, 23, 59, 59)
    return start, end


def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        ts = pd.to_datetime(text, errors="coerce")
    except Exception:
        ts = None
    if ts is None or pd.isna(ts):
        return None
    return ts.to_pydatetime()


def _window_from_row(row: dict[str, Any], plugin_id: str) -> AccountingWindow | None:
    month_key = str(row.get("accounting_month") or "").strip()
    if not month_key:
        return None
    month_start, month_end = _month_bounds(month_key)
    if month_start is None:
        return None
    static_start = _parse_dt(
        row.get("close_start_default")
        or row.get("close_start")
        or row.get("close_start_static")
    )
    static_end = _parse_dt(
        row.get("close_end_default")
        or row.get("close_end")
        or row.get("close_end_static")
    )
    dynamic_start = _parse_dt(
        row.get("close_start_dynamic")
        or row.get("close_start")
        or row.get("close_start_default")
    )
    dynamic_end = _parse_dt(
        row.get("close_end_dynamic")
        or row.get("close_end")
        or row.get("close_end_default")
    )
    source = str(row.get("source") or plugin_id).strip() or plugin_id
    fallback_reason = str(row.get("fallback_reason") or "").strip() or None
    confidence: float | None
    try:
        confidence = float(row.get("confidence")) if row.get("confidence") is not None else None
    except (TypeError, ValueError):
        confidence = None
    return AccountingWindow(
        accounting_month=month_key,
        accounting_month_start_ts=month_start,
        accounting_month_end_ts=month_end,
        close_static_start_ts=static_start,
        close_static_end_ts=static_end,
        close_dynamic_start_ts=dynamic_start,
        close_dynamic_end_ts=dynamic_end,
        source_plugin=source,
        confidence=confidence,
        fallback_reason=fallback_reason,
    )


def load_accounting_windows_from_run(
    run_dir: Path,
    *,
    plugin_ids: Iterable[str] | None = None,
) -> list[AccountingWindow]:
    ordered = list(plugin_ids or DEFAULT_CLOSE_WINDOW_PLUGIN_IDS)
    windows_by_month: dict[str, AccountingWindow] = {}
    for plugin_id in ordered:
        path = run_dir / "artifacts" / plugin_id / "close_windows.csv"
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    window = _window_from_row(row, plugin_id)
                    if window is None:
                        continue
                    current = windows_by_month.get(window.accounting_month)
                    if current is None:
                        windows_by_month[window.accounting_month] = window
                        continue
                    cur_conf = float(current.confidence or 0.0)
                    new_conf = float(window.confidence or 0.0)
                    if new_conf >= cur_conf:
                        windows_by_month[window.accounting_month] = window
        except Exception:
            continue
    out = list(windows_by_month.values())
    out.sort(key=lambda x: x.accounting_month)
    return out


def infer_accounting_windows_from_timestamps(
    timestamps: Any,
    *,
    roll_day: int | None = None,
) -> list[AccountingWindow]:
    ts = pd.to_datetime(timestamps, errors="coerce", utc=False)
    if not isinstance(ts, pd.Series):
        ts = pd.Series(ts)
    ts = ts.dropna()
    if ts.empty:
        return []
    inferred_roll_day = int(roll_day) if isinstance(roll_day, int) else infer_roll_day_from_timestamps(ts)
    cohort = assign_accounting_month(ts, roll_day=inferred_roll_day)
    windows: list[AccountingWindow] = []
    for month_key in sorted(set(str(v) for v in cohort.dropna().unique())):
        month_start, month_end = _month_bounds(month_key)
        if month_start is None or month_end is None:
            continue
        prev_year = month_start.year
        prev_month = month_start.month - 1
        if prev_month <= 0:
            prev_month = 12
            prev_year -= 1
        close_static_start = datetime(prev_year, prev_month, 20)
        close_static_end = month_start + timedelta(days=max(0, inferred_roll_day - 1), hours=23, minutes=59, seconds=59)
        windows.append(
            AccountingWindow(
                accounting_month=month_key,
                accounting_month_start_ts=month_start,
                accounting_month_end_ts=month_end,
                close_static_start_ts=close_static_start,
                close_static_end_ts=close_static_end,
                close_dynamic_start_ts=close_static_start,
                close_dynamic_end_ts=close_static_end,
                source_plugin="inferred_from_timestamps",
                confidence=0.35,
                fallback_reason="no_accounting_month_markers_detected",
            )
        )
    return windows


def window_ranges(
    windows: Iterable[AccountingWindow],
    *,
    kind: str,
) -> list[tuple[datetime, datetime]]:
    out: list[tuple[datetime, datetime]] = []
    for w in windows:
        if kind == "accounting_month":
            start = w.accounting_month_start_ts
            end = w.accounting_month_end_ts
        elif kind == "close_static":
            start = w.close_static_start_ts
            end = w.close_static_end_ts
        else:
            start = w.close_dynamic_start_ts
            end = w.close_dynamic_end_ts
        if isinstance(start, datetime) and isinstance(end, datetime) and end >= start:
            out.append((start, end))
    return out
