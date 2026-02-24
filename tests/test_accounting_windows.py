from __future__ import annotations

from datetime import datetime

from statistic_harness.core.accounting_windows import (
    assign_accounting_month,
    infer_accounting_windows_from_timestamps,
    infer_roll_day_from_timestamps,
    parse_accounting_month_from_params,
)


def test_parse_accounting_month_from_params_prefers_explicit_key() -> None:
    params = {
        "run mode": "normal",
        "posting period": "2026-01",
        "accounting month": "202602",
    }
    out = parse_accounting_month_from_params(params)
    assert out == datetime(2026, 2, 1)


def test_parse_accounting_month_from_params_zero_shot_fallback() -> None:
    params = {
        "foo": "x",
        "bar_period_code": "2025-11",
    }
    out = parse_accounting_month_from_params(params)
    assert out == datetime(2025, 11, 1)


def test_parse_accounting_month_ignores_invalid_year_zero_tokens() -> None:
    params = {
        "weird_token": "000000",
        "period_code": "2026-01",
    }
    out = parse_accounting_month_from_params(params)
    assert out == datetime(2026, 1, 1)


def test_parse_accounting_month_ignores_far_reference_months() -> None:
    params = {
        "period": "1900",
    }
    out = parse_accounting_month_from_params(
        params, reference_ts=datetime(2025, 8, 30, 12, 0, 0)
    )
    assert out is None


def test_parse_accounting_month_prefers_near_reference_month() -> None:
    params = {
        "period_a": "1900",
        "period_b": "2025-08",
    }
    out = parse_accounting_month_from_params(
        params, reference_ts=datetime(2025, 8, 30, 12, 0, 0)
    )
    assert out == datetime(2025, 8, 1)


def test_infer_roll_day_and_assign_accounting_month() -> None:
    timestamps = [
        "2026-01-31 23:00:00",
        "2026-02-01 01:00:00",
        "2026-02-03 09:00:00",
        "2026-02-06 13:00:00",
        "2026-02-28 22:00:00",
        "2026-03-02 07:00:00",
        "2026-03-06 07:00:00",
    ]
    roll_day = infer_roll_day_from_timestamps(timestamps, min_day=2, max_day=8, default_day=5)
    assert 2 <= roll_day <= 8
    cohort = assign_accounting_month(timestamps, roll_day=roll_day).tolist()
    assert len(cohort) == len(timestamps)
    assert all(isinstance(v, str) and len(v) == 7 for v in cohort if v)


def test_infer_accounting_windows_from_timestamps_emits_month_windows() -> None:
    timestamps = [
        "2026-01-25 01:00:00",
        "2026-01-30 05:00:00",
        "2026-02-03 10:00:00",
        "2026-02-26 09:00:00",
        "2026-03-03 12:00:00",
    ]
    windows = infer_accounting_windows_from_timestamps(timestamps, roll_day=5)
    assert windows
    assert windows[0].accounting_month
    assert windows[0].accounting_month_start_ts is not None
    assert windows[0].close_dynamic_start_ts is not None
