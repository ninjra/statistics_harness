import datetime as dt

import pandas as pd

from plugins.analysis_close_cycle_start_backtrack_v1.plugin import Plugin
from tests.conftest import make_context


def _acct_param(year: int, month: int) -> str:
    return f"Accounting Month(dt)={year:04d}-{month:02d}-01"


def _prev_month(year: int, month: int) -> tuple[int, int]:
    if month == 1:
        return year - 1, 12
    return year, month - 1


def _build_month_rows(
    *,
    roll_ts: dt.datetime,
    close_start_day: int,
    include_noise: bool = True,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    month_value = _acct_param(roll_ts.year, roll_ts.month)
    prev_year, prev_month = _prev_month(roll_ts.year, roll_ts.month)
    prev_month_value = _acct_param(prev_year, prev_month)

    # Roll marker event.
    rows.append(
        {
            "PROCESS_ID": "roll_marker",
            "PARAM_DESCR_LIST": month_value,
            "START_DT": roll_ts.isoformat(),
            "QUEUE_DT": (roll_ts - dt.timedelta(minutes=5)).isoformat(),
        }
    )

    # Repeating close processes before roll; these should define dynamic close start.
    current = dt.datetime(roll_ts.year, roll_ts.month, 1, 9, 0, 0)
    while current.date() >= dt.date(prev_year, prev_month, close_start_day):
        for proc in ("close_a", "close_b"):
            rows.append(
                {
                    "PROCESS_ID": proc,
                    "PARAM_DESCR_LIST": prev_month_value,
                    "START_DT": current.isoformat(),
                    "QUEUE_DT": (current - dt.timedelta(minutes=15)).isoformat(),
                }
            )
        current -= dt.timedelta(days=1)

    if include_noise:
        noise_ts = roll_ts - dt.timedelta(days=20)
        rows.append(
            {
                "PROCESS_ID": "noise_job",
                "PARAM_DESCR_LIST": prev_month_value,
                "START_DT": noise_ts.isoformat(),
                "QUEUE_DT": (noise_ts - dt.timedelta(minutes=10)).isoformat(),
            }
        )

    return rows


def test_backtrack_infers_non_default_start(run_dir):
    rows: list[dict[str, str]] = []
    rows.extend(_build_month_rows(roll_ts=dt.datetime(2026, 2, 2, 8, 0, 0), close_start_day=18))
    rows.extend(_build_month_rows(roll_ts=dt.datetime(2026, 3, 2, 8, 0, 0), close_start_day=18))
    rows.extend(_build_month_rows(roll_ts=dt.datetime(2026, 4, 2, 8, 0, 0), close_start_day=18))

    df = pd.DataFrame(rows)
    ctx = make_context(
        run_dir,
        df,
        {
            "close_start_day": 20,
            "lookback_days": 21,
            "min_signature_months": 3,
            "min_streak_days": 3,
            "max_gap_days": 1,
        },
    )
    result = Plugin().run(ctx)

    assert result.status == "ok"
    assert int(result.metrics.get("months_detected") or 0) >= 3
    assert int(result.metrics.get("months_backtracked") or 0) >= 1

    windows = [f for f in result.findings if f.get("kind") == "close_cycle_start_backtracked"]
    assert windows
    backtracked = [w for w in windows if str(w.get("source") or "").startswith("backtracked")]
    assert backtracked
    start_days = [pd.to_datetime(w.get("close_start_dynamic")).day for w in backtracked]
    assert any(day == 18 for day in start_days)


def test_backtrack_falls_back_without_signatures(run_dir):
    rows = _build_month_rows(roll_ts=dt.datetime(2026, 2, 2, 8, 0, 0), close_start_day=18, include_noise=False)
    df = pd.DataFrame(rows)
    ctx = make_context(
        run_dir,
        df,
        {
            "close_start_day": 20,
            "lookback_days": 21,
            "min_signature_months": 3,
            "min_streak_days": 3,
            "max_gap_days": 1,
        },
    )
    result = Plugin().run(ctx)

    assert result.status == "ok"
    assert int(result.metrics.get("months_detected") or 0) >= 1
    assert int(result.metrics.get("months_fallback") or 0) >= 1

    windows = [f for f in result.findings if f.get("kind") == "close_cycle_start_backtracked"]
    assert windows
    fallback = [w for w in windows if w.get("fallback_reason")]
    assert fallback
    assert pd.to_datetime(fallback[0].get("close_start_dynamic")).day == 20
