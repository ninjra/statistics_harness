from __future__ import annotations

import pandas as pd

from statistic_harness.core.close_cycle import (
    baseline_target_spillover_masks,
    compute_close_month,
)


def test_compute_close_month_shifts_days_leq_end_to_previous_month() -> None:
    ts = pd.to_datetime(["2026-01-05T00:00:00", "2026-01-20T00:00:00"])
    close_month = compute_close_month(ts, baseline_close_end_day=5)
    assert list(close_month) == ["2025-12", "2026-01"]


def test_spillover_masks_baseline_wrap_vs_target_no_wrap() -> None:
    ts = pd.to_datetime(
        [
            "2026-01-02T00:00:00",  # baseline close (20..5), spillover vs target (20..31)
            "2026-01-25T00:00:00",  # baseline close, in target
            "2026-01-10T00:00:00",  # open
        ]
    )
    base, target, spill = baseline_target_spillover_masks(
        ts, baseline_close_start_day=20, baseline_close_end_day=5, target_close_end_day=31
    )
    assert base.tolist() == [True, True, False]
    assert target.tolist() == [False, True, False]
    assert spill.tolist() == [True, False, False]

