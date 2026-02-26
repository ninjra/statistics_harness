from __future__ import annotations

import pandas as pd
import pytest

from statistic_harness.core.payout_report import _safe_to_datetime_series, build_payout_report


def test_safe_to_datetime_series_handles_arrow_backed_series() -> None:
    pytest.importorskip("pyarrow")
    series = pd.Series(
        ["2026-01-01 10:00:00.000", "2026-01-01 10:00:05.000", None],
        dtype="string[pyarrow]",
    )
    parsed = _safe_to_datetime_series(series)
    assert int(parsed.notna().sum()) == 2


def test_build_payout_report_works_with_arrow_backed_columns() -> None:
    pytest.importorskip("pyarrow")
    df = pd.DataFrame(
        {
            "PROCESS_ID": pd.Series(["JBPREPAY", "JBPREPAY", "OTHER"], dtype="string[pyarrow]"),
            "PARAM_DESCR_LIST": pd.Series(["A", "B", "C"], dtype="string[pyarrow]"),
            "QUEUE_DT": pd.Series(
                ["2026-01-01 10:00:00.000", "2026-01-01 10:01:00.000", "2026-01-01 10:02:00.000"],
                dtype="string[pyarrow]",
            ),
            "START_DT": pd.Series(
                ["2026-01-01 10:00:05.000", "2026-01-01 10:01:05.000", "2026-01-01 10:02:05.000"],
                dtype="string[pyarrow]",
            ),
            "END_DT": pd.Series(
                ["2026-01-01 10:00:20.000", "2026-01-01 10:01:20.000", "2026-01-01 10:02:20.000"],
                dtype="string[pyarrow]",
            ),
        }
    )

    report = build_payout_report(df)
    assert report.get("status") == "ok"
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    assert int(metrics.get("payout_rows") or 0) == 2
    assert int(metrics.get("unique_payout_keys") or 0) == 2
