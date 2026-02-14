import csv
import datetime as dt
from pathlib import Path

import pandas as pd

from plugins.analysis_close_cycle_window_resolver.plugin import Plugin
from tests.conftest import make_context


def _write_backtrack_csv(run_dir: Path) -> None:
    out_dir = run_dir / "artifacts" / "analysis_close_cycle_start_backtrack_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "close_windows.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "accounting_month",
                "roll_timestamp",
                "close_start_default",
                "close_end_default",
                "close_start_dynamic",
                "close_end_dynamic",
                "close_window_days_dynamic",
                "close_end_delta_days",
                "source",
                "confidence",
                "fallback_reason",
                "signature_processes",
                "signature_coverage",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "accounting_month": "2026-02",
                "roll_timestamp": "2026-02-02T08:00:00",
                "close_start_default": "2026-01-20T00:00:00",
                "close_end_default": "2026-02-05T00:00:00",
                "close_start_dynamic": "2026-01-18T00:00:00",
                "close_end_dynamic": "2026-02-02T08:00:00",
                "close_window_days_dynamic": "15.33",
                "close_end_delta_days": "-2.67",
                "source": "backtracked_signature",
                "confidence": "0.89",
                "fallback_reason": "",
                "signature_processes": "close_a,close_b",
                "signature_coverage": "0.42",
            }
        )


def test_resolver_prefers_backtracked_start_when_available(run_dir):
    _write_backtrack_csv(run_dir)

    rows = [
        {
            "PROCESS_ID": "roll_marker",
            "PARAM_DESCR_LIST": "Accounting Month(dt)=2026-02-01",
            "START_DT": dt.datetime(2026, 2, 2, 8, 0, 0).isoformat(),
            "QUEUE_DT": dt.datetime(2026, 2, 2, 7, 55, 0).isoformat(),
        },
        {
            "PROCESS_ID": "roll_marker",
            "PARAM_DESCR_LIST": "Accounting Month(dt)=2026-02-01",
            "START_DT": dt.datetime(2026, 2, 1, 8, 0, 0).isoformat(),
            "QUEUE_DT": dt.datetime(2026, 2, 1, 7, 55, 0).isoformat(),
        },
    ]
    df = pd.DataFrame(rows)
    ctx = make_context(
        run_dir,
        df,
        {
            "close_start_day": 20,
            "close_end_day": 5,
            "indicator_min_months": 1,
        },
    )

    result = Plugin().run(ctx)
    assert result.status == "ok"

    resolved = [f for f in result.findings if f.get("kind") == "close_cycle_window_resolved"]
    assert resolved
    first = resolved[0]
    assert pd.to_datetime(first.get("close_start_dynamic")).day == 18
    assert str(first.get("source") or "") == "backtracked_signature"
    assert float(first.get("confidence") or 0.0) >= 0.89
