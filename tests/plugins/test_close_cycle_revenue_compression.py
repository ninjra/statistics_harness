import datetime as dt

import pandas as pd

from plugins.analysis_close_cycle_revenue_compression.plugin import Plugin
from tests.conftest import make_context


def _make_rows(days, process, queue_wait_days=7, service_days=1):
    rows = []
    for day in days:
        queue = dt.datetime(2026, 1, day, 0, 0, 0)
        start = queue + dt.timedelta(days=queue_wait_days)
        end = start + dt.timedelta(days=service_days)
        rows.append(
            {
                "PROCESS": process,
                "QUEUE_DT": queue.isoformat(),
                "START_DT": start.isoformat(),
                "END_DT": end.isoformat(),
                "HOST": "h1",
            }
        )
    return rows


def test_revenue_compression_model_required_scale(run_dir):
    rows = []
    rows += _make_rows([20, 21], "revenue")
    df = pd.DataFrame(rows)

    settings = {
        "close_window_mode": "override",
        "close_cycle_start_day": 20,
        "close_cycle_end_day": 27,
        "target_days": 7,
        "max_scale": 5,
        "revenue_process_patterns": ["revenue"],
        "min_month_rows": 1,
    }
    ctx = make_context(run_dir, df, settings)
    result = Plugin().run(ctx)

    assert result.status == "ok"
    assert result.findings
    finding = result.findings[0]
    assert finding["decision"] == "modeled"
    required_scale = finding.get("worst_month_required_scale")
    assert required_scale is not None
    assert 1.15 < required_scale < 1.2


def test_revenue_compression_not_applicable_when_no_match(run_dir):
    rows = []
    rows += _make_rows([20, 21], "other")
    df = pd.DataFrame(rows)

    settings = {
        "close_window_mode": "override",
        "close_cycle_start_day": 20,
        "close_cycle_end_day": 27,
        "target_days": 7,
        "max_scale": 5,
        "revenue_process_patterns": ["revenue"],
    }
    ctx = make_context(run_dir, df, settings)
    result = Plugin().run(ctx)

    assert result.status == "ok"
    assert result.findings
    assert result.findings[0]["decision"] == "not_applicable"
