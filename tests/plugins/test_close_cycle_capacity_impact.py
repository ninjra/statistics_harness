import datetime as dt

import pandas as pd

from plugins.analysis_close_cycle_capacity_impact.plugin import Plugin
from tests.conftest import make_context


def _make_rows(days, hosts, duration_sec, process):
    rows = []
    for day in days:
        for host in hosts:
            start = dt.datetime(2026, 1, day, 0, 0, 0)
            end = start + dt.timedelta(seconds=duration_sec)
            rows.append(
                {
                    "PROCESS_ID": process,
                    "LOCAL_MACHINE_ID": host,
                    "START_DT": start.isoformat(),
                    "END_DT": end.isoformat(),
                }
            )
    return rows


def _base_settings():
    return {
        "close_window_mode": "override",
        "close_cycle_start_day": 20,
        "close_cycle_end_day": 24,
        "bucket_size": "day",
        "min_bucket_rows": 1,
        "min_buckets_per_group": 2,
        "min_buckets_per_month": 1,
        "min_months": 1,
        "bootstrap_samples": 200,
        "alpha": 0.2,
    }


def test_capacity_impact_not_applicable_without_third_host(run_dir):
    rows = _make_rows(range(20, 25), ["h1", "h2"], 10, "PROC")
    df = pd.DataFrame(rows)

    settings = _base_settings()
    ctx = make_context(run_dir, df, settings)
    result = Plugin().run(ctx)

    assert result.status == "ok"
    assert result.findings
    assert all(item["decision"] == "not_applicable" for item in result.findings)


def test_capacity_impact_detects_reduction_with_third_host(run_dir):
    rows = []
    rows += _make_rows([20, 21, 22], ["h1", "h2", "h3"], 7, "PROC")
    rows += _make_rows([23, 24], ["h1", "h2"], 10, "PROC")
    df = pd.DataFrame(rows)

    settings = _base_settings()
    settings.update(
        {
            "target_reduction": 0.30,
            "tolerance": 0.2,
            "max_js_divergence": 0.9,
        }
    )
    ctx = make_context(run_dir, df, settings)
    result = Plugin().run(ctx)

    assert result.status == "ok"
    detected = [item for item in result.findings if item["decision"] == "detected"]
    assert detected


def test_capacity_impact_confounded_by_process_mix(run_dir):
    rows = []
    rows += _make_rows([20, 21, 22], ["h1", "h2", "h3"], 7, "PROC_A")
    rows += _make_rows([23, 24], ["h1", "h2"], 10, "PROC_B")
    df = pd.DataFrame(rows)

    settings = _base_settings()
    settings.update(
        {
            "target_reduction": 0.30,
            "tolerance": 0.2,
            "max_js_divergence": 0.1,
        }
    )
    ctx = make_context(run_dir, df, settings)
    result = Plugin().run(ctx)

    assert result.status == "ok"
    assert all(item["decision"] == "not_applicable" for item in result.findings)
