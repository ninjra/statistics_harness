import datetime as dt

import pandas as pd

from plugins.analysis_close_cycle_capacity_model.plugin import Plugin
from tests.conftest import make_context


def _make_rows(days, hosts, process="PROC"):
    rows = []
    for day in days:
        for host in hosts:
            start = dt.datetime(2026, 1, day, 0, 0, 0)
            end = start + dt.timedelta(hours=1)
            queue = start - dt.timedelta(hours=2)
            eligible = start - dt.timedelta(hours=1)
            rows.append(
                {
                    "PROCESS_ID": process,
                    "LOCAL_MACHINE_ID": host,
                    "START_DT": start.isoformat(),
                    "END_DT": end.isoformat(),
                    "QUEUE_DT": queue.isoformat(),
                    "ELIGIBLE_DT": eligible.isoformat(),
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
        "min_months": 1,
        "baseline_host_count": 2,
        "added_hosts": 1,
        "target_reduction": 0.30,
        "tolerance": 0.15,
    }


def test_capacity_model_emits_modeled_findings(run_dir):
    rows = _make_rows(range(20, 25), ["h1", "h2"])
    df = pd.DataFrame(rows)

    ctx = make_context(run_dir, df, _base_settings())
    result = Plugin().run(ctx)

    assert result.status == "ok"
    modeled = [
        item
        for item in result.findings
        if item.get("decision") == "modeled" and item.get("metric_type") == "queue_to_end"
    ]
    assert modeled
    assert any(item.get("target_met") for item in modeled)


def test_capacity_model_handles_missing_queue_columns(run_dir):
    rows = []
    for day in range(20, 25):
        for host in ["h1", "h2"]:
            start = dt.datetime(2026, 1, day, 0, 0, 0)
            end = start + dt.timedelta(hours=1)
            rows.append(
                {
                    "PROCESS_ID": "PROC",
                    "LOCAL_MACHINE_ID": host,
                    "START_DT": start.isoformat(),
                    "END_DT": end.isoformat(),
                }
            )
    df = pd.DataFrame(rows)

    ctx = make_context(run_dir, df, _base_settings())
    result = Plugin().run(ctx)

    assert result.status == "ok"
    not_applicable = [
        item
        for item in result.findings
        if item.get("metric_type") in {"queue_to_end", "eligible_to_end"}
    ]
    assert not_applicable
    assert all(item.get("decision") == "not_applicable" for item in not_applicable)

    ttc_modeled = [
        item
        for item in result.findings
        if item.get("metric_type") == "ttc" and item.get("decision") == "modeled"
    ]
    assert ttc_modeled
