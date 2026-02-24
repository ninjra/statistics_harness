import datetime as dt

import pandas as pd

from plugins.analysis_process_server_levers_v1.plugin import Plugin
from tests.conftest import make_context


def test_process_server_levers_emits_dynamic_add_server_and_assignment(run_dir):
    rows = []

    # Open period: lighter workload.
    for day in range(10, 15):
        base = dt.datetime(2026, 1, day, 8, 0, 0)
        for i in range(20):
            start = base + dt.timedelta(minutes=i)
            queue = start - dt.timedelta(seconds=20)
            rows.append(
                {
                    "PROCESS_CODE": "mailblast",
                    "HOST_NODE": "lane_a" if i % 2 == 0 else "lane_b",
                    "QUEUE_TS": queue.isoformat(),
                    "START_TS": start.isoformat(),
                    "END_TS": (start + dt.timedelta(seconds=35)).isoformat(),
                }
            )

    # Close period: heavier backlog and cross-lane contention.
    for day in range(20, 27):
        base = dt.datetime(2026, 1, day, 8, 0, 0)
        for i in range(120):
            start = base + dt.timedelta(minutes=i)
            queue = start - dt.timedelta(seconds=180 if i % 3 == 0 else 120)
            rows.append(
                {
                    "PROCESS_CODE": "mailblast",
                    "HOST_NODE": "lane_a" if i % 4 != 0 else "lane_b",
                    "QUEUE_TS": queue.isoformat(),
                    "START_TS": start.isoformat(),
                    "END_TS": (start + dt.timedelta(seconds=40)).isoformat(),
                }
            )
        for i in range(40):
            start = base + dt.timedelta(minutes=200 + i)
            queue = start - dt.timedelta(seconds=40)
            rows.append(
                {
                    "PROCESS_CODE": "closecore",
                    "HOST_NODE": "lane_a" if i % 2 == 0 else "lane_b",
                    "QUEUE_TS": queue.isoformat(),
                    "START_TS": start.isoformat(),
                    "END_TS": (start + dt.timedelta(seconds=50)).isoformat(),
                }
            )

    df = pd.DataFrame(rows)
    ctx = make_context(run_dir, df, settings={"min_process_runs": 10, "min_modeled_delta_hours": 0.01})
    result = Plugin().run(ctx)

    assert result.status == "ok"
    actionable = [
        f
        for f in result.findings
        if f.get("kind") == "actionable_ops_lever"
        and str(f.get("process_norm") or "") == "mailblast"
    ]
    assert actionable
    action_types = {str(f.get("action_type") or "") for f in actionable}
    assert "add_server" in action_types
    assert "tune_schedule" in action_types
    assert all(float(f.get("expected_delta_seconds") or 0.0) > 0.0 for f in actionable)
    assert all("delta_hours_accounting_month" in f for f in actionable)
    assert all("delta_hours_close_dynamic" in f for f in actionable)


def test_process_server_levers_returns_not_applicable_for_missing_required_columns(run_dir):
    df = pd.DataFrame([{"PROCESS_CODE": "mailblast", "QUEUE_TS": "2026-01-21T10:00:00"}])
    ctx = make_context(run_dir, df, settings={})
    result = Plugin().run(ctx)

    assert result.status == "ok"
    assert result.metrics.get("not_applicable_reason") == "missing_required_columns"
    assert any(str(item.get("decision") or "") == "not_applicable" for item in result.findings)
