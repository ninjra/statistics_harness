from __future__ import annotations

import datetime as dt

import pandas as pd

from conftest import make_context
from statistic_harness.core.report import _build_known_issue_recommendations


def test_known_issue_close_cycle_contention_uses_process_removal_fallback(run_dir):
    rows = []
    for day in range(10, 16):
        date = dt.datetime(2026, 1, day, 8, 0, 0)
        for idx in range(12):
            rows.append(
                {
                    "PROCESS_ID": "other_job",
                    "QUEUE_DT": (date + dt.timedelta(minutes=idx)).isoformat(),
                    "START_DT": (date + dt.timedelta(minutes=idx, seconds=5)).isoformat(),
                    "END_DT": (date + dt.timedelta(minutes=idx, seconds=65)).isoformat(),
                }
            )
        for idx in range(2):
            rows.append(
                {
                    "PROCESS_ID": "qemail",
                    "QUEUE_DT": (date + dt.timedelta(minutes=200 + idx)).isoformat(),
                    "START_DT": (date + dt.timedelta(minutes=200 + idx, seconds=5)).isoformat(),
                    "END_DT": (date + dt.timedelta(minutes=200 + idx, seconds=25)).isoformat(),
                }
            )
    df = pd.DataFrame(rows)
    ctx = make_context(run_dir, df, settings={})

    report = {
        "lineage": {"input": {"dataset_version_id": ctx.dataset_version_id}},
        "plugins": {
            "analysis_close_cycle_contention": {
                "findings": [],
            }
        },
        "known_issues": {
            "expected_findings": [
                {
                    "plugin_id": "analysis_close_cycle_contention",
                    "kind": "close_cycle_contention",
                    "where": {"process": "qemail"},
                    "min_count": 1,
                    "max_count": 1,
                    "title": "known_issue_qemail_schedule",
                }
            ]
        },
    }

    payload = _build_known_issue_recommendations(report, ctx.storage)
    items = payload.get("items") or []
    assert items
    item = items[0]
    assert item.get("status") == "confirmed"
    assert int(item.get("observed_count") or 0) >= 1
    assert float(item.get("modeled_general_percent") or 0.0) > 0.0
