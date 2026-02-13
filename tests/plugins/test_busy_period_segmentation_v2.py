import datetime as dt
import json

import pandas as pd

from plugins.analysis_busy_period_segmentation_v2.plugin import Plugin
from tests.conftest import make_context


def test_busy_period_segmentation_merges_intervals(run_dir):
    rows = [
        {
            "process": "alpha",
            "host": "host1",
            "queue_dt": dt.datetime(2026, 1, 1, 8, 0, 0),
            "start_dt": dt.datetime(2026, 1, 1, 8, 2, 0),
        },
        {
            "process": "alpha",
            "host": "host1",
            "queue_dt": dt.datetime(2026, 1, 1, 8, 1, 0),
            "start_dt": dt.datetime(2026, 1, 1, 8, 4, 0),
        },
        {
            "process": "beta",
            "host": "host2",
            "queue_dt": dt.datetime(2026, 1, 1, 9, 0, 0),
            "start_dt": dt.datetime(2026, 1, 1, 9, 2, 0),
        },
    ]
    df = pd.DataFrame(rows)
    df["queue_dt"] = df["queue_dt"].astype(str)
    df["start_dt"] = df["start_dt"].astype(str)

    ctx = make_context(
        run_dir,
        df,
        {"wait_threshold_seconds": 60, "gap_tolerance_seconds": 60},
    )
    result = Plugin().run(ctx)
    assert result.status == "ok"

    payload = json.loads(
        (run_dir / "artifacts" / "analysis_busy_period_segmentation_v2" / "busy_periods.json").read_text(
            encoding="utf-8"
        )
    )
    busy_periods = payload.get("busy_periods") or []
    assert len(busy_periods) == 2
    assert float(busy_periods[0].get("total_over_threshold_wait_sec")) == 180.0
