import datetime as dt

import pandas as pd

from plugins.analysis_tail_isolation.plugin import Plugin
from tests.conftest import make_context


def test_tail_isolation_process_dimension(run_dir):
    rows = [
        {
            "process": "alpha",
            "queue_time": dt.datetime(2026, 1, 1, 8, 0, 0),
            "start_time": dt.datetime(2026, 1, 1, 8, 2, 0),
        },
        {
            "process": "alpha",
            "queue_time": dt.datetime(2026, 1, 1, 9, 0, 0),
            "start_time": dt.datetime(2026, 1, 1, 9, 3, 0),
        },
        {
            "process": "beta",
            "queue_time": dt.datetime(2026, 1, 1, 10, 0, 0),
            "start_time": dt.datetime(2026, 1, 1, 10, 0, 30),
        },
    ]
    df = pd.DataFrame(rows)
    df["queue_time"] = df["queue_time"].astype(str)
    df["start_time"] = df["start_time"].astype(str)

    ctx = make_context(run_dir, df, {"wait_threshold_seconds": 60})
    result = Plugin().run(ctx)

    assert result.status == "ok"
    assert result.metrics["tail_rows"] > 0
    findings = [
        f
        for f in result.findings
        if f.get("kind") == "tail_isolation" and f.get("dimension") == "process"
    ]
    assert findings
    assert findings[0]["key"] == "alpha"
