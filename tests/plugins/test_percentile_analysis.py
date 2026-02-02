import datetime as dt

import pandas as pd

from plugins.analysis_percentile_analysis.plugin import Plugin
from tests.conftest import make_context


def test_percentile_analysis_outputs_stats(run_dir):
    rows = [
        {
            "process": "alpha",
            "module": "m1",
            "queue_time": dt.datetime(2026, 1, 1, 8, 0, 0),
            "start_time": dt.datetime(2026, 1, 1, 8, 0, 30),
            "end_time": dt.datetime(2026, 1, 1, 8, 2, 0),
        },
        {
            "process": "alpha",
            "module": "m1",
            "queue_time": dt.datetime(2026, 1, 1, 9, 0, 0),
            "start_time": dt.datetime(2026, 1, 1, 9, 1, 0),
            "end_time": dt.datetime(2026, 1, 1, 9, 4, 0),
        },
        {
            "process": "beta",
            "module": "m2",
            "queue_time": dt.datetime(2026, 1, 1, 10, 0, 0),
            "start_time": dt.datetime(2026, 1, 1, 10, 0, 10),
            "end_time": dt.datetime(2026, 1, 1, 10, 0, 40),
        },
    ]
    df = pd.DataFrame(rows)
    for col in ["queue_time", "start_time", "end_time"]:
        df[col] = df[col].astype(str)

    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)

    assert result.status == "ok"
    stats = [f for f in result.findings if f.get("kind") == "percentile_stats"]
    assert stats
    assert any(f.get("process") == "alpha" for f in stats)
    assert all("eligible_wait_p95" in f for f in stats)
