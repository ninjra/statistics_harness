import datetime as dt

import pandas as pd

from plugins.analysis_attribution.plugin import Plugin
from tests.conftest import make_context


def test_attribution_emits_dimension_findings(run_dir):
    rows = [
        {
            "process": "alpha",
            "module": "m1",
            "user": "u1",
            "queue_time": dt.datetime(2026, 1, 1, 8, 0, 0),
            "start_time": dt.datetime(2026, 1, 1, 8, 2, 0),
        },
        {
            "process": "alpha",
            "module": "m1",
            "user": "u2",
            "queue_time": dt.datetime(2026, 1, 1, 9, 0, 0),
            "start_time": dt.datetime(2026, 1, 1, 9, 3, 0),
        },
        {
            "process": "beta",
            "module": "m2",
            "user": "u3",
            "queue_time": dt.datetime(2026, 1, 1, 10, 0, 0),
            "start_time": dt.datetime(2026, 1, 1, 10, 0, 10),
        },
    ]
    df = pd.DataFrame(rows)
    for col in ["queue_time", "start_time"]:
        df[col] = df[col].astype(str)

    ctx = make_context(run_dir, df, {"wait_threshold_seconds": 60})
    result = Plugin().run(ctx)

    assert result.status == "ok"
    findings = [
        f
        for f in result.findings
        if f.get("kind") == "attribution" and f.get("dimension") == "process"
    ]
    assert findings
    assert findings[0]["key"] == "alpha"
