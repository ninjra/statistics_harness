import datetime as dt

import pandas as pd

from plugins.analysis_concurrency_reconstruction.plugin import Plugin
from tests.conftest import make_context


def test_concurrency_reconstruction_peaks(run_dir):
    rows = [
        {
            "HOST": "h1",
            "START_DT": dt.datetime(2026, 1, 1, 0, 0, 0),
            "END_DT": dt.datetime(2026, 1, 1, 0, 10, 0),
        },
        {
            "HOST": "h1",
            "START_DT": dt.datetime(2026, 1, 1, 0, 5, 0),
            "END_DT": dt.datetime(2026, 1, 1, 0, 15, 0),
        },
        {
            "HOST": "h1",
            "START_DT": dt.datetime(2026, 1, 1, 0, 12, 0),
            "END_DT": dt.datetime(2026, 1, 1, 0, 20, 0),
        },
    ]
    df = pd.DataFrame(rows)
    df["START_DT"] = df["START_DT"].astype(str)
    df["END_DT"] = df["END_DT"].astype(str)

    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)

    assert result.status == "ok"
    findings = [
        f
        for f in result.findings
        if f.get("kind") == "concurrency_summary" and f.get("host") == "h1"
    ]
    assert findings
    assert findings[0]["peak_concurrency"] == 2
