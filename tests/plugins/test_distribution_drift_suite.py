import datetime as dt

import pandas as pd

from plugins.analysis_distribution_drift_suite.plugin import Plugin
from tests.conftest import make_context


def test_distribution_drift_detected(run_dir):
    n = 400
    df = pd.DataFrame(
        {
            "metric": [0.0] * 200 + [1.5] * 200,
            "category": ["a"] * 200 + ["b"] * 200,
            "ts": [dt.datetime(2026, 1, 1) + dt.timedelta(minutes=i) for i in range(n)],
        }
    )
    df["ts"] = df["ts"].astype(str)
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert result.findings
