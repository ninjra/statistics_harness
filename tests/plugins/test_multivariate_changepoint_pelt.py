import datetime as dt

import pandas as pd

from plugins.analysis_multivariate_changepoint_pelt.plugin import Plugin
from tests.conftest import make_context


def test_changepoint_detected(run_dir):
    n = 300
    df = pd.DataFrame(
        {
            "a": [0.0] * 100 + [2.0] * 100 + [0.0] * 100,
            "b": [0.0] * 100 + [1.5] * 100 + [0.0] * 100,
            "ts": [dt.datetime(2026, 1, 1) + dt.timedelta(minutes=i) for i in range(n)],
        }
    )
    df["ts"] = df["ts"].astype(str)
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert result.findings
