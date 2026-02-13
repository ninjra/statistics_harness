import datetime as dt

import pandas as pd

from plugins.analysis_multivariate_control_charts.plugin import Plugin
from tests.conftest import make_context


def test_multivariate_control_detects_joint_shift(run_dir):
    n = 300
    data = {
        "a": [0.0] * 150 + [2.0] * 150,
        "b": [0.0] * 150 + [2.0] * 150,
        "c": [0.0] * n,
        "ts": [dt.datetime(2026, 1, 1) + dt.timedelta(minutes=i) for i in range(n)],
    }
    df = pd.DataFrame(data)
    df["ts"] = df["ts"].astype(str)
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert result.findings
