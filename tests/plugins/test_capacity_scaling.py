import datetime as dt

import pandas as pd

from plugins.analysis_capacity_scaling.plugin import Plugin
from tests.conftest import make_context


def test_capacity_scaling_models_wait(run_dir):
    rows = [
        {
            "HOST": "h1",
            "QUEUE_DT": dt.datetime(2026, 1, 1, 0, 0, 0),
            "START_DT": dt.datetime(2026, 1, 1, 0, 1, 0),
        },
        {
            "HOST": "h1",
            "QUEUE_DT": dt.datetime(2026, 1, 1, 0, 2, 0),
            "START_DT": dt.datetime(2026, 1, 1, 0, 4, 0),
        },
    ]
    df = pd.DataFrame(rows)
    df["QUEUE_DT"] = df["QUEUE_DT"].astype(str)
    df["START_DT"] = df["START_DT"].astype(str)

    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)

    assert result.status == "ok"
    assert result.metrics["scale_factor"] == 2.0
    assert result.findings
    assert result.findings[0]["measurement_type"] == "modeled"
    assert "assumptions" in result.findings[0]
