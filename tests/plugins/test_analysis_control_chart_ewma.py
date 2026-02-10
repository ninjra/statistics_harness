import datetime as dt

import pandas as pd

from plugins.analysis_control_chart_ewma.plugin import Plugin
from tests.conftest import make_context


def _sample_df() -> pd.DataFrame:
    rows = 200
    return pd.DataFrame(
        {
            "metric": [0.1] * 100 + [2.0] * 100,
            "metric2": [1.0] * 50 + [3.0] * 150,
            "category": ["A"] * 100 + ["B"] * 100,
            "text": ["Error code 500"] * 100 + ["Timeout at step 3"] * 100,
            "case_id": [i // 4 for i in range(rows)],
            "ts": [
                dt.datetime(2026, 1, 1) + dt.timedelta(minutes=i)
                for i in range(rows)
            ],
        }
    )


def test_analysis_control_chart_ewma_smoke(run_dir):
    df = _sample_df()
    df["ts"] = df["ts"].astype(str)
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status in ("ok", "skipped")
