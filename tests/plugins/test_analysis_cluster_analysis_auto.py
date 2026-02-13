import datetime as dt

import pandas as pd

from plugins.analysis_cluster_analysis_auto.plugin import Plugin
from tests.conftest import make_context


def _sample_df() -> pd.DataFrame:
    rows = 240
    # Includes some PII-shaped strings to ensure redaction paths are exercised.
    return pd.DataFrame(
        {
            "metric": [0.1] * 120 + [2.0] * 120,
            "metric2": [1.0] * 60 + [3.0] * 180,
            "category": ["A"] * 120 + ["B"] * 120,
            "email": ["user@example.com"] * rows,
            "uuid": ["123e4567-e89b-12d3-a456-426614174000"] * rows,
            "case_id": [i // 6 for i in range(rows)],
            "ts": [dt.datetime(2026, 1, 1) + dt.timedelta(minutes=i) for i in range(rows)],
            "x": [float(i % 20) for i in range(rows)],
            "y": [float(i % 12) for i in range(rows)],
        }
    )


def test_analysis_cluster_analysis_auto_smoke(run_dir):
    df = _sample_df()
    df["ts"] = df["ts"].astype(str)
    ctx = make_context(run_dir, df, {}, run_seed=1337)
    result = Plugin().run(ctx)
    assert result.status in ("ok", "skipped", "degraded")
