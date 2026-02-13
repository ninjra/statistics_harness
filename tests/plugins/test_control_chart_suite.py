import datetime as dt

import pandas as pd

from plugins.analysis_control_chart_suite.plugin import Plugin
from tests.conftest import make_context


def test_control_chart_detects_shift(run_dir):
    values = [0.1] * 100 + [3.0] * 30
    df = pd.DataFrame(
        {
            "metric": values,
            "ts": [dt.datetime(2026, 1, 1) + dt.timedelta(minutes=i) for i in range(len(values))],
        }
    )
    df["ts"] = df["ts"].astype(str)
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert any(f.get("title", "").startswith("Shift detected") for f in result.findings)


def test_control_chart_group_by(run_dir):
    df = pd.DataFrame(
        {
            "metric": [0.0] * 50 + [5.0] * 50 + [0.0] * 100,
            "group": ["A"] * 100 + ["B"] * 100,
            "ts": [dt.datetime(2026, 1, 1) + dt.timedelta(minutes=i) for i in range(200)],
        }
    )
    df["ts"] = df["ts"].astype(str)
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert any("group" in (f.get("where") or {}) for f in result.findings)
