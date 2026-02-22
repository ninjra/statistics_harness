from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from plugins.analysis_control_chart_suite.plugin import Plugin as ControlChartPlugin
from plugins.analysis_distribution_drift_suite.plugin import Plugin as DriftPlugin
from plugins.analysis_multivariate_control_charts.plugin import Plugin as MultivariatePlugin
from conftest import make_context


def _wide_grouped_df(rows: int = 6000) -> pd.DataFrame:
    base = dt.datetime(2026, 1, 1)
    frame = pd.DataFrame(
        {
            "metric": [1.0 + (i % 7) * 0.03 for i in range(rows)],
            "metric2": [0.7 + (i % 5) * 0.04 for i in range(rows)],
            "metric3": [0.2 + (i % 9) * 0.02 for i in range(rows)],
            "grp": ["A" if i % 2 == 0 else "B" for i in range(rows)],
            "ts": [base + dt.timedelta(minutes=i) for i in range(rows)],
        }
    )
    for idx in range(20):
        frame[f"text_{idx}"] = ["x" * 24 for _ in range(rows)]
    frame["ts"] = frame["ts"].astype(str)
    return frame


@pytest.mark.parametrize(
    "plugin_cls",
    [ControlChartPlugin, DriftPlugin, MultivariatePlugin],
)
def test_group_slicing_plugins_do_not_error_on_wide_grouped_data(run_dir, plugin_cls) -> None:
    df = _wide_grouped_df()
    ctx = make_context(run_dir, df, {})
    ctx.settings = {"group_by": ["grp"], "max_groups": 2, "max_findings": 5}
    result = plugin_cls().run(ctx)
    assert result.status in {"ok", "skipped", "degraded"}
    assert result.status != "error"
