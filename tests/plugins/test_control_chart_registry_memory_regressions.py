from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from conftest import make_context
from plugins.analysis_control_chart_cusum.plugin import Plugin as CusumPlugin
from plugins.analysis_control_chart_ewma.plugin import Plugin as EwmaPlugin
from plugins.analysis_control_chart_individuals.plugin import Plugin as IndividualsPlugin


def _wide_grouped_df(rows: int = 7000) -> pd.DataFrame:
    base = dt.datetime(2026, 1, 1)
    frame = pd.DataFrame(
        {
            "metric": [1.0 + (i % 11) * 0.02 for i in range(rows)],
            "grp": ["A" if i % 2 == 0 else "B" for i in range(rows)],
            "ts": [base + dt.timedelta(minutes=i) for i in range(rows)],
        }
    )
    for idx in range(25):
        frame[f"text_{idx}"] = ["Y" * 32 for _ in range(rows)]
    frame["ts"] = frame["ts"].astype(str)
    return frame


@pytest.mark.parametrize("plugin_cls", [IndividualsPlugin, EwmaPlugin, CusumPlugin])
def test_registry_control_chart_plugins_do_not_error_on_wide_grouped_data(run_dir, plugin_cls) -> None:
    df = _wide_grouped_df()
    ctx = make_context(run_dir, df, {})
    ctx.settings = {"group_by": ["grp"], "max_groups": 2, "max_findings": 5}
    result = plugin_cls().run(ctx)
    assert result.status in {"ok", "skipped", "degraded"}
    assert result.status != "error"

