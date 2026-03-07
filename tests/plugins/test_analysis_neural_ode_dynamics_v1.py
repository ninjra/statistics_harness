from __future__ import annotations

import numpy as np
import pandas as pd

from plugins.analysis_neural_ode_dynamics_v1.plugin import Plugin
from tests.conftest import make_context


def _time_series_df() -> pd.DataFrame:
    """Synthetic exponential decay with 80 points."""
    t = np.linspace(0, 5, 80)
    values = 10.0 * np.exp(-0.3 * t) + np.random.RandomState(7).normal(0, 0.1, size=80)
    return pd.DataFrame({"signal": values, "tag": ["s"] * 80})


def test_neural_ode_smoke(run_dir) -> None:
    df = _time_series_df()
    ctx = make_context(run_dir, df, {"n_iters": 30}, run_seed=42)
    result = Plugin().run(ctx)
    assert result.status in ("ok", "na", "degraded")
    if result.status == "ok":
        assert "rmse" in result.metrics
        assert "mae" in result.metrics
        assert len(result.findings) >= 1
        assert result.findings[0]["kind"] == "time_series"
        assert result.findings[0]["measurement_type"] == "measured"


def test_neural_ode_too_few_points(run_dir) -> None:
    df = pd.DataFrame({"a": pd.Series(list(range(10)), dtype=float)})
    ctx = make_context(run_dir, df, {}, run_seed=42)
    result = Plugin().run(ctx)
    assert result.status == "na"
