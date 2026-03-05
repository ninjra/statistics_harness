from __future__ import annotations

import numpy as np
import pandas as pd

from plugins.analysis_reservoir_computing_esn_v1.plugin import Plugin
from tests.conftest import make_context


def _time_series_df() -> pd.DataFrame:
    """Synthetic sine wave with 100 points."""
    rng = np.random.RandomState(42)
    t = np.linspace(0, 4 * np.pi, 100)
    values = np.sin(t) + rng.normal(0, 0.05, size=100)
    return pd.DataFrame({"value": values, "label": ["x"] * 100})


def test_reservoir_esn_smoke(run_dir) -> None:
    df = _time_series_df()
    ctx = make_context(run_dir, df, {}, run_seed=42)
    result = Plugin().run(ctx)
    assert result.status in ("ok", "skipped", "degraded")
    if result.status == "ok":
        assert "rmse" in result.metrics
        assert "mae" in result.metrics
        assert len(result.findings) >= 1
        assert result.findings[0]["kind"] == "time_series"
        assert result.findings[0]["measurement_type"] == "measured"


def test_reservoir_esn_empty(run_dir) -> None:
    df = pd.DataFrame({"a": pd.Series([], dtype=float)})
    ctx = make_context(run_dir, df, {}, run_seed=42)
    result = Plugin().run(ctx)
    assert result.status == "skipped"
