import numpy as np
import pandas as pd
import pytest

from plugins.analysis_sindy_dynamics_discovery_v1.plugin import Plugin
from tests.conftest import make_context

pytest.importorskip("pysindy")


def _sample_df() -> pd.DataFrame:
    """Simple linear dynamics: dx/dt ~ -0.5*x, dy/dt ~ 0.3*y."""
    dt = 0.1
    n = 100
    x = np.zeros(n)
    y = np.zeros(n)
    x[0], y[0] = 2.0, 1.0
    for i in range(1, n):
        x[i] = x[i - 1] + (-0.5 * x[i - 1]) * dt
        y[i] = y[i - 1] + (0.3 * y[i - 1]) * dt
    return pd.DataFrame({"x": x, "y": y})


def test_sindy_dynamics_discovery_smoke(run_dir):
    df = _sample_df()
    ctx = make_context(run_dir, df, {"dt": 0.1, "threshold": 0.05}, run_seed=42)
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert len(result.findings) == 1
    f = result.findings[0]
    assert f["kind"] == "time_series"
    assert f["measurement_type"] == "measured"
    assert "equations" in f
    assert len(f["equations"]) == 2
    assert "coefficients" in f


def test_sindy_dynamics_discovery_skips_empty(run_dir):
    df = pd.DataFrame({"a": pd.Series([], dtype=float)})
    ctx = make_context(run_dir, df, {}, run_seed=42)
    result = Plugin().run(ctx)
    assert result.status == "skipped"
