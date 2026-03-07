import numpy as np
import pandas as pd
import pytest

from plugins.analysis_temporal_fusion_transformer_v1.plugin import Plugin
from tests.conftest import make_context

pytest.importorskip("torch")


def _sample_df() -> pd.DataFrame:
    """Sine wave with 100 points -- enough to exceed MIN_POINTS=50."""
    rng = np.random.RandomState(42)
    t = np.linspace(0, 4 * np.pi, 100)
    values = np.sin(t) + rng.normal(0, 0.1, size=100)
    return pd.DataFrame({"value": values})


def test_temporal_fusion_transformer_smoke(run_dir):
    df = _sample_df()
    ctx = make_context(run_dir, df, {"n_epochs": 10}, run_seed=42)
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert len(result.findings) == 1
    f = result.findings[0]
    assert f["kind"] == "time_series"
    assert f["measurement_type"] == "measured"
    assert "rmse" in f
    assert "mae" in f
    assert f["train_size"] == 80
    assert f["test_size"] > 0


def test_temporal_fusion_transformer_skips_short_series(run_dir):
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
    ctx = make_context(run_dir, df, {}, run_seed=42)
    result = Plugin().run(ctx)
    assert result.status == "na"
