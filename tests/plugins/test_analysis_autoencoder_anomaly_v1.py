from __future__ import annotations

import numpy as np
import pandas as pd

from plugins.analysis_autoencoder_anomaly_v1.plugin import Plugin
from tests.conftest import make_context


def _numeric_df_with_outliers() -> pd.DataFrame:
    """50 normal rows + 5 outlier rows across 3 numeric columns."""
    rng = np.random.RandomState(42)
    n_normal = 50
    n_outlier = 5
    normal = rng.randn(n_normal, 3)
    outliers = rng.randn(n_outlier, 3) * 10 + 20
    data = np.vstack([normal, outliers])
    return pd.DataFrame(data, columns=["feat_a", "feat_b", "feat_c"])


def test_autoencoder_anomaly_smoke(run_dir) -> None:
    df = _numeric_df_with_outliers()
    ctx = make_context(run_dir, df, {"epochs": 30}, run_seed=42)
    result = Plugin().run(ctx)
    assert result.status in ("ok", "na", "degraded")
    if result.status == "ok":
        assert "n_anomalies" in result.metrics
        assert "anomaly_rate" in result.metrics
        assert len(result.findings) >= 1
        assert result.findings[0]["kind"] == "anomaly"
        assert result.findings[0]["measurement_type"] == "measured"


def test_autoencoder_anomaly_empty(run_dir) -> None:
    df = pd.DataFrame({"a": pd.Series([], dtype=float)})
    ctx = make_context(run_dir, df, {}, run_seed=42)
    result = Plugin().run(ctx)
    assert result.status == "na"
