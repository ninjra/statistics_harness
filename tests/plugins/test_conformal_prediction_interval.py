import numpy as np
import pandas as pd

from plugins.analysis_conformal_prediction_interval_v1.plugin import Plugin
from tests.conftest import make_context


def test_conformal_prediction_smoke(run_dir):
    rng = np.random.RandomState(42)
    n = 200
    x1 = rng.randn(n)
    x2 = rng.randn(n)
    y = 2 * x1 + x2 + rng.randn(n) * 0.5
    df = pd.DataFrame({"x1": x1, "x2": x2, "target": y})
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert result.metrics["empirical_coverage"] > 0.8
    assert result.findings[0]["kind"] == "distribution"


def test_conformal_skips_insufficient_data(run_dir):
    df = pd.DataFrame({"x": range(10), "y": range(10)})
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "na"
