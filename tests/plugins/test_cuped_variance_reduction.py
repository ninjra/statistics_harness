import numpy as np
import pandas as pd
from plugins.analysis_cuped_variance_reduction_v1.plugin import Plugin
from tests.conftest import make_context


def test_cuped_variance_reduction_smoke(run_dir):
    rng = np.random.RandomState(42)
    n = 200
    # Pre and post metrics with strong correlation (CUPED should reduce variance)
    pre = rng.randn(n) * 10
    treatment_effect = 2.0
    noise = rng.randn(n) * 1.0
    post = pre + treatment_effect + noise  # highly correlated with pre
    df = pd.DataFrame({"pre_metric": pre, "post_metric": post})
    ctx = make_context(run_dir, df, {"pre_metric": "pre_metric", "post_metric": "post_metric"})
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert result.findings
    assert result.findings[0]["kind"] == "distribution"
    assert result.findings[0]["variance_reduction_ratio"] > 0.5  # should be substantial
    assert abs(result.findings[0]["theta"] - 1.0) < 0.5  # theta close to 1 for y = x + noise


def test_cuped_variance_reduction_skips_empty(run_dir):
    df = pd.DataFrame({"a": pd.Series([], dtype=float)})
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "skipped"
