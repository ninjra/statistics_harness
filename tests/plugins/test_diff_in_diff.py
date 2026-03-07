import numpy as np
import pandas as pd
from plugins.analysis_diff_in_diff_v1.plugin import Plugin
from tests.conftest import make_context

def test_did_detects_treatment_effect(run_dir):
    rng = np.random.RandomState(42)
    n = 200
    treatment = rng.binomial(1, 0.5, n).astype(float)
    post = rng.binomial(1, 0.5, n).astype(float)
    effect = 5.0
    y = 10 + 2 * treatment + 3 * post + effect * treatment * post + rng.randn(n)
    df = pd.DataFrame({"outcome": y, "treatment": treatment, "post": post})
    ctx = make_context(run_dir, df, {"outcome_column": "outcome", "treatment_column": "treatment", "post_column": "post"})
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert result.findings[0]["kind"] == "counterfactual"
    assert abs(result.findings[0]["did_estimate"] - effect) < 2.0

def test_did_skips_without_binary_columns(run_dir):
    df = pd.DataFrame({"x": np.random.randn(50)})
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "na"
