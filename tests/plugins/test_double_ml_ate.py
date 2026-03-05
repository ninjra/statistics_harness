import numpy as np
import pandas as pd
from plugins.analysis_double_ml_ate_v1.plugin import Plugin
from tests.conftest import make_context

def test_dml_estimates_treatment_effect(run_dir):
    rng = np.random.RandomState(42)
    n = 300
    x1 = rng.randn(n)
    x2 = rng.randn(n)
    treatment = (rng.rand(n) > 0.5).astype(float)
    effect = 3.0
    y = 2 * x1 + x2 + effect * treatment + rng.randn(n) * 0.5
    df = pd.DataFrame({"x1": x1, "x2": x2, "treatment": treatment, "outcome": y})
    ctx = make_context(run_dir, df, {"treatment_column": "treatment", "outcome_column": "outcome"})
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert result.findings[0]["kind"] == "causal"
    assert abs(result.findings[0]["ate"] - effect) < 2.0

def test_dml_skips_without_treatment(run_dir):
    df = pd.DataFrame({"x": np.random.randn(100), "y": np.random.randn(100)})
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "skipped"
