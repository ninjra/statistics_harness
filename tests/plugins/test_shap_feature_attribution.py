import numpy as np
import pandas as pd
from plugins.analysis_shap_feature_attribution_v1.plugin import Plugin
from tests.conftest import make_context

def test_shap_feature_attribution_smoke(run_dir):
    rng = np.random.RandomState(42)
    n = 100
    x1 = rng.randn(n)
    x2 = rng.randn(n)
    x3 = rng.randn(n)
    y = 3 * x1 + 0.5 * x2 + rng.randn(n) * 0.1  # x1 dominant
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "target": y})
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert result.findings
    assert result.findings[0]["kind"] == "role_inference"
    assert result.findings[0]["feature"] == "x1"

def test_shap_skips_on_insufficient_columns(run_dir):
    df = pd.DataFrame({"x": [1, 2, 3]})
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "na"

def test_shap_skips_on_empty_dataset(run_dir):
    df = pd.DataFrame({"a": pd.Series([], dtype=float)})
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "na"
