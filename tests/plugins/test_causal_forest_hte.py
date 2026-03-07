import numpy as np
import pandas as pd
import pytest
from plugins.analysis_causal_forest_hte_v1.plugin import Plugin
from tests.conftest import make_context


def test_causal_forest_hte_smoke(run_dir):
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
    # econml may not be installed; accept ok or skipped
    assert result.status in ("ok", "na")
    if result.status == "ok":
        assert result.findings[0]["kind"] == "causal"
        assert result.findings[0]["measurement_type"] == "measured"
        assert "hte_std" in result.findings[0]
        assert "feature_importance" in result.findings[0]


def test_causal_forest_hte_skips_empty_dataframe(run_dir):
    df = pd.DataFrame({"a": pd.Series([], dtype=float)})
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "na"
