import numpy as np
import pandas as pd
from plugins.analysis_instrumental_variables_2sls_v1.plugin import Plugin
from tests.conftest import make_context


def test_iv2sls_estimates_causal_effect(run_dir):
    rng = np.random.RandomState(42)
    n = 500
    # z is the instrument, correlated with x but not directly with y
    z = rng.randn(n)
    u = rng.randn(n)  # unobserved confounder
    x = 0.8 * z + 0.5 * u + rng.randn(n) * 0.3  # endogenous
    true_effect = 2.0
    y = true_effect * x + 1.5 * u + rng.randn(n) * 0.5  # outcome
    w = rng.randn(n)  # exogenous control
    df = pd.DataFrame({"y": y, "x": x, "z": z, "w": w})
    ctx = make_context(
        run_dir,
        df,
        {
            "outcome_column": "y",
            "endogenous_column": "x",
            "instrument_column": "z",
            "exogenous_columns": ["w"],
        },
    )
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert result.findings[0]["kind"] == "causal"
    assert result.findings[0]["measurement_type"] == "measured"
    # IV estimate should be in the neighborhood of true_effect
    assert abs(result.findings[0]["coefficient"] - true_effect) < 2.0


def test_iv2sls_skips_without_settings(run_dir):
    df = pd.DataFrame({"x": np.random.randn(100), "y": np.random.randn(100)})
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "skipped"
