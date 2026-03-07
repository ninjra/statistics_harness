import numpy as np
import pandas as pd
from plugins.analysis_regression_discontinuity_v1.plugin import Plugin
from tests.conftest import make_context

def test_rd_detects_discontinuity(run_dir):
    rng = np.random.RandomState(42)
    n = 200
    x = rng.uniform(-5, 5, n)
    cutoff = 0.0
    y = 2 * x + 5 * (x >= cutoff) + rng.randn(n) * 0.5  # jump of 5 at cutoff
    df = pd.DataFrame({"running": x, "outcome": y})
    ctx = make_context(run_dir, df, {"running_column": "running", "outcome_column": "outcome", "cutoff": 0.0})
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert result.findings[0]["kind"] == "counterfactual"
    assert abs(result.findings[0]["rd_estimate"] - 5.0) < 3.0

def test_rd_skips_single_column(run_dir):
    df = pd.DataFrame({"x": np.random.randn(50)})
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "na"
