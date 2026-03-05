import numpy as np
import pandas as pd
from plugins.analysis_interrupted_time_series_v1.plugin import Plugin
from tests.conftest import make_context

def test_its_detects_level_change(run_dir):
    rng = np.random.RandomState(42)
    n = 100
    pre = rng.randn(50) + 10
    post = rng.randn(50) + 15  # level shift of ~5
    y = np.concatenate([pre, post])
    df = pd.DataFrame({"outcome": y})
    ctx = make_context(run_dir, df, {"outcome_column": "outcome", "intervention_index": 50})
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert result.findings[0]["kind"] == "counterfactual"
    assert result.findings[0]["level_change"] > 0

def test_its_skips_short_series(run_dir):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "skipped"
