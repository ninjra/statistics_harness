import numpy as np
import pandas as pd
from plugins.analysis_shapley_interactions_v1.plugin import Plugin
from tests.conftest import make_context


def test_shapley_finds_interaction_pairs(run_dir):
    rng = np.random.RandomState(42)
    n = 200
    x1 = rng.randn(n)
    x2 = rng.randn(n)
    x3 = rng.randn(n)
    # y depends on x1*x2 interaction
    y = 2.0 * x1 * x2 + 0.5 * x3 + rng.randn(n) * 0.3
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "target": y})
    ctx = make_context(
        run_dir, df, {"target_column": "target", "max_rows": 200, "top_n": 5}
    )
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert result.findings[0]["kind"] == "role_inference"
    assert result.findings[0]["measurement_type"] == "measured"
    assert len(result.findings[0]["top_interactions"]) > 0


def test_shapley_skips_too_few_columns(run_dir):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "skipped"
