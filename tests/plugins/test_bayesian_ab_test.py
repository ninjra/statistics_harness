import numpy as np
import pandas as pd
from plugins.analysis_bayesian_ab_test_v1.plugin import Plugin
from tests.conftest import make_context


def test_bayesian_ab_test_smoke(run_dir):
    rng = np.random.RandomState(42)
    n = 500
    group = rng.choice(["A", "B"], size=n)
    # Group B has higher conversion rate
    outcome = np.where(
        group == "A",
        rng.binomial(1, 0.10, size=n),
        rng.binomial(1, 0.20, size=n),
    )
    df = pd.DataFrame({"group": group, "converted": outcome})
    ctx = make_context(run_dir, df, {"group_column": "group", "outcome_column": "converted"})
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert result.findings
    assert result.findings[0]["kind"] == "distribution"
    assert result.findings[0]["prob_b_greater_than_a"] > 0.5
    assert result.findings[0]["expected_lift"] > 0


def test_bayesian_ab_test_skips_empty(run_dir):
    df = pd.DataFrame({"a": pd.Series([], dtype=float)})
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "na"
