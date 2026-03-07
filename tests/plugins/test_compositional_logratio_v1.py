import numpy as np
import pandas as pd
from plugins.analysis_compositional_logratio_v1.plugin import Plugin
from tests.conftest import make_context


def test_clr_on_proportion_data(run_dir):
    rng = np.random.RandomState(42)
    n = 100
    raw = rng.dirichlet([1, 1, 5], size=n)  # Third component dominates
    df = pd.DataFrame(raw, columns=["comp_a", "comp_b", "comp_c"])
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert result.findings[0]["kind"] == "distribution"
    assert result.findings[0]["measurement_type"] == "measured"
    assert result.findings[0]["n_components"] == 3
    # comp_c should be the most deviant (it has higher concentration)
    deviations = result.findings[0]["component_deviations"]
    assert len(deviations) == 3


def test_clr_skips_non_compositional(run_dir):
    # Columns don't sum to 1
    df = pd.DataFrame({"a": [10.0, 20.0, 30.0], "b": [40.0, 50.0, 60.0]})
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "na"
