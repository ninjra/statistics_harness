import numpy as np
import pandas as pd
from plugins.analysis_synthetic_control_v1.plugin import Plugin
from tests.conftest import make_context

def test_synthetic_control_detects_effect(run_dir):
    rng = np.random.RandomState(42)
    n = 100
    donor1 = rng.randn(n).cumsum()
    donor2 = rng.randn(n).cumsum()
    treated = 0.5 * donor1 + 0.5 * donor2 + rng.randn(n) * 0.1
    # Add treatment effect after midpoint
    treated[50:] += 5.0
    df = pd.DataFrame({"treated": treated, "donor1": donor1, "donor2": donor2})
    ctx = make_context(run_dir, df, {"treated_column": "treated", "intervention_index": 50})
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert result.findings[0]["kind"] == "counterfactual"
    assert result.findings[0]["mean_treatment_effect"] > 2.0

def test_synthetic_control_skips_insufficient(run_dir):
    df = pd.DataFrame({"x": [1.0, 2.0]})
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "skipped"
