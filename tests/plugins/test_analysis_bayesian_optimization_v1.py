import numpy as np
import pandas as pd

from plugins.analysis_bayesian_optimization_v1.plugin import Plugin
from tests.conftest import make_context


def _sample_df() -> pd.DataFrame:
    rng = np.random.RandomState(42)
    n = 100
    x1 = rng.uniform(0, 10, size=n)
    x2 = rng.uniform(0, 5, size=n)
    # Objective: quadratic with known optimum near x1=5, x2=2.5
    objective = -((x1 - 5) ** 2) - ((x2 - 2.5) ** 2) + 50 + rng.normal(0, 0.5, size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "objective": objective})


def test_bayesian_optimization_smoke(run_dir):
    df = _sample_df()
    ctx = make_context(run_dir, df, {"objective_column": "objective", "param_columns": ["x1", "x2"]}, run_seed=42)
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding["kind"] == "distribution"
    assert finding["measurement_type"] == "measured"
    assert "best_params" in finding
    assert "best_target" in finding
    assert result.metrics["n_params"] == 2


def test_bayesian_optimization_skips_empty(run_dir):
    df = pd.DataFrame({"a": pd.Series([], dtype=float)})
    ctx = make_context(run_dir, df, {}, run_seed=42)
    result = Plugin().run(ctx)
    assert result.status == "skipped"
