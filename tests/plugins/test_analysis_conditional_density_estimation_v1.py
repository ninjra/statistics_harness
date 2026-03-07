import numpy as np
import pandas as pd

from plugins.analysis_conditional_density_estimation_v1.plugin import Plugin
from tests.conftest import make_context


def _sample_df() -> pd.DataFrame:
    rng = np.random.RandomState(42)
    n = 200
    groups = rng.choice(["high", "low"], size=n)
    # Different distributions per group
    values = np.where(
        np.array(groups) == "high",
        rng.normal(10, 2, size=n),
        rng.normal(3, 1, size=n),
    )
    return pd.DataFrame({"group": groups, "value": values})


def test_conditional_density_estimation_smoke(run_dir):
    df = _sample_df()
    ctx = make_context(run_dir, df, {"group_column": "group", "value_column": "value"}, run_seed=42)
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding["kind"] == "distribution"
    assert finding["measurement_type"] == "measured"
    assert finding["n_groups"] == 2
    assert len(finding["kl_divergences"]) == 1
    assert finding["max_symmetric_kl"] > 0
    assert result.metrics["n_groups"] == 2


def test_conditional_density_estimation_skips_empty(run_dir):
    df = pd.DataFrame({"a": pd.Series([], dtype=float)})
    ctx = make_context(run_dir, df, {}, run_seed=42)
    result = Plugin().run(ctx)
    assert result.status == "na"
