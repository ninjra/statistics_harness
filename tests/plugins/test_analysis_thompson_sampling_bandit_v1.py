import numpy as np
import pandas as pd

from plugins.analysis_thompson_sampling_bandit_v1.plugin import Plugin
from tests.conftest import make_context


def _sample_df() -> pd.DataFrame:
    rng = np.random.RandomState(42)
    n = 200
    groups = rng.choice(["arm_A", "arm_B", "arm_C"], size=n)
    # arm_A has 70% success, arm_B 40%, arm_C 20%
    probs = {"arm_A": 0.7, "arm_B": 0.4, "arm_C": 0.2}
    outcomes = np.array([rng.binomial(1, probs[g]) for g in groups], dtype=float)
    return pd.DataFrame({"group": groups, "reward": outcomes})


def test_thompson_sampling_bandit_smoke(run_dir):
    df = _sample_df()
    ctx = make_context(run_dir, df, {"group_column": "group", "outcome_column": "reward"}, run_seed=42)
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding["kind"] == "distribution"
    assert finding["measurement_type"] == "measured"
    assert finding["best_arm"] == "arm_A"
    assert finding["prob_best_arm"] > 0.5
    assert finding["n_arms"] == 3
    assert result.metrics["n_arms"] == 3


def test_thompson_sampling_bandit_skips_empty(run_dir):
    df = pd.DataFrame({"a": pd.Series([], dtype=float)})
    ctx = make_context(run_dir, df, {}, run_seed=42)
    result = Plugin().run(ctx)
    assert result.status == "skipped"
