import pandas as pd
import pytest

from plugins.analysis_fairness_bias_detection_v1.plugin import Plugin
from tests.conftest import make_context

pytest.importorskip("fairlearn")


def _sample_df() -> pd.DataFrame:
    """Binary outcome with clear group disparity."""
    n = 200
    group = ["A"] * 100 + ["B"] * 100
    # Group A has 80% positive rate, Group B has 40%
    outcome = [1] * 80 + [0] * 20 + [1] * 40 + [0] * 60
    return pd.DataFrame({"group": group, "outcome": outcome})


def test_fairness_bias_detection_smoke(run_dir):
    df = _sample_df()
    ctx = make_context(run_dir, df, {"sensitive_column": "group", "outcome_column": "outcome"}, run_seed=42)
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert len(result.findings) == 1
    f = result.findings[0]
    assert f["kind"] == "distribution"
    assert f["measurement_type"] == "measured"
    assert "demographic_parity_difference" in f
    assert "equalized_odds_difference" in f
    # There is measurable disparity
    assert abs(f["demographic_parity_difference"]) > 0.0


def test_fairness_bias_detection_skips_empty(run_dir):
    df = pd.DataFrame({"a": pd.Series([], dtype=float)})
    ctx = make_context(run_dir, df, {}, run_seed=42)
    result = Plugin().run(ctx)
    assert result.status == "skipped"
