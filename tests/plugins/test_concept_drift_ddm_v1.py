import numpy as np
import pandas as pd
from plugins.analysis_concept_drift_ddm_v1.plugin import Plugin
from tests.conftest import make_context


def test_ddm_detects_drift_on_shifted_data(run_dir):
    rng = np.random.RandomState(42)
    # First segment: mean=0, second segment: mean=5 (clear shift)
    segment1 = rng.normal(0, 1, 200)
    segment2 = rng.normal(5, 1, 200)
    values = np.concatenate([segment1, segment2])
    df = pd.DataFrame({"metric": values})
    ctx = make_context(run_dir, df, {"target_column": "metric"})
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert result.findings[0]["kind"] == "changepoint"
    assert result.findings[0]["measurement_type"] == "measured"
    assert result.findings[0]["n_observations"] == 400


def test_ddm_skips_empty_dataframe(run_dir):
    df = pd.DataFrame({"a": pd.Series([], dtype=float)})
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "skipped"
