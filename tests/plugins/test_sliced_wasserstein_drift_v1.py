import numpy as np
import pandas as pd
from plugins.analysis_sliced_wasserstein_drift_v1.plugin import Plugin
from tests.conftest import make_context


def test_swd_detects_distribution_shift(run_dir):
    rng = np.random.RandomState(42)
    # First half from N(0,1), second half from N(3,1) -- clear shift
    values = np.concatenate([rng.normal(0, 1, 100), rng.normal(3, 1, 100)])
    df = pd.DataFrame({"metric": values})
    ctx = make_context(run_dir, df, {"target_column": "metric"})
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert result.findings[0]["kind"] == "distribution"
    assert result.findings[0]["measurement_type"] == "measured"
    assert result.findings[0]["sliced_wasserstein_distance"] > 0.5


def test_swd_skips_empty_dataframe(run_dir):
    df = pd.DataFrame({"a": pd.Series([], dtype=float)})
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "skipped"
