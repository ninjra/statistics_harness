import numpy as np
import pandas as pd
import pytest

from plugins.analysis_functional_data_analysis_v1.plugin import Plugin
from tests.conftest import make_context

pytest.importorskip("skfda")


def _sample_df() -> pd.DataFrame:
    """20 functions across 5 grid points, with 2 outlier rows."""
    rng = np.random.RandomState(42)
    data = rng.normal(0, 1, size=(20, 5))
    # Make rows 0 and 1 clear outliers
    data[0] = [10, 10, 10, 10, 10]
    data[1] = [-10, -10, -10, -10, -10]
    cols = [f"v{i}" for i in range(5)]
    return pd.DataFrame(data, columns=cols)


def test_functional_data_analysis_smoke(run_dir):
    df = _sample_df()
    ctx = make_context(run_dir, df, {}, run_seed=42)
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert len(result.findings) == 1
    f = result.findings[0]
    assert f["kind"] == "distribution"
    assert f["measurement_type"] == "measured"
    assert f["n_functions"] == 20
    assert f["n_grid_points"] == 5
    assert "functional_mean" in f
    assert "functional_variance" in f
    assert f["n_outliers"] >= 0


def test_functional_data_analysis_skips_single_column(run_dir):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    ctx = make_context(run_dir, df, {}, run_seed=42)
    result = Plugin().run(ctx)
    assert result.status == "na"
