import numpy as np
import pandas as pd

from plugins.analysis_information_bottleneck_v1.plugin import Plugin
from tests.conftest import make_context


def _sample_df() -> pd.DataFrame:
    rng = np.random.RandomState(42)
    n = 150
    # 3 clusters with different target distributions
    cluster = rng.choice([0, 1, 2], size=n)
    f1 = cluster * 3.0 + rng.normal(0, 0.5, size=n)
    f2 = cluster * 2.0 + rng.normal(0, 0.3, size=n)
    f3 = rng.normal(0, 1, size=n)  # noise feature
    target = cluster * 10.0 + rng.normal(0, 1, size=n)
    return pd.DataFrame({"f1": f1, "f2": f2, "f3": f3, "target": target})


def test_information_bottleneck_smoke(run_dir):
    df = _sample_df()
    ctx = make_context(run_dir, df, {"target_column": "target"}, run_seed=42)
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding["kind"] == "cluster"
    assert finding["measurement_type"] == "measured"
    assert finding["best_k"] >= 2
    assert finding["best_mutual_information"] > 0
    assert len(finding["information_distortion_curve"]) > 0
    assert result.metrics["n_features"] == 3


def test_information_bottleneck_skips_empty(run_dir):
    df = pd.DataFrame({"a": pd.Series([], dtype=float)})
    ctx = make_context(run_dir, df, {}, run_seed=42)
    result = Plugin().run(ctx)
    assert result.status == "na"
