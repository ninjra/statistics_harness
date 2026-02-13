import pandas as pd

from plugins.analysis_robust_pca_sparse_outliers.plugin import Plugin
from tests.conftest import make_context


def test_robust_pca_detects_outlier(run_dir):
    df = pd.DataFrame(
        {
            "a": [0.0] * 50 + [10.0],
            "b": [0.0] * 50 + [10.0],
            "c": [0.0] * 50 + [10.0],
        }
    )
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert result.findings
