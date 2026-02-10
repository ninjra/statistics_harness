import pandas as pd

from plugins.analysis_isolation_forest_anomaly.plugin import Plugin
from tests.conftest import make_context


def test_isolation_forest_flags_outlier(run_dir):
    df = pd.DataFrame(
        {
            "a": [0.0] * 100 + [10.0],
            "b": [0.0] * 100 + [10.0],
        }
    )
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert result.findings
