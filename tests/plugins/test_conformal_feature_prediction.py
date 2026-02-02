import pandas as pd

from plugins.analysis_conformal_feature_prediction.plugin import Plugin
from tests.conftest import make_context


def test_conformal_feature_prediction(run_dir):
    df = pd.read_csv("tests/fixtures/synth_linear.csv")
    ctx = make_context(run_dir, df, {"alpha": 0.2})
    result = Plugin().run(ctx)
    assert result.status in {"ok", "skipped"}
    if result.status == "ok":
        assert any(f["kind"] == "anomaly" for f in result.findings)
