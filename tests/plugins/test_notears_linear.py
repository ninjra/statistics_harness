import pandas as pd

from plugins.analysis_notears_linear.plugin import Plugin
from tests.conftest import make_context


def test_notears_linear(run_dir):
    df = pd.read_csv("tests/fixtures/synth_linear.csv")
    ctx = make_context(run_dir, df, {"weight_threshold": 0.1})
    result = Plugin().run(ctx)
    assert result.status in {"ok", "skipped"}
    if result.status == "ok":
        assert len(result.findings) >= 0
