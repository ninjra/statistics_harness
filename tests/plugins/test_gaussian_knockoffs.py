import pandas as pd

from plugins.analysis_gaussian_knockoffs.plugin import Plugin
from tests.conftest import make_context


def test_gaussian_knockoffs(run_dir):
    df = pd.read_csv("tests/fixtures/synth_linear.csv")
    ctx = make_context(run_dir, df, {"target_column": "y"})
    result = Plugin().run(ctx)
    assert result.status in {"ok", "skipped"}
    if result.status == "ok":
        assert any(f["selected"] for f in result.findings)
