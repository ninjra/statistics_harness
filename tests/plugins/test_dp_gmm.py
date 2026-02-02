import pandas as pd

from plugins.analysis_dp_gmm.plugin import Plugin
from tests.conftest import make_context


def test_dp_gmm(run_dir):
    df = pd.read_csv("tests/fixtures/synth_clusters.csv")
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status in {"ok", "skipped"}
    if result.status == "ok":
        assert result.metrics["clusters"] >= 2
