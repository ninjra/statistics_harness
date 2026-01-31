import pandas as pd

from plugins.analysis_graph_topology_curves.plugin import Plugin
from tests.conftest import make_context


def test_graph_topology_curves(run_dir):
    df = pd.read_csv("tests/fixtures/synth_clusters.csv")
    ctx = make_context(run_dir, df, {"max_points": 10, "n_thresholds": 5})
    result = Plugin().run(ctx)
    assert result.status in {"ok", "skipped"}
    if result.status == "ok":
        assert result.metrics["beta1_peak"] >= 0
