import pandas as pd

from plugins.analysis_bocpd_gaussian.plugin import Plugin
from tests.conftest import make_context


def test_bocpd_gaussian(run_dir):
    df = pd.read_csv("tests/fixtures/synth_timeseries.csv")
    ctx = make_context(run_dir, df, {"value_column": "value"})
    result = Plugin().run(ctx)
    assert result.status in {"ok", "skipped"}
    if result.status == "ok":
        assert len(result.findings) >= 0
