import pandas as pd

from plugins.profile_basic.plugin import Plugin
from tests.conftest import make_context


def test_profile_basic(run_dir):
    df = pd.read_csv("tests/fixtures/synth_linear.csv")
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert any(a.path.endswith("columns.json") for a in result.artifacts)
