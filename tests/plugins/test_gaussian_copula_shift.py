import pandas as pd

from plugins.analysis_gaussian_copula_shift.plugin import Plugin
from tests.conftest import make_context


def test_gaussian_copula_shift(run_dir):
    df = pd.read_csv("tests/fixtures/synth_shift_corr.csv")
    ctx = make_context(run_dir, df, {"max_pairs": 1, "n_permutations": 5})
    result = Plugin().run(ctx)
    assert result.status in {"ok", "skipped"}
    if result.status == "ok":
        assert any(f["pair"] == ["a", "b"] for f in result.findings)
