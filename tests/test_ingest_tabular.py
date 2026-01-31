from pathlib import Path

import pandas as pd

from plugins.ingest_tabular.plugin import Plugin
from tests.conftest import make_context


def test_ingest_tabular_csv(run_dir):
    df = pd.read_csv("tests/fixtures/synth_linear.csv")
    ctx = make_context(run_dir, df, {"input_file": "tests/fixtures/synth_linear.csv"})
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert (run_dir / "dataset" / "canonical.csv").exists()


def test_ingest_tabular_xlsx(run_dir, tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    xlsx_path = tmp_path / "data.xlsx"
    df.to_excel(xlsx_path, index=False)
    ctx = make_context(run_dir, df, {"input_file": str(xlsx_path)})
    result = Plugin().run(ctx)
    assert result.status == "ok"
