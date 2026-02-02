import pandas as pd

from plugins.ingest_tabular.plugin import Plugin
from tests.conftest import make_context


def test_ingest_tabular_csv(run_dir):
    df = pd.read_csv("tests/fixtures/synth_linear.csv")
    ctx = make_context(
        run_dir, df, {"input_file": "tests/fixtures/synth_linear.csv"}, populate=False
    )
    result = Plugin().run(ctx)
    assert result.status == "ok"
    version = ctx.storage.get_dataset_version(ctx.dataset_version_id)
    assert version
    assert int(version["row_count"]) == len(df)


def test_ingest_tabular_xlsx(run_dir, tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    xlsx_path = tmp_path / "data.xlsx"
    df.to_excel(xlsx_path, index=False)
    ctx = make_context(run_dir, df, {"input_file": str(xlsx_path)}, populate=False)
    result = Plugin().run(ctx)
    assert result.status == "ok"
    version = ctx.storage.get_dataset_version(ctx.dataset_version_id)
    assert version
    assert int(version["row_count"]) == len(df)


def test_ingest_tabular_json_lines(run_dir, tmp_path):
    path = tmp_path / "data.json"
    path.write_text(
        '{"a": 1, "b": "x"}\n{"a": 2, "b": "y"}\n',
        encoding="utf-8",
    )
    ctx = make_context(run_dir, pd.DataFrame({"a": [1], "b": ["x"]}), {"input_file": str(path)}, populate=False)
    result = Plugin().run(ctx)
    assert result.status == "ok"
    version = ctx.storage.get_dataset_version(ctx.dataset_version_id)
    assert version
    assert int(version["row_count"]) == 2


def test_ingest_tabular_weird_headers(run_dir, tmp_path):
    df = pd.DataFrame(
        {
            "select": [1, 2],
            "bad name": [3, 4],
            "a;b": [5, 6],
        }
    )
    path = tmp_path / "weird.csv"
    df.to_csv(path, index=False)
    ctx = make_context(run_dir, df, {"input_file": str(path)}, populate=False)
    result = Plugin().run(ctx)
    assert result.status == "ok"
    columns = ctx.storage.fetch_dataset_columns(ctx.dataset_version_id)
    assert [col["safe_name"] for col in columns] == ["c1", "c2", "c3"]
