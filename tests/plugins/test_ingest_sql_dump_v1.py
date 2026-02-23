from __future__ import annotations

from pathlib import Path

import pandas as pd

from tests.conftest import make_context
from plugins.ingest_sql_dump_v1.plugin import Plugin
from statistic_harness.core.dataset_io import DatasetAccessor


def test_ingest_sql_dump_v1_imports_rows(run_dir: Path, tmp_path: Path) -> None:
    ctx = make_context(run_dir, pd.DataFrame({"x": [1]}), settings={}, run_seed=11)
    sql_path = tmp_path / "input.sql"
    sql_path.write_text(
        "\n".join(
            [
                "CREATE TABLE src (name TEXT, amount REAL);",
                "INSERT INTO src (name, amount) VALUES ('A', 10.0), ('B', 20.5);",
            ]
        ),
        encoding="utf-8",
    )
    ctx.settings = {"input_path": str(sql_path), "chunk_rows": 10}
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert any(a.path.endswith("canonical_import_manifest.json") for a in result.artifacts)
    accessor = DatasetAccessor(ctx.storage, str(ctx.dataset_version_id))
    frame = accessor.load()
    assert len(frame) == 2

