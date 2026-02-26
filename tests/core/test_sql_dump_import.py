from __future__ import annotations

from pathlib import Path

import pytest

from statistic_harness.core.sql_dump_import import import_sql_dump


def test_sql_dump_import_core_rejects_attach(tmp_path: Path) -> None:
    sql = tmp_path / "bad.sql"
    sql.write_text(
        "CREATE TABLE t (a TEXT); ATTACH DATABASE 'x.db' AS x; INSERT INTO t (a) VALUES ('ok');",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        import_sql_dump(sql)

