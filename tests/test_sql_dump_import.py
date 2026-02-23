from __future__ import annotations

from pathlib import Path

import pytest

from statistic_harness.core.sql_dump_import import import_sql_dump


def test_import_sql_dump_parses_create_and_insert(tmp_path: Path) -> None:
    sql = tmp_path / "sample.sql"
    sql.write_text(
        "\n".join(
            [
                "CREATE TABLE donations (donor TEXT, amount REAL, note TEXT);",
                "INSERT INTO donations (donor, amount, note) VALUES ('Alice', 10.5, 'A''B'), ('Bob', NULL, 'ok');",
            ]
        ),
        encoding="utf-8",
    )
    seen_tables: list[str] = []
    seen_rows: list[list[object]] = []

    def on_create(parsed) -> None:
        seen_tables.append(parsed.table_name)

    def on_insert(table: str, columns, rows) -> None:
        assert table == "donations"
        assert columns == ["donor", "amount", "note"]
        seen_rows.extend(rows)

    manifest = import_sql_dump(
        sql,
        on_create_table=on_create,
        on_insert_rows=on_insert,
        chunk_rows=100,
    )
    assert manifest["create_statements"] == 1
    assert manifest["insert_statements"] == 1
    assert manifest["rows_inserted"] == 2
    assert seen_tables == ["donations"]
    assert seen_rows[0][0] == "Alice"
    assert seen_rows[0][2] == "A'B"
    assert seen_rows[1][1] is None


def test_import_sql_dump_rejects_unsafe_statements(tmp_path: Path) -> None:
    sql = tmp_path / "unsafe.sql"
    sql.write_text(
        "CREATE TABLE t (a TEXT); PRAGMA foreign_keys=OFF; INSERT INTO t (a) VALUES ('x');",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        import_sql_dump(sql)


def test_import_sql_dump_supports_insert_without_columns_when_create_known(tmp_path: Path) -> None:
    sql = tmp_path / "implicit_cols.sql"
    sql.write_text(
        "\n".join(
            [
                "CREATE TABLE t (a TEXT, b INTEGER);",
                "INSERT INTO t VALUES ('x', 1), ('y', 2);",
            ]
        ),
        encoding="utf-8",
    )
    captured: list[tuple[list[str] | None, list[list[object]]]] = []

    def on_insert(_table: str, columns, rows) -> None:
        captured.append((columns, rows))

    manifest = import_sql_dump(sql, on_insert_rows=on_insert)
    assert manifest["rows_inserted"] == 2
    assert captured
    assert captured[0][0] == ["a", "b"]
    assert captured[0][1][1][0] == "y"

