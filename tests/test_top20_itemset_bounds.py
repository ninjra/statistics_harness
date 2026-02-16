import sqlite3

from statistic_harness.core.top20_plugins import (
    _boolean_item_matrix,
    _fetch_limited_entity_key_rows,
)


def test_fetch_limited_entity_key_rows_prefers_dense_entities() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("create table parameter_kv (entity_id integer, key text)")
    rows = [
        (1, "a"),
        (1, "b"),
        (1, "c"),
        (2, "a"),
        (2, "b"),
        (3, "a"),
        (4, "a"),
        (4, "b"),
        (4, "c"),
        (4, "d"),
    ]
    conn.executemany("insert into parameter_kv(entity_id, key) values (?, ?)", rows)
    conn.commit()

    ent_rows, total = _fetch_limited_entity_key_rows(
        conn,
        ["a", "b", "c", "d"],
        max_entities=2,
    )
    assert total == 4
    assert {int(r["entity_id"]) for r in ent_rows} == {1, 4}

    df, used = _boolean_item_matrix(ent_rows, ["a", "b", "c", "d"])
    assert used == 2
    assert df.shape == (2, 4)
    assert int(df.values.sum()) == 7


def test_fetch_limited_entity_key_rows_empty_input() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("create table parameter_kv (entity_id integer, key text)")
    conn.commit()
    ent_rows, total = _fetch_limited_entity_key_rows(conn, [], max_entities=10)
    assert ent_rows == []
    assert total == 0
