from __future__ import annotations

import sqlite3
from pathlib import Path

from statistic_harness.core.storage import Storage


def test_ensure_dataset_row_index_index_creates_index(tmp_path):
    db_path = tmp_path / "appdata" / "state.sqlite"
    storage = Storage(db_path)

    table_name = "dataset_test"
    columns = [
        {"safe_name": "c1", "sqlite_type": "TEXT", "original_name": "a", "dtype": "object", "column_id": 1},
        {"safe_name": "c2", "sqlite_type": "REAL", "original_name": "b", "dtype": "float64", "column_id": 2},
    ]
    storage.create_dataset_table(table_name, columns)
    storage.ensure_dataset_row_index_index(table_name)

    con = sqlite3.connect(db_path)
    try:
        # Assert at least one index references row_index.
        idxs = con.execute(f"pragma index_list('{table_name}')").fetchall()
        assert idxs
        found = False
        for (seq, name, unique, origin, partial) in idxs:
            cols = [r[2] for r in con.execute(f"pragma index_info('{name}')").fetchall()]
            if "row_index" in cols:
                found = True
                break
        assert found
    finally:
        con.close()

