import sqlite3

import pytest

from statistic_harness.core.storage import Storage
from statistic_harness.core.utils import quote_identifier


def test_append_only_triggers(tmp_path):
    storage = Storage(tmp_path / "state.sqlite")
    table_name = "dataset_test"
    columns = [
        {
            "column_id": 1,
            "safe_name": "c1",
            "original_name": "a",
            "dtype": "int64",
            "sqlite_type": "INTEGER",
        }
    ]

    with storage.connection() as conn:
        storage.create_dataset_table(table_name, columns, conn)
        storage.add_append_only_triggers(table_name, conn)
        storage.insert_dataset_rows(table_name, ["c1"], [(0, None, 1)], conn)

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                f"DELETE FROM {quote_identifier(table_name)} WHERE row_index = 0"
            )

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                f"UPDATE {quote_identifier(table_name)} SET c1 = 2 WHERE row_index = 0"
            )
