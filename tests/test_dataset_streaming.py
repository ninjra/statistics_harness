from __future__ import annotations

import pandas as pd

from statistic_harness.core.dataset_io import DatasetAccessor
from statistic_harness.core.storage import Storage
from statistic_harness.core.utils import now_iso


def _build_dataset_accessor(tmp_path, *, row_count_meta: int = 23) -> DatasetAccessor:
    storage = Storage(tmp_path / "state.sqlite")
    project_id = "p1"
    dataset_id = "d1"
    dataset_version_id = "dv1"
    table_name = f"dataset_{dataset_version_id}"
    storage.ensure_project(project_id, project_id, now_iso())
    storage.ensure_dataset(dataset_id, project_id, dataset_id, now_iso())

    df = pd.DataFrame({"a": list(range(23)), "b": [str(i) for i in range(23)]})

    with storage.connection() as conn:
        storage.ensure_dataset_version(
            dataset_version_id, dataset_id, now_iso(), table_name, dataset_id, conn
        )
        columns_meta = [
            {
                "column_id": 1,
                "safe_name": "c1",
                "original_name": "a",
                "dtype": "int",
                "sqlite_type": "INTEGER",
            },
            {
                "column_id": 2,
                "safe_name": "c2",
                "original_name": "b",
                "dtype": "str",
                "sqlite_type": "TEXT",
            },
        ]
        storage.create_dataset_table(table_name, columns_meta, conn)
        storage.replace_dataset_columns(dataset_version_id, columns_meta, conn)
        safe_columns = [col["safe_name"] for col in columns_meta]
        rows = [(idx, None, int(row.a), row.b) for idx, row in df.iterrows()]
        storage.insert_dataset_rows(table_name, safe_columns, rows, conn)
        storage.update_dataset_version_stats(dataset_version_id, int(row_count_meta), 2, conn)

    return DatasetAccessor(storage, dataset_version_id)


def test_dataset_accessor_iter_batches_roundtrip(tmp_path) -> None:
    accessor = _build_dataset_accessor(tmp_path, row_count_meta=23)

    batches = list(accessor.iter_batches(batch_size=7))
    combined = pd.concat(batches).sort_index()
    full = accessor.load()
    pd.testing.assert_frame_equal(combined, full)


def test_dataset_accessor_large_dataset_load_requires_override(tmp_path, monkeypatch) -> None:
    accessor = _build_dataset_accessor(tmp_path, row_count_meta=1_000_001)
    monkeypatch.delenv("STAT_HARNESS_ALLOW_FULL_DF", raising=False)
    monkeypatch.delenv("STAT_HARNESS_MAX_FULL_DF_ROWS", raising=False)
    try:
        accessor.load(columns=["a"])
    except RuntimeError as exc:
        assert "Refusing to load full dataset into memory" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected large-dataset full load refusal")


def test_dataset_accessor_large_dataset_load_override_allows(tmp_path, monkeypatch) -> None:
    accessor = _build_dataset_accessor(tmp_path, row_count_meta=1_000_001)
    monkeypatch.setenv("STAT_HARNESS_ALLOW_FULL_DF", "1")
    df = accessor.load(columns=["a"])
    assert list(df.columns) == ["a"]
    assert len(df.index) == 23
