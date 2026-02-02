from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from statistic_harness.core.dataset_io import DatasetAccessor
from statistic_harness.core.storage import Storage
from statistic_harness.core.types import PluginContext
from statistic_harness.core.utils import now_iso


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    (run_dir / "dataset").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    return run_dir


def make_context(
    run_dir: Path, df: pd.DataFrame, settings: dict, populate: bool = True
) -> PluginContext:
    storage = Storage(run_dir / "state.sqlite")
    project_id = "test-project"
    dataset_id = "test_dataset"
    dataset_version_id = "test_dataset"
    table_name = f"dataset_{dataset_version_id}"
    storage.ensure_project(project_id, project_id, now_iso())
    storage.ensure_dataset(dataset_id, project_id, dataset_id, now_iso())

    with storage.connection() as conn:
        storage.ensure_dataset_version(
            dataset_version_id, dataset_id, now_iso(), table_name, dataset_id, conn
        )
        if populate:
            columns_meta = []
            for idx, (col, dtype) in enumerate(df.dtypes.items(), start=1):
                columns_meta.append(
                    {
                        "column_id": idx,
                        "safe_name": f"c{idx}",
                        "original_name": str(col),
                        "dtype": str(dtype),
                        "sqlite_type": "REAL"
                        if pd.api.types.is_numeric_dtype(dtype)
                        else "TEXT",
                    }
                )
            storage.create_dataset_table(table_name, columns_meta, conn)
            storage.replace_dataset_columns(dataset_version_id, columns_meta, conn)
            safe_columns = [col["safe_name"] for col in columns_meta]
            rows = []
            for row_index, row in enumerate(df.itertuples(index=False, name=None)):
                rows.append((row_index, None, *row))
            storage.insert_dataset_rows(table_name, safe_columns, rows, conn)
            storage.update_dataset_version_stats(
                dataset_version_id, len(df), len(columns_meta), conn
            )

    accessor = DatasetAccessor(storage, dataset_version_id)

    def logger(msg: str) -> None:
        pass

    return PluginContext(
        run_id="test-run",
        run_dir=run_dir,
        settings=settings,
        run_seed=42,
        logger=logger,
        storage=storage,
        dataset_loader=accessor.load,
        project_id=project_id,
        dataset_id=dataset_id,
        dataset_version_id=dataset_version_id,
        input_hash=dataset_id,
    )
