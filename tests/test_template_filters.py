import json

import pandas as pd

from statistic_harness.core.dataset_io import resolve_dataset_accessor
from statistic_harness.core.storage import Storage
from statistic_harness.core.utils import now_iso


def test_template_filters_by_project(tmp_path):
    storage = Storage(tmp_path / "state.sqlite")
    template_id = storage.create_template(
        name="FilterTemplate",
        fields=[{"name": "value", "dtype": "float"}],
        description=None,
        version=None,
        created_at=now_iso(),
    )
    template = storage.fetch_template(template_id)
    assert template
    table_name = template["table_name"]
    fields = storage.fetch_template_fields(template_id)
    safe_cols = [field["safe_name"] for field in fields]

    project_a = "project_a"
    project_b = "project_b"
    storage.ensure_project(project_a, "fp_a", now_iso())
    storage.ensure_project(project_b, "fp_b", now_iso())
    storage.ensure_dataset("dataset_a", project_a, "fp_a", now_iso())
    storage.ensure_dataset("dataset_b", project_b, "fp_b", now_iso())
    storage.ensure_dataset_version("dv_a", "dataset_a", now_iso(), "dataset_dv_a", "hash_a")
    storage.ensure_dataset_version("dv_b", "dataset_b", now_iso(), "dataset_dv_b", "hash_b")

    rows = [
        ("dv_a", 0, json.dumps({"value": 1.0}), 1.0),
        ("dv_b", 0, json.dumps({"value": 2.0}), 2.0),
    ]
    storage.insert_template_rows(table_name, safe_cols, rows)

    aggregate_id = storage.ensure_template_aggregate_dataset(
        template_id, now_iso(), filters={"project_ids": [project_a]}
    )
    accessor, _ = resolve_dataset_accessor(storage, aggregate_id)
    df = accessor.load()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df["dataset_version_id"].iloc[0] == "dv_a"
