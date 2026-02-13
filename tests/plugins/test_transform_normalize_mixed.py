import json

import pandas as pd

from plugins.transform_normalize_mixed.plugin import Plugin
from statistic_harness.core.utils import now_iso, quote_identifier
from tests.conftest import make_context


def test_transform_normalize_mixed_basic(run_dir):
    df = pd.DataFrame(
        {
            "Name": [" Foo ", "Bar"],
            "Amount": ["1,000", "2"],
            "ID": ["001", "002"],
        }
    )
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "ok"

    dataset_template = ctx.storage.fetch_dataset_template(ctx.dataset_version_id)
    assert dataset_template
    assert dataset_template["status"] == "ready"
    template = ctx.storage.fetch_template(int(dataset_template["template_id"]))
    assert template
    fields = ctx.storage.fetch_template_fields(int(dataset_template["template_id"]))
    assert len(fields) == 3

    name_col = fields[0]["safe_name"]
    amount_col = fields[1]["safe_name"]
    id_col = fields[2]["safe_name"]

    with ctx.storage.connection() as conn:
        cur = conn.execute(
            f"""
            SELECT {quote_identifier(name_col)}, {quote_identifier(amount_col)}, {quote_identifier(id_col)}
            FROM {quote_identifier(template['table_name'])}
            WHERE dataset_version_id = ? AND row_index = ?
            """,
            (ctx.dataset_version_id, 0),
        )
        row = cur.fetchone()
        assert row is not None
        assert row[0] == "foo"
        assert isinstance(row[1], (int, float))
        assert row[2] == "001"


def test_transform_normalize_mixed_includes_source_classification(run_dir):
    df = pd.DataFrame({"Name": ["Foo"], "Amount": ["1"]})
    ctx = make_context(run_dir, df, {})
    ctx.storage.ensure_dataset_version(
        ctx.dataset_version_id,
        ctx.dataset_id,
        now_iso(),
        f"dataset_{ctx.dataset_version_id}",
        ctx.dataset_id,
        source_classification="synthetic",
    )

    result = Plugin().run(ctx)
    assert result.status == "ok"
    dataset_template = ctx.storage.fetch_dataset_template(ctx.dataset_version_id)
    assert dataset_template is not None
    payload = json.loads(str(dataset_template.get("mapping_json") or "{}"))
    assert payload.get("source", {}).get("classification") == "synthetic"


def test_transform_normalize_mixed_sets_quorum_erp_from_field_structure(run_dir):
    df = pd.DataFrame(
        {
            "PROCESS_QUEUE_ID": [1, 2],
            "PROCESS_ID": ["A", "B"],
            "STATUS_CD": ["DONE", "DONE"],
            "LOCAL_MACHINE_ID": ["H1", "H2"],
            "QUEUE_DT": ["2026-01-01T00:00:00", "2026-01-01T00:01:00"],
            "START_DT": ["2026-01-01T00:02:00", "2026-01-01T00:03:00"],
            "END_DT": ["2026-01-01T00:04:00", "2026-01-01T00:05:00"],
        }
    )
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "ok"
    project = ctx.storage.fetch_project(ctx.project_id)
    assert project is not None
    assert str(project.get("erp_type") or "") == "quorum"
