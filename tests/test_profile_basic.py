import pandas as pd

from plugins.profile_basic.plugin import Plugin
from tests.conftest import make_context


def test_profile_basic(run_dir):
    df = pd.read_csv("tests/fixtures/synth_linear.csv")
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert any(a.path.endswith("columns.json") for a in result.artifacts)


def test_profile_basic_pii_tags(run_dir):
    df = pd.DataFrame(
        {
            "email": ["user@example.com", "admin@example.com"],
            "value": [1, 2],
        }
    )
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "ok"
    columns = ctx.storage.fetch_dataset_columns(ctx.dataset_version_id)
    email_col = next(col for col in columns if col["original_name"] == "email")
    assert "email" in (email_col.get("pii_tags") or [])
    pii_entities = ctx.storage.fetch_pii_entities(ctx.tenant_id or "default")
    raw_values = {entry["raw_value"] for entry in pii_entities}
    assert "user@example.com" in raw_values
