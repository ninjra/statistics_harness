import pandas as pd

from plugins.profile_basic.plugin import Plugin as ProfilePlugin
from tests.conftest import make_context


def test_row_trace_with_parameters(run_dir):
    df = pd.DataFrame({"params": ["k=v", "x=1"], "value": [1, 2]})
    ctx = make_context(run_dir, df, {})
    ProfilePlugin().run(ctx)

    payload = ctx.storage.fetch_row_trace(ctx.dataset_version_id, 0)
    assert payload["rows"]
    row = payload["rows"][0]
    assert row["row_index"] == 0
    assert "params" in row["values"]
    assert row["parameters"]
