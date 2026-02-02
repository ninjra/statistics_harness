import pandas as pd

from plugins.profile_basic.plugin import Plugin
from tests.conftest import make_context


def test_parameter_normalization(run_dir):
    df = pd.DataFrame(
        {
            "params": ["a=1; b=2", "b=2; a=1"],
            "value": [10, 12],
        }
    )
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "ok"

    with ctx.storage.connection() as conn:
        entities = conn.execute("SELECT COUNT(*) FROM parameter_entities").fetchone()
        assert entities and int(entities[0]) == 1
        links = conn.execute(
            "SELECT COUNT(*) FROM row_parameter_link WHERE dataset_version_id = ?",
            (ctx.dataset_version_id,),
        ).fetchone()
        assert links and int(links[0]) == 2
        kv_rows = conn.execute(
            "SELECT COUNT(*) FROM parameter_kv WHERE key IN ('a', 'b')"
        ).fetchone()
        assert kv_rows and int(kv_rows[0]) >= 2
