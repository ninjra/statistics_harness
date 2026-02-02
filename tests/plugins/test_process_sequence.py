import pandas as pd

from plugins.analysis_process_sequence.plugin import Plugin
from tests.conftest import make_context


def test_process_sequence_plugin(run_dir):
    df = pd.DataFrame(
        {
            "case_id": [1, 1, 2, 2],
            "activity": ["A", "B", "A", "C"],
            "timestamp": [1, 2, 1, 2],
        }
    )
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert any(f["kind"] == "process_variant" for f in result.findings)
