import pandas as pd

from plugins.analysis_log_template_mining_drain.plugin import Plugin
from tests.conftest import make_context


def test_log_template_mining_extracts_templates(run_dir):
    df = pd.DataFrame(
        {
            "message": [
                "Error user 123 failed login",
                "Error user 124 failed login",
                "Warning code 99 timeout",
            ]
        }
    )
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert result.findings
