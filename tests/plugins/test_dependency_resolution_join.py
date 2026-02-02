import pandas as pd

from plugins.analysis_dependency_resolution_join.plugin import Plugin
from tests.conftest import make_context


def test_dependency_resolution_join_emits_lag_summary(run_dir):
    df = pd.DataFrame(
        [
            {
                "dep_id": "ROOT",
                "process_id": "A",
                "start_ts": "2026-01-01 00:00:00",
                "end_ts": "2026-01-01 00:10:00",
            },
            {
                "dep_id": "A",
                "process_id": "B",
                "start_ts": "2026-01-01 00:10:00",
                "end_ts": "2026-01-01 00:20:00",
            },
            {
                "dep_id": "A",
                "process_id": "C",
                "start_ts": "2026-01-01 00:15:00",
                "end_ts": "2026-01-01 00:25:00",
            },
        ]
    )
    df = df[["process_id", "dep_id", "start_ts", "end_ts"]]

    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)

    assert result.status == "ok"
    assert result.findings
    finding = result.findings[0]
    assert finding["kind"] == "dependency_lag_summary"
    assert finding["dependency_rows"] == 2
    assert finding["near_zero_ratio"] >= 0.5
