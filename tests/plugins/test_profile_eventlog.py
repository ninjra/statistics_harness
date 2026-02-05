import pandas as pd

from plugins.profile_eventlog.plugin import Plugin
from tests.conftest import make_context


def test_profile_eventlog_role_inference(run_dir):
    df = pd.DataFrame(
        {
            "QUEUE_DT": [
                "2026-01-20T01:00:00",
                "2026-01-20T02:00:00",
                "2026-01-21T01:30:00",
            ],
            "START_DT": [
                "2026-01-20T01:01:00",
                "2026-01-20T02:05:00",
                "2026-01-21T01:45:00",
            ],
            "END_DT": [
                "2026-01-20T01:04:00",
                "2026-01-20T02:08:00",
                "2026-01-21T01:50:00",
            ],
            "PROCESS_ID": ["qemail", "qpec", "qpec"],
            "USER_ID": ["u1", "u2", "u1"],
        }
    )
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert "time_to_completion" in result.metrics
    candidates = ctx.storage.fetch_dataset_role_candidates(ctx.dataset_version_id)
    assert any(c["role"] == "start_time" for c in candidates)
    columns = ctx.storage.fetch_dataset_columns(ctx.dataset_version_id)
    assert any(col.get("role") in {"queue_time", "start_time", "end_time"} for col in columns)
