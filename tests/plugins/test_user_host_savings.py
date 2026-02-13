import pandas as pd

from plugins.analysis_user_host_savings.plugin import Plugin
from tests.conftest import make_context


def test_user_host_savings_detects_groups(run_dir):
    base = pd.Timestamp("2024-01-01 00:00:00")
    rows = []
    for idx in range(4):
        eligible = base + pd.Timedelta(minutes=idx * 5)
        start = eligible + pd.Timedelta(seconds=200)
        rows.append(
            {
                "user_name": "user_a",
                "host_id": "h1",
                "eligible_ts": eligible.isoformat(),
                "start_ts": start.isoformat(),
            }
        )
    for idx in range(4):
        eligible = base + pd.Timedelta(hours=1, minutes=idx * 5)
        start = eligible + pd.Timedelta(seconds=20)
        rows.append(
            {
                "user_name": "user_b",
                "host_id": "h2",
                "eligible_ts": eligible.isoformat(),
                "start_ts": start.isoformat(),
            }
        )
    df = pd.DataFrame(rows)
    ctx = make_context(
        run_dir,
        df=df,
        settings={"min_runs": 2, "wait_threshold_seconds": 60},
    )
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert any(
        isinstance(item, dict) and item.get("kind") == "user_host_savings"
        for item in result.findings
    )
