import pandas as pd

from plugins.analysis_process_counterfactuals.plugin import Plugin
from tests.conftest import make_context


def test_process_counterfactuals_detect_savings(run_dir):
    base = pd.Timestamp("2024-01-01 00:00:00")
    waits = [200, 190, 180, 70, 80]
    rows = []
    for idx, wait in enumerate(waits):
        eligible = base + pd.Timedelta(minutes=idx * 5)
        start = eligible + pd.Timedelta(seconds=wait)
        rows.append(
            {
                "process_id": "alpha",
                "eligible_ts": eligible.isoformat(),
                "start_ts": start.isoformat(),
                "host_id": "h1",
            }
        )
    for idx in range(5):
        eligible = base + pd.Timedelta(hours=1, minutes=idx * 5)
        start = eligible + pd.Timedelta(seconds=20)
        rows.append(
            {
                "process_id": "beta",
                "eligible_ts": eligible.isoformat(),
                "start_ts": start.isoformat(),
                "host_id": "h2",
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
        isinstance(item, dict) and item.get("kind") == "process_counterfactual"
        for item in result.findings
    )
