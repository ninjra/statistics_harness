import datetime as dt

import pandas as pd

from plugins.analysis_queue_delay_decomposition.plugin import Plugin
from tests.conftest import make_context


def test_queue_delay_decomposition_targets_qemail(run_dir):
    rows = []
    # Close cycle days
    for day in range(20, 23):
        queue_ts = dt.datetime(2026, 1, day, 8, 0, 0)
        for idx in range(3):
            rows.append(
                {
                    "process": "qemail",
                    "queue_dt": queue_ts + dt.timedelta(minutes=idx),
                    "start_dt": queue_ts + dt.timedelta(minutes=65 + idx),
                    "dep_id": None,
                }
            )
    # Open cycle days
    for day in range(10, 12):
        queue_ts = dt.datetime(2026, 1, day, 8, 0, 0)
        for idx in range(2):
            rows.append(
                {
                    "process": "qemail",
                    "queue_dt": queue_ts + dt.timedelta(minutes=idx),
                    "start_dt": queue_ts + dt.timedelta(minutes=10 + idx),
                    "dep_id": None,
                }
            )
    # Another process
    rows.append(
        {
            "process": "other",
            "queue_dt": dt.datetime(2026, 1, 20, 9, 0, 0),
            "start_dt": dt.datetime(2026, 1, 20, 9, 5, 0),
            "dep_id": None,
        }
    )

    df = pd.DataFrame(rows)
    df["queue_dt"] = df["queue_dt"].astype(str)
    df["start_dt"] = df["start_dt"].astype(str)

    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "ok"

    findings = [
        f
        for f in result.findings
        if f.get("kind") == "eligible_wait_process_stats"
        and f.get("process") == "qemail"
    ]
    assert findings
    impact = [f for f in result.findings if f.get("kind") == "eligible_wait_impact"]
    assert impact
