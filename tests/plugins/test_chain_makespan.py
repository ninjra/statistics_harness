import datetime as dt

import pandas as pd

from plugins.analysis_chain_makespan.plugin import Plugin
from tests.conftest import make_context


def test_chain_makespan_outputs_gaps(run_dir):
    rows = [
        {
            "MASTER_ID": "chain_a",
            "START_DT": dt.datetime(2026, 1, 1, 0, 0, 0),
            "END_DT": dt.datetime(2026, 1, 1, 0, 5, 0),
        },
        {
            "MASTER_ID": "chain_a",
            "START_DT": dt.datetime(2026, 1, 1, 0, 10, 0),
            "END_DT": dt.datetime(2026, 1, 1, 0, 15, 0),
        },
        {
            "MASTER_ID": "chain_b",
            "START_DT": dt.datetime(2026, 1, 1, 0, 0, 0),
            "END_DT": dt.datetime(2026, 1, 1, 0, 3, 0),
        },
    ]
    df = pd.DataFrame(rows)
    df["START_DT"] = df["START_DT"].astype(str)
    df["END_DT"] = df["END_DT"].astype(str)

    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)

    assert result.status == "ok"
    assert result.metrics["chains"] == 2
    findings = [f for f in result.findings if f.get("kind") == "chain_makespan"]
    assert findings
    chain_a = [f for f in findings if f.get("sequence_id") == "chain_a"][0]
    assert chain_a["idle_gap_seconds"] > 0
    assert chain_a["measurement_type"] == "measured"
