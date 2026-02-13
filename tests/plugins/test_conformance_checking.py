import datetime as dt

import pandas as pd

from plugins.analysis_conformance_checking.plugin import Plugin
from tests.conftest import make_context


def test_conformance_detects_unexpected_transition(run_dir):
    rows = []
    for i in range(10):
        rows.append({"case_id": i, "activity": "A", "ts": dt.datetime(2026, 1, 1, 8, 0, 0)})
        rows.append({"case_id": i, "activity": "B", "ts": dt.datetime(2026, 1, 1, 8, 1, 0)})
    rows.append({"case_id": 999, "activity": "A", "ts": dt.datetime(2026, 1, 1, 9, 0, 0)})
    rows.append({"case_id": 999, "activity": "C", "ts": dt.datetime(2026, 1, 1, 9, 1, 0)})
    df = pd.DataFrame(rows)
    df["ts"] = df["ts"].astype(str)
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert result.findings
