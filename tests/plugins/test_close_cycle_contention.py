import datetime as dt

import pandas as pd

from plugins.analysis_close_cycle_contention.plugin import Plugin
from tests.conftest import make_context


def _add_day(rows, date, qemail_count, other_duration):
    for idx in range(10):
        rows.append(
            {
                "process": "qpec_job",
                "timestamp": date + dt.timedelta(minutes=idx),
                "duration": other_duration,
                "server": "qpec1",
                "params": "job=main",
            }
        )
    for idx in range(qemail_count):
        rows.append(
            {
                "process": "qemail",
                "timestamp": date + dt.timedelta(minutes=60 + idx),
                "duration": 1,
                "server": "qpec1" if idx % 2 == 0 else "qpec2",
                "params": "noop=true",
            }
        )


def test_close_cycle_contention_detects_qemail(run_dir):
    rows = []
    # Open cycle days: Jan 10-19, Feb 6-10
    for day in range(10, 20):
        date = dt.datetime(2026, 1, day, 8, 0, 0)
        _add_day(rows, date, qemail_count=2, other_duration=10)
    for day in range(6, 11):
        date = dt.datetime(2026, 2, day, 8, 0, 0)
        _add_day(rows, date, qemail_count=2, other_duration=10)

    # Close cycle days: Jan 20-31, Feb 1-5 with varying load
    for day in range(20, 32):
        date = dt.datetime(2026, 1, day, 8, 0, 0)
        qemail_count = 2 + (day % 4)
        other_duration = 10 + qemail_count * 4
        _add_day(rows, date, qemail_count=qemail_count, other_duration=other_duration)
    for day in range(1, 6):
        date = dt.datetime(2026, 2, day, 8, 0, 0)
        qemail_count = 2 + (day % 4)
        other_duration = 10 + qemail_count * 4
        _add_day(rows, date, qemail_count=qemail_count, other_duration=other_duration)

    df = pd.DataFrame(rows)
    df["timestamp"] = df["timestamp"].astype(str)
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "ok"
    contention = [f for f in result.findings if f.get("kind") == "close_cycle_contention"]
    assert contention
    assert any(f.get("process") == "qemail" for f in contention)
