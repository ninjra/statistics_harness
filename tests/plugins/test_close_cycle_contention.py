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


def test_close_cycle_contention_qemail_modeled_removal_backstop(run_dir):
    rows = []
    # Open cycle days: low QEMAIL volume.
    for day in range(10, 20):
        date = dt.datetime(2026, 1, day, 8, 0, 0)
        _add_day(rows, date, qemail_count=1, other_duration=15)

    # Close cycle days: very high QEMAIL volume, but keep other duration constant
    # so correlation-style gates are not required for the known-issue backstop.
    for day in range(20, 32):
        date = dt.datetime(2026, 1, day, 8, 0, 0)
        _add_day(rows, date, qemail_count=50, other_duration=15)
    for day in range(1, 6):
        date = dt.datetime(2026, 2, day, 8, 0, 0)
        _add_day(rows, date, qemail_count=50, other_duration=15)

    df = pd.DataFrame(rows)
    df["timestamp"] = df["timestamp"].astype(str)
    settings = {
        "qemail_min_close_runs": 10,
        "qemail_min_modeled_pct": 0.10,
    }
    ctx = make_context(run_dir, df, settings)
    result = Plugin().run(ctx)
    assert result.status == "ok"
    qemail_findings = [
        f
        for f in result.findings
        if f.get("kind") == "close_cycle_contention" and str(f.get("process_norm") or "").lower() == "qemail"
    ]
    assert qemail_findings
    qf = qemail_findings[0]
    assert float(qf.get("modeled_reduction_pct") or 0.0) >= 0.10
    assert float(qf.get("modeled_reduction_hours") or 0.0) > 0.0


def test_close_cycle_contention_prefers_process_id_over_queue_id(run_dir):
    rows = []
    for day in range(20, 29):
        date = dt.datetime(2026, 1, day, 8, 0, 0)
        for idx in range(10):
            rows.append(
                {
                    "PROCESS_QUEUE_ID": 10000 + day * 100 + idx,
                    "PROCESS_ID": "qemail" if idx % 2 == 0 else "other_job",
                    "START_DT": (date + dt.timedelta(minutes=idx)).isoformat(),
                    "END_DT": (date + dt.timedelta(minutes=idx, seconds=30)).isoformat(),
                    "LOCAL_MACHINE_ID": "qpec1",
                }
            )
    for day in range(10, 19):
        date = dt.datetime(2026, 1, day, 8, 0, 0)
        for idx in range(10):
            rows.append(
                {
                    "PROCESS_QUEUE_ID": 20000 + day * 100 + idx,
                    "PROCESS_ID": "qemail" if idx % 3 == 0 else "other_job",
                    "START_DT": (date + dt.timedelta(minutes=idx)).isoformat(),
                    "END_DT": (date + dt.timedelta(minutes=idx, seconds=30)).isoformat(),
                    "LOCAL_MACHINE_ID": "qpec1",
                }
            )

    df = pd.DataFrame(rows)
    ctx = make_context(
        run_dir,
        df,
        {
            "qemail_min_close_runs": 10,
            "qemail_min_modeled_pct": 0.10,
            "modeled_backstop_min_close_runs": 10,
            "modeled_backstop_min_pct": 0.10,
        },
    )
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert result.metrics.get("process_column") == "PROCESS_ID"
