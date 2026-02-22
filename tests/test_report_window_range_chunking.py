from __future__ import annotations

from datetime import datetime, timedelta
import sqlite3

from statistic_harness.core.report import _sum_duration_hours_for_ranges


def _duration_expr(start_col: str, end_col: str) -> str:
    return (
        f"(CASE WHEN {start_col} IS NOT NULL AND {end_col} IS NOT NULL "
        f"AND julianday({end_col}) >= julianday({start_col}) "
        f"THEN (julianday({end_col}) - julianday({start_col})) * 24.0 ELSE 0.0 END)"
    )


def test_sum_duration_hours_for_ranges_chunks_large_window_sets() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE ds (ts TEXT, start_dt TEXT, end_dt TEXT, process_id TEXT)"
    )

    base = datetime(2026, 1, 1, 0, 0, 0)
    rows: list[tuple[str, str, str, str]] = []
    for i in range(100):
        ts = base + timedelta(minutes=i)
        start = ts
        end = ts + timedelta(hours=1)
        process_id = "target" if i % 2 == 0 else "other"
        rows.append(
            (
                ts.strftime("%Y-%m-%d %H:%M:%S"),
                start.strftime("%Y-%m-%d %H:%M:%S"),
                end.strftime("%Y-%m-%d %H:%M:%S"),
                process_id,
            )
        )
    conn.executemany(
        "INSERT INTO ds(ts, start_dt, end_dt, process_id) VALUES (?, ?, ?, ?)", rows
    )

    ranges: list[tuple[datetime, datetime]] = []
    for i in range(1200):
        start = base + timedelta(minutes=i)
        end = start + timedelta(seconds=30)
        ranges.append((start, end))

    expr = _duration_expr("start_dt", "end_dt")
    total = _sum_duration_hours_for_ranges(
        conn,
        table_name="ds",
        duration_expr=expr,
        timestamp_col="ts",
        ranges=ranges,
    )
    filtered = _sum_duration_hours_for_ranges(
        conn,
        table_name="ds",
        duration_expr=expr,
        timestamp_col="ts",
        ranges=ranges,
        extra_predicate_sql="LOWER(process_id)=?",
        extra_params=["target"],
    )

    assert total > 0.0
    assert filtered > 0.0
    assert filtered < total

