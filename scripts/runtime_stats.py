#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import sqlite3
import statistics
from pathlib import Path


def _parse_iso(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="appdata/state.sqlite")
    ap.add_argument("--dataset-version-id", required=True)
    ap.add_argument("--current-run-id", default="")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--status-run-id", action="append", default=[])
    args = ap.parse_args()

    db_path = Path(args.db)
    con = sqlite3.connect(db_path)
    rows = con.execute(
        """
        SELECT run_id, COALESCE(started_at, created_at) AS started_at, completed_at, status
        FROM runs
        WHERE dataset_version_id = ?
          AND status IN ('completed', 'partial')
          AND completed_at IS NOT NULL
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (args.dataset_version_id, int(args.limit)),
    ).fetchall()

    samples: list[tuple[str, float, str]] = []
    for run_id, started_at, completed_at, status in rows:
        start_dt = _parse_iso(str(started_at) if started_at is not None else None)
        end_dt = _parse_iso(str(completed_at) if completed_at is not None else None)
        if start_dt is None or end_dt is None:
            continue
        minutes = (end_dt - start_dt).total_seconds() / 60.0
        if minutes <= 0:
            continue
        samples.append((str(run_id), float(minutes), str(status)))

    samples.reverse()
    current_run_id = str(args.current_run_id or "").strip()
    history = [m for rid, m, _ in samples if rid != current_run_id]
    current = next((m for rid, m, _ in samples if rid == current_run_id), None)
    avg = float(statistics.mean(history)) if history else None
    std = float(statistics.pstdev(history)) if len(history) > 1 else (0.0 if history else None)

    print(f"samples={len(samples)}")
    print(f"history={len(history)}")
    if current is not None:
        print(f"current_minutes={current:.3f}")
    if avg is not None:
        print(f"avg_minutes={avg:.3f}")
    if std is not None:
        print(f"std_minutes={std:.3f}")
    if current is not None and avg is not None and avg > 0:
        delta_pct = ((current - avg) / avg) * 100.0
        print(f"delta_vs_avg_pct={delta_pct:+.2f}")
    latest = samples[-5:]
    for rid, minutes, status in latest:
        print(f"run={rid} status={status} minutes={minutes:.3f}")

    for run_id in [str(x).strip() for x in args.status_run_id if str(x).strip()]:
        rows = con.execute(
            """
            SELECT status, COUNT(*)
            FROM plugin_executions
            WHERE run_id = ?
            GROUP BY status
            ORDER BY status
            """,
            (run_id,),
        ).fetchall()
        joined = ",".join(f"{status}:{count}" for status, count in rows)
        print(f"status_counts[{run_id}]={joined}")

        final_rows = con.execute(
            """
            SELECT status, COUNT(*) FROM (
                SELECT pr.status
                FROM plugin_results_v2 pr
                JOIN (
                    SELECT plugin_id, MAX(result_id) AS max_id
                    FROM plugin_results_v2
                    WHERE run_id = ?
                    GROUP BY plugin_id
                ) latest
                ON pr.plugin_id = latest.plugin_id
                AND pr.result_id = latest.max_id
                WHERE pr.run_id = ?
            )
            GROUP BY status
            ORDER BY status
            """,
            (run_id, run_id),
        ).fetchall()
        final_joined = ",".join(f"{status}:{count}" for status, count in final_rows)
        print(f"final_status_counts[{run_id}]={final_joined}")

        final_items = con.execute(
            """
            SELECT pr.plugin_id, pr.status
            FROM plugin_results_v2 pr
            JOIN (
                SELECT plugin_id, MAX(result_id) AS max_id
                FROM plugin_results_v2
                WHERE run_id = ?
                GROUP BY plugin_id
            ) latest
            ON pr.plugin_id = latest.plugin_id
            AND pr.result_id = latest.max_id
            WHERE pr.run_id = ? AND pr.status IN ('error', 'degraded')
            ORDER BY pr.status DESC, pr.plugin_id
            """,
            (run_id, run_id),
        ).fetchall()
        for plugin_id, status in final_items:
            print(f"final_issue[{run_id}]={status}:{plugin_id}")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
