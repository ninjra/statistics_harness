#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class RunStatus:
    run_id: str
    status: str | None
    created_at: str | None
    input_filename: str | None
    elapsed_seconds: float | None
    expected_plugins: list[str]
    expected_executable_plugins: list[str]
    plugin_total: int
    plugin_done: int
    plugin_running: int
    running_plugins: list[str]
    running_details: list[tuple[str, str | None]]
    outputs_present: dict[str, bool]
    eta_seconds: float | None


def _connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def _get_run_status(appdata_dir: Path, run_id: str) -> RunStatus:
    db_path = appdata_dir / "state.sqlite"
    con = _connect(db_path)
    try:
        row = con.execute(
            "select status, created_at, input_filename from runs where run_id=?",
            (run_id,),
        ).fetchone()
        status = str(row["status"]) if row is not None else None
        created_at = str(row["created_at"]) if (row is not None and row["created_at"] is not None) else None
        input_filename = (
            str(row["input_filename"]) if (row is not None and row["input_filename"] is not None) else None
        )
        elapsed_seconds: float | None = None
        if created_at:
            try:
                started = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                elapsed_seconds = max(0.0, (now - started).total_seconds())
            except Exception:
                elapsed_seconds = None

        # Expected plugin list is recorded once via the run_fingerprint event.
        expected_plugins: list[str] = []
        evt = con.execute(
            "select payload_json from events where run_id=? and kind='run_fingerprint' order by created_at desc limit 1",
            (run_id,),
        ).fetchone()
        if evt is not None and evt["payload_json"]:
            try:
                payload = json.loads(evt["payload_json"])
                if isinstance(payload, dict) and isinstance(payload.get("plugins"), list):
                    expected_plugins = [str(x) for x in payload["plugins"] if isinstance(x, str)]
            except Exception:
                expected_plugins = []

        # DB-only runs never execute ingest_tabular (file-driven), even if listed as a dependency.
        expected_executable = sorted(set(expected_plugins))
        if input_filename and input_filename.startswith("db://"):
            expected_executable = [p for p in expected_executable if p != "ingest_tabular"]

        plugin_total = int(
            con.execute(
                "select count(*) as c from plugin_executions where run_id=?",
                (run_id,),
            ).fetchone()["c"]
        )
        plugin_done = int(
            con.execute(
                "select count(*) as c from plugin_executions where run_id=? and status in "
                "('ok','skipped','degraded','error','aborted')",
                (run_id,),
            ).fetchone()["c"]
        )
        plugin_running = int(
            con.execute(
                "select count(*) as c from plugin_executions where run_id=? and status='running'",
                (run_id,),
            ).fetchone()["c"]
        )
        running_plugins = [
            str(r["plugin_id"])
            for r in con.execute(
                "select plugin_id from plugin_executions where run_id=? and status='running' order by plugin_id",
                (run_id,),
            ).fetchall()
        ]
        running_details = [
            (str(r["plugin_id"]), (str(r["started_at"]) if r["started_at"] is not None else None))
            for r in con.execute(
                "select plugin_id, started_at from plugin_executions where run_id=? and status='running' order by plugin_id",
                (run_id,),
            ).fetchall()
        ]

        # ETA (very rough): estimate throughput from wall-clock elapsed time and completed plugins.
        eta_seconds: float | None = None
        # Avoid wildly misleading ETAs early in a run. Require at least a few completed plugins.
        if status == "running" and elapsed_seconds is not None and expected_executable and plugin_done >= 5:
            try:
                elapsed = max(1.0, float(elapsed_seconds))
                remaining = max(0, len(set(expected_executable)) - plugin_done)
                throughput = plugin_done / elapsed  # plugins/sec (wall clock)
                if throughput > 0:
                    eta_seconds = remaining / throughput
            except Exception:
                eta_seconds = None
    finally:
        con.close()

    run_dir = appdata_dir / "runs" / run_id

    def _journal_pid_and_started_at() -> tuple[int | None, str | None]:
        journal = run_dir / "journal.json"
        if not journal.exists():
            return None, None
        try:
            payload = json.loads(journal.read_text(encoding="utf-8"))
        except Exception:
            return None, None
        if not isinstance(payload, dict):
            return None, None
        pid = payload.get("pid")
        started_at = payload.get("started_at") or payload.get("created_at")
        if isinstance(pid, int) and pid > 0:
            return pid, (str(started_at) if started_at is not None else None)
        return None, (str(started_at) if started_at is not None else None)

    def _pid_is_same_process(pid: int, started_at: str | None) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        if not started_at:
            return True
        try:
            from datetime import timedelta

            run_started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            if run_started.tzinfo is None:
                run_started = run_started.replace(tzinfo=timezone.utc)
            stat_path = Path(f"/proc/{pid}/stat")
            uptime_path = Path("/proc/uptime")
            if not stat_path.exists() or not uptime_path.exists():
                return True
            stat = stat_path.read_text(encoding="utf-8", errors="ignore")
            parts = stat.split()
            if len(parts) < 22:
                return True
            start_ticks = int(parts[21])
            hz = int(os.sysconf(os.sysconf_names["SC_CLK_TCK"]))
            uptime_s = float(uptime_path.read_text(encoding="utf-8", errors="ignore").split()[0])
            age_s = max(0.0, uptime_s - (float(start_ticks) / float(hz)))
            proc_started = datetime.now(timezone.utc) - timedelta(seconds=age_s)
            return abs((proc_started - run_started).total_seconds()) <= 180.0
        except Exception:
            return True

    # Fail closed: if DB says "running" but the runner PID is not alive (or appears reused),
    # treat the run as aborted/stale for status reporting.
    effective_status = status
    if status == "running":
        pid, started_at = _journal_pid_and_started_at()
        if isinstance(pid, int) and pid > 0 and _pid_is_same_process(pid, started_at):
            effective_status = "running"
        else:
            effective_status = "aborted"

    # If we detected a stale run, suppress "running plugin" counts to avoid misleading output.
    stale_aborted = (status == "running" and effective_status == "aborted")
    if stale_aborted:
        plugin_running = 0
        running_plugins = []
        running_details = []

    outputs_present = {
        "report_md": (run_dir / "report.md").is_file(),
        "report_json": (run_dir / "report.json").is_file(),
        "answers_summary_json": (run_dir / "answers_summary.json").is_file(),
        "answers_recommendations_md": (run_dir / "answers_recommendations.md").is_file(),
    }

    return RunStatus(
        run_id=run_id,
        status=effective_status,
        created_at=created_at,
        input_filename=input_filename,
        elapsed_seconds=elapsed_seconds,
        expected_plugins=sorted(set(expected_plugins)),
        expected_executable_plugins=expected_executable,
        plugin_total=plugin_total,
        plugin_done=plugin_done,
        plugin_running=plugin_running,
        running_plugins=running_plugins,
        running_details=running_details,
        outputs_present=outputs_present,
        eta_seconds=eta_seconds,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--appdata", type=Path, default=Path("appdata"))
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()

    rs = _get_run_status(args.appdata, args.run_id)
    print(
        f"run_id={rs.run_id} status={rs.status} plugins={rs.plugin_total} done={rs.plugin_done} running={rs.plugin_running}"
    )
    if rs.created_at:
        print(f"run_created_at={rs.created_at}")
    if rs.elapsed_seconds is not None:
        print(f"run_elapsed_minutes~={rs.elapsed_seconds/60.0:.1f}")
    if rs.input_filename:
        print(f"run_input={rs.input_filename}")
    if rs.expected_plugins:
        print(f"expected_plugins_all={len(rs.expected_plugins)}")
    if rs.expected_executable_plugins:
        print(f"expected_plugins_executable={len(rs.expected_executable_plugins)}")
        pct = (rs.plugin_done / max(1, len(rs.expected_executable_plugins))) * 100.0
        print(f"progress_pct={pct:.1f}")
    if rs.eta_seconds is not None:
        # Keep it simple and stable for shell parsing.
        eta_min = rs.eta_seconds / 60.0
        print(f"eta_minutes~={eta_min:.1f}")
        if rs.elapsed_seconds is not None:
            print(f"estimated_total_minutes~={(rs.elapsed_seconds/60.0)+eta_min:.1f}")
    if rs.running_plugins:
        print("running_plugins=" + ",".join(rs.running_plugins))
    for plugin_id, started_at in rs.running_details:
        if started_at:
            print(f"running_started_at[{plugin_id}]={started_at}")
    for k, v in rs.outputs_present.items():
        print(f"{k}={int(v)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
