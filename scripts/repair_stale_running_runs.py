#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


@dataclass(frozen=True)
class RepairResult:
    scanned: int
    repaired: int
    repaired_run_ids: list[str]


def _connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


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
        # Tolerance: within 3 minutes of journal started_at.
        return abs((proc_started - run_started).total_seconds()) <= 180.0
    except Exception:
        return True


def _read_journal(run_dir: Path) -> tuple[int | None, str | None]:
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
    pid_i = pid if isinstance(pid, int) else None
    started_s = str(started_at) if started_at is not None else None
    return pid_i, started_s


def _parse_iso8601(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def repair_stale_runs(appdata: Path, startup_grace_seconds: int = 180) -> RepairResult:
    db_path = appdata / "state.sqlite"
    runs_root = appdata / "runs"
    con = _connect(db_path)
    repaired: list[str] = []
    scanned = 0
    try:
        cols = [r["name"] for r in con.execute("pragma table_info(runs)").fetchall()]
        has_completed_at = "completed_at" in cols
        has_error_json = "error_json" in cols
        now = datetime.now(timezone.utc).isoformat()

        rows = con.execute(
            "select run_id, created_at from runs where status='running' order by created_at",
        ).fetchall()
        for r in rows:
            run_id = str(r["run_id"])
            scanned += 1
            pid, started_at = _read_journal(runs_root / run_id)
            if not (isinstance(pid, int) and pid > 0):
                # Fresh runs may be inserted before their run journal has pid metadata.
                # Fail closed for genuinely stale runs, but avoid aborting active startup.
                created_at = _parse_iso8601(str(r["created_at"]) if r["created_at"] is not None else None)
                if created_at is not None:
                    age_s = (datetime.now(timezone.utc) - created_at).total_seconds()
                    if age_s <= float(max(0, int(startup_grace_seconds))):
                        continue
            if isinstance(pid, int) and pid > 0 and _pid_is_same_process(pid, started_at):
                continue
            # Stale: mark aborted and also abort any still-running plugin executions.
            set_parts = ["status=?"]
            params: list[object] = ["aborted"]
            if has_completed_at:
                set_parts.append("completed_at=coalesce(completed_at, ?)")
                params.append(now)
            if has_error_json:
                set_parts.append("error_json=coalesce(error_json, ?)")
                params.append(
                    json.dumps(
                        {
                            "type": "CrashRecovery",
                            "message": "Marked aborted by repair_stale_running_runs.py (runner pid not alive).",
                        },
                        sort_keys=True,
                    )
                )
            params.append(run_id)
            con.execute(
                "update runs set " + ", ".join(set_parts) + " where run_id=?",
                tuple(params),
            )
            con.execute(
                "update plugin_executions set status='aborted', completed_at=coalesce(completed_at, ?), "
                "duration_ms=coalesce(duration_ms, 0), stderr=coalesce(stderr, ?) "
                "where run_id=? and status='running'",
                (now, "Aborted by stale-run repair.", run_id),
            )
            repaired.append(run_id)

        if repaired:
            con.commit()
    finally:
        con.close()
    return RepairResult(scanned=scanned, repaired=len(repaired), repaired_run_ids=repaired)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--appdata", type=Path, default=Path("appdata"))
    ap.add_argument("--startup-grace-seconds", type=int, default=180)
    args = ap.parse_args()
    res = repair_stale_runs(args.appdata, startup_grace_seconds=int(args.startup_grace_seconds))
    print(f"scanned={res.scanned} repaired={res.repaired}")
    for rid in res.repaired_run_ids:
        print(f"repaired_run_id={rid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
