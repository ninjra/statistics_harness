#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from statistic_harness.core.plain_report import build_plain_report
from statistic_harness.core.storage import Storage
from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import now_iso


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--appdata", type=Path, default=Path("appdata"))
    args = ap.parse_args()

    run_id = str(args.run_id).strip()
    appdata = Path(args.appdata)
    run_dir = appdata / "runs" / run_id
    report_path = run_dir / "report.json"
    if not report_path.exists():
        raise SystemExit(f"Missing report.json: {report_path}")

    text = build_plain_report(report_path)
    out_path = run_dir / "plain_report.md"
    out_path.write_text(text, encoding="utf-8")

    storage = Storage(appdata / "state.sqlite", tenant_id=None, mode="rw", initialize=False)
    plugin_id = "report_plain_english_v1"

    executed_at = now_iso()

    # Update the latest execution row for this plugin to ok (if present).
    execution_id = None
    with storage.connection() as conn:
        row = conn.execute(
            "select execution_id from plugin_executions where run_id=? and plugin_id=? order by execution_id desc limit 1",
            (run_id, plugin_id),
        ).fetchone()
        if row:
            execution_id = int(row[0])
    if execution_id is not None:
        storage.update_plugin_execution(
            execution_id=execution_id,
            completed_at=executed_at,
            duration_ms=0,
            status="ok",
            exit_code=0,
            cpu_user=None,
            cpu_system=None,
            max_rss=None,
            warnings_count=None,
            stdout="",
            stderr="",
        )

    # Preserve hashes/version if available so the DB has consistent metadata.
    plugin_version = None
    code_hash = None
    settings_hash = None
    dataset_hash = None
    with storage.connection() as conn:
        row = conn.execute(
            "select plugin_version, code_hash, settings_hash, dataset_hash from plugin_results_v2 where run_id=? and plugin_id=? order by result_id desc limit 1",
            (run_id, plugin_id),
        ).fetchone()
        if row:
            plugin_version = row[0]
            code_hash = row[1]
            settings_hash = row[2]
            dataset_hash = row[3]

    result = PluginResult(
        status="ok",
        summary="Wrote plain_report.md (repair)",
        metrics={},
        findings=[],
        artifacts=[
            PluginArtifact(path="plain_report.md", type="markdown", description="Plain-English report"),
        ],
        error=None,
        references=[],
        debug={"repair": True},
        budget=None,
    )
    storage.save_plugin_result(
        run_id,
        plugin_id,
        plugin_version,
        executed_at,
        code_hash,
        settings_hash,
        dataset_hash,
        result,
        execution_fingerprint=None,
    )

    # If this was the only failure, flip the run status back to completed.
    latest = storage.fetch_plugin_results(run_id)
    any_errors = any(str(r.get("status") or "") == "error" for r in latest)
    storage.update_run_status(run_id, status="partial" if any_errors else "completed", error=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
