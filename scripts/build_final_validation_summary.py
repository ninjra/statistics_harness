from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

import yaml


def _status_counts(conn: sqlite3.Connection, run_id: str) -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT status, COUNT(*) AS n
        FROM plugin_results_v2
        WHERE run_id = ?
        GROUP BY status
        """,
        (run_id,),
    ).fetchall()
    counts: dict[str, int] = {}
    for status, n in rows:
        key = str(status or "unknown")
        counts[key] = int(n or 0)
    return counts


def _plugin_meta() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for manifest in sorted(Path("plugins").glob("*/plugin.yaml")):
        try:
            payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        plugin_id = str(payload.get("id") or manifest.parent.name)
        capabilities = payload.get("capabilities")
        caps = [str(v).strip() for v in capabilities] if isinstance(capabilities, list) else []
        out[plugin_id] = {
            "type": str(payload.get("type") or ""),
            "diagnostic_only": "diagnostic_only" in caps,
            "sql_assist_required": "sql_assist_required" in caps,
        }
    return out


def _parse_findings_count(raw: Any) -> int:
    if isinstance(raw, list):
        return len(raw)
    if isinstance(raw, str):
        try:
            loaded = json.loads(raw)
        except Exception:
            return 0
        return len(loaded) if isinstance(loaded, list) else 0
    return 0


def build_summary(run_id: str) -> dict[str, object]:
    con = sqlite3.connect("appdata/state.sqlite")
    con.row_factory = sqlite3.Row
    try:
        run = con.execute(
            "SELECT status, created_at, completed_at FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        counts = _status_counts(con, run_id)
        expected = int(
            con.execute(
                "SELECT COUNT(*) FROM plugin_executions WHERE run_id = ?",
                (run_id,),
            ).fetchone()[0]
            or 0
        )
        done = int(sum(counts.values()))
        missing = max(expected - done, 0)
        plugin_rows = con.execute(
            """
            SELECT plugin_id, status, findings_json, summary
            FROM plugin_results_v2
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchall()
        meta = _plugin_meta()
        quality_violations: list[str] = []
        sql_assist_failures: list[str] = []
        for row in plugin_rows:
            plugin_id = str(row["plugin_id"] or "")
            status = str(row["status"] or "").lower()
            m = meta.get(plugin_id, {})
            plugin_type = str(m.get("type") or "")
            diagnostic_only = bool(m.get("diagnostic_only"))
            sql_assist_required = bool(m.get("sql_assist_required"))
            findings_count = _parse_findings_count(row["findings_json"])
            if (
                status == "ok"
                and plugin_type == "analysis"
                and not diagnostic_only
                and findings_count == 0
            ):
                quality_violations.append(plugin_id)
            if sql_assist_required and status in {"error", "aborted"}:
                sql_assist_failures.append(plugin_id)
        quality_violations = sorted(set(quality_violations))
        sql_assist_failures = sorted(set(sql_assist_failures))
        run_dir = Path("appdata") / "runs" / run_id
        manifest_summary: dict[str, Any] = {}
        manifest_path = run_dir / "run_manifest.json"
        if manifest_path.exists():
            try:
                manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                maybe = manifest_payload.get("summary")
                if isinstance(maybe, dict):
                    manifest_summary = dict(maybe)
            except Exception:
                manifest_summary = {}
        return {
            "run_id": run_id,
            "run_status": str(run["status"]) if run else None,
            "created_at": str(run["created_at"]) if run else None,
            "completed_at": str(run["completed_at"]) if run else None,
            "plugin_status_counts": counts,
            "expected_plugins": expected,
            "completed_plugin_results": done,
            "missing_plugin_results": missing,
            "analysis_ok_without_findings_count": int(len(quality_violations)),
            "analysis_ok_without_findings_plugins": quality_violations,
            "sql_assist_required_failure_count": int(len(sql_assist_failures)),
            "sql_assist_required_failure_plugins": sql_assist_failures,
            "run_manifest_summary": manifest_summary,
        }
    finally:
        con.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    summary = build_summary(str(args.run_id))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(summary, indent=2, sort_keys=True)
    out_path.write_text(payload, encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
