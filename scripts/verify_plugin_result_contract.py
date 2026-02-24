#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "appdata" / "state.sqlite"

_PASS = {"ok"}
_NA = {"na"}
_FAIL = {"error", "aborted", "degraded", "skipped"}
_KNOWN = _PASS | _NA | _FAIL


def _connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def _loads_obj(text: Any) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        return {}
    try:
        payload = json.loads(text)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _loads_list(text: Any) -> list[dict[str, Any]]:
    if not isinstance(text, str) or not text.strip():
        return []
    try:
        payload = json.loads(text)
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def _classify(status: str) -> str:
    s = str(status or "").strip().lower()
    if s in _PASS:
        return "pass"
    if s in _NA:
        return "na"
    if s in _FAIL:
        return "fail"
    return "fail"


def _run_row(conn: sqlite3.Connection, run_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT run_id, status, dataset_version_id, created_at, completed_at FROM runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    return dict(row) if row else None


def _latest_execution_rows(conn: sqlite3.Connection, run_id: str) -> dict[str, dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT e.plugin_id, e.status, e.exit_code
        FROM plugin_executions e
        JOIN (
          SELECT plugin_id, MAX(execution_id) AS max_execution_id
          FROM plugin_executions
          WHERE run_id = ?
          GROUP BY plugin_id
        ) m
          ON e.plugin_id = m.plugin_id AND e.execution_id = m.max_execution_id
        WHERE e.run_id = ?
        """,
        (run_id, run_id),
    ).fetchall()
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        plugin_id = str(row["plugin_id"] or "").strip()
        if not plugin_id:
            continue
        out[plugin_id] = {
            "execution_status": str(row["status"] or "").strip().lower(),
            "exit_code": row["exit_code"],
        }
    return out


def _latest_result_rows(conn: sqlite3.Connection, run_id: str) -> dict[str, dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT r.plugin_id, r.status, r.summary, r.findings_json, r.error_json
        FROM plugin_results_v2 r
        JOIN (
          SELECT plugin_id, MAX(result_id) AS max_result_id
          FROM plugin_results_v2
          WHERE run_id = ?
          GROUP BY plugin_id
        ) m
          ON r.plugin_id = m.plugin_id AND r.result_id = m.max_result_id
        WHERE r.run_id = ?
        """,
        (run_id, run_id),
    ).fetchall()
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        plugin_id = str(row["plugin_id"] or "").strip()
        if not plugin_id:
            continue
        out[plugin_id] = {
            "result_status": str(row["status"] or "").strip().lower(),
            "summary": str(row["summary"] or "").strip(),
            "findings": _loads_list(row["findings_json"]),
            "error": _loads_obj(row["error_json"]),
        }
    return out


def _report_maps(run_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    report_path = run_dir / "report.json"
    if not report_path.exists():
        return {}, {}
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return {}, {}
    if not isinstance(payload, dict):
        return {}, {}
    plugins = payload.get("plugins") if isinstance(payload.get("plugins"), dict) else {}
    recs = payload.get("recommendations") if isinstance(payload.get("recommendations"), dict) else {}
    explanations = recs.get("explanations") if isinstance(recs.get("explanations"), dict) else {}
    explanation_items = explanations.get("items") if isinstance(explanations.get("items"), list) else []
    explanation_by_plugin: dict[str, dict[str, Any]] = {}
    for row in explanation_items:
        if not isinstance(row, dict):
            continue
        plugin_id = str(row.get("plugin_id") or "").strip()
        if plugin_id and plugin_id not in explanation_by_plugin:
            explanation_by_plugin[plugin_id] = row
    typed_plugins = {
        str(pid).strip(): value
        for pid, value in plugins.items()
        if str(pid).strip() and isinstance(value, dict)
    }
    return typed_plugins, explanation_by_plugin


def _finding_reason_code(findings: list[dict[str, Any]]) -> str:
    for row in findings:
        code = str(row.get("reason_code") or "").strip()
        if code:
            return code
    return ""


def audit_plugin_contract(db_path: Path, run_id: str) -> dict[str, Any]:
    con = _connect(db_path)
    try:
        run = _run_row(con, run_id)
        if not isinstance(run, dict):
            raise SystemExit(f"Run not found: {run_id}")
        execution_rows = _latest_execution_rows(con, run_id)
        result_rows = _latest_result_rows(con, run_id)
    finally:
        con.close()

    run_dir = ROOT / "appdata" / "runs" / run_id
    report_plugins, explanations = _report_maps(run_dir)

    plugin_ids = sorted(set(execution_rows.keys()) | set(result_rows.keys()) | set(report_plugins.keys()))
    rows: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()

    for plugin_id in plugin_ids:
        execution = execution_rows.get(plugin_id, {})
        result = result_rows.get(plugin_id, {})
        report_plugin = report_plugins.get(plugin_id, {})
        explanation = explanations.get(plugin_id, {})

        result_status = str(result.get("result_status") or execution.get("execution_status") or "").strip().lower()
        classification = _classify(result_status)
        class_counts[classification] += 1
        status_counts[result_status or "missing"] += 1

        findings = result.get("findings") if isinstance(result.get("findings"), list) else []
        finding_reason = _finding_reason_code(findings)
        explanation_reason = str(explanation.get("reason_code") or "").strip()
        summary = str(result.get("summary") or "").strip()
        has_error = bool(result.get("error"))
        has_report_snapshot = bool(report_plugin)

        reason_code = explanation_reason or finding_reason
        deterministic_reason_ok = bool(reason_code or summary or has_error)

        row_violations: list[str] = []
        if not result:
            row_violations.append("MISSING_RESULT_RECORD")
        if result_status and result_status not in _KNOWN:
            row_violations.append("UNKNOWN_RESULT_STATUS")
        if classification == "na":
            if not bool(reason_code):
                row_violations.append("NA_REASON_CODE_MISSING")
        if classification == "fail":
            if not deterministic_reason_ok:
                row_violations.append("FAIL_REASON_MISSING")
        if not has_report_snapshot:
            row_violations.append("REPORT_PLUGIN_SNAPSHOT_MISSING")

        if row_violations:
            violations.append({"plugin_id": plugin_id, "violations": row_violations})

        rows.append(
            {
                "plugin_id": plugin_id,
                "result_status": result_status or None,
                "classification": classification,
                "summary": summary or None,
                "reason_code": reason_code or None,
                "has_error_payload": bool(has_error),
                "has_report_snapshot": bool(has_report_snapshot),
                "violations": row_violations,
            }
        )

    payload = {
        "schema_version": "plugin_result_contract.v1",
        "run": run,
        "plugin_count": int(len(rows)),
        "classification_counts": {k: int(v) for k, v in sorted(class_counts.items())},
        "result_status_counts": {k: int(v) for k, v in sorted(status_counts.items())},
        "violation_count": int(len(violations)),
        "violations": violations,
        "plugins": rows,
        "ok": int(len(violations)) == 0,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify per-plugin result contract: pass/fail/na with deterministic reasons."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--out-json", default="")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    db_path = Path(str(args.db)).resolve()
    if not db_path.exists():
        raise SystemExit(f"Missing DB: {db_path}")

    payload = audit_plugin_contract(db_path, str(args.run_id).strip())
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    out_json = str(args.out_json).strip()
    if out_json:
        out_path = Path(out_json)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if bool(args.strict) and not bool(payload.get("ok")):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
