from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


def _resolve_run_id(conn: sqlite3.Connection, requested: str | None) -> str:
    run_id = str(requested or "").strip()
    if run_id:
        return run_id
    row = conn.execute(
        "SELECT run_id FROM runs ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    if not row:
        raise SystemExit("No runs found in state.sqlite")
    return str(row[0])


def _parse_findings(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, str):
        try:
            loaded = json.loads(raw)
        except Exception:
            return []
        if isinstance(loaded, list):
            return [item for item in loaded if isinstance(item, dict)]
    return []


def _plugin_rows(conn: sqlite3.Connection, run_id: str) -> list[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    return list(
        conn.execute(
            """
            SELECT plugin_id, status, findings_json
            FROM plugin_results_v2
            WHERE run_id = ?
            ORDER BY plugin_id
            """,
            (run_id,),
        ).fetchall()
    )


def validate_run(db_path: Path, run_id: str) -> dict[str, Any]:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = _plugin_rows(conn, run_id)
    finally:
        conn.close()

    blank_kind_plugins: dict[str, int] = {}
    missing_measurement_plugins: dict[str, int] = {}
    for row in rows:
        plugin_id = str(row["plugin_id"] or "").strip()
        findings = _parse_findings(row["findings_json"])
        blank = sum(1 for f in findings if not str(f.get("kind") or "").strip())
        if blank > 0:
            blank_kind_plugins[plugin_id] = int(blank)
        missing_measurement = sum(
            1 for f in findings if not str(f.get("measurement_type") or "").strip()
        )
        if missing_measurement > 0:
            missing_measurement_plugins[plugin_id] = int(missing_measurement)

    violations = []
    if blank_kind_plugins:
        violations.append("blank_kind_findings")
    if missing_measurement_plugins:
        violations.append("missing_measurement_type")

    return {
        "run_id": run_id,
        "plugin_count": len(rows),
        "blank_kind_findings_count": int(sum(blank_kind_plugins.values())),
        "plugins_with_blank_kind_findings": blank_kind_plugins,
        "missing_measurement_type_count": int(sum(missing_measurement_plugins.values())),
        "plugins_with_missing_measurement_type": missing_measurement_plugins,
        "violations": violations,
        "ok": not violations,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="appdata/state.sqlite")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    db_path = Path(str(args.db))
    if not db_path.exists():
        raise SystemExit(f"Missing database: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        run_id = _resolve_run_id(conn, str(args.run_id or ""))
    finally:
        conn.close()

    result = validate_run(db_path, run_id)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if str(args.out or "").strip():
        out_path = Path(str(args.out))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if bool(result.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
