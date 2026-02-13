#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_appdata() -> Path:
    return _repo_root() / "appdata"


def _load_declared_plugins(plugins_dir: Path) -> dict[str, dict[str, Any]]:
    declared: dict[str, dict[str, Any]] = {}
    manifest_paths = sorted(plugins_dir.glob("*/plugin.yaml")) + sorted(
        plugins_dir.glob("*/plugin.json")
    )
    for manifest in manifest_paths:
        try:
            if manifest.suffix.lower() in {".yaml", ".yml"}:
                payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
            else:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        plugin_id = str(payload.get("id") or payload.get("plugin_id") or "").strip()
        if not plugin_id:
            continue
        declared[plugin_id] = {
            "plugin_id": plugin_id,
            "type": str(payload.get("type") or ""),
            "version": str(payload.get("version") or ""),
            "entrypoint": str(payload.get("entrypoint") or ""),
            "manifest_path": str(manifest.relative_to(_repo_root())),
        }
    return declared


def _latest_run_id(conn: sqlite3.Connection) -> str | None:
    row = conn.execute(
        """
        SELECT r.run_id
        FROM runs r
        WHERE EXISTS (
            SELECT 1 FROM plugin_results pr WHERE pr.run_id = r.run_id
        )
        ORDER BY r.created_at DESC
        LIMIT 1
        """
    ).fetchone()
    if not row:
        return None
    return str(row[0])


def _load_run_plugin_rows(conn: sqlite3.Connection, run_id: str) -> list[sqlite3.Row]:
    rows = conn.execute(
        """
        SELECT plugin_id, status, summary, findings_json, artifacts_json, error_json
        FROM plugin_results
        WHERE run_id = ?
        ORDER BY plugin_id
        """,
        (run_id,),
    ).fetchall()
    return list(rows)


def _json_len(payload: str | None, expected: type) -> int:
    if not isinstance(payload, str) or not payload.strip():
        return 0
    try:
        obj = json.loads(payload)
    except Exception:
        return 0
    if isinstance(obj, expected):
        return len(obj)
    return 0


@dataclass
class PluginAuditRow:
    plugin_id: str
    declared: bool
    plugin_type: str
    version: str
    status: str
    finding_count: int
    artifact_count: int
    has_error: bool
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "plugin_id": self.plugin_id,
            "declared": self.declared,
            "plugin_type": self.plugin_type,
            "version": self.version,
            "status": self.status,
            "finding_count": self.finding_count,
            "artifact_count": self.artifact_count,
            "has_error": self.has_error,
            "summary": self.summary,
        }


def _build_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Plugin Functional Audit")
    lines.append("")
    lines.append(f"- generated_at_utc: {payload.get('generated_at_utc')}")
    lines.append(f"- run_id: {payload.get('run_id')}")
    lines.append(f"- total_declared_plugins: {payload.get('total_declared_plugins')}")
    lines.append(f"- total_executed_plugins: {payload.get('total_executed_plugins')}")
    lines.append(f"- missing_from_run: {payload.get('missing_from_run')}")
    lines.append("")
    lines.append("## Status Counts")
    status_counts = payload.get("status_counts") or {}
    for key in sorted(status_counts.keys()):
        lines.append(f"- {key}: {status_counts[key]}")
    lines.append("")
    lines.append("## Plugin Rows")
    lines.append("| plugin_id | declared | type | status | findings | artifacts | has_error |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | --- |")
    for row in payload.get("items") or []:
        lines.append(
            f"| {row.get('plugin_id')} | {row.get('declared')} | {row.get('plugin_type')} | "
            f"{row.get('status')} | {row.get('finding_count')} | {row.get('artifact_count')} | "
            f"{row.get('has_error')} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit plugin functional execution for a run (or latest run)."
    )
    parser.add_argument("--appdata", default=str(_default_appdata()))
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()

    appdata = Path(args.appdata).resolve()
    state_db = appdata / "state.sqlite"
    if not state_db.exists():
        raise SystemExit(f"state DB not found: {state_db}")

    plugins_dir = _repo_root() / "plugins"
    declared = _load_declared_plugins(plugins_dir)

    conn = sqlite3.connect(state_db)
    conn.row_factory = sqlite3.Row
    try:
        run_id = str(args.run_id or "").strip()
        if not run_id:
            run_id = _latest_run_id(conn) or ""
        rows = _load_run_plugin_rows(conn, run_id) if run_id else []
        by_id: dict[str, PluginAuditRow] = {}
        for row in rows:
            plugin_id = str(row["plugin_id"])
            manifest = declared.get(plugin_id) or {}
            by_id[plugin_id] = PluginAuditRow(
                plugin_id=plugin_id,
                declared=plugin_id in declared,
                plugin_type=str(manifest.get("type") or ""),
                version=str(manifest.get("version") or ""),
                status=str(row["status"] or ""),
                finding_count=_json_len(row["findings_json"], list),
                artifact_count=_json_len(row["artifacts_json"], list),
                has_error=bool(str(row["error_json"] or "").strip()),
                summary=str(row["summary"] or ""),
            )

        for plugin_id, manifest in declared.items():
            if plugin_id in by_id:
                continue
            by_id[plugin_id] = PluginAuditRow(
                plugin_id=plugin_id,
                declared=True,
                plugin_type=str(manifest.get("type") or ""),
                version=str(manifest.get("version") or ""),
                status="not_executed",
                finding_count=0,
                artifact_count=0,
                has_error=False,
                summary="",
            )

        items = [row.to_dict() for _, row in sorted(by_id.items(), key=lambda kv: kv[0])]
        status_counts = Counter(str(item.get("status") or "") for item in items)
        missing_from_run = int(sum(1 for item in items if item.get("status") == "not_executed"))
        payload = {
            "generated_at_utc": _now_iso(),
            "run_id": run_id or None,
            "state_db": str(state_db),
            "total_declared_plugins": int(len(declared)),
            "total_executed_plugins": int(len(rows)),
            "missing_from_run": missing_from_run,
            "status_counts": dict(sorted(status_counts.items(), key=lambda kv: kv[0])),
            "items": items,
        }

        run_dir = appdata / "runs" / (run_id or "_no_run")
        out_dir = run_dir / "audit"
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "plugin_functional_audit.json"
        md_path = out_dir / "plugin_functional_audit.md"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        md_path.write_text(_build_markdown(payload), encoding="utf-8")

        print(f"run_id={run_id or 'none'}")
        print(f"audit_json={json_path}")
        print(f"audit_md={md_path}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
