#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _sqlite_connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    if not _table_exists(conn, table):
        return set()
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {str(row["name"] or "") for row in rows}


def _load_declared_plugins(plugins_dir: Path, root: Path) -> dict[str, dict[str, Any]]:
    declared: dict[str, dict[str, Any]] = {}
    manifest_paths = sorted(plugins_dir.glob("*/plugin.yaml")) + sorted(plugins_dir.glob("*/plugin.json"))
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
            "plugin_type": str(payload.get("type") or "").strip() or None,
            "plugin_version": str(payload.get("version") or "").strip() or None,
            "entrypoint": str(payload.get("entrypoint") or "").strip() or None,
            "manifest_path": str(manifest.relative_to(root)).replace("\\", "/"),
        }
    return declared


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_run_row(conn: sqlite3.Connection, run_id: str) -> dict[str, Any]:
    columns = _table_columns(conn, "runs")
    if not columns:
        raise SystemExit("Missing table: runs")
    wanted = [
        "run_id",
        "status",
        "dataset_version_id",
        "run_seed",
        "requested_run_seed",
        "input_hash",
        "created_at",
    ]
    selected = [col for col in wanted if col in columns]
    if "run_id" not in selected:
        raise SystemExit("runs table does not contain run_id")
    sql = f"SELECT {', '.join(selected)} FROM runs WHERE run_id = ? LIMIT 1"
    row = conn.execute(sql, (run_id,)).fetchone()
    if row is None:
        raise SystemExit(f"Run not found: {run_id}")
    out = {col: row[col] for col in selected}
    out.setdefault("status", None)
    out.setdefault("dataset_version_id", None)
    out.setdefault("run_seed", 0)
    out.setdefault("requested_run_seed", None)
    out.setdefault("input_hash", None)
    out.setdefault("created_at", None)
    return out


def _load_execution_rows(conn: sqlite3.Connection, run_id: str) -> dict[str, dict[str, Any]]:
    columns = _table_columns(conn, "plugin_executions")
    if not columns or "plugin_id" not in columns or "run_id" not in columns:
        return {}
    wanted = ["execution_id", "plugin_id", "status"]
    selected = [col for col in wanted if col in columns]
    sql = f"SELECT {', '.join(selected)} FROM plugin_executions WHERE run_id = ?"
    rows = conn.execute(sql, (run_id,)).fetchall()
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        plugin_id = str(row["plugin_id"] or "").strip()
        if not plugin_id:
            continue
        current = out.setdefault(
            plugin_id,
            {
                "execution_count": 0,
                "latest_execution_id": None,
                "execution_status": None,
            },
        )
        current["execution_count"] = int(current["execution_count"]) + 1
        eid = row["execution_id"] if "execution_id" in row.keys() else None
        status = str(row["status"] or "").strip().lower() if "status" in row.keys() else ""
        if current["latest_execution_id"] is None:
            current["latest_execution_id"] = int(eid) if isinstance(eid, int) else None
            current["execution_status"] = status or None
            continue
        if isinstance(eid, int) and isinstance(current["latest_execution_id"], int):
            if eid >= int(current["latest_execution_id"]):
                current["latest_execution_id"] = eid
                current["execution_status"] = status or current["execution_status"]
            continue
        current["execution_status"] = status or current["execution_status"]
    return out


def _load_result_rows(conn: sqlite3.Connection, run_id: str) -> dict[str, dict[str, Any]]:
    columns = _table_columns(conn, "plugin_results_v2")
    required = {"run_id", "plugin_id"}
    if not columns or not required.issubset(columns):
        return {}
    wanted = [
        "result_id",
        "plugin_id",
        "status",
        "plugin_version",
        "code_hash",
        "settings_hash",
        "execution_fingerprint",
        "dataset_hash",
    ]
    selected = [col for col in wanted if col in columns]
    sql = f"SELECT {', '.join(selected)} FROM plugin_results_v2 WHERE run_id = ?"
    rows = conn.execute(sql, (run_id,)).fetchall()
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        plugin_id = str(row["plugin_id"] or "").strip()
        if not plugin_id:
            continue
        rid = row["result_id"] if "result_id" in row.keys() else None
        prev = out.get(plugin_id)
        replace = prev is None
        if not replace and isinstance(rid, int) and isinstance(prev.get("latest_result_id"), int):
            replace = rid >= int(prev["latest_result_id"])
        if not replace and "latest_result_id" not in (prev or {}):
            replace = True
        if not replace:
            continue
        out[plugin_id] = {
            "latest_result_id": int(rid) if isinstance(rid, int) else None,
            "result_status": str(row["status"] or "").strip().lower() or None,
            "plugin_version": str(row["plugin_version"] or "").strip() or None
            if "plugin_version" in row.keys()
            else None,
            "code_hash": str(row["code_hash"] or "").strip() or None
            if "code_hash" in row.keys()
            else None,
            "settings_hash": str(row["settings_hash"] or "").strip() or None
            if "settings_hash" in row.keys()
            else None,
            "execution_fingerprint": str(row["execution_fingerprint"] or "").strip() or None
            if "execution_fingerprint" in row.keys()
            else None,
            "dataset_hash": str(row["dataset_hash"] or "").strip() or None
            if "dataset_hash" in row.keys()
            else None,
        }
    return out


def _plugin_rows_from_run_manifest(run_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    plugins = run_manifest.get("plugins")
    if not isinstance(plugins, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in plugins:
        if not isinstance(row, dict):
            continue
        plugin_id = str(row.get("plugin_id") or "").strip()
        if not plugin_id:
            continue
        out[plugin_id] = {
            "result_status": str(row.get("status") or "").strip().lower() or None,
            "plugin_version": str(row.get("plugin_version") or "").strip() or None,
            "code_hash": str(row.get("code_hash") or "").strip() or None,
            "settings_hash": str(row.get("settings_hash") or "").strip() or None,
            "execution_fingerprint": str(row.get("execution_fingerprint") or "").strip() or None,
        }
    return out


def build_plugin_run_manifest(root: Path, run_id: str, db_path: Path) -> dict[str, Any]:
    run_dir = root / "appdata" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest = _load_json(run_manifest_path)
    declared = _load_declared_plugins(root / "plugins", root)
    with _sqlite_connect(db_path) as conn:
        run_row = _load_run_row(conn, run_id)
        execution_rows = _load_execution_rows(conn, run_id)
        result_rows = _load_result_rows(conn, run_id)
    run_manifest_rows = _plugin_rows_from_run_manifest(run_manifest)

    plugin_ids = sorted(set(declared) | set(execution_rows) | set(result_rows) | set(run_manifest_rows))
    items: list[dict[str, Any]] = []
    for plugin_id in plugin_ids:
        declared_row = declared.get(plugin_id, {})
        execution_row = execution_rows.get(plugin_id, {})
        result_row = result_rows.get(plugin_id, {})
        run_manifest_row = run_manifest_rows.get(plugin_id, {})
        execution_count = int(execution_row.get("execution_count") or 0)
        result_status = result_row.get("result_status") or run_manifest_row.get("result_status")
        has_result = bool(result_row) or bool(result_status)
        row = {
            "plugin_id": plugin_id,
            "declared": bool(plugin_id in declared),
            "executed": bool(execution_count > 0 or has_result),
            "has_result": bool(has_result),
            "execution_count": execution_count,
            "execution_status": execution_row.get("execution_status"),
            "result_status": result_status,
            "plugin_type": declared_row.get("plugin_type"),
            "plugin_version": result_row.get("plugin_version")
            or run_manifest_row.get("plugin_version")
            or declared_row.get("plugin_version"),
            "entrypoint": declared_row.get("entrypoint"),
            "code_hash": result_row.get("code_hash") or run_manifest_row.get("code_hash"),
            "settings_hash": result_row.get("settings_hash") or run_manifest_row.get("settings_hash"),
            "execution_fingerprint": result_row.get("execution_fingerprint")
            or run_manifest_row.get("execution_fingerprint"),
            "dataset_hash": result_row.get("dataset_hash"),
            "latest_execution_id": execution_row.get("latest_execution_id"),
            "latest_result_id": result_row.get("latest_result_id"),
        }
        items.append(row)

    registry_material = [
        {
            "plugin_id": row["plugin_id"],
            "declared": bool(row["declared"]),
            "plugin_type": row["plugin_type"],
            "plugin_version": row["plugin_version"],
            "entrypoint": row["entrypoint"],
        }
        for row in items
    ]
    dataset_hash = str(run_row.get("input_hash") or "").strip() or None
    if dataset_hash is None:
        dataset_hash = (
            str(run_manifest.get("input", {}).get("input_hash") or "").strip()
            if isinstance(run_manifest.get("input"), dict)
            else ""
        ) or None
    if dataset_hash is None:
        for row in items:
            candidate = str(row.get("dataset_hash") or "").strip()
            if candidate:
                dataset_hash = candidate
                break

    payload = {
        "schema_version": "plugin_run_manifest.v1",
        "generated_at": _now_iso(),
        "run_id": run_id,
        "run_seed": int(run_row.get("run_seed") or 0),
        "requested_run_seed": (
            int(run_row["requested_run_seed"])
            if isinstance(run_row.get("requested_run_seed"), int)
            else None
        ),
        "dataset_hash": dataset_hash,
        "dataset_version_id": (
            str(run_row.get("dataset_version_id") or "").strip() or None
        ),
        "run_status": str(run_row.get("status") or "").strip() or None,
        "run_manifest_path": (
            str(run_manifest_path.relative_to(root)).replace("\\", "/")
            if run_manifest_path.exists()
            else None
        ),
        "plugin_registry_hash": _json_hash(registry_material),
        "plugin_count": int(len(items)),
        "plugins": items,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build canonical plugin_run_manifest.json for a run."
    )
    parser.add_argument("--root", default=str(_repo_root()))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--db", default="")
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    db_path = Path(args.db).resolve() if str(args.db).strip() else root / "appdata" / "state.sqlite"
    if not db_path.exists():
        raise SystemExit(f"Missing DB: {db_path}")
    payload = build_plugin_run_manifest(root=root, run_id=str(args.run_id), db_path=db_path)
    out_path = (
        Path(args.out_json).resolve()
        if str(args.out_json).strip()
        else root / "appdata" / "runs" / str(args.run_id) / "plugin_run_manifest.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"run_id={payload.get('run_id')}")
    print(f"plugin_count={payload.get('plugin_count')}")
    print(f"out_json={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
