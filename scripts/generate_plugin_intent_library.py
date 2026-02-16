#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
PLUGINS_ROOT = ROOT / "plugins"
FUNCTIONALITY_MATRIX_PATH = ROOT / "docs" / "plugins_functionality_matrix.json"
DATA_ACCESS_MATRIX_PATH = ROOT / "docs" / "plugin_data_access_matrix.json"
SQL_ASSIST_MATRIX_PATH = ROOT / "docs" / "sql_assist_adoption_matrix.json"
DEFAULT_JSON_OUT = ROOT / "docs" / "plugin_intent_library.json"
DEFAULT_MD_OUT = ROOT / "docs" / "plugin_intent_library.md"


@dataclass(frozen=True)
class PluginIntentRow:
    plugin_id: str
    plugin_type: str
    name: str
    description: str
    entrypoint: str
    depends_on: list[str]
    capabilities: list[str]
    access_contracts: list[str]
    sql_intent: str
    sql_intent_source: str
    uses_sql_effective: bool
    manifest_path: str
    config_schema_path: str | None
    output_schema_path: str | None


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return payload


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _manifest_rows(plugins_root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for pdir in sorted([p for p in plugins_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        manifest_path = pdir / "plugin.yaml"
        if not manifest_path.exists():
            continue
        manifest = _read_yaml(manifest_path)
        plugin_id = str(manifest.get("id") or pdir.name).strip()
        if not plugin_id:
            continue
        out[plugin_id] = {
            "manifest": manifest,
            "manifest_path": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
            "config_schema_path": str((pdir / "config.schema.json").relative_to(ROOT)).replace("\\", "/")
            if (pdir / "config.schema.json").exists()
            else None,
            "output_schema_path": str((pdir / "output.schema.json").relative_to(ROOT)).replace("\\", "/")
            if (pdir / "output.schema.json").exists()
            else None,
        }
    return out


def build_rows(
    plugins_root: Path,
    functionality_matrix: dict[str, Any],
    data_access_matrix: dict[str, Any],
    sql_assist_matrix: dict[str, Any],
) -> list[PluginIntentRow]:
    manifests = _manifest_rows(plugins_root)

    functional_plugins = functionality_matrix.get("plugins") or {}
    if not isinstance(functional_plugins, dict):
        functional_plugins = {}

    access_rows_raw = data_access_matrix.get("plugins") or []
    access_by_plugin: dict[str, dict[str, Any]] = {}
    if isinstance(access_rows_raw, list):
        for row in access_rows_raw:
            if not isinstance(row, dict):
                continue
            plugin_id = str(row.get("plugin_id") or "").strip()
            if plugin_id:
                access_by_plugin[plugin_id] = row

    sql_rows_raw = sql_assist_matrix.get("plugins") or []
    sql_by_plugin: dict[str, dict[str, Any]] = {}
    if isinstance(sql_rows_raw, list):
        for row in sql_rows_raw:
            if not isinstance(row, dict):
                continue
            plugin_id = str(row.get("plugin_id") or "").strip()
            if plugin_id:
                sql_by_plugin[plugin_id] = row

    plugin_ids = sorted(set(functional_plugins.keys()) | set(manifests.keys()) | set(access_by_plugin.keys()) | set(sql_by_plugin.keys()))
    rows: list[PluginIntentRow] = []
    for plugin_id in plugin_ids:
        functional = functional_plugins.get(plugin_id) if isinstance(functional_plugins.get(plugin_id), dict) else {}
        manifest_meta = manifests.get(plugin_id, {})
        manifest = manifest_meta.get("manifest") if isinstance(manifest_meta.get("manifest"), dict) else {}
        access_row = access_by_plugin.get(plugin_id, {})
        sql_row = sql_by_plugin.get(plugin_id, {})

        plugin_type = str(functional.get("type") or manifest.get("type") or access_row.get("plugin_type") or sql_row.get("plugin_type") or "").strip()
        name = str(functional.get("name") or manifest.get("name") or plugin_id).strip()
        description = str(functional.get("description") or manifest.get("description") or "").strip()
        entrypoint = str(functional.get("entrypoint") or manifest.get("entrypoint") or "").strip()
        depends_on = sorted([str(x) for x in (functional.get("depends_on") or manifest.get("depends_on") or []) if isinstance(x, str)])
        capabilities = sorted([str(x) for x in (functional.get("capabilities") or manifest.get("capabilities") or []) if isinstance(x, str)])
        access_contracts = sorted([str(x) for x in (access_row.get("access_contracts") or []) if isinstance(x, str)])
        sql_intent = str(sql_row.get("sql_intent") or "").strip()
        sql_intent_source = str(sql_row.get("sql_intent_source") or "").strip()
        uses_sql_effective = bool(sql_row.get("uses_sql_effective", False))

        rows.append(
            PluginIntentRow(
                plugin_id=plugin_id,
                plugin_type=plugin_type,
                name=name,
                description=description,
                entrypoint=entrypoint,
                depends_on=depends_on,
                capabilities=capabilities,
                access_contracts=access_contracts,
                sql_intent=sql_intent,
                sql_intent_source=sql_intent_source,
                uses_sql_effective=uses_sql_effective,
                manifest_path=str(manifest_meta.get("manifest_path") or ""),
                config_schema_path=manifest_meta.get("config_schema_path"),
                output_schema_path=manifest_meta.get("output_schema_path"),
            )
        )
    return rows


def build_payload(rows: list[PluginIntentRow]) -> dict[str, Any]:
    by_type: dict[str, int] = {}
    by_sql_intent: dict[str, int] = {}
    for row in rows:
        by_type[row.plugin_type] = by_type.get(row.plugin_type, 0) + 1
        by_sql_intent[row.sql_intent] = by_sql_intent.get(row.sql_intent, 0) + 1

    return {
        "schema_version": "plugin_intent_library.v1",
        "generated_by": "scripts/generate_plugin_intent_library.py",
        "plugin_count": len(rows),
        "counts": {
            "by_type": dict(sorted(by_type.items(), key=lambda item: item[0])),
            "by_sql_intent": dict(sorted(by_sql_intent.items(), key=lambda item: item[0])),
        },
        "plugins": [
            {
                "plugin_id": row.plugin_id,
                "plugin_type": row.plugin_type,
                "name": row.name,
                "description": row.description,
                "entrypoint": row.entrypoint,
                "depends_on": row.depends_on,
                "capabilities": row.capabilities,
                "access_contracts": row.access_contracts,
                "sql_intent": row.sql_intent,
                "sql_intent_source": row.sql_intent_source,
                "uses_sql_effective": row.uses_sql_effective,
                "manifest_path": row.manifest_path,
                "config_schema_path": row.config_schema_path,
                "output_schema_path": row.output_schema_path,
            }
            for row in rows
        ],
    }


def build_markdown(rows: list[PluginIntentRow]) -> str:
    lines: list[str] = []
    lines.append("# Plugin Intent Library")
    lines.append("")
    lines.append("Consolidated plugin purpose and contract index generated from existing matrices/manifests.")
    lines.append("")
    lines.append(f"- Total plugins: {len(rows)}")
    lines.append("")
    lines.append("| plugin_id | type | name | depends_on | access_contracts | sql_intent | uses_sql_effective |")
    lines.append("|---|---|---|---:|---|---|---:|")
    for row in rows:
        access_contracts = ", ".join(row.access_contracts) if row.access_contracts else "none"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.plugin_id}`",
                    row.plugin_type or "unknown",
                    row.name.replace("|", "/"),
                    str(len(row.depends_on)),
                    access_contracts.replace("|", "/"),
                    (row.sql_intent or "unspecified").replace("|", "/"),
                    "1" if row.uses_sql_effective else "0",
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _stable_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate consolidated plugin intent library artifacts.")
    parser.add_argument("--plugins-root", type=Path, default=PLUGINS_ROOT)
    parser.add_argument("--functionality-matrix", type=Path, default=FUNCTIONALITY_MATRIX_PATH)
    parser.add_argument("--data-access-matrix", type=Path, default=DATA_ACCESS_MATRIX_PATH)
    parser.add_argument("--sql-assist-matrix", type=Path, default=SQL_ASSIST_MATRIX_PATH)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    rows = build_rows(
        args.plugins_root.resolve(),
        _read_json(args.functionality_matrix.resolve()),
        _read_json(args.data_access_matrix.resolve()),
        _read_json(args.sql_assist_matrix.resolve()),
    )
    payload = build_payload(rows)
    json_text = _stable_json(payload)
    md_text = build_markdown(rows)

    json_out = args.json_out.resolve()
    md_out = args.md_out.resolve()
    if args.verify:
        if not json_out.exists() or not md_out.exists():
            return 2
        if json_out.read_text(encoding="utf-8") != json_text:
            return 2
        return 0 if md_out.read_text(encoding="utf-8") == md_text else 2

    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json_text, encoding="utf-8")
    md_out.write_text(md_text, encoding="utf-8")
    print(f"json_out={json_out}")
    print(f"md_out={md_out}")
    print(f"plugins={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

