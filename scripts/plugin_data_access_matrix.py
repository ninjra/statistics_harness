#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
ACCESS_OVERRIDE_PATH = ROOT / "docs" / "plugin_data_access_overrides.json"

CONTRACT_DATASET_LOADER = "dataset_loader"
CONTRACT_ITER_BATCHES = "iter_batches"
CONTRACT_SQL_ASSIST = "sql_assist"
CONTRACT_SQL_DIRECT = "sql_direct"
CONTRACT_ARTIFACT_ONLY = "artifact_only"
CONTRACT_ORCHESTRATION_ONLY = "orchestration_only"

ALLOWED_CONTRACTS = {
    CONTRACT_DATASET_LOADER,
    CONTRACT_ITER_BATCHES,
    CONTRACT_SQL_ASSIST,
    CONTRACT_SQL_DIRECT,
    CONTRACT_ARTIFACT_ONLY,
    CONTRACT_ORCHESTRATION_ONLY,
}


@dataclass(frozen=True)
class PluginAccess:
    plugin_id: str
    plugin_type: str
    uses_dataset_loader: bool
    uses_dataset_loader_unbounded: bool
    uses_dataset_iter_batches: bool
    uses_sql_direct: bool
    uses_sql_assist: bool
    access_contracts: tuple[str, ...]
    contract_sources: tuple[str, ...]
    unclassified: bool


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _manifest(manifest_path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(_read_text(manifest_path))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _plugin_type(manifest_payload: dict[str, Any]) -> str:
    value = manifest_payload.get("type")
    return value.strip() if isinstance(value, str) else ""


def _plugin_depends_on(manifest_payload: dict[str, Any]) -> list[str]:
    raw = manifest_payload.get("depends_on")
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


def _scan_plugin_py(text: str) -> tuple[bool, bool, bool, bool, bool]:
    uses_loader = "ctx.dataset_loader" in text
    uses_batches = "ctx.dataset_iter_batches" in text
    uses_sql_assist = ("ctx.sql" in text) or ("ctx.sql_exec" in text)

    # Indirect dataset access via the stat_plugins registry (common pattern for analysis plugins).
    if (
        "statistic_harness.core.stat_plugins.registry" in text
        and "run_plugin" in text
        and "run_plugin(" in text
    ):
        uses_loader = True
        # Registry currently calls ctx.dataset_loader() unbounded unless settings pass row_limit.
        unbounded = True
    else:
        unbounded = False

    # Heuristic "unbounded": `ctx.dataset_loader()` with no args.
    if uses_loader:
        for needle in ("ctx.dataset_loader()", "ctx.dataset_loader( )"):
            if needle in text:
                unbounded = True
                break
        if not unbounded:
            # crude: find dataset_loader( and see if first non-space after '(' is ')'
            idx = 0
            while True:
                idx = text.find("ctx.dataset_loader(", idx)
                if idx == -1:
                    break
                j = idx + len("ctx.dataset_loader(")
                while j < len(text) and text[j] in {" ", "\t", "\n", "\r"}:
                    j += 1
                if j < len(text) and text[j] == ")":
                    unbounded = True
                    break
                idx = j

    # Direct SQL: look for storage.connection() and SELECT/INSERT patterns.
    uses_sql = "storage.connection" in text or "ctx.storage.connection" in text
    uses_sql = uses_sql and ("SELECT " in text or "INSERT " in text or "DELETE " in text or "UPDATE " in text)

    return uses_loader, unbounded, uses_batches, uses_sql, uses_sql_assist


def _load_contract_overrides(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    rows = payload.get("contracts") if isinstance(payload, dict) else None
    if not isinstance(rows, dict):
        return {}
    out: dict[str, list[str]] = {}
    for plugin_id, contracts in rows.items():
        if not isinstance(plugin_id, str):
            continue
        if not isinstance(contracts, list):
            continue
        normalized: list[str] = []
        for value in contracts:
            if not isinstance(value, str):
                continue
            contract = value.strip()
            if contract in ALLOWED_CONTRACTS and contract not in normalized:
                normalized.append(contract)
        if normalized:
            out[plugin_id] = normalized
    return out


def _infer_contracts(
    *,
    plugin_type: str,
    depends_on: list[str],
    uses_loader: bool,
    uses_batches: bool,
    uses_sql: bool,
    uses_sql_assist: bool,
) -> tuple[list[str], list[str]]:
    contracts: list[str] = []
    sources: list[str] = []
    if uses_loader:
        contracts.append(CONTRACT_DATASET_LOADER)
        sources.append("detected:dataset_loader")
    if uses_batches:
        contracts.append(CONTRACT_ITER_BATCHES)
        sources.append("detected:iter_batches")
    if uses_sql_assist:
        contracts.append(CONTRACT_SQL_ASSIST)
        sources.append("detected:sql_assist")
    if uses_sql:
        contracts.append(CONTRACT_SQL_DIRECT)
        sources.append("detected:sql_direct")

    if contracts:
        return contracts, sources

    ptype = plugin_type.strip().lower()
    if ptype in {"planner", "llm"}:
        return [CONTRACT_ORCHESTRATION_ONLY], ["inferred:type_orchestration"]
    if ptype == "report":
        return [CONTRACT_ARTIFACT_ONLY], ["inferred:type_report"]
    if ptype == "analysis" and depends_on:
        return [CONTRACT_ORCHESTRATION_ONLY], ["inferred:depends_on"]
    if ptype == "transform":
        return [CONTRACT_ARTIFACT_ONLY], ["inferred:type_transform"]
    return [CONTRACT_ARTIFACT_ONLY], ["inferred:default"]


def generate(plugins_root: Path) -> list[PluginAccess]:
    overrides = _load_contract_overrides(ACCESS_OVERRIDE_PATH)
    out: list[PluginAccess] = []
    for pdir in sorted([p for p in plugins_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        manifest = pdir / "plugin.yaml"
        entry = pdir / "plugin.py"
        if not manifest.exists() or not entry.exists():
            continue
        manifest_payload = _manifest(manifest)
        ptype = _plugin_type(manifest_payload)
        depends_on = _plugin_depends_on(manifest_payload)
        text = _read_text(entry)
        uses_loader, unbounded, uses_batches, uses_sql, uses_sql_assist = _scan_plugin_py(text)
        contracts, sources = _infer_contracts(
            plugin_type=ptype,
            depends_on=depends_on,
            uses_loader=uses_loader,
            uses_batches=uses_batches,
            uses_sql=uses_sql,
            uses_sql_assist=uses_sql_assist,
        )
        override_contracts = overrides.get(pdir.name, [])
        for contract in override_contracts:
            if contract not in contracts:
                contracts.append(contract)
        if override_contracts:
            sources.append("override:contracts")
        contracts = sorted(set(contracts))
        sources = sorted(set(sources))
        out.append(
            PluginAccess(
                plugin_id=pdir.name,
                plugin_type=ptype,
                uses_dataset_loader=uses_loader,
                uses_dataset_loader_unbounded=unbounded,
                uses_dataset_iter_batches=uses_batches,
                uses_sql_direct=uses_sql,
                uses_sql_assist=uses_sql_assist,
                access_contracts=tuple(contracts),
                contract_sources=tuple(sources),
                unclassified=(len(contracts) == 0),
            )
        )
    return out


def _as_json(items: list[PluginAccess]) -> dict[str, Any]:
    contract_counts = {contract: 0 for contract in sorted(ALLOWED_CONTRACTS)}
    unclassified = 0
    for item in items:
        if item.unclassified:
            unclassified += 1
        for contract in item.access_contracts:
            if contract in contract_counts:
                contract_counts[contract] += 1
    return {
        "plugin_count": len(items),
        "unclassified_count": unclassified,
        "contract_counts": contract_counts,
        "plugins": [
            {
                "plugin_id": i.plugin_id,
                "plugin_type": i.plugin_type,
                "uses_dataset_loader": i.uses_dataset_loader,
                "uses_dataset_loader_unbounded": i.uses_dataset_loader_unbounded,
                "uses_dataset_iter_batches": i.uses_dataset_iter_batches,
                "uses_sql_direct": i.uses_sql_direct,
                "uses_sql_assist": i.uses_sql_assist,
                "access_contracts": list(i.access_contracts),
                "contract_sources": list(i.contract_sources),
                "unclassified": i.unclassified,
            }
            for i in items
        ],
    }


def _as_md(items: list[PluginAccess]) -> str:
    lines: list[str] = []
    lines.append("# Plugin Data Access Matrix")
    lines.append("")
    lines.append("Generated by `scripts/plugin_data_access_matrix.py`.")
    lines.append("")
    lines.append("| Plugin | Type | Contracts | contract_sources | dataset_loader | loader_unbounded | iter_batches | direct_sql | sql_assist |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|")
    for i in items:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{i.plugin_id}`",
                    i.plugin_type or "",
                    ", ".join(i.access_contracts),
                    ", ".join(i.contract_sources),
                    str(int(i.uses_dataset_loader)),
                    str(int(i.uses_dataset_loader_unbounded)),
                    str(int(i.uses_dataset_iter_batches)),
                    str(int(i.uses_sql_direct)),
                    str(int(i.uses_sql_assist)),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugins-root", default="plugins")
    ap.add_argument("--out-json", default="docs/plugin_data_access_matrix.json")
    ap.add_argument("--out-md", default="docs/plugin_data_access_matrix.md")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    items = generate((ROOT / args.plugins_root).resolve())
    payload = _as_json(items)
    out_json = (ROOT / args.out_json).resolve()
    out_md = (ROOT / args.out_md).resolve()
    json_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    md_text = _as_md(items)

    if args.verify:
        if not items:
            return 2
        if any(i.unclassified for i in items):
            return 2
        if not out_json.exists() or out_json.read_text(encoding="utf-8") != json_text:
            return 2
        if not out_md.exists() or out_md.read_text(encoding="utf-8") != md_text:
            return 2
        return 0

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json_text, encoding="utf-8")
    out_md.write_text(md_text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
