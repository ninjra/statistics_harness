#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "docs" / "repo_improvements_execution_plan_v1.json"
DEFAULT_OUT = ROOT / "docs" / "repo_improvements_scaffold_plan_v1.json"
PLUGINS = ROOT / "plugins"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _config_schema() -> dict[str, Any]:
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "notes": {"type": ["string", "null"], "default": None}
        },
        "additionalProperties": True,
    }


def _output_schema() -> dict[str, Any]:
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["status", "summary", "metrics", "findings", "artifacts", "budget", "error", "references", "debug"],
        "properties": {
            "status": {"type": "string"},
            "summary": {"type": "string"},
            "metrics": {"type": "object"},
            "findings": {"type": "array", "items": {"type": "object"}},
            "artifacts": {"type": "array", "items": {"type": "object"}},
            "budget": {"type": "object"},
            "error": {"type": ["object", "null"]},
            "references": {"type": "array", "items": {"type": "object"}},
            "debug": {"type": "object"},
        },
        "additionalProperties": True,
    }


def _plugin_code(plugin_id: str) -> str:
    return (
        "from __future__ import annotations\n\n"
        "from statistic_harness.core.types import PluginResult\n\n\n"
        "class Plugin:\n"
        "    def run(self, ctx) -> PluginResult:\n"
        "        return PluginResult(\n"
        "            status='degraded',\n"
        f"            summary='Scaffold placeholder for {plugin_id}',\n"
        "            metrics={'implemented': 0},\n"
        "            findings=[],\n"
        "            artifacts=[],\n"
        "            references=[],\n"
        "            debug={'placeholder': True},\n"
        "        )\n"
    )


def _manifest(plugin_id: str) -> dict[str, Any]:
    return {
        "id": plugin_id,
        "name": f"Repo Improvement Scaffold {plugin_id}",
        "version": "0.1.0",
        "type": "analysis",
        "entrypoint": "plugin.py:Plugin",
        "depends_on": [],
        "settings": {
            "description": "Generated scaffold for repo improvements wave planning.",
            "defaults": {},
        },
        "capabilities": ["needs_eventlog"],
        "config_schema": "config.schema.json",
        "output_schema": "output.schema.json",
        "sandbox": {"no_network": True, "fs_allowlist": ["appdata", "plugins", "run_dir"]},
    }


def _selected_plugin_ids(plan: dict[str, Any], wave: str, ids: list[str]) -> list[str]:
    if ids:
        return sorted({x for x in ids if x.strip()})
    out: set[str] = set()
    for row in plan.get("items") or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("wave")) != wave:
            continue
        for pid in row.get("missing_proposed_plugins") or []:
            if isinstance(pid, str) and pid.strip():
                out.add(pid.strip())
    return sorted(out)


def build_plan_payload(plan: dict[str, Any], wave: str, ids: list[str]) -> dict[str, Any]:
    selected = _selected_plugin_ids(plan, wave, ids)
    to_create = [pid for pid in selected if not (PLUGINS / pid).exists()]
    existing = [pid for pid in selected if (PLUGINS / pid).exists()]
    return {
        "generated_by": "scripts/scaffold_repo_improvement_plugins.py",
        "source_execution_plan": "docs/repo_improvements_execution_plan_v1.json",
        "wave": wave,
        "selected_plugin_ids": selected,
        "to_create": to_create,
        "already_existing": existing,
    }


def apply_scaffolds(plugin_ids: list[str]) -> None:
    for pid in plugin_ids:
        pdir = PLUGINS / pid
        pdir.mkdir(parents=True, exist_ok=True)
        _write_yaml(pdir / "plugin.yaml", _manifest(pid))
        (pdir / "plugin.py").write_text(_plugin_code(pid), encoding="utf-8")
        _write_json(pdir / "config.schema.json", _config_schema())
        _write_json(pdir / "output.schema.json", _output_schema())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--wave", default="wave_1")
    ap.add_argument("--id", dest="ids", action="append", default=[])
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    plan = _read_json(args.plan)
    payload = build_plan_payload(plan, args.wave, args.ids)
    out = args.out.resolve()

    if args.verify:
        if not out.exists():
            return 2
        return 0 if _read_json(out) == payload else 2

    if args.apply:
        apply_scaffolds(payload.get("to_create") or [])
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_json(out, payload)
    print(f"out={out}")
    print(f"selected={len(payload.get('selected_plugin_ids') or [])}")
    print(f"to_create={len(payload.get('to_create') or [])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
