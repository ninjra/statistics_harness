#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


REPO = Path(__file__).resolve().parents[1]
PLUGINS = REPO / "plugins"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    _write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_yaml(path: Path, payload: Any) -> None:
    _write(path, yaml.safe_dump(payload, sort_keys=False))


def _output_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": True,
        "required": [
            "status",
            "summary",
            "metrics",
            "findings",
            "artifacts",
            "budget",
            "error",
            "references",
            "debug",
        ],
        "properties": {
            "status": {"type": "string"},
            "summary": {"type": "string"},
            "metrics": {"type": "object"},
            "findings": {"type": "array"},
            "artifacts": {"type": "array"},
            "budget": {"type": "object"},
            "error": {"type": ["object", "null"]},
            "references": {"type": "array"},
            "debug": {"type": "object"},
        },
    }


SPECS: list[dict[str, Any]] = [
    {
        "id": "ingest_sql_dump_v1",
        "name": "SQL Dump Ingest v1",
        "type": "ingest",
        "depends_on": [],
        "capabilities": ["needs_file", "needs_sql_dump"],
        "defaults": {
            "input_path": None,
            "table_name": None,
            "max_rows": None,
            "chunk_rows": 50000,
            "encoding": "utf-8",
            "dialect": "sqlite_like",
        },
    },
    {
        "id": "transform_entity_resolution_map_v1",
        "name": "Entity Resolution Map v1",
        "type": "transform",
        "depends_on": [],
        "capabilities": ["needs_eventlog", "needs_cross_dataset"],
        "defaults": {
            "datasets": {},
            "fields": [],
            "batch_size": 100000,
            "fuzzy_threshold": 82,
            "min_token_len": 4,
            "token_overlap_min_ratio": 0.6,
            "min_overlap_tokens": 2,
            "max_alias_rows": 200000,
        },
    },
    {
        "id": "transform_cross_dataset_link_graph_v1",
        "name": "Cross Dataset Link Graph v1",
        "type": "transform",
        "depends_on": ["transform_entity_resolution_map_v1"],
        "capabilities": ["needs_eventlog", "needs_cross_dataset"],
        "defaults": {
            "datasets": {},
            "edges": [],
            "batch_size": 100000,
            "fuzzy_threshold": 82,
            "min_token_len": 4,
            "token_overlap_min_ratio": 0.6,
            "min_overlap_tokens": 2,
            "include_row_payload_excerpt": False,
        },
    },
    {
        "id": "analysis_bundled_donations_v1",
        "name": "Bundled Donations v1",
        "type": "analysis",
        "depends_on": ["transform_cross_dataset_link_graph_v1"],
        "capabilities": ["needs_eventlog", "needs_cross_dataset"],
        "defaults": {
            "contributions_dataset_version_id": None,
            "employer": None,
            "candidate_id": None,
            "donation_date": None,
            "amount": None,
            "donor_name": None,
            "min_donors": 3,
            "batch_size": 100000,
        },
    },
    {
        "id": "analysis_contribution_limit_flags_v1",
        "name": "Contribution Limit Flags v1",
        "type": "analysis",
        "depends_on": ["transform_cross_dataset_link_graph_v1"],
        "capabilities": ["needs_eventlog", "needs_cross_dataset"],
        "defaults": {
            "contributions_dataset_version_id": None,
            "donor_id_fields": ["donor_last", "donor_first", "donor_address"],
            "amount_field": None,
            "date_field": None,
            "annual_limit": 1000.0,
            "min_excess": 0.0,
            "batch_size": 100000,
        },
    },
    {
        "id": "analysis_vendor_influence_breadth_v1",
        "name": "Vendor Influence Breadth v1",
        "type": "analysis",
        "depends_on": ["transform_cross_dataset_link_graph_v1"],
        "capabilities": ["needs_cross_dataset"],
        "defaults": {
            "cross_links_path": None,
            "vendor_entity_type": "org",
            "top_k": 50,
        },
    },
    {
        "id": "analysis_vendor_politician_timing_permutation_v1",
        "name": "Vendor Politician Timing Permutation v1",
        "type": "analysis",
        "depends_on": ["transform_cross_dataset_link_graph_v1"],
        "capabilities": ["needs_eventlog", "needs_cross_dataset", "needs_timestamp"],
        "defaults": {
            "contracts_dataset_version_id": None,
            "contributions_dataset_version_id": None,
            "vendor_field": None,
            "award_date_field": None,
            "candidate_id_field": None,
            "donation_date_field": None,
            "amount_field": None,
            "cross_links_path": None,
            "min_donations": 3,
            "n_permutations": 2000,
            "max_permutations": 5000,
            "rng_seed": 0,
            "top_k": 200,
            "batch_size": 100000,
        },
    },
    {
        "id": "analysis_red_flags_refined_v1",
        "name": "Red Flags Refined v1",
        "type": "analysis",
        "depends_on": [
            "analysis_bundled_donations_v1",
            "analysis_contribution_limit_flags_v1",
            "analysis_vendor_politician_timing_permutation_v1",
        ],
        "capabilities": ["needs_cross_dataset"],
        "defaults": {
            "sole_source_methods": ["Sole Source", "Limited Competition", "Emergency", "Exempt"],
            "max_p_value_for_timing_flag": 0.1,
            "min_effect_size_for_timing_flag": 0.5,
        },
    },
    {
        "id": "report_evidence_index_v1",
        "name": "Evidence Index v1",
        "type": "report",
        "depends_on": ["analysis_red_flags_refined_v1"],
        "capabilities": ["needs_reporting"],
        "defaults": {},
    },
]


def _config_schema(defaults: dict[str, Any]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for key, value in defaults.items():
        if isinstance(value, bool):
            typ = "boolean"
        elif isinstance(value, int):
            typ = "integer"
        elif isinstance(value, float):
            typ = "number"
        elif isinstance(value, list):
            typ = "array"
        elif isinstance(value, dict):
            typ = "object"
        else:
            typ = "string"
        properties[key] = {"type": [typ, "null"], "default": value}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": True,
        "properties": properties,
    }


def main() -> int:
    for spec in SPECS:
        pid = spec["id"]
        pdir = PLUGINS / pid
        pdir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "id": pid,
            "name": spec["name"],
            "version": "0.1.0",
            "type": spec["type"],
            "entrypoint": "plugin.py:Plugin",
            "depends_on": spec.get("depends_on", []),
            "settings": {
                "description": spec["name"],
                "defaults": spec.get("defaults", {}),
            },
            "capabilities": spec.get("capabilities", []),
            "config_schema": "config.schema.json",
            "output_schema": "output.schema.json",
            "sandbox": {"no_network": True, "fs_allowlist": ["appdata", "plugins", "run_dir"]},
        }
        _write_yaml(pdir / "plugin.yaml", manifest)
        _write_json(pdir / "config.schema.json", _config_schema(spec.get("defaults", {})))
        _write_json(pdir / "output.schema.json", _output_schema())
        _write(
            pdir / "plugin.py",
            (
                "from __future__ import annotations\n\n"
                "from statistic_harness.core.openplanter_pack import run_openplanter_plugin\n\n\n"
                "class Plugin:\n"
                "    def run(self, ctx):\n"
                f"        return run_openplanter_plugin({pid!r}, ctx)\n"
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

