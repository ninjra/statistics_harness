#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from statistic_harness.core.pipeline import Pipeline
from statistic_harness.core.tenancy import get_tenant_context


OPENPLANTER_PLUGIN_IDS = [
    "profile_basic",
    "transform_entity_resolution_map_v1",
    "transform_cross_dataset_link_graph_v1",
    "analysis_bundled_donations_v1",
    "analysis_contribution_limit_flags_v1",
    "analysis_vendor_influence_breadth_v1",
    "analysis_vendor_politician_timing_permutation_v1",
    "analysis_red_flags_refined_v1",
    "report_evidence_index_v1",
    "report_bundle",
]


def _load_mapping(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Invalid mapping payload (object required): {path}")
    return payload


def _validate_mapping(mapping: dict[str, Any], strict: bool) -> None:
    if not strict:
        return
    roles = mapping.get("roles")
    if not isinstance(roles, dict):
        raise SystemExit("Strict mode requires roles mapping with contracts/contributions")
    required = {"contracts", "contributions"}
    missing = sorted(role for role in required if role not in roles)
    if missing:
        raise SystemExit(f"Strict mode missing required roles: {', '.join(missing)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the cross-dataset Openplanter plugin pack deterministically."
    )
    parser.add_argument("--input", required=True, help="Input dataset path (CSV/XLSX/etc)")
    parser.add_argument("--appdata", default="", help="Override STAT_HARNESS_APPDATA")
    parser.add_argument("--run-seed", type=int, default=20260226)
    parser.add_argument("--mapping-json", default="", help="Optional role mapping payload")
    parser.add_argument("--strict-mapping", action="store_true")
    args = parser.parse_args()

    input_path = Path(str(args.input)).resolve()
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    if str(args.appdata).strip():
        appdata = Path(str(args.appdata)).resolve()
        appdata.mkdir(parents=True, exist_ok=True)
        os.environ["STAT_HARNESS_APPDATA"] = str(appdata)

    mapping_path = Path(str(args.mapping_json)).resolve() if str(args.mapping_json).strip() else None
    mapping = _load_mapping(mapping_path)
    _validate_mapping(mapping, strict=bool(args.strict_mapping))

    tenant_ctx = get_tenant_context()
    pipeline = Pipeline(tenant_ctx.appdata_root, Path("plugins"), tenant_id=tenant_ctx.tenant_id)
    settings: dict[str, Any] = dict(mapping)
    run_id = pipeline.run(
        input_path,
        OPENPLANTER_PLUGIN_IDS,
        settings=settings,
        run_seed=int(args.run_seed),
    )
    run_dir = tenant_ctx.tenant_root / "runs" / run_id
    print(json.dumps({"run_id": run_id, "run_dir": str(run_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

