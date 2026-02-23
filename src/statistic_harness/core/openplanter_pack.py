from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .cross_dataset_pack import (
    DatasetSpec,
    ensure_roles,
    iter_rows,
    matches_when,
    parse_amount,
    parse_dataset_specs,
    parse_dt,
    resolve_column,
)
from .entity_resolution import (
    build_token_inverted_index,
    match_entity,
    normalize_org_name,
)
from .evidence_links import (
    evidence_link,
    register_artifact,
    row_ref,
    stable_edge_id,
    stable_entity_id,
)
from .report_v2_utils import claim_id
from .types import PluginArtifact, PluginContext, PluginError, PluginResult
from .utils import file_sha256, now_iso, write_json


PACK_PLUGIN_IDS = {
    "ingest_sql_dump_v1",
    "transform_entity_resolution_map_v1",
    "transform_cross_dataset_link_graph_v1",
    "analysis_bundled_donations_v1",
    "analysis_contribution_limit_flags_v1",
    "analysis_vendor_influence_breadth_v1",
    "analysis_vendor_politician_timing_permutation_v1",
    "analysis_red_flags_refined_v1",
    "report_evidence_index_v1",
}


def _error_result(code: str, message: str, *, debug: dict[str, Any] | None = None) -> PluginResult:
    return PluginResult(
        status="error",
        summary=message,
        metrics={},
        findings=[],
        artifacts=[],
        error=PluginError(type=code, message=message, traceback=""),
        debug=dict(debug or {}),
    )


def _ok_result(
    summary: str,
    *,
    metrics: dict[str, Any],
    findings: list[dict[str, Any]],
    artifacts: list[PluginArtifact],
    debug: dict[str, Any] | None = None,
) -> PluginResult:
    return PluginResult(
        status="ok",
        summary=summary,
        metrics=dict(metrics or {}),
        findings=list(findings or []),
        artifacts=list(artifacts or []),
        error=None,
        debug=dict(debug or {}),
    )


def _hash_seed(seed: str) -> int:
    return int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16], 16)


def _artifact_meta(registered: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in registered:
        meta = getattr(item, "metadata", None)
        if isinstance(meta, dict):
            out.append(meta)
    return out


def _find_artifact(run_dir: Path, plugin_id: str, filename: str) -> Path | None:
    path = run_dir / "artifacts" / plugin_id / filename
    if path.exists():
        return path
    return None


def _csv_row_count(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            total = sum(1 for _ in reader)
        return max(0, total - 1)
    except Exception:
        return 0


def _jsonl_row_count(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)
    except Exception:
        return 0


def _resolve_dataset_column(
    ctx: PluginContext,
    dataset_version_id: str,
    explicit: str | None,
    fallbacks: list[str],
) -> str | None:
    return resolve_column(
        ctx.storage,
        dataset_version_id,
        explicit=explicit,
        candidates=fallbacks,
    )


def _dataset_specs_or_error(
    ctx: PluginContext,
    required_roles: list[str],
) -> tuple[dict[str, Any], PluginResult | None]:
    specs = parse_dataset_specs(ctx.settings)
    missing = ensure_roles(specs, required_roles)
    if missing:
        return {}, _error_result(
            "MissingDatasetRoles",
            f"Missing required dataset roles: {', '.join(sorted(missing))}",
            debug={"required_roles": required_roles, "provided_roles": sorted(specs.keys())},
        )
    return specs, None


def _strict_prereq(ctx: PluginContext) -> bool:
    return bool(ctx.settings.get("strict_prerequisites", False))


def _run_transform_entity_resolution_map(ctx: PluginContext) -> PluginResult:
    specs = parse_dataset_specs(ctx.settings)
    if not specs and ctx.dataset_version_id:
        specs = {"current": DatasetSpec(role="current", dataset_version_id=str(ctx.dataset_version_id))}
    fields = ctx.settings.get("fields")
    if not isinstance(fields, list) or not fields:
        auto_fields: list[dict[str, Any]] = []
        first = next(iter(specs.values()), None)
        if first:
            cols = [str(c.get("original_name") or "") for c in (ctx.storage.fetch_dataset_columns(first.dataset_version_id) or [])]
            for col in cols:
                low = col.lower()
                if any(token in low for token in ("vendor", "employer", "donor", "company", "org")):
                    auto_fields.append(
                        {
                            "role": first.role,
                            "field": col,
                            "entity_type": "org",
                            "key": col,
                        }
                    )
        fields = auto_fields

    entity_stats: dict[str, dict[str, Any]] = {}
    alias_rows: list[dict[str, Any]] = []
    scanned = 0
    for spec in fields:
        if not isinstance(spec, dict):
            continue
        role = str(spec.get("role") or "").strip()
        field = str(spec.get("field") or "").strip()
        entity_type = str(spec.get("entity_type") or "org").strip() or "org"
        key_name = str(spec.get("key") or role or "unknown")
        when = spec.get("when") if isinstance(spec.get("when"), dict) else None
        if role not in specs:
            if _strict_prereq(ctx):
                return _error_result("MissingDatasetRole", f"Field mapping role '{role}' is not configured in settings.datasets")
            continue
        dvid = specs[role].dataset_version_id
        if not field:
            if _strict_prereq(ctx):
                return _error_result("MissingFieldMapping", f"Field mapping for role '{role}' is missing 'field'")
            continue
        resolved_field = _resolve_dataset_column(ctx, dvid, field, [field])
        if not resolved_field:
            if _strict_prereq(ctx):
                return _error_result("MissingDatasetColumn", f"Column '{field}' not found in dataset role '{role}' ({dvid})")
            continue
        columns = [resolved_field]
        if when:
            for wk in when.keys():
                name = wk[: -len("_in")] if wk.endswith("_in") else wk
                if isinstance(name, str) and name.strip() and name not in columns:
                    columns.append(name.strip())
        for row in iter_rows(
            ctx.storage,
            dvid,
            columns=columns,
            batch_size=int(ctx.settings.get("batch_size", 100_000)),
            sql=ctx.sql,
        ):
            scanned += 1
            if not matches_when(row, when):
                continue
            raw = row.get(resolved_field)
            norm = normalize_org_name(raw)
            if not norm:
                continue
            e_id = stable_entity_id(entity_type, norm)
            ref = row_ref(dvid, int(row["row_index"]))
            bucket = entity_stats.setdefault(
                e_id,
                {
                    "entity_id": e_id,
                    "entity_type": entity_type,
                    "canonical_name": norm,
                    "alias_count": 0,
                    "source_count": 0,
                    "roles": set(),
                    "sample_refs": [],
                },
            )
            bucket["alias_count"] = int(bucket["alias_count"]) + 1
            bucket["source_count"] = int(bucket["source_count"]) + 1
            bucket["roles"].add(role)
            if len(bucket["sample_refs"]) < 8:
                bucket["sample_refs"].append(ref)
            if len(alias_rows) < int(ctx.settings.get("max_alias_rows", 200_000)):
                alias_rows.append(
                    {
                        "entity_id": e_id,
                        "entity_type": entity_type,
                        "canonical_name": norm,
                        "alias_value": str(raw or ""),
                        "role": role,
                        "key": key_name,
                        "dataset_version_id": dvid,
                        "row_ref": ref,
                    }
                )
    entities: list[dict[str, Any]] = []
    for entity_id in sorted(entity_stats.keys()):
        raw = entity_stats[entity_id]
        entities.append(
            {
                "entity_id": raw["entity_id"],
                "entity_type": raw["entity_type"],
                "canonical_name": raw["canonical_name"],
                "alias_count": int(raw["alias_count"]),
                "source_count": int(raw["source_count"]),
                "roles": sorted(raw["roles"]),
                "sample_refs": list(raw["sample_refs"]),
            }
        )

    artifacts_dir = ctx.artifacts_dir("transform_entity_resolution_map_v1")
    map_path = artifacts_dir / "entity_map.json"
    aliases_path = artifacts_dir / "entity_aliases.csv"
    write_json(
        map_path,
        {
            "generated_at_utc": now_iso(),
            "entities": entities,
            "stats": {
                "entity_count": len(entities),
                "alias_rows": len(alias_rows),
                "scanned_rows": scanned,
            },
        },
    )
    with aliases_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "entity_id",
                "entity_type",
                "canonical_name",
                "alias_value",
                "role",
                "key",
                "dataset_version_id",
                "row_ref",
            ],
        )
        writer.writeheader()
        for row in alias_rows:
            writer.writerow(row)

    reg_map = register_artifact(ctx, map_path, description="Cross-dataset entity map", mime="application/json", producer_plugin_id="transform_entity_resolution_map_v1", record_count=len(entities))
    reg_alias = register_artifact(ctx, aliases_path, description="Flattened entity aliases", mime="text/csv", producer_plugin_id="transform_entity_resolution_map_v1", record_count=len(alias_rows))
    findings = [
        {
            "id": claim_id(f"entity_map_stats:{ctx.run_id}:{ctx.dataset_version_id or 'none'}"),
            "kind": "entity_map_stats",
            "severity": "info",
            "summary": f"Built {len(entities)} canonical entities from {len(alias_rows)} aliases.",
            "measurement_type": "measured",
            "evidence": {
                "artifact_path": reg_map.artifact.path,
                "dataset_version_id": ctx.dataset_version_id,
            },
        }
    ]
    return _ok_result(
        f"Built entity map with {len(entities)} entities",
        metrics={"entity_count": len(entities), "alias_rows": len(alias_rows), "scanned_rows": scanned},
        findings=findings,
        artifacts=[reg_map.artifact, reg_alias.artifact],
        debug={"artifact_manifest": _artifact_meta([reg_map, reg_alias])},
    )


def _run_transform_cross_dataset_link_graph(ctx: PluginContext) -> PluginResult:
    specs = parse_dataset_specs(ctx.settings)
    if not specs and ctx.dataset_version_id:
        specs = {"current": DatasetSpec(role="current", dataset_version_id=str(ctx.dataset_version_id))}
    edges = ctx.settings.get("edges")
    if not isinstance(edges, list) or not edges:
        edges = []
        first = next(iter(specs.values()), None)
        if first:
            cols = {str(c.get("original_name") or "").lower(): str(c.get("original_name") or "") for c in (ctx.storage.fetch_dataset_columns(first.dataset_version_id) or [])}
            vendor = cols.get("vendor_name1") or cols.get("vendor") or cols.get("vendor_name")
            employer = cols.get("employer") or cols.get("employer_name")
            if vendor and employer:
                edges.append(
                    {
                        "left": {"role": first.role, "field": vendor},
                        "right": {"role": first.role, "field": employer},
                        "relation": "vendor_employer",
                    }
                )

    fuzzy_threshold = int(ctx.settings.get("fuzzy_threshold", 82))
    min_token_len = int(ctx.settings.get("min_token_len", 4))
    overlap_min_ratio = float(ctx.settings.get("token_overlap_min_ratio", 0.6))
    min_overlap_tokens = int(ctx.settings.get("min_overlap_tokens", 2))
    batch_size = int(ctx.settings.get("batch_size", 100_000))

    artifacts_dir = ctx.artifacts_dir("transform_cross_dataset_link_graph_v1")
    csv_path = artifacts_dir / "cross_links.csv"
    jsonl_path = artifacts_dir / "cross_links.jsonl"
    summary_path = artifacts_dir / "cross_link_summary.json"

    edge_count = 0
    relation_counts: Counter[str] = Counter()
    match_counts: Counter[str] = Counter()
    conf_counts: Counter[str] = Counter()

    with csv_path.open("w", encoding="utf-8", newline="") as csv_handle, jsonl_path.open(
        "w", encoding="utf-8"
    ) as jsonl_handle:
        writer = csv.DictWriter(
            csv_handle,
            fieldnames=[
                "edge_id",
                "relation",
                "left_entity_id",
                "right_entity_id",
                "left_norm",
                "right_norm",
                "match_type",
                "confidence_tier",
                "fuzzy_score",
                "overlap_ratio",
                "left_ref",
                "right_ref",
                "left_role",
                "right_role",
                "left_dataset_version_id",
                "right_dataset_version_id",
            ],
        )
        writer.writeheader()
        for edge_spec in edges:
            if not isinstance(edge_spec, dict):
                continue
            left = edge_spec.get("left") if isinstance(edge_spec.get("left"), dict) else {}
            right = edge_spec.get("right") if isinstance(edge_spec.get("right"), dict) else {}
            relation = str(edge_spec.get("relation") or "linked").strip() or "linked"
            when = edge_spec.get("when") if isinstance(edge_spec.get("when"), dict) else None
            left_role = str(left.get("role") or "").strip()
            right_role = str(right.get("role") or "").strip()
            left_field = str(left.get("field") or "").strip()
            right_field = str(right.get("field") or "").strip()
            if not left_role or not right_role or not left_field or not right_field:
                if _strict_prereq(ctx):
                    return _error_result("InvalidEdgeSpec", "Every edge spec must include left/right role and field")
                continue
            if left_role not in specs or right_role not in specs:
                if _strict_prereq(ctx):
                    return _error_result("MissingDatasetRole", f"Edge role mapping missing for relation '{relation}'")
                continue
            left_dvid = specs[left_role].dataset_version_id
            right_dvid = specs[right_role].dataset_version_id
            resolved_left = _resolve_dataset_column(ctx, left_dvid, left_field, [left_field])
            resolved_right = _resolve_dataset_column(ctx, right_dvid, right_field, [right_field])
            if not resolved_left or not resolved_right:
                if _strict_prereq(ctx):
                    return _error_result("MissingDatasetColumn", f"Edge mapping columns missing for relation '{relation}'")
                continue

            left_norm_to_id: dict[str, str] = {}
            left_id_to_norm: dict[str, str] = {}
            left_id_to_ref: dict[str, str] = {}
            left_rows_for_index: list[dict[str, Any]] = []
            left_when_cols = []
            if when:
                for wk in when.keys():
                    col = wk[: -len("_in")] if wk.endswith("_in") else wk
                    if isinstance(col, str) and col and col != resolved_left:
                        left_when_cols.append(col)
            for row in iter_rows(
                ctx.storage,
                left_dvid,
                columns=[resolved_left, *left_when_cols],
                batch_size=batch_size,
                sql=ctx.sql,
            ):
                if not matches_when(row, when):
                    continue
                norm = normalize_org_name(row.get(resolved_left))
                if not norm:
                    continue
                entity_id = stable_entity_id("org", norm)
                left_norm_to_id.setdefault(norm, entity_id)
                left_id_to_norm.setdefault(entity_id, norm)
                left_id_to_ref.setdefault(entity_id, row_ref(left_dvid, int(row["row_index"])))
            for entity_id, norm in left_id_to_norm.items():
                left_rows_for_index.append({"key": entity_id, "name": norm})
            token_index = build_token_inverted_index(left_rows_for_index, "name", key_field="key", min_token_len=min_token_len)

            right_when_cols = []
            if when:
                for wk in when.keys():
                    col = wk[: -len("_in")] if wk.endswith("_in") else wk
                    if isinstance(col, str) and col and col != resolved_right:
                        right_when_cols.append(col)
            for row in iter_rows(
                ctx.storage,
                right_dvid,
                columns=[resolved_right, *right_when_cols],
                batch_size=batch_size,
                sql=ctx.sql,
            ):
                if not matches_when(row, when):
                    continue
                right_norm = normalize_org_name(row.get(resolved_right))
                if not right_norm:
                    continue
                if right_norm in left_norm_to_id:
                    left_entity_id = left_norm_to_id[right_norm]
                    m_type = "exact"
                    conf = "high"
                    fuzzy = 100.0
                    overlap_ratio = 1.0
                else:
                    matched = match_entity(
                        right_norm,
                        left_id_to_norm,
                        token_index=token_index,
                        fuzzy_threshold=fuzzy_threshold,
                        min_token_len=min_token_len,
                        token_overlap_min_ratio=overlap_min_ratio,
                        min_overlap_tokens=min_overlap_tokens,
                    )
                    if not matched:
                        continue
                    left_entity_id = matched.candidate_key
                    m_type = matched.match_type
                    conf = matched.confidence_tier
                    fuzzy = float(matched.fuzzy_score)
                    overlap_ratio = float(matched.overlap_ratio)
                right_entity_id = stable_entity_id("org", right_norm)
                l_ref = left_id_to_ref.get(left_entity_id, "")
                r_ref = row_ref(right_dvid, int(row["row_index"]))
                edge_id = stable_edge_id(relation, left_entity_id, right_entity_id, r_ref)
                edge_row = {
                    "edge_id": edge_id,
                    "relation": relation,
                    "left_entity_id": left_entity_id,
                    "right_entity_id": right_entity_id,
                    "left_norm": left_id_to_norm.get(left_entity_id, ""),
                    "right_norm": right_norm,
                    "match_type": m_type,
                    "confidence_tier": conf,
                    "fuzzy_score": round(fuzzy, 4),
                    "overlap_ratio": round(overlap_ratio, 6),
                    "left_ref": l_ref,
                    "right_ref": r_ref,
                    "left_role": left_role,
                    "right_role": right_role,
                    "left_dataset_version_id": left_dvid,
                    "right_dataset_version_id": right_dvid,
                }
                writer.writerow(edge_row)
                jsonl_handle.write(json.dumps(edge_row, ensure_ascii=False, sort_keys=True) + "\n")
                edge_count += 1
                relation_counts[relation] += 1
                match_counts[m_type] += 1
                conf_counts[conf] += 1

    summary_payload = {
        "generated_at_utc": now_iso(),
        "edge_count": edge_count,
        "relation_counts": dict(sorted(relation_counts.items())),
        "match_type_counts": dict(sorted(match_counts.items())),
        "confidence_tier_counts": dict(sorted(conf_counts.items())),
    }
    write_json(summary_path, summary_payload)
    reg_csv = register_artifact(ctx, csv_path, description="Cross-dataset links (CSV)", mime="text/csv", producer_plugin_id="transform_cross_dataset_link_graph_v1", record_count=_csv_row_count(csv_path))
    reg_jsonl = register_artifact(ctx, jsonl_path, description="Cross-dataset links (JSONL)", mime="application/jsonl", producer_plugin_id="transform_cross_dataset_link_graph_v1", record_count=_jsonl_row_count(jsonl_path))
    reg_sum = register_artifact(ctx, summary_path, description="Cross-link summary stats", mime="application/json", producer_plugin_id="transform_cross_dataset_link_graph_v1")
    findings = [
        {
            "id": claim_id(f"cross_link_summary:{ctx.run_id}:{edge_count}"),
            "kind": "cross_link_summary",
            "severity": "info",
            "summary": f"Generated {edge_count} cross-dataset links.",
            "measurement_type": "measured",
            "evidence": {
                "artifact_path": reg_sum.artifact.path,
                "relation_counts": summary_payload["relation_counts"],
                "match_type_counts": summary_payload["match_type_counts"],
            },
        }
    ]
    return _ok_result(
        f"Generated {edge_count} cross-dataset links",
        metrics={"edge_count": edge_count, "relation_types": len(relation_counts)},
        findings=findings,
        artifacts=[reg_csv.artifact, reg_jsonl.artifact, reg_sum.artifact],
        debug={"artifact_manifest": _artifact_meta([reg_csv, reg_jsonl, reg_sum])},
    )


def _run_analysis_bundled_donations(ctx: PluginContext) -> PluginResult:
    contrib_dvid = str(ctx.settings.get("contributions_dataset_version_id") or "").strip()
    if not contrib_dvid:
        specs = parse_dataset_specs(ctx.settings)
        if "contributions" in specs:
            contrib_dvid = specs["contributions"].dataset_version_id
    if not contrib_dvid and ctx.dataset_version_id:
        contrib_dvid = str(ctx.dataset_version_id)
    if not contrib_dvid:
        return _error_result("MissingConfig", "contributions_dataset_version_id (or datasets.contributions) is required")
    employer_col = _resolve_dataset_column(ctx, contrib_dvid, str(ctx.settings.get("employer") or ""), ["employer", "employer_name"])
    candidate_col = _resolve_dataset_column(ctx, contrib_dvid, str(ctx.settings.get("candidate_id") or ""), ["candidate_id", "candidate", "candidate_name"])
    date_col = _resolve_dataset_column(ctx, contrib_dvid, str(ctx.settings.get("donation_date") or ""), ["donation_date", "date", "trans_date"])
    amount_col = _resolve_dataset_column(ctx, contrib_dvid, str(ctx.settings.get("amount") or ""), ["amount", "donation_amount"])
    donor_col = _resolve_dataset_column(ctx, contrib_dvid, str(ctx.settings.get("donor_name") or ""), ["donor_name", "name"])
    if not employer_col or not candidate_col or not date_col or not amount_col:
        if _strict_prereq(ctx):
            return _error_result("MissingDatasetColumn", "Required contribution columns could not be resolved")
        return _ok_result(
            "Bundling analysis unavailable: required columns missing",
            metrics={"events": 0},
            findings=[],
            artifacts=[],
            debug={"reason": "MISSING_COLUMNS"},
        )

    min_donors = int(ctx.settings.get("min_donors", 3))
    batch_size = int(ctx.settings.get("batch_size", 100_000))
    artifacts_dir = ctx.artifacts_dir("analysis_bundled_donations_v1")
    scratch_path = artifacts_dir / "bundling.sqlite"
    csv_path = artifacts_dir / "bundling_events.csv"
    json_path = artifacts_dir / "bundling_events.json"

    scanned = 0
    accepted = 0
    with sqlite3.connect(scratch_path) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS agg (
                employer_norm TEXT NOT NULL,
                donation_day TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                donor_count INTEGER NOT NULL,
                total_amount REAL NOT NULL,
                sample_ref TEXT NOT NULL,
                PRIMARY KEY(employer_norm, donation_day, candidate_id)
            )
            """
        )
        for row in iter_rows(
            ctx.storage,
            contrib_dvid,
            columns=[employer_col, candidate_col, date_col, amount_col, *( [donor_col] if donor_col else [])],
            batch_size=batch_size,
            sql=ctx.sql,
        ):
            scanned += 1
            employer_norm = normalize_org_name(row.get(employer_col))
            candidate_id = str(row.get(candidate_col) or "").strip()
            dt = parse_dt(row.get(date_col))
            amount = parse_amount(row.get(amount_col))
            if not employer_norm or not candidate_id or dt is None or amount is None:
                continue
            donation_day = dt.date().isoformat()
            ref = row_ref(contrib_dvid, int(row["row_index"]))
            con.execute(
                """
                INSERT INTO agg(employer_norm, donation_day, candidate_id, donor_count, total_amount, sample_ref)
                VALUES (?, ?, ?, 1, ?, ?)
                ON CONFLICT(employer_norm, donation_day, candidate_id) DO UPDATE SET
                    donor_count = donor_count + 1,
                    total_amount = total_amount + excluded.total_amount
                """,
                (employer_norm, donation_day, candidate_id, float(amount), ref),
            )
            accepted += 1
        rows = con.execute(
            """
            SELECT employer_norm, donation_day, candidate_id, donor_count, total_amount, sample_ref
            FROM agg
            WHERE donor_count >= ?
            ORDER BY total_amount DESC, donor_count DESC
            LIMIT 5000
            """,
            (min_donors,),
        ).fetchall()

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "employer_norm",
                "donation_day",
                "candidate_id",
                "donor_count",
                "total_amount",
                "sample_ref",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "employer_norm": row[0],
                    "donation_day": row[1],
                    "candidate_id": row[2],
                    "donor_count": int(row[3]),
                    "total_amount": round(float(row[4]), 2),
                    "sample_ref": row[5],
                }
            )
    write_json(
        json_path,
        {
            "generated_at_utc": now_iso(),
            "min_donors": min_donors,
            "event_count": len(rows),
            "top_events": [
                {
                    "employer_norm": r[0],
                    "donation_day": r[1],
                    "candidate_id": r[2],
                    "donor_count": int(r[3]),
                    "total_amount": round(float(r[4]), 2),
                    "sample_ref": r[5],
                }
                for r in rows[:100]
            ],
        },
    )

    reg_csv = register_artifact(ctx, csv_path, description="Bundled donation events", mime="text/csv", producer_plugin_id="analysis_bundled_donations_v1", record_count=len(rows))
    reg_json = register_artifact(ctx, json_path, description="Bundling summary", mime="application/json", producer_plugin_id="analysis_bundled_donations_v1")
    findings = []
    for idx, row in enumerate(rows[:25]):
        findings.append(
            {
                "id": claim_id(f"bundling:{row[0]}:{row[1]}:{row[2]}"),
                "kind": "bundling_event",
                "severity": "high" if int(row[3]) >= max(5, min_donors) else "medium",
                "summary": f"{row[0]} made {int(row[3])} contributions to {row[2]} on {row[1]}.",
                "measurement_type": "measured",
                "delta_hours": 0.0,
                "modeled_delta_hours": 0.0,
                "evidence": {
                    "artifact_path": reg_csv.artifact.path,
                    "row_index": idx,
                    "sample_ref": row[5],
                },
            }
        )
    return _ok_result(
        f"Detected {len(rows)} bundling events",
        metrics={"events": len(rows), "rows_scanned": scanned, "rows_accepted": accepted},
        findings=findings,
        artifacts=[reg_csv.artifact, reg_json.artifact],
        debug={"artifact_manifest": _artifact_meta([reg_csv, reg_json])},
    )


def _run_analysis_contribution_limit_flags(ctx: PluginContext) -> PluginResult:
    contrib_dvid = str(ctx.settings.get("contributions_dataset_version_id") or "").strip()
    if not contrib_dvid:
        specs = parse_dataset_specs(ctx.settings)
        if "contributions" in specs:
            contrib_dvid = specs["contributions"].dataset_version_id
    if not contrib_dvid and ctx.dataset_version_id:
        contrib_dvid = str(ctx.dataset_version_id)
    if not contrib_dvid:
        return _error_result("MissingConfig", "contributions_dataset_version_id (or datasets.contributions) is required")
    donor_fields = ctx.settings.get("donor_id_fields")
    if not isinstance(donor_fields, list) or not donor_fields:
        donor_fields = ["donor_last", "donor_first", "donor_address"]
    donor_cols: list[str] = []
    for field in donor_fields:
        col = _resolve_dataset_column(ctx, contrib_dvid, str(field), [str(field)])
        if col:
            donor_cols.append(col)
    amount_col = _resolve_dataset_column(ctx, contrib_dvid, str(ctx.settings.get("amount_field") or ""), ["amount", "donation_amount"])
    date_col = _resolve_dataset_column(ctx, contrib_dvid, str(ctx.settings.get("date_field") or ""), ["donation_date", "date", "trans_date"])
    if not donor_cols or not amount_col or not date_col:
        if _strict_prereq(ctx):
            return _error_result("MissingDatasetColumn", "Donor ID/date/amount columns could not be resolved")
        return _ok_result(
            "Contribution limit analysis unavailable: required columns missing",
            metrics={"flags": 0},
            findings=[],
            artifacts=[],
            debug={"reason": "MISSING_COLUMNS"},
        )
    annual_limit = float(ctx.settings.get("annual_limit", 1000.0))
    min_excess = float(ctx.settings.get("min_excess", 0.0))
    batch_size = int(ctx.settings.get("batch_size", 100_000))

    artifacts_dir = ctx.artifacts_dir("analysis_contribution_limit_flags_v1")
    scratch_path = artifacts_dir / "limit_flags.sqlite"
    csv_path = artifacts_dir / "contribution_limit_flags.csv"
    json_path = artifacts_dir / "contribution_limit_flags.json"

    scanned = 0
    with sqlite3.connect(scratch_path) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS agg (
                donor_key TEXT NOT NULL,
                year INTEGER NOT NULL,
                total_amount REAL NOT NULL,
                donation_count INTEGER NOT NULL,
                sample_ref TEXT NOT NULL,
                PRIMARY KEY(donor_key, year)
            )
            """
        )
        for row in iter_rows(
            ctx.storage,
            contrib_dvid,
            columns=[*donor_cols, date_col, amount_col],
            batch_size=batch_size,
            sql=ctx.sql,
        ):
            scanned += 1
            dt = parse_dt(row.get(date_col))
            amount = parse_amount(row.get(amount_col))
            if dt is None or amount is None:
                continue
            donor_parts = [str(row.get(col) or "").strip().upper() for col in donor_cols]
            donor_key = "|".join(part for part in donor_parts if part)
            if not donor_key:
                continue
            year = int(dt.year)
            ref = row_ref(contrib_dvid, int(row["row_index"]))
            con.execute(
                """
                INSERT INTO agg(donor_key, year, total_amount, donation_count, sample_ref)
                VALUES (?, ?, ?, 1, ?)
                ON CONFLICT(donor_key, year) DO UPDATE SET
                    total_amount = total_amount + excluded.total_amount,
                    donation_count = donation_count + 1
                """,
                (donor_key, year, float(amount), ref),
            )
        rows = con.execute(
            """
            SELECT donor_key, year, total_amount, donation_count, sample_ref
            FROM agg
            WHERE total_amount > ?
            ORDER BY (total_amount - ?) DESC, donation_count DESC
            LIMIT 5000
            """,
            (annual_limit + min_excess, annual_limit),
        ).fetchall()

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["donor_key", "year", "total_amount", "annual_limit", "excess_amount", "donation_count", "sample_ref"],
        )
        writer.writeheader()
        for row in rows:
            total_amount = float(row[2])
            excess = total_amount - annual_limit
            writer.writerow(
                {
                    "donor_key": row[0],
                    "year": int(row[1]),
                    "total_amount": round(total_amount, 2),
                    "annual_limit": round(annual_limit, 2),
                    "excess_amount": round(excess, 2),
                    "donation_count": int(row[3]),
                    "sample_ref": row[4],
                }
            )
    write_json(
        json_path,
        {
            "generated_at_utc": now_iso(),
            "annual_limit": annual_limit,
            "min_excess": min_excess,
            "flag_count": len(rows),
        },
    )
    reg_csv = register_artifact(ctx, csv_path, description="Contribution limit exceedance flags", mime="text/csv", producer_plugin_id="analysis_contribution_limit_flags_v1", record_count=len(rows))
    reg_json = register_artifact(ctx, json_path, description="Contribution limit summary", mime="application/json", producer_plugin_id="analysis_contribution_limit_flags_v1")
    findings = []
    for idx, row in enumerate(rows[:25]):
        total_amount = float(row[2])
        excess = total_amount - annual_limit
        findings.append(
            {
                "id": claim_id(f"contrib_limit:{row[0]}:{row[1]}"),
                "kind": "contribution_limit_flag",
                "severity": "high" if excess >= 2 * annual_limit else "medium",
                "summary": f"Donor exceeded annual limit by {excess:.2f} in {int(row[1])}.",
                "measurement_type": "measured",
                "modeled_delta_hours": 0.0,
                "evidence": {"artifact_path": reg_csv.artifact.path, "row_index": idx, "sample_ref": row[4]},
            }
        )
    return _ok_result(
        f"Flagged {len(rows)} donor-year annual limit exceedances",
        metrics={"flags": len(rows), "rows_scanned": scanned, "annual_limit": annual_limit},
        findings=findings,
        artifacts=[reg_csv.artifact, reg_json.artifact],
        debug={"artifact_manifest": _artifact_meta([reg_csv, reg_json])},
    )


def _run_analysis_vendor_influence_breadth(ctx: PluginContext) -> PluginResult:
    path_raw = str(ctx.settings.get("cross_links_path") or "").strip()
    path = Path(path_raw) if path_raw else (_find_artifact(ctx.run_dir, "transform_cross_dataset_link_graph_v1", "cross_links.csv") or Path(""))
    if not path or not path.exists():
        if _strict_prereq(ctx):
            return _error_result("MissingPrerequisiteArtifact", "cross_links.csv not found (set cross_links_path or run transform_cross_dataset_link_graph_v1 first)")
        return _ok_result(
            "Vendor influence breadth unavailable: cross_links.csv missing",
            metrics={"vendors_ranked": 0},
            findings=[],
            artifacts=[],
            debug={"reason": "MISSING_CROSS_LINKS"},
        )
    top_k = int(ctx.settings.get("top_k", 50))
    agg: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            vendor = str(row.get("left_entity_id") or "").strip()
            if not vendor:
                continue
            candidate = str(row.get("right_entity_id") or "").strip()
            bucket = agg.setdefault(
                vendor,
                {
                    "vendor_entity_id": vendor,
                    "vendor_norm": str(row.get("left_norm") or ""),
                    "link_count": 0,
                    "candidates": set(),
                    "relations": Counter(),
                },
            )
            bucket["link_count"] = int(bucket["link_count"]) + 1
            if candidate:
                bucket["candidates"].add(candidate)
            bucket["relations"][str(row.get("relation") or "linked")] += 1
    rows = sorted(
        [
            {
                "vendor_entity_id": v["vendor_entity_id"],
                "vendor_norm": v["vendor_norm"],
                "link_count": int(v["link_count"]),
                "unique_candidate_count": len(v["candidates"]),
                "top_relation": v["relations"].most_common(1)[0][0] if v["relations"] else "linked",
            }
            for v in agg.values()
        ],
        key=lambda x: (-int(x["unique_candidate_count"]), -int(x["link_count"]), str(x["vendor_entity_id"])),
    )
    rows = rows[: max(1, top_k)]
    artifacts_dir = ctx.artifacts_dir("analysis_vendor_influence_breadth_v1")
    csv_path = artifacts_dir / "shared_donor_networks.csv"
    json_path = artifacts_dir / "vendor_influence_breadth.json"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["vendor_entity_id", "vendor_norm", "link_count", "unique_candidate_count", "top_relation"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    write_json(
        json_path,
        {
            "generated_at_utc": now_iso(),
            "vendor_count": len(rows),
            "top_vendors": rows[:100],
            "source_artifact": str(path),
        },
    )
    reg_csv = register_artifact(ctx, csv_path, description="Vendor influence breadth ranking", mime="text/csv", producer_plugin_id="analysis_vendor_influence_breadth_v1", record_count=len(rows))
    reg_json = register_artifact(ctx, json_path, description="Vendor influence breadth summary", mime="application/json", producer_plugin_id="analysis_vendor_influence_breadth_v1")
    findings = [
        {
            "id": claim_id(f"vendor_breadth:{row['vendor_entity_id']}"),
            "kind": "vendor_influence_breadth",
            "severity": "medium" if int(row["unique_candidate_count"]) >= 2 else "low",
            "summary": f"{row['vendor_norm']} reaches {int(row['unique_candidate_count'])} unique linked entities.",
            "measurement_type": "measured",
            "modeled_delta_hours": 0.0,
            "evidence": {"artifact_path": reg_csv.artifact.path},
        }
        for row in rows[:25]
    ]
    return _ok_result(
        f"Computed influence breadth for {len(rows)} vendors",
        metrics={"vendors_ranked": len(rows), "source_rows": sum(int(v["link_count"]) for v in rows)},
        findings=findings,
        artifacts=[reg_csv.artifact, reg_json.artifact],
        debug={"artifact_manifest": _artifact_meta([reg_csv, reg_json])},
    )


def _nearest_day_distance(target: datetime, points: list[datetime]) -> float:
    if not points:
        return math.inf
    return min(abs((target - p).total_seconds()) / 86400.0 for p in points)


def _run_analysis_vendor_politician_timing_permutation(ctx: PluginContext) -> PluginResult:
    specs, err = _dataset_specs_or_error(ctx, [])
    if err:
        return err
    contracts_dvid = str(ctx.settings.get("contracts_dataset_version_id") or "").strip()
    contributions_dvid = str(ctx.settings.get("contributions_dataset_version_id") or "").strip()
    if not contracts_dvid and "contracts" in specs:
        contracts_dvid = specs["contracts"].dataset_version_id
    if not contributions_dvid and "contributions" in specs:
        contributions_dvid = specs["contributions"].dataset_version_id
    if not contracts_dvid and ctx.dataset_version_id:
        contracts_dvid = str(ctx.dataset_version_id)
    if not contributions_dvid and ctx.dataset_version_id:
        contributions_dvid = str(ctx.dataset_version_id)
    if not contracts_dvid or not contributions_dvid:
        return _error_result("MissingConfig", "contracts/contributions dataset ids must be provided (directly or via settings.datasets)")

    vendor_col = _resolve_dataset_column(ctx, contracts_dvid, str(ctx.settings.get("vendor_field") or ""), ["vendor_name1", "vendor", "vendor_name"])
    award_col = _resolve_dataset_column(ctx, contracts_dvid, str(ctx.settings.get("award_date_field") or ""), ["award_date", "contract_date", "date"])
    candidate_col = _resolve_dataset_column(ctx, contributions_dvid, str(ctx.settings.get("candidate_id_field") or ""), ["candidate_id", "candidate", "candidate_name"])
    donation_col = _resolve_dataset_column(ctx, contributions_dvid, str(ctx.settings.get("donation_date_field") or ""), ["donation_date", "date", "trans_date"])
    amount_col = _resolve_dataset_column(ctx, contributions_dvid, str(ctx.settings.get("amount_field") or ""), ["amount", "donation_amount"])
    if not vendor_col or not award_col or not candidate_col or not donation_col:
        if _strict_prereq(ctx):
            return _error_result("MissingDatasetColumn", "Required vendor/candidate/date columns could not be resolved for timing analysis")
        return _ok_result(
            "Timing permutation unavailable: required columns missing",
            metrics={"pair_count": 0},
            findings=[],
            artifacts=[],
            debug={"reason": "MISSING_COLUMNS"},
        )

    cross_links_path_raw = str(ctx.settings.get("cross_links_path") or "").strip()
    cross_links_path = Path(cross_links_path_raw) if cross_links_path_raw else _find_artifact(ctx.run_dir, "transform_cross_dataset_link_graph_v1", "cross_links.csv")
    if cross_links_path is None or not cross_links_path.exists():
        if _strict_prereq(ctx):
            return _error_result("MissingPrerequisiteArtifact", "cross_links.csv is required for vendor-to-contribution linkage in timing analysis")
        return _ok_result(
            "Timing permutation unavailable: cross_links.csv missing",
            metrics={"pair_count": 0},
            findings=[],
            artifacts=[],
            debug={"reason": "MISSING_CROSS_LINKS"},
        )

    row_to_vendor: dict[str, str] = {}
    with cross_links_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ref = str(row.get("right_ref") or "").strip()
            vendor_id = str(row.get("left_entity_id") or "").strip()
            if ref and vendor_id:
                row_to_vendor[ref] = vendor_id

    vendor_awards: dict[str, list[datetime]] = defaultdict(list)
    for row in iter_rows(ctx.storage, contracts_dvid, columns=[vendor_col, award_col], batch_size=int(ctx.settings.get("batch_size", 100_000)), sql=ctx.sql):
        vendor_norm = normalize_org_name(row.get(vendor_col))
        if not vendor_norm:
            continue
        dt = parse_dt(row.get(award_col))
        if dt is None:
            continue
        vendor_awards[stable_entity_id("org", vendor_norm)].append(dt)

    pair_donations: dict[tuple[str, str], list[datetime]] = defaultdict(list)
    pair_amounts: dict[tuple[str, str], float] = defaultdict(float)
    for row in iter_rows(ctx.storage, contributions_dvid, columns=[candidate_col, donation_col, *( [amount_col] if amount_col else [])], batch_size=int(ctx.settings.get("batch_size", 100_000)), sql=ctx.sql):
        candidate = str(row.get(candidate_col) or "").strip()
        dt = parse_dt(row.get(donation_col))
        if not candidate or dt is None:
            continue
        ref = row_ref(contributions_dvid, int(row["row_index"]))
        vendor_id = row_to_vendor.get(ref)
        if not vendor_id:
            continue
        key = (vendor_id, candidate)
        pair_donations[key].append(dt)
        amt = parse_amount(row.get(amount_col)) if amount_col else None
        if amt is not None:
            pair_amounts[key] += float(amt)

    min_donations = int(ctx.settings.get("min_donations", 3))
    n_permutations = max(100, int(ctx.settings.get("n_permutations", 2000)))
    n_permutations = min(n_permutations, int(ctx.settings.get("max_permutations", 5000)))
    rng_seed = int(ctx.settings.get("rng_seed", 0))
    top_k = int(ctx.settings.get("top_k", 200))

    rows: list[dict[str, Any]] = []
    for (vendor_id, candidate), donations in pair_donations.items():
        if len(donations) < min_donations:
            continue
        awards = vendor_awards.get(vendor_id, [])
        if not awards:
            continue
        observed = sum(_nearest_day_distance(d, awards) for d in donations) / float(len(donations))
        start = min(awards)
        end = max(awards)
        span_days = max(1.0, (end - start).total_seconds() / 86400.0)
        if rng_seed == 0:
            seed = _hash_seed(f"{ctx.run_seed}:{vendor_id}:{candidate}")
        else:
            seed = int(rng_seed)
        rng = random.Random(seed)
        null_sum = 0.0
        null_sq_sum = 0.0
        le_count = 0
        for _ in range(n_permutations):
            rand_awards = [start + timedelta(days=rng.uniform(0.0, span_days)) for _ in awards]
            perm_mean = sum(_nearest_day_distance(d, rand_awards) for d in donations) / float(len(donations))
            null_sum += perm_mean
            null_sq_sum += perm_mean * perm_mean
            if perm_mean <= observed:
                le_count += 1
        null_mean = null_sum / float(n_permutations)
        null_var = max(0.0, (null_sq_sum / float(n_permutations)) - (null_mean * null_mean))
        null_std = math.sqrt(null_var)
        effect = (null_mean - observed) / null_std if null_std > 1e-12 else 0.0
        p_value = le_count / float(n_permutations)
        rows.append(
            {
                "vendor_entity_id": vendor_id,
                "candidate_id": candidate,
                "donation_count": len(donations),
                "observed_mean_abs_days": round(observed, 6),
                "null_mean_abs_days": round(null_mean, 6),
                "effect_size": round(effect, 6),
                "p_value": round(p_value, 8),
                "total_amount": round(pair_amounts.get((vendor_id, candidate), 0.0), 2),
                "seed": int(seed),
            }
        )
    rows.sort(key=lambda r: (float(r["p_value"]), -float(r["effect_size"]), -int(r["donation_count"])))
    rows = rows[: max(1, top_k)]

    artifacts_dir = ctx.artifacts_dir("analysis_vendor_politician_timing_permutation_v1")
    json_path = artifacts_dir / "politician_timing_analysis.json"
    csv_path = artifacts_dir / "vendor_politician_timing.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0].keys()) if rows else ["vendor_entity_id", "candidate_id", "donation_count", "observed_mean_abs_days", "null_mean_abs_days", "effect_size", "p_value", "total_amount", "seed"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    write_json(
        json_path,
        {
            "generated_at_utc": now_iso(),
            "min_donations": min_donations,
            "n_permutations": n_permutations,
            "pair_count": len(rows),
            "rows": rows[:1000],
        },
    )
    reg_json = register_artifact(ctx, json_path, description="Vendor-politician timing permutation summary", mime="application/json", producer_plugin_id="analysis_vendor_politician_timing_permutation_v1")
    reg_csv = register_artifact(ctx, csv_path, description="Vendor-politician timing permutation rows", mime="text/csv", producer_plugin_id="analysis_vendor_politician_timing_permutation_v1", record_count=len(rows))
    findings = []
    for idx, row in enumerate(rows[:25]):
        findings.append(
            {
                "id": claim_id(f"timing_signal:{row['vendor_entity_id']}:{row['candidate_id']}"),
                "kind": "timing_clustering_signal",
                "severity": "high" if float(row["p_value"]) <= 0.05 and float(row["effect_size"]) >= 1.0 else "medium",
                "summary": f"Timing clustering for vendor {row['vendor_entity_id']} and candidate {row['candidate_id']} (p={row['p_value']}, effect={row['effect_size']}).",
                "measurement_type": "modeled",
                "modeled_scope": "pair",
                "modeled_assumptions": "Permutation null over observed vendor award-date interval.",
                "baseline_host_count": 1,
                "modeled_host_count": 1,
                "baseline_value": float(row["observed_mean_abs_days"]),
                "modeled_value": float(row["null_mean_abs_days"]),
                "delta_value": float(row["null_mean_abs_days"]) - float(row["observed_mean_abs_days"]),
                "unit": "days",
                "modeled_delta_hours": max(0.0, float(row["null_mean_abs_days"]) - float(row["observed_mean_abs_days"])) * 24.0,
                "evidence": {"artifact_path": reg_csv.artifact.path, "row_index": idx},
            }
        )
    return _ok_result(
        f"Computed timing permutation signals for {len(rows)} vendor-candidate pairs",
        metrics={"pair_count": len(rows), "n_permutations": n_permutations},
        findings=findings,
        artifacts=[reg_json.artifact, reg_csv.artifact],
        debug={"artifact_manifest": _artifact_meta([reg_json, reg_csv])},
    )


def _run_analysis_red_flags_refined(ctx: PluginContext) -> PluginResult:
    bundling_path = _find_artifact(ctx.run_dir, "analysis_bundled_donations_v1", "bundling_events.csv")
    limits_path = _find_artifact(ctx.run_dir, "analysis_contribution_limit_flags_v1", "contribution_limit_flags.csv")
    timing_path = _find_artifact(ctx.run_dir, "analysis_vendor_politician_timing_permutation_v1", "vendor_politician_timing.csv")
    links_path = _find_artifact(ctx.run_dir, "transform_cross_dataset_link_graph_v1", "cross_links.csv")
    missing = [name for name, p in [("bundling_events.csv", bundling_path), ("contribution_limit_flags.csv", limits_path), ("vendor_politician_timing.csv", timing_path), ("cross_links.csv", links_path)] if p is None or not p.exists()]
    if missing:
        if _strict_prereq(ctx):
            return _error_result("MissingPrerequisiteArtifact", f"Missing prerequisite artifacts: {', '.join(missing)}")
        return _ok_result(
            f"Red-flag refinement unavailable: missing prerequisites ({', '.join(missing)})",
            metrics={"flag_count": 0},
            findings=[],
            artifacts=[],
            debug={"reason": "MISSING_PREREQUISITES", "missing": missing},
        )

    sole_source_methods = [str(v).strip().upper() for v in (ctx.settings.get("sole_source_methods") or ["Sole Source", "Limited Competition", "Emergency", "Exempt"]) if str(v).strip()]
    max_p = float(ctx.settings.get("max_p_value_for_timing_flag", 0.1))
    min_effect = float(ctx.settings.get("min_effect_size_for_timing_flag", 0.5))

    employer_to_vendor: dict[str, str] = {}
    with links_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            relation = str(row.get("relation") or "")
            if relation != "vendor_employer":
                continue
            employer_norm = normalize_org_name(row.get("right_norm"))
            vendor_id = str(row.get("left_entity_id") or "").strip()
            if employer_norm and vendor_id:
                employer_to_vendor.setdefault(employer_norm, vendor_id)

    bundle_by_pair: Counter[tuple[str, str]] = Counter()
    with bundling_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            employer_norm = normalize_org_name(row.get("employer_norm"))
            candidate = str(row.get("candidate_id") or "").strip()
            vendor = employer_to_vendor.get(employer_norm)
            if vendor and candidate:
                bundle_by_pair[(vendor, candidate)] += int(row.get("donor_count") or 0)

    limit_pressure = 0
    with limits_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for _ in reader:
            limit_pressure += 1

    flagged: list[dict[str, Any]] = []
    with timing_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            vendor = str(row.get("vendor_entity_id") or "").strip()
            candidate = str(row.get("candidate_id") or "").strip()
            if not vendor or not candidate:
                continue
            p = float(row.get("p_value") or 1.0)
            effect = float(row.get("effect_size") or 0.0)
            if p > max_p or effect < min_effect:
                continue
            bundle_score = int(bundle_by_pair.get((vendor, candidate), 0))
            score = (max(0.0, min_effect - p) * 10.0) + max(0.0, effect) + (0.1 * bundle_score) + (0.01 * limit_pressure)
            flagged.append(
                {
                    "vendor_entity_id": vendor,
                    "candidate_id": candidate,
                    "timing_p_value": p,
                    "timing_effect_size": effect,
                    "bundle_signal": bundle_score,
                    "global_limit_pressure_count": limit_pressure,
                    "sole_source_methods": ";".join(sorted(set(sole_source_methods))),
                    "risk_score": round(score, 6),
                }
            )
    flagged.sort(key=lambda x: (-float(x["risk_score"]), float(x["timing_p_value"]), -float(x["timing_effect_size"])))

    artifacts_dir = ctx.artifacts_dir("analysis_red_flags_refined_v1")
    csv_path = artifacts_dir / "red_flags_refined.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(flagged[0].keys()) if flagged else ["vendor_entity_id", "candidate_id", "timing_p_value", "timing_effect_size", "bundle_signal", "global_limit_pressure_count", "sole_source_methods", "risk_score"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in flagged:
            writer.writerow(row)
    reg_csv = register_artifact(ctx, csv_path, description="Refined multi-factor red flags", mime="text/csv", producer_plugin_id="analysis_red_flags_refined_v1", record_count=len(flagged))
    findings = []
    for idx, row in enumerate(flagged[:25]):
        findings.append(
            {
                "id": claim_id(f"red_flag:{row['vendor_entity_id']}:{row['candidate_id']}"),
                "kind": "multi_factor_red_flag",
                "severity": "high" if float(row["risk_score"]) >= 2.0 else "medium",
                "summary": f"Multi-factor red flag for vendor {row['vendor_entity_id']} and candidate {row['candidate_id']} (score={row['risk_score']}).",
                "measurement_type": "modeled",
                "modeled_scope": "pair",
                "modeled_assumptions": "Timing clustering + bundling + systemwide limit-pressure heuristic.",
                "baseline_host_count": 1,
                "modeled_host_count": 1,
                "baseline_value": 0.0,
                "modeled_value": float(row["risk_score"]),
                "delta_value": float(row["risk_score"]),
                "unit": "risk_score",
                "modeled_delta_hours": max(0.0, float(row["timing_effect_size"])) * 0.5,
                "evidence": {"artifact_path": reg_csv.artifact.path, "row_index": idx},
            }
        )
    return _ok_result(
        f"Computed {len(flagged)} multi-factor red flags",
        metrics={"flag_count": len(flagged), "limit_pressure_count": limit_pressure},
        findings=findings,
        artifacts=[reg_csv.artifact],
        debug={"artifact_manifest": _artifact_meta([reg_csv])},
    )


def _run_report_evidence_index(ctx: PluginContext) -> PluginResult:
    artifacts_dir = ctx.run_dir / "artifacts"
    if not artifacts_dir.exists():
        return _error_result("MissingArtifactsDirectory", "Run artifacts directory does not exist")
    rows: list[dict[str, Any]] = []
    for plugin_id in sorted(PACK_PLUGIN_IDS):
        pdir = artifacts_dir / plugin_id
        if not pdir.exists():
            continue
        for path in sorted(pdir.rglob("*")):
            if not path.is_file():
                continue
            rel = str(path.relative_to(ctx.run_dir))
            ext = path.suffix.lower()
            if ext == ".csv":
                count = _csv_row_count(path)
            elif ext == ".jsonl":
                count = _jsonl_row_count(path)
            elif ext == ".json":
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
                        count = len(payload.get("rows"))
                    else:
                        count = 1
                except Exception:
                    count = 0
            else:
                count = 0
            rows.append(
                {
                    "plugin_id": plugin_id,
                    "path": rel,
                    "sha256": file_sha256(path),
                    "size_bytes": int(path.stat().st_size),
                    "record_count": int(count),
                }
            )

    out_json = ctx.artifacts_dir("report_evidence_index_v1") / "evidence_index.json"
    out_md = ctx.artifacts_dir("report_evidence_index_v1") / "evidence_index.md"
    write_json(out_json, {"generated_at_utc": now_iso(), "artifacts": rows})
    lines = ["# Evidence Index", "", f"Generated: {now_iso()}", "", "| plugin_id | path | record_count | sha256 |", "| --- | --- | ---: | --- |"]
    for row in rows:
        lines.append(f"| {row['plugin_id']} | {row['path']} | {row['record_count']} | {row['sha256']} |")
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    reg_json = register_artifact(ctx, out_json, description="Cross-dataset evidence index (json)", mime="application/json", producer_plugin_id="report_evidence_index_v1", record_count=len(rows))
    reg_md = register_artifact(ctx, out_md, description="Cross-dataset evidence index (markdown)", mime="text/markdown", producer_plugin_id="report_evidence_index_v1", record_count=len(rows))
    findings = [
        {
            "id": claim_id(f"evidence_index:{ctx.run_id}:{len(rows)}"),
            "kind": "evidence_index",
            "severity": "info",
            "summary": f"Indexed {len(rows)} artifacts for cross-dataset plugin pack.",
            "measurement_type": "measured",
            "evidence": {"artifact_path": reg_json.artifact.path},
        }
    ]
    return _ok_result(
        f"Indexed {len(rows)} cross-dataset artifacts",
        metrics={"artifact_count": len(rows)},
        findings=findings,
        artifacts=[reg_json.artifact, reg_md.artifact],
        debug={"artifact_manifest": _artifact_meta([reg_json, reg_md])},
    )


def _run_ingest_sql_dump(ctx: PluginContext) -> PluginResult:
    from .sql_dump_import import ParsedCreateTable, import_sql_dump

    input_path = str(ctx.settings.get("input_path") or "").strip()
    if not input_path:
        return _error_result("MissingConfig", "settings.input_path is required")
    sql_path = Path(input_path)
    if not sql_path.exists():
        return _error_result("MissingInputFile", f"SQL input file not found: {input_path}")

    dataset_version_id = ctx.dataset_version_id
    if not dataset_version_id:
        return _error_result("MissingDatasetVersion", "dataset_version_id is required for SQL ingest")
    chunk_rows = int(ctx.settings.get("chunk_rows", 50_000))
    max_rows = ctx.settings.get("max_rows")
    max_rows_int = int(max_rows) if isinstance(max_rows, (int, float)) and int(max_rows) > 0 else None
    table_name = f"dataset_{dataset_version_id}"
    parsed_columns: list[str] = []
    rows_inserted = 0
    with ctx.storage.connection() as con:
        ctx.storage.reset_dataset_version(
            dataset_version_id,
            table_name=table_name,
            data_hash=ctx.input_hash,
            created_at=now_iso(),
            conn=con,
        )

        def on_create(parsed: ParsedCreateTable) -> None:
            nonlocal parsed_columns
            if not parsed_columns and parsed.columns:
                parsed_columns = list(parsed.columns)
                columns_meta = []
                for idx, col in enumerate(parsed_columns, start=1):
                    columns_meta.append(
                        {
                            "column_id": idx,
                            "safe_name": f"c{idx}",
                            "original_name": str(col),
                            "source_original_name": str(col),
                            "dtype": "TEXT",
                            "sqlite_type": "TEXT",
                        }
                    )
                ctx.storage.create_dataset_table(table_name, columns_meta, con)
                ctx.storage.add_append_only_triggers(table_name, con)
                ctx.storage.replace_dataset_columns(dataset_version_id, columns_meta, con)

        def on_insert(_tbl: str, columns: list[str] | None, rows: list[list[Any]]) -> None:
            nonlocal rows_inserted, parsed_columns
            if not parsed_columns and columns:
                parsed_columns = list(columns)
                columns_meta = []
                for idx, col in enumerate(parsed_columns, start=1):
                    columns_meta.append(
                        {
                            "column_id": idx,
                            "safe_name": f"c{idx}",
                            "original_name": str(col),
                            "source_original_name": str(col),
                            "dtype": "TEXT",
                            "sqlite_type": "TEXT",
                        }
                    )
                ctx.storage.create_dataset_table(table_name, columns_meta, con)
                ctx.storage.add_append_only_triggers(table_name, con)
                ctx.storage.replace_dataset_columns(dataset_version_id, columns_meta, con)
            if not parsed_columns:
                raise ValueError("Could not determine SQL column list from CREATE TABLE or INSERT statements")
            safe_columns = [f"c{i+1}" for i in range(len(parsed_columns))]
            batch = []
            for row_values in rows:
                padded = list(row_values[: len(parsed_columns)])
                if len(padded) < len(parsed_columns):
                    padded.extend([None] * (len(parsed_columns) - len(padded)))
                row_obj = {parsed_columns[i]: padded[i] for i in range(len(parsed_columns))}
                batch.append((rows_inserted, json.dumps(row_obj, ensure_ascii=False), *padded))
                rows_inserted += 1
            ctx.storage.insert_dataset_rows(table_name, safe_columns, batch, con)

        manifest = import_sql_dump(
            sql_path,
            encoding=str(ctx.settings.get("encoding", "utf-8")),
            max_rows=max_rows_int,
            chunk_rows=max(1, chunk_rows),
            on_create_table=on_create,
            on_insert_rows=on_insert,
        )
        ctx.storage.update_dataset_version_stats(dataset_version_id, rows_inserted, len(parsed_columns), conn=con)
    artifacts_dir = ctx.artifacts_dir("ingest_sql_dump_v1")
    manifest_path = artifacts_dir / "canonical_import_manifest.json"
    write_json(
        manifest_path,
        {
            "generated_at_utc": now_iso(),
            "dataset_version_id": dataset_version_id,
            "table_name": table_name,
            "rows_inserted": rows_inserted,
            "columns": parsed_columns,
            "sql_import": manifest,
            "input_path": str(sql_path),
        },
    )
    reg_manifest = register_artifact(ctx, manifest_path, description="SQL dump canonical import manifest", mime="application/json", producer_plugin_id="ingest_sql_dump_v1")
    findings = [
        {
            "id": claim_id(f"sql_ingest:{dataset_version_id}:{rows_inserted}"),
            "kind": "sql_ingest_summary",
            "severity": "info",
            "summary": f"Imported {rows_inserted} rows from SQL dump into {table_name}.",
            "measurement_type": "measured",
            "evidence": {"artifact_path": reg_manifest.artifact.path},
        }
    ]
    return _ok_result(
        f"Imported {rows_inserted} rows from SQL dump",
        metrics={"rows_inserted": rows_inserted, "column_count": len(parsed_columns)},
        findings=findings,
        artifacts=[reg_manifest.artifact],
        debug={"artifact_manifest": _artifact_meta([reg_manifest])},
    )


def run_openplanter_plugin(plugin_id: str, ctx: PluginContext) -> PluginResult:
    if plugin_id == "ingest_sql_dump_v1":
        return _run_ingest_sql_dump(ctx)
    if plugin_id == "transform_entity_resolution_map_v1":
        return _run_transform_entity_resolution_map(ctx)
    if plugin_id == "transform_cross_dataset_link_graph_v1":
        return _run_transform_cross_dataset_link_graph(ctx)
    if plugin_id == "analysis_bundled_donations_v1":
        return _run_analysis_bundled_donations(ctx)
    if plugin_id == "analysis_contribution_limit_flags_v1":
        return _run_analysis_contribution_limit_flags(ctx)
    if plugin_id == "analysis_vendor_influence_breadth_v1":
        return _run_analysis_vendor_influence_breadth(ctx)
    if plugin_id == "analysis_vendor_politician_timing_permutation_v1":
        return _run_analysis_vendor_politician_timing_permutation(ctx)
    if plugin_id == "analysis_red_flags_refined_v1":
        return _run_analysis_red_flags_refined(ctx)
    if plugin_id == "report_evidence_index_v1":
        return _run_report_evidence_index(ctx)
    return _error_result("UnknownPluginId", f"Unsupported OpenPlanter plugin id: {plugin_id}")
