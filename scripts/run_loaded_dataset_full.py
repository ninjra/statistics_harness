from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import statistics
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from statistic_harness.core.pipeline import Pipeline
from statistic_harness.core.erp_inference import (
    infer_erp_type_from_field_names,
    normalize_field_name,
)
from statistic_harness.core.tenancy import get_tenant_context
from statistic_harness.core.utils import make_run_id


REPO_ROOT = Path(__file__).resolve().parents[1]

ROLE_GROUP_PROCESS = ("process_name", "process", "process_id")
ROLE_GROUP_TIME = ("queue_time", "start_time", "end_time")
ROLE_GROUP_CASE = ("master_id",)
ROLE_GROUP_USER = ("user_id",)
ROLE_GROUP_HOST = ("host_id",)

_STRUCTURAL_ROLE_REQUIREMENTS: dict[str, tuple[tuple[str, ...], ...]] = {
    "analysis_association_rules_apriori_v1": (ROLE_GROUP_PROCESS,),
    "analysis_biclustering_cheng_church_v1": (ROLE_GROUP_PROCESS,),
    "analysis_burst_modeling_hawkes_v1": (ROLE_GROUP_PROCESS, ROLE_GROUP_TIME),
    "analysis_busy_period_segmentation_v2": (ROLE_GROUP_PROCESS, ROLE_GROUP_TIME),
    "analysis_constrained_clustering_cop_kmeans_v1": (ROLE_GROUP_PROCESS,),
    "analysis_daily_pattern_alignment_dtw_v1": (ROLE_GROUP_PROCESS, ROLE_GROUP_TIME),
    "analysis_dependency_community_leiden_v1": (
        ROLE_GROUP_PROCESS,
        ROLE_GROUP_TIME,
        ROLE_GROUP_CASE,
    ),
    "analysis_dependency_community_louvain_v1": (
        ROLE_GROUP_PROCESS,
        ROLE_GROUP_TIME,
        ROLE_GROUP_CASE,
    ),
    "analysis_dependency_critical_path_v1": (
        ROLE_GROUP_PROCESS,
        ROLE_GROUP_TIME,
        ROLE_GROUP_CASE,
    ),
    "analysis_distribution_shift_wasserstein_v1": (ROLE_GROUP_PROCESS, ROLE_GROUP_TIME),
    "analysis_frequent_itemsets_fpgrowth_v1": (ROLE_GROUP_PROCESS,),
    "analysis_graph_min_cut_partition_v1": (
        ROLE_GROUP_PROCESS,
        ROLE_GROUP_TIME,
        ROLE_GROUP_CASE,
    ),
    "analysis_param_near_duplicate_minhash_v1": (ROLE_GROUP_PROCESS,),
    "analysis_param_near_duplicate_simhash_v1": (ROLE_GROUP_PROCESS,),
    "analysis_process_counterfactuals": (ROLE_GROUP_PROCESS, ROLE_GROUP_TIME),
    "analysis_process_sequence_bottlenecks": (
        ROLE_GROUP_PROCESS,
        ROLE_GROUP_TIME,
        ROLE_GROUP_CASE,
    ),
    "analysis_retry_rate_hotspots_v1": (ROLE_GROUP_PROCESS, ROLE_GROUP_TIME),
    "analysis_sequence_grammar_sequitur_v1": (
        ROLE_GROUP_PROCESS,
        ROLE_GROUP_TIME,
        ROLE_GROUP_CASE,
    ),
    "analysis_sequential_patterns_prefixspan_v1": (
        ROLE_GROUP_PROCESS,
        ROLE_GROUP_TIME,
        ROLE_GROUP_CASE,
    ),
    "analysis_similarity_graph_spectral_clustering_v1": (ROLE_GROUP_PROCESS,),
    "analysis_user_host_savings": (
        ROLE_GROUP_PROCESS,
        ROLE_GROUP_TIME,
        ROLE_GROUP_USER,
        ROLE_GROUP_HOST,
    ),
}


def _debug_stage(label: str) -> None:
    raw = os.environ.get("STAT_HARNESS_DEBUG_STARTUP", "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        print(f"STAGE={label}", flush=True)


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _infer_structural_role(field_name: str) -> str | None:
    name = str(field_name or "").strip().lower()
    if not name:
        return None
    tokenized = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    tokens = [tok for tok in tokenized.split("_") if tok]
    token_set = set(tokens)

    def has_any(values: set[str]) -> bool:
        return any(v in token_set for v in values)

    if has_any({"queue"}) and has_any({"dt", "time", "date", "ts", "timestamp"}):
        return "queue_time"
    if has_any({"start", "begin", "started"}) and has_any({"dt", "time", "date", "ts", "timestamp"}):
        return "start_time"
    if has_any({"end", "finish", "finished", "complete", "completed", "stop"}) and has_any(
        {"dt", "time", "date", "ts", "timestamp"}
    ):
        return "end_time"
    if tokenized == "process_id":
        return "process_name"
    if has_any({"dependency", "dep", "parent", "prereq", "precede"}):
        return "dependency_id"
    if has_any({"master", "workflow", "case", "trace"}):
        return "master_id"
    if tokenized == "process_queue_id" or (has_any({"queue"}) and has_any({"process"}) and has_any({"id"})):
        return "master_id"
    if has_any({"module", "mod"}):
        return "module_code"
    if has_any({"user", "operator", "owner"}):
        return "user_id"
    if has_any({"host", "server", "node", "machine", "worker"}):
        return "host_id"
    if has_any({"status", "state", "result", "outcome"}):
        return "status"
    if has_any({"process", "activity", "task", "job", "step", "action", "proc"}):
        return "process_name"
    return None


def _resolve_structural_roles(conn: sqlite3.Connection, dataset_version_id: str) -> dict[str, str]:
    safe_to_role: dict[str, str] = {}
    rows = conn.execute(
        """
        SELECT safe_name, original_name, role
        FROM dataset_columns
        WHERE dataset_version_id = ?
        ORDER BY column_id
        """,
        (dataset_version_id,),
    ).fetchall()
    for row in rows:
        safe = str(row["safe_name"] or "").strip()
        if not safe:
            continue
        role = str(row["role"] or "").strip()
        if role:
            safe_to_role[safe] = role

    template_row = conn.execute(
        """
        SELECT template_id, mapping_json
        FROM dataset_templates
        WHERE dataset_version_id = ?
        ORDER BY updated_at DESC
        LIMIT 1
        """,
        (dataset_version_id,),
    ).fetchone()
    if not template_row:
        for row in rows:
            safe = str(row["safe_name"] or "").strip()
            if not safe or safe in safe_to_role:
                continue
            inferred = _infer_structural_role(str(row["original_name"] or ""))
            if inferred:
                safe_to_role[safe] = inferred
        return safe_to_role

    mapping_json = str(template_row["mapping_json"] or "").strip()
    template_id = template_row["template_id"]
    name_to_safe: dict[str, str] = {}
    if template_id is not None:
        tf_rows = conn.execute(
            """
            SELECT name, safe_name
            FROM template_fields
            WHERE template_id = ?
            ORDER BY field_id
            """,
            (int(template_id),),
        ).fetchall()
        for row in tf_rows:
            name = str(row["name"] or "").strip()
            safe = str(row["safe_name"] or "").strip()
            if name and safe:
                name_to_safe[name] = safe

    try:
        payload = json.loads(mapping_json) if mapping_json else {}
    except json.JSONDecodeError:
        payload = {}
    mapping = payload.get("mapping") if isinstance(payload, dict) else {}
    if not isinstance(mapping, dict):
        mapping = {}

    for field_name, meta in mapping.items():
        if not isinstance(field_name, str) or not field_name.strip():
            continue
        field_name = field_name.strip()
        template_safe = name_to_safe.get(field_name)
        if not template_safe and isinstance(meta, dict):
            candidate = str(meta.get("safe_name") or "").strip()
            if candidate:
                template_safe = candidate
        if not template_safe:
            continue
        source_safe = None
        if isinstance(meta, dict):
            source_safe = str(meta.get("safe_name") or "").strip() or None
        current = safe_to_role.get(source_safe or template_safe)
        inferred = _infer_structural_role(field_name)
        temporal_roles = {"queue_time", "start_time", "end_time"}
        chosen = current
        if inferred and (
            not current
            or current in {"id", "numeric", "parameter"}
            or (inferred in temporal_roles and current in temporal_roles and current != inferred)
            or (inferred == "process_name" and current == "process_id")
        ):
            chosen = inferred
        if not chosen:
            chosen = inferred
        if chosen:
            safe_to_role[template_safe] = chosen

    for row in rows:
        safe = str(row["safe_name"] or "").strip()
        if not safe or safe in safe_to_role:
            continue
        inferred = _infer_structural_role(str(row["original_name"] or ""))
        if inferred:
            safe_to_role[safe] = inferred
    return safe_to_role


def _group_satisfied(roles_present: set[str], group: tuple[str, ...]) -> bool:
    return any(role in roles_present for role in group)


def _structural_preflight(
    db_path: Path,
    dataset_version_id: str,
    plugin_ids: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    conn = _connect(db_path)
    try:
        safe_to_role = _resolve_structural_roles(conn, dataset_version_id)
    finally:
        conn.close()

    roles_present = {str(v).strip().lower() for v in safe_to_role.values() if str(v).strip()}
    blockers: list[dict[str, Any]] = []
    checked = sorted(set(plugin_ids) & set(_STRUCTURAL_ROLE_REQUIREMENTS.keys()))
    for plugin_id in checked:
        groups = _STRUCTURAL_ROLE_REQUIREMENTS[plugin_id]
        missing: list[list[str]] = []
        for group in groups:
            if not _group_satisfied(roles_present, group):
                missing.append(list(group))
        if missing:
            blockers.append(
                {
                    "plugin_id": plugin_id,
                    "missing_role_groups": missing,
                }
            )

    report = {
        "dataset_version_id": dataset_version_id,
        "checked_plugins": checked,
        "checked_count": len(checked),
        "blocking_count": len(blockers),
        "blockers": blockers,
        "roles_present": sorted(roles_present),
        "safe_to_role_count": len(safe_to_role),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "structural_preflight.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def _runtime_trend(db_path: Path, dataset_version_id: str, run_id: str) -> dict[str, Any]:
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT run_id, started_at, created_at, completed_at
            FROM runs
            WHERE dataset_version_id = ?
              AND status IN ('completed', 'partial')
              AND completed_at IS NOT NULL
            ORDER BY created_at
            """,
            (dataset_version_id,),
        ).fetchall()
    finally:
        conn.close()

    samples: list[tuple[str, float]] = []
    for row in rows:
        start_dt = _parse_ts(row["started_at"]) or _parse_ts(row["created_at"])
        end_dt = _parse_ts(row["completed_at"])
        if start_dt is None or end_dt is None:
            continue
        minutes = (end_dt - start_dt).total_seconds() / 60.0
        if minutes > 0.0:
            samples.append((str(row["run_id"]), float(minutes)))

    if not samples:
        return {}

    current = next((m for rid, m in samples if rid == run_id), None)
    history = [m for rid, m in samples if rid != run_id]
    avg = float(statistics.mean(history)) if history else None
    stddev = float(statistics.pstdev(history)) if len(history) > 1 else (0.0 if history else None)
    delta_pct = None
    if isinstance(current, float) and isinstance(avg, float) and avg > 0.0:
        delta_pct = ((current - avg) / avg) * 100.0
    return {
        "current_minutes": current,
        "historical_count": len(history),
        "historical_avg_minutes": avg,
        "historical_stddev_minutes": stddev,
        "delta_vs_avg_percent": delta_pct,
    }


def _latest_dataset_version_row(db_path: Path) -> dict[str, Any] | None:
    conn = _connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT dv.dataset_version_id,
                   dv.dataset_id,
                   dv.created_at,
                   dv.table_name,
                   dv.row_count,
                   dv.column_count,
                   dv.data_hash,
                   dv.raw_format_id,
                   dv.source_classification,
                   d.project_id,
                   p.erp_type
            FROM dataset_versions dv
            LEFT JOIN datasets d
              ON d.dataset_id = dv.dataset_id
             AND d.tenant_id = dv.tenant_id
            LEFT JOIN projects p
              ON p.project_id = d.project_id
             AND p.tenant_id = d.tenant_id
            ORDER BY dv.row_count DESC, dv.created_at DESC
            LIMIT 1
            """
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def _dataset_version_row(db_path: Path, dataset_version_id: str) -> dict[str, Any] | None:
    conn = _connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT dv.dataset_version_id,
                   dv.dataset_id,
                   dv.created_at,
                   dv.table_name,
                   dv.row_count,
                   dv.column_count,
                   dv.data_hash,
                   dv.raw_format_id,
                   dv.source_classification,
                   d.project_id,
                   p.erp_type
            FROM dataset_versions dv
            LEFT JOIN datasets d
              ON d.dataset_id = dv.dataset_id
             AND d.tenant_id = dv.tenant_id
            LEFT JOIN projects p
              ON p.project_id = d.project_id
             AND p.tenant_id = d.tenant_id
            WHERE dv.dataset_version_id = ?
            LIMIT 1
            """,
            (dataset_version_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def _dataset_semantic_field_names(
    conn: sqlite3.Connection, dataset_version_id: str
) -> list[str]:
    template_row = conn.execute(
        """
        SELECT template_id
        FROM dataset_templates
        WHERE dataset_version_id = ?
        ORDER BY updated_at DESC
        LIMIT 1
        """,
        (dataset_version_id,),
    ).fetchone()
    if template_row and template_row["template_id"] is not None:
        rows = conn.execute(
            """
            SELECT name
            FROM template_fields
            WHERE template_id = ?
            ORDER BY field_id
            """,
            (int(template_row["template_id"]),),
        ).fetchall()
        names = [
            str(row["name"]).strip()
            for row in rows
            if isinstance(row["name"], str) and str(row["name"]).strip()
        ]
        if names:
            return names

    rows = conn.execute(
        """
        SELECT original_name
        FROM dataset_columns
        WHERE dataset_version_id = ?
        ORDER BY column_id
        """,
        (dataset_version_id,),
    ).fetchall()
    return [
        str(row["original_name"]).strip()
        for row in rows
        if isinstance(row["original_name"], str) and str(row["original_name"]).strip()
    ]


def _schema_signature(field_names: list[str]) -> str:
    normalized = [normalize_field_name(name) for name in field_names if normalize_field_name(name)]
    payload = json.dumps(normalized, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _dataset_identity(
    conn: sqlite3.Connection, dataset_row: dict[str, Any]
) -> dict[str, Any]:
    dataset_version_id = str(dataset_row.get("dataset_version_id") or "")
    names = _dataset_semantic_field_names(conn, dataset_version_id)
    inferred_erp = infer_erp_type_from_field_names(names)
    configured_erp = str(dataset_row.get("erp_type") or "").strip().lower()
    effective_erp = configured_erp if configured_erp and configured_erp != "unknown" else inferred_erp
    return {
        "dataset_version_id": dataset_version_id,
        "field_names": names,
        "field_count": len(names),
        "schema_signature": _schema_signature(names) if names else "",
        "configured_erp_type": configured_erp or "unknown",
        "inferred_erp_type": inferred_erp,
        "erp_type": effective_erp or "unknown",
    }


def _baseline_comparison(
    db_path: Path, dataset_row: dict[str, Any]
) -> dict[str, Any]:
    conn = _connect(db_path)
    try:
        target = _dataset_identity(conn, dataset_row)
        target_id = str(target.get("dataset_version_id") or "")
        target_created = _parse_ts(dataset_row.get("created_at"))
        target_erp = str(target.get("erp_type") or "unknown")
        target_sig = str(target.get("schema_signature") or "")

        candidates_rows = conn.execute(
            """
            SELECT dv.dataset_version_id,
                   dv.dataset_id,
                   dv.created_at,
                   dv.row_count,
                   dv.column_count,
                   dv.source_classification,
                   d.project_id,
                   p.erp_type
            FROM dataset_versions dv
            LEFT JOIN datasets d
              ON d.dataset_id = dv.dataset_id
             AND d.tenant_id = dv.tenant_id
            LEFT JOIN projects p
              ON p.project_id = d.project_id
             AND p.tenant_id = d.tenant_id
            WHERE dv.dataset_version_id != ?
            """,
            (target_id,),
        ).fetchall()

        ranked: list[tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]]] = []
        for row in candidates_rows:
            cand_row = dict(row)
            if str(cand_row.get("source_classification") or "").strip().lower() != "real":
                continue
            cand = _dataset_identity(conn, cand_row)
            cand_erp = str(cand.get("erp_type") or "unknown")
            cand_sig = str(cand.get("schema_signature") or "")
            same_erp = target_erp != "unknown" and cand_erp == target_erp
            same_sig = bool(target_sig) and cand_sig == target_sig
            if not same_erp and not same_sig:
                continue
            cand_created = _parse_ts(cand_row.get("created_at"))
            prior = bool(
                target_created
                and cand_created
                and cand_created <= target_created
            )
            time_distance = None
            if target_created and cand_created:
                time_distance = abs((target_created - cand_created).total_seconds())
            ranked.append(
                (
                    (
                        0 if same_erp else 1,
                        0 if same_sig else 1,
                        0 if prior else 1,
                        float(time_distance) if isinstance(time_distance, (int, float)) else float("inf"),
                        str(cand_row.get("dataset_version_id") or ""),
                    ),
                    cand_row,
                    cand,
                )
            )

        if not ranked:
            return {
                "available": False,
                "dataset_version_id": target_id,
                "erp_type": target_erp,
                "inferred_erp_type": target.get("inferred_erp_type"),
                "configured_erp_type": target.get("configured_erp_type"),
                "reason": "no_real_baseline_for_same_erp_or_schema",
            }

        ranked.sort(key=lambda item: item[0])
        baseline_row, baseline_identity = ranked[0][1], ranked[0][2]
        baseline_id = str(baseline_row.get("dataset_version_id") or "")
        row_count_target = int(dataset_row.get("row_count") or 0)
        row_count_baseline = int(baseline_row.get("row_count") or 0)
        col_count_target = int(dataset_row.get("column_count") or 0)
        col_count_baseline = int(baseline_row.get("column_count") or 0)
        delta_rows = row_count_target - row_count_baseline
        delta_cols = col_count_target - col_count_baseline
        row_ratio = (
            (float(row_count_target) / float(row_count_baseline))
            if row_count_baseline > 0
            else None
        )

        return {
            "available": True,
            "dataset_version_id": target_id,
            "baseline_dataset_version_id": baseline_id,
            "erp_type": target_erp,
            "inferred_erp_type": target.get("inferred_erp_type"),
            "configured_erp_type": target.get("configured_erp_type"),
            "baseline_inferred_erp_type": baseline_identity.get("inferred_erp_type"),
            "baseline_configured_erp_type": baseline_identity.get("configured_erp_type"),
            "baseline_match_basis": {
                "same_erp": bool(
                    target_erp != "unknown"
                    and str(baseline_identity.get("erp_type") or "unknown") == target_erp
                ),
                "same_schema_signature": bool(
                    target_sig
                    and str(baseline_identity.get("schema_signature") or "") == target_sig
                ),
            },
            "target_source_classification": str(
                dataset_row.get("source_classification") or "unknown"
            ),
            "baseline_source_classification": str(
                baseline_row.get("source_classification") or "unknown"
            ),
            "target_row_count": row_count_target,
            "baseline_row_count": row_count_baseline,
            "row_count_delta": delta_rows,
            "row_count_ratio": row_ratio,
            "target_column_count": col_count_target,
            "baseline_column_count": col_count_baseline,
            "column_count_delta": delta_cols,
        }
    finally:
        conn.close()


def _discover_plugin_ids(types: set[str]) -> list[str]:
    plugin_ids: list[str] = []
    for manifest in sorted((REPO_ROOT / "plugins").glob("*/plugin.yaml")):
        data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        pid = data.get("id")
        ptype = data.get("type")
        if isinstance(pid, str) and isinstance(ptype, str) and ptype in types:
            plugin_ids.append(pid)
    return sorted(set(plugin_ids))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _get_plugin_block(report: dict[str, Any], plugin_id: str) -> dict[str, Any]:
    plugins = report.get("plugins")
    if not isinstance(plugins, dict):
        return {}
    block = plugins.get(plugin_id)
    return block if isinstance(block, dict) else {}


def _top_findings(report: dict[str, Any], plugin_id: str, n: int = 8) -> list[dict[str, Any]]:
    block = _get_plugin_block(report, plugin_id)
    items = block.get("findings")
    if not isinstance(items, list):
        return []
    out: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            out.append(item)
        if len(out) >= n:
            break
    return out


def _extract_recommendations(report: dict[str, Any]) -> dict[str, Any]:
    recs = report.get("recommendations")
    if isinstance(recs, dict):
        return recs
    return {"summary": "No recommendations block found.", "items": []}


def _parse_exclude_processes(raw: str | None) -> list[str]:
    if not isinstance(raw, str) or not raw.strip():
        return []
    out: list[str] = []
    seen: set[str] = set()
    for token in re.split(r"[,\s;]+", raw.strip()):
        value = token.strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _extract_known_issue_checks(recs: dict[str, Any]) -> dict[str, Any]:
    known = recs.get("known")
    if not isinstance(known, dict):
        return {"status": "none", "summary": "No known-issue checks found.", "items": [], "totals": {}}
    items_raw = known.get("items")
    items = [i for i in items_raw if isinstance(i, dict)] if isinstance(items_raw, list) else []
    status_counts: dict[str, int] = {}
    normalized: list[dict[str, Any]] = []
    for item in items:
        status = str(item.get("status") or "unknown")
        status_counts[status] = int(status_counts.get(status, 0)) + 1
        expected = item.get("expected") if isinstance(item.get("expected"), dict) else {}
        normalized.append(
            {
                "status": status,
                "title": str(item.get("title") or "").strip(),
                "plugin_id": str(item.get("plugin_id") or "").strip(),
                "kind": str(item.get("kind") or "").strip(),
                "observed_count": item.get("observed_count"),
                "evidence_source": str(item.get("evidence_source") or "").strip(),
                "min_count": expected.get("min_count"),
                "max_count": expected.get("max_count"),
                "modeled_percent": item.get("modeled_percent"),
                "modeled_general_percent": item.get("modeled_general_percent"),
                "modeled_close_percent": item.get("modeled_close_percent"),
                "recommendation": str(item.get("recommendation") or "").strip(),
            }
        )
    failing = int(sum(v for k, v in status_counts.items() if k != "confirmed"))
    return {
        "status": str(known.get("status") or ""),
        "summary": str(known.get("summary") or ""),
        "items": normalized,
        "totals": {
            "total": int(len(normalized)),
            "confirmed": int(status_counts.get("confirmed", 0)),
            "failing": failing,
            "by_status": status_counts,
        },
    }


def _extract_ideaspace_route_map(run_dir: Path, max_steps: int = 12) -> dict[str, Any]:
    energy_path = run_dir / "artifacts" / "analysis_ideaspace_energy_ebm_v1" / "energy_state_vector.json"
    verified_path = run_dir / "artifacts" / "analysis_ebm_action_verifier_v1" / "verified_actions.json"
    out: dict[str, Any] = {"available": False, "summary": "No Kona route map artifacts found.", "actions": []}

    energy: dict[str, Any] = {}
    verified: dict[str, Any] = {}
    if energy_path.exists():
        try:
            payload = _load_json(energy_path)
            if isinstance(payload, dict):
                energy = payload
        except Exception:
            energy = {}
    if verified_path.exists():
        try:
            payload = _load_json(verified_path)
            if isinstance(payload, dict):
                verified = payload
        except Exception:
            verified = {}

    entities = energy.get("entities") if isinstance(energy.get("entities"), list) else []
    entity_all = None
    for item in entities:
        if isinstance(item, dict) and str(item.get("entity_key") or "") == "ALL":
            entity_all = item
            break
    if entity_all is None and entities:
        first = entities[0]
        if isinstance(first, dict):
            entity_all = first

    route_actions: list[dict[str, Any]] = []
    raw_actions = verified.get("verified_actions") if isinstance(verified.get("verified_actions"), list) else []
    for rec in raw_actions:
        if not isinstance(rec, dict):
            continue
        delta = rec.get("delta_energy")
        before = rec.get("energy_before")
        pct = None
        if isinstance(delta, (int, float)) and isinstance(before, (int, float)) and float(before) > 0.0:
            pct = max(0.0, min(100.0, (float(delta) / float(before)) * 100.0))
        route_actions.append(
            {
                "lever_id": str(rec.get("lever_id") or "").strip(),
                "title": str(rec.get("title") or "").strip(),
                "action": str(rec.get("action") or "").strip(),
                "target": str(rec.get("target") or "").strip(),
                "delta_energy": float(delta) if isinstance(delta, (int, float)) else None,
                "energy_before": float(before) if isinstance(before, (int, float)) else None,
                "energy_after": float(rec.get("energy_after")) if isinstance(rec.get("energy_after"), (int, float)) else None,
                "modeled_percent": float(pct) if isinstance(pct, (int, float)) else None,
                "confidence": float(rec.get("confidence")) if isinstance(rec.get("confidence"), (int, float)) else None,
            }
        )
    route_actions.sort(
        key=lambda r: (
            -float(r.get("delta_energy") or 0.0),
            -float(r.get("modeled_percent") or 0.0),
            str(r.get("lever_id") or ""),
        )
    )
    route_actions = route_actions[: max(1, int(max_steps))]

    current: dict[str, Any] = {}
    if isinstance(entity_all, dict):
        current = {
            "entity_key": str(entity_all.get("entity_key") or ""),
            "energy_total": entity_all.get("energy_total"),
            "energy_gap": entity_all.get("energy_gap"),
            "energy_constraints": entity_all.get("energy_constraints"),
            "observed": entity_all.get("observed") if isinstance(entity_all.get("observed"), dict) else {},
            "ideal": entity_all.get("ideal") if isinstance(entity_all.get("ideal"), dict) else {},
        }

    available = bool(current or route_actions)
    out = {
        "available": available,
        "summary": "Current-to-ideal route extracted from Kona EBM artifacts."
        if available
        else "No Kona route map artifacts found.",
        "ideal_mode": str(energy.get("ideal_mode") or "").strip(),
        "current": current,
        "actions": route_actions,
    }
    return out


def _render_recommendations_md(
    recs: dict[str, Any],
    known_checks: dict[str, Any],
    route_map: dict[str, Any] | None = None,
    erp_baseline: dict[str, Any] | None = None,
    max_items: int = 40,
) -> str:
    summary = str(recs.get("summary") or "").strip()
    lines = []
    lines.append("# Recommendations (From report.json)")
    if summary:
        lines.append("")
        lines.append(summary)

    if isinstance(erp_baseline, dict):
        lines.append("")
        lines.append("## ERP Baseline Comparison")
        if erp_baseline.get("available"):
            erp_type = str(erp_baseline.get("erp_type") or "unknown")
            baseline_id = str(erp_baseline.get("baseline_dataset_version_id") or "")
            lines.append(f"- ERP type: {erp_type}")
            lines.append(f"- Baseline dataset_version_id: `{baseline_id}`")
            basis = erp_baseline.get("baseline_match_basis")
            if isinstance(basis, dict):
                lines.append(
                    "- Match basis: "
                    + f"same_erp={bool(basis.get('same_erp'))}, "
                    + f"same_schema_signature={bool(basis.get('same_schema_signature'))}"
                )
            lines.append(
                "- Rows: "
                + f"target={int(erp_baseline.get('target_row_count') or 0):,}, "
                + f"baseline={int(erp_baseline.get('baseline_row_count') or 0):,}, "
                + f"delta={int(erp_baseline.get('row_count_delta') or 0):+,}"
            )
            row_ratio = erp_baseline.get("row_count_ratio")
            if isinstance(row_ratio, (int, float)):
                lines.append(f"- Row ratio (target/baseline): {float(row_ratio):.3f}x")
            lines.append(
                "- Columns: "
                + f"target={int(erp_baseline.get('target_column_count') or 0)}, "
                + f"baseline={int(erp_baseline.get('baseline_column_count') or 0)}, "
                + f"delta={int(erp_baseline.get('column_count_delta') or 0):+}"
            )
        else:
            lines.append(
                "- No same-ERP real baseline dataset found for comparison in local state."
            )

    def _as_items(block: Any) -> list[dict[str, Any]]:
        if isinstance(block, dict) and isinstance(block.get("items"), list):
            return [i for i in block["items"] if isinstance(i, dict)]
        if isinstance(block, list):
            return [i for i in block if isinstance(i, dict)]
        return []

    known = _as_items(recs.get("known"))
    discovery = _as_items(recs.get("discovery"))
    flat = _as_items(recs.get("items"))
    explanations = _as_items(recs.get("explanations"))
    explanations = _as_items(recs.get("explanations"))
    explanations = _as_items(recs.get("explanations"))

    sections: list[tuple[str, list[dict[str, Any]]]] = []
    if known or discovery:
        discovery_close = [
            item for item in discovery if str(item.get("scope_class") or "").strip().lower() == "close_specific"
        ]
        discovery_general = [
            item for item in discovery if str(item.get("scope_class") or "").strip().lower() != "close_specific"
        ]
        sections.append(("Discovery (Close-Specific)", discovery_close))
        sections.append(("Discovery (General)", discovery_general))
        sections.append(("Known", known))
    else:
        sections.append(("Recommendations", flat))

    for title, items in sections:
        if not items:
            continue
        lines.append("")
        lines.append(f"## {title}")
        for item in items[:max_items]:
            txt = str(item.get("recommendation") or item.get("text") or "").strip()
            if not txt:
                continue
            lines.append(f"- {txt}")
            plugin_id = item.get("plugin_id")
            kind = item.get("kind")
            where = item.get("where")
            if isinstance(where, dict) and where:
                proc = (
                    where.get("process_norm")
                    or where.get("process")
                    or where.get("process_id")
                    or where.get("transition")
                )
                if isinstance(proc, str) and proc.strip():
                    lines.append(f"  Applies to: {proc.strip()}")
            impact = item.get("impact_hours")
            if isinstance(impact, (int, float)) and float(impact) > 0:
                lines.append(f"  Potential size (upper bound): ~{float(impact):.2f} hours")
            obviousness_rank = str(item.get("obviousness_rank") or "").strip()
            obviousness_score = item.get("obviousness_score")
            if obviousness_rank:
                if isinstance(obviousness_score, (int, float)):
                    lines.append(
                        f"  Obviousness: {obviousness_rank} ({float(obviousness_score):.2f}; lower is better)"
                    )
                else:
                    lines.append(f"  Obviousness: {obviousness_rank}")
            modeled_pct = item.get("modeled_percent")
            modeled_delta = item.get("modeled_delta_hours")
            modeled_basis = item.get("modeled_basis_hours")
            if isinstance(modeled_pct, (int, float)):
                modeled_text = f"{float(modeled_pct):.2f}%"
                if isinstance(modeled_delta, (int, float)) and isinstance(modeled_basis, (int, float)) and float(modeled_basis) > 0.0:
                    modeled_text += (
                        f" ({float(modeled_delta):.2f}h / {float(modeled_basis):.2f}h baseline)"
                    )
                lines.append(f"  Modeled improvement: {modeled_text}")
            else:
                reason = str(item.get("not_modeled_reason") or "").strip()
                if reason:
                    lines.append(f"  Modeled improvement: not available ({reason})")
            scope_class = str(item.get("scope_class") or "").strip()
            if scope_class:
                lines.append(f"  Scope: {scope_class}")
            vsteps = item.get("validation_steps")
            if isinstance(vsteps, list):
                steps = [s.strip() for s in vsteps if isinstance(s, str) and s.strip()]
                if steps:
                    lines.append("  Validation:")
                    for s in steps[:3]:
                        lines.append(f"  - {s}")
            if isinstance(plugin_id, str) and plugin_id:
                src = f"{plugin_id}" + (f":{kind}" if isinstance(kind, str) and kind else "")
                lines.append(f"  Source: {src}")

    if explanations:
        lines.append("")
        lines.append("## Non-Actionable Explanations")
        for item in explanations[: max_items * 2]:
            if not isinstance(item, dict):
                continue
            plugin_id = str(item.get("plugin_id") or "unknown").strip() or "unknown"
            reason = str(item.get("reason_code") or "unspecified").strip() or "unspecified"
            explanation = str(item.get("plain_english_explanation") or "").strip()
            next_step = str(item.get("recommended_next_step") or "").strip()
            lines.append(f"- `{plugin_id}` ({reason})")
            if explanation:
                lines.append(f"  - explanation: {explanation}")
            if next_step:
                lines.append(f"  - next_step: {next_step}")
            downstream = item.get("downstream_plugins")
            if isinstance(downstream, list):
                rendered = [str(v).strip() for v in downstream if str(v).strip()]
                lines.append(
                    "  - downstream_plugins: "
                    + (", ".join(rendered) if rendered else "(none)")
                )

    if isinstance(route_map, dict) and bool(route_map.get("available")):
        lines.append("")
        lines.append("## Ideaspace Route Map (Current -> Ideal)")
        ideal_mode = str(route_map.get("ideal_mode") or "").strip()
        if ideal_mode:
            lines.append(f"- ideal mode: {ideal_mode}")
        current = route_map.get("current") if isinstance(route_map.get("current"), dict) else {}
        if current:
            lines.append(
                "- current energy: total="
                + f"{float(current.get('energy_total') or 0.0):.4f}, "
                + f"gap={float(current.get('energy_gap') or 0.0):.4f}, "
                + f"constraints={float(current.get('energy_constraints') or 0.0):.4f}"
            )
        actions = route_map.get("actions") if isinstance(route_map.get("actions"), list) else []
        if actions:
            lines.append("")
            lines.append("| Step | Lever | Target | Modeled reduction | Confidence |")
            lines.append("|---|---|---|---:|---:|")
            for idx, action in enumerate(actions[:12], start=1):
                if not isinstance(action, dict):
                    continue
                lever = str(action.get("lever_id") or "").strip() or "unknown"
                target = str(action.get("target") or "").strip() or "ALL"
                pct = action.get("modeled_percent")
                delta = action.get("delta_energy")
                reduction = "n/a"
                if isinstance(pct, (int, float)) and isinstance(delta, (int, float)):
                    reduction = f"{float(pct):.2f}% ({float(delta):.4f} energy)"
                elif isinstance(pct, (int, float)):
                    reduction = f"{float(pct):.2f}%"
                elif isinstance(delta, (int, float)):
                    reduction = f"{float(delta):.4f} energy"
                conf = action.get("confidence")
                conf_txt = f"{float(conf):.2f}" if isinstance(conf, (int, float)) else "n/a"
                lines.append(f"| {idx} | `{lever}` | `{target}` | {reduction} | {conf_txt} |")
        else:
            lines.append("- no modeled route actions available in this run.")

    lines.append("")
    lines.append("## Known-Issue Detection")
    known_summary = str(known_checks.get("summary") or "").strip()
    totals = known_checks.get("totals") if isinstance(known_checks.get("totals"), dict) else {}
    total = int(totals.get("total") or 0)
    confirmed = int(totals.get("confirmed") or 0)
    failing = int(totals.get("failing") or 0)
    lines.append(f"- total checks: {total}")
    lines.append(f"- confirmed: {confirmed}")
    lines.append(f"- failing: {failing}")
    if known_summary:
        lines.append(f"- summary: {known_summary}")
    items = known_checks.get("items") if isinstance(known_checks.get("items"), list) else []
    if items:
        lines.append("")
        lines.append("| Status | Issue | Plugin | Kind | Observed | Modeled | Expected |")
        lines.append("|---|---|---|---|---:|---|---|")
        for item in items[:max_items]:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status") or "")
            title = str(item.get("title") or "")
            plugin_id = str(item.get("plugin_id") or "")
            kind = str(item.get("kind") or "")
            observed = item.get("observed_count")
            min_count = item.get("min_count")
            max_count = item.get("max_count")
            modeled = "n/a"
            close_pct = item.get("modeled_close_percent")
            general_pct = item.get("modeled_general_percent")
            if isinstance(close_pct, (int, float)) or isinstance(general_pct, (int, float)):
                parts: list[str] = []
                if isinstance(close_pct, (int, float)):
                    parts.append(f"close={float(close_pct):.2f}%")
                if isinstance(general_pct, (int, float)):
                    parts.append(f"general={float(general_pct):.2f}%")
                modeled = ", ".join(parts)
            elif isinstance(item.get("modeled_percent"), (int, float)):
                modeled = f"{float(item.get('modeled_percent')):.2f}%"
            expected = f"min={min_count}, max={max_count}"
            lines.append(
                f"| {status} | {title} | `{plugin_id}` | `{kind}` | {observed if observed is not None else 'n/a'} | {modeled} | {expected} |"
            )
            recommendation = str(item.get("recommendation") or "").strip()
            if recommendation:
                lines.append(f"  - Follow-up: {recommendation}")
    return "\n".join(lines).rstrip() + "\n"


def _plain_process(item: dict[str, Any]) -> str:
    where = item.get("where") if isinstance(item.get("where"), dict) else {}
    for key in ("process_norm", "process", "process_id", "activity"):
        value = where.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "this process"


def _plain_hours(item: dict[str, Any]) -> str:
    touches = item.get("modeled_user_touches_reduced")
    touches_n = float(touches) if isinstance(touches, (int, float)) and float(touches) > 0.0 else 0.0
    value = item.get("modeled_delta_hours")
    if not isinstance(value, (int, float)) or float(value) <= 0:
        value = item.get("impact_hours")
    if not isinstance(value, (int, float)) or float(value) <= 0:
        value = item.get("modeled_delta")
    if isinstance(value, (int, float)) and float(value) > 0:
        if touches_n > 0.0:
            return (
                f"about {float(value):.2f} queue-delay hours, plus ~{touches_n:.0f} manual runs eliminated"
            )
        return f"about {float(value):.2f} hours"
    if touches_n > 0.0:
        return f"~{touches_n:.0f} manual runs eliminated (human effort relief)"
    return "an unknown amount"


def _plain_validation(item: dict[str, Any]) -> str:
    vsteps = item.get("validation_steps")
    if isinstance(vsteps, list):
        for step in vsteps:
            if isinstance(step, str) and step.strip():
                return step.strip()
    return "Re-run the harness and compare before vs after metrics."


def _plain_recommendation_fields(item: dict[str, Any]) -> tuple[str, str]:
    action_type = str(item.get("action_type") or item.get("action") or "").strip()
    process = _plain_process(item)
    evidence = {}
    ev_raw = item.get("evidence")
    if isinstance(ev_raw, list) and ev_raw and isinstance(ev_raw[0], dict):
        evidence = ev_raw[0]
    key = str(evidence.get("key") or item.get("key") or "").strip()
    runs = evidence.get("runs_with_key") or item.get("runs_with_key")
    unique_values = evidence.get("unique_values") or item.get("unique_values")
    coverage = evidence.get("coverage") if isinstance(evidence.get("coverage"), (int, float)) else item.get("coverage")
    unique_ratio = evidence.get("unique_ratio") if isinstance(evidence.get("unique_ratio"), (int, float)) else item.get("unique_ratio")

    if action_type in {"batch_input", "batch_input_refactor"}:
        key_name = key or "the input key"
        change = f"Convert `{process}` to batch mode so one run can handle a list of `{key_name}` values."
        why_parts: list[str] = []
        if isinstance(runs, (int, float)):
            why_parts.append(f"it ran {int(runs):,} times")
        if isinstance(unique_values, (int, float)):
            why_parts.append(f"across {int(unique_values):,} unique values")
        if isinstance(coverage, (int, float)):
            why_parts.append(f"with {float(coverage) * 100.0:.1f}% coverage")
        if isinstance(unique_ratio, (int, float)):
            why_parts.append(f"and {float(unique_ratio) * 100.0:.1f}% uniqueness")
        why = (
            "This is a one-value-per-run sweep pattern"
            + (": " + ", ".join(why_parts) if why_parts else ".")
        )
        return change, why

    if action_type == "batch_group_candidate":
        targets = item.get("target_process_ids")
        if not isinstance(targets, list):
            targets = evidence.get("target_process_ids")
        target_tokens = (
            [str(t).strip() for t in targets if isinstance(t, str) and t.strip()]
            if isinstance(targets, list)
            else [process]
        )
        rpt_targets = [t for t in target_tokens if t.strip().lower().startswith("rpt_")]
        po_targets = [t for t in target_tokens if t.strip().lower().startswith("po")]
        other_targets = [
            t
            for t in target_tokens
            if t not in rpt_targets and t not in po_targets
        ]
        if rpt_targets and (po_targets or other_targets):
            grouped = po_targets + other_targets
            grouped_text = ", ".join(grouped)
            rpt_text = ", ".join(rpt_targets)
            change = (
                f"Convert these process_ids to batch input first: {grouped_text}. "
                f"Keep `{rpt_text}` as a separate orchestration/report anchor recommendation."
            )
        else:
            target_list = ", ".join(target_tokens)
            change = f"Convert these process_ids to batch input first: {target_list}."
        why = (
            "These steps look like one-by-one payout processing in the same close-month chain, "
            "so batching should cut repeated job launches."
        )
        return change, why

    if action_type in {"batch_or_cache", "dedupe_or_cache", "throttle_or_dedupe"}:
        return (
            f"Reduce repeat work in `{process}` using batching, caching, or dedupe.",
            "The same work pattern appears many times and adds avoidable queue delay.",
        )

    if action_type in {"route_process", "reschedule", "schedule_shift_target"}:
        return (
            f"Move `{process}` to a better host or run window.",
            "Current placement/time has worse wait or run times than alternatives.",
        )

    if action_type in {"unblock_dependency_chain", "reduce_transition_gap"}:
        return (
            f"Fix the handoff dependency around `{process}` so downstream work starts sooner.",
            "The process chain is showing wait buildup at a specific handoff.",
        )

    if action_type == "reduce_spillover_past_eom":
        return (
            "Target month-end spillover processes with structural changes first.",
            "The close window is leaking work into days after month-end.",
        )

    text = str(item.get("recommendation") or "").strip()
    if text:
        return text, "This recommendation comes from measured plugin evidence."
    return f"Review `{process}` for a targeted fix.", "Measured evidence suggests this process is a driver."


def _plain_linkage_key(item: dict[str, Any]) -> str:
    action_type = str(item.get("action_type") or item.get("action") or "").strip().lower()
    if action_type not in {"batch_input", "batch_input_refactor"}:
        return ""
    ev_raw = item.get("evidence")
    evidence = ev_raw[0] if isinstance(ev_raw, list) and ev_raw and isinstance(ev_raw[0], dict) else {}
    key = str(evidence.get("key") or item.get("key") or "").strip().lower()
    user = str(
        item.get("affected_user_primary")
        or item.get("top_user_redacted")
        or evidence.get("top_user_redacted")
        or ""
    ).strip().lower()
    month = str(
        item.get("close_month")
        or item.get("month")
        or evidence.get("close_month")
        or evidence.get("month")
        or ""
    ).strip().lower()
    if not any([key, user, month]):
        return ""
    return f"{key}|{user}|{month}"


def _plain_linked_display_group(current_process: str, group: list[str]) -> list[str]:
    current = str(current_process or "").strip().lower()
    tokens = [str(v).strip().lower() for v in group if str(v).strip()]
    if current.startswith("po"):
        subset = [v for v in tokens if v.startswith("po")]
        return subset if subset else tokens
    if current.startswith("rpt_"):
        subset = [v for v in tokens if v.startswith("rpt_")]
        return subset if subset else tokens
    return tokens


def _render_recommendations_plain_md(
    recs: dict[str, Any],
    known_checks: dict[str, Any],
    route_map: dict[str, Any] | None = None,
    erp_baseline: dict[str, Any] | None = None,
    max_items: int = 30,
) -> str:
    lines: list[str] = []
    lines.append("# Recommendations (Plain Language)")
    lines.append("")
    lines.append("This version is written for easier reading (high school to early college level).")

    if isinstance(erp_baseline, dict):
        lines.append("")
        lines.append("## ERP Baseline Comparison (Plain)")
        if erp_baseline.get("available"):
            lines.append(
                f"- This dataset is treated as ERP type `{str(erp_baseline.get('erp_type') or 'unknown')}`."
            )
            lines.append(
                "- It is compared against baseline dataset "
                + f"`{str(erp_baseline.get('baseline_dataset_version_id') or '')}` "
                + "from the same ERP family."
            )
            lines.append(
                "- Row count changed by "
                + f"{int(erp_baseline.get('row_count_delta') or 0):+,} rows "
                + f"(target={int(erp_baseline.get('target_row_count') or 0):,}, "
                + f"baseline={int(erp_baseline.get('baseline_row_count') or 0):,})."
            )
            lines.append(
                "- Column count changed by "
                + f"{int(erp_baseline.get('column_count_delta') or 0):+} columns."
            )
        else:
            lines.append(
                "- No same-ERP real baseline dataset was found in local state."
            )

    def _as_items(block: Any) -> list[dict[str, Any]]:
        if isinstance(block, dict) and isinstance(block.get("items"), list):
            return [i for i in block["items"] if isinstance(i, dict)]
        if isinstance(block, list):
            return [i for i in block if isinstance(i, dict)]
        return []

    known = _as_items(recs.get("known"))
    discovery = _as_items(recs.get("discovery"))
    flat = _as_items(recs.get("items"))
    explanations = _as_items(recs.get("explanations"))
    sections: list[tuple[str, list[dict[str, Any]]]] = []
    if known or discovery:
        discovery_close = [
            item for item in discovery if str(item.get("scope_class") or "").strip().lower() == "close_specific"
        ]
        discovery_general = [
            item for item in discovery if str(item.get("scope_class") or "").strip().lower() != "close_specific"
        ]
        sections.append(("Discovery (Close-Specific)", discovery_close))
        sections.append(("Discovery (General)", discovery_general))
        sections.append(("Known Issues", known))
    else:
        sections.append(("Recommendations", flat))

    for title, items in sections:
        if not items:
            continue
        linked_groups: dict[str, list[str]] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            lk = _plain_linkage_key(item)
            if not lk:
                continue
            proc = _plain_process(item).strip().lower()
            if not proc:
                continue
            linked_groups.setdefault(lk, [])
            if proc not in linked_groups[lk]:
                linked_groups[lk].append(proc)
        lines.append("")
        lines.append(f"## {title}")
        for idx, item in enumerate(items[:max_items], start=1):
            change, why = _plain_recommendation_fields(item)
            lines.append(f"{idx}. What to change: {change}")
            lines.append(f"   Why this matters: {why}")
            lk = _plain_linkage_key(item)
            linked_group = _plain_linked_display_group(_plain_process(item), linked_groups.get(lk, []))
            if lk and len(linked_group) > 1:
                linked = ", ".join(linked_group)
                lines.append(
                    f"   Linked recommendation set: {linked} (same parameter/user sweep family)."
                )
            lines.append(f"   Expected benefit: {_plain_hours(item)}")
            touches_reduced = item.get("modeled_user_touches_reduced")
            if isinstance(touches_reduced, (int, float)) and float(touches_reduced) > 0.0:
                user_hint = str(item.get("affected_user_primary") or "").strip()
                user_suffix = f" for user `{user_hint}`" if user_hint else ""
                lines.append(
                    f"   User effort reduction: ~{float(touches_reduced):.0f} fewer manual runs per close cycle{user_suffix}."
                )
                if float(touches_reduced) >= 100.0:
                    lines.append(
                        "   Human effort note: queue-delay hours understate this because it removes repetitive manual execution workload."
                    )
            close_cycle_saved = item.get("modeled_close_hours_saved_cycle")
            if isinstance(close_cycle_saved, (int, float)) and float(close_cycle_saved) > 0.0:
                lines.append(
                    f"   Close-window time saved: ~{float(close_cycle_saved):.2f} hours per close cycle."
                )
            month_saved = item.get("modeled_close_hours_saved_month")
            if isinstance(month_saved, (int, float)) and float(month_saved) > 0.0:
                lines.append(
                    f"   Monthly time saved: ~{float(month_saved):.2f} hours."
                )
            annual_saved = item.get("modeled_close_hours_saved_annualized")
            if isinstance(annual_saved, (int, float)) and float(annual_saved) > 0.0:
                lines.append(
                    f"   Annualized time saved: ~{float(annual_saved):.2f} hours."
                )
            contention_pct = item.get("modeled_contention_reduction_pct_close")
            if isinstance(contention_pct, (int, float)) and float(contention_pct) > 0.0:
                lines.append(
                    f"   Contention reduction: ~{float(contention_pct):.2f}% less close-window queue pressure."
                )
            modeled_pct = item.get("modeled_percent")
            if isinstance(modeled_pct, (int, float)):
                lines.append(f"   Modeled improvement percent: {float(modeled_pct):.2f}%")
            else:
                reason = str(item.get("not_modeled_reason") or "").strip()
                if reason:
                    lines.append(f"   Modeled improvement percent: not available ({reason})")
            value_score = item.get("value_score_v2")
            if not isinstance(value_score, (int, float)):
                value_score = item.get("client_value_score")
            if isinstance(value_score, (int, float)):
                lines.append(f"   Priority score: {float(value_score):.2f} (higher = better client value)")
            efficiency_pct = item.get("modeled_efficiency_gain_pct")
            if isinstance(efficiency_pct, (int, float)):
                lines.append(f"   Efficiency gain (%): {float(efficiency_pct):.2f}%")
            manual_pct = item.get("modeled_manual_run_reduction_pct")
            if isinstance(manual_pct, (int, float)):
                lines.append(f"   Manual-work reduction (%): {float(manual_pct):.2f}%")
            opp_class = str(item.get("opportunity_class") or "").strip()
            if opp_class:
                lines.append(f"   Recommendation class: {opp_class}")
            rank = str(item.get("obviousness_rank") or "").strip()
            score = item.get("obviousness_score")
            if rank:
                if isinstance(score, (int, float)):
                    lines.append(
                        f"   Obviousness rank: {rank} ({float(score):.2f}; lower means less obvious)"
                    )
                else:
                    lines.append(f"   Obviousness rank: {rank}")
            lines.append(f"   How to check: {_plain_validation(item)}")
            plugin_id = str(item.get("plugin_id") or "").strip()
            kind = str(item.get("kind") or "").strip()
            if plugin_id:
                src = plugin_id + (f":{kind}" if kind else "")
                lines.append(f"   Source: {src}")

    if explanations:
        lines.append("")
        lines.append("## Non-Actionable Explanations (Plain)")
        for idx, item in enumerate(explanations[: max_items * 2], start=1):
            if not isinstance(item, dict):
                continue
            plugin_id = str(item.get("plugin_id") or "unknown").strip() or "unknown"
            reason = str(item.get("reason_code") or "unspecified").strip() or "unspecified"
            explanation = str(item.get("plain_english_explanation") or "").strip()
            next_step = str(item.get("recommended_next_step") or "").strip()
            lines.append(f"{idx}. Plugin: `{plugin_id}`")
            lines.append(f"   Why not actionable: {reason}")
            if explanation:
                lines.append(f"   Explanation: {explanation}")
            if next_step:
                lines.append(f"   Next step: {next_step}")
            downstream = item.get("downstream_plugins")
            if isinstance(downstream, list):
                rendered = [str(v).strip() for v in downstream if str(v).strip()]
                lines.append(
                    "   Downstream plugins: "
                    + (", ".join(rendered) if rendered else "(none)")
                )

    if isinstance(route_map, dict) and bool(route_map.get("available")):
        lines.append("")
        lines.append("## Ideaspace Route Map (Simple View)")
        ideal_mode = str(route_map.get("ideal_mode") or "").strip()
        if ideal_mode:
            lines.append(f"- Ideal mode used: {ideal_mode}")
        current = route_map.get("current") if isinstance(route_map.get("current"), dict) else {}
        if current:
            lines.append(
                "- Current energy score: "
                + f"{float(current.get('energy_total') or 0.0):.2f} "
                + "(lower is better)."
            )
        actions = route_map.get("actions") if isinstance(route_map.get("actions"), list) else []
        if actions:
            lines.append("- Best route steps from current state toward ideal state:")
            for idx, action in enumerate(actions[:8], start=1):
                if not isinstance(action, dict):
                    continue
                title = str(action.get("title") or action.get("action") or "Action").strip()
                target = str(action.get("target") or "ALL").strip() or "ALL"
                pct = action.get("modeled_percent")
                if isinstance(pct, (int, float)):
                    lines.append(f"  {idx}. {title} (target: {target}, modeled gain: {float(pct):.2f}%)")
                else:
                    lines.append(f"  {idx}. {title} (target: {target})")
        else:
            lines.append("- No route steps were modeled in this run.")

    totals = known_checks.get("totals") if isinstance(known_checks.get("totals"), dict) else {}
    if totals:
        lines.append("")
        lines.append("## Known-Issue Detection")
        lines.append(f"- Total checks: {int(totals.get('total') or 0)}")
        lines.append(f"- Confirmed: {int(totals.get('confirmed') or 0)}")
        lines.append(f"- Failing: {int(totals.get('failing') or 0)}")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    _debug_stage("main_start")
    env_known_mode = str(os.environ.get("STAT_HARNESS_KNOWN_ISSUES_MODE", "on")).strip().lower()
    if env_known_mode not in {"on", "off"}:
        env_known_mode = "on"
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-version-id", default="")
    parser.add_argument("--run-seed", type=int, default=123)
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional run id. If omitted, a new run id is generated and printed immediately.",
    )
    parser.add_argument(
        "--plugin-set",
        choices=["auto", "full"],
        default="full",
        help="auto=planner-selected; full=run all non-ingest plugins (profile+planner+transform+analysis+report+llm)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if a cache hit exists (ignores STAT_HARNESS_REUSE_CACHE).",
    )
    parser.add_argument(
        "--exclude-processes",
        default="",
        help="Comma/space/semicolon-separated process ids or patterns to exclude from recommendations.",
    )
    parser.add_argument(
        "--recommendations-top-n",
        type=int,
        default=0,
        help="Optional cap on discovery recommendations kept in report output (0 = no cap).",
    )
    parser.add_argument(
        "--recommendations-min-relevance",
        type=float,
        default=0.0,
        help="Optional minimum relevance score for discovery recommendations (0 = no threshold).",
    )
    parser.add_argument(
        "--recommendations-allow-action-types",
        default="",
        help="Optional allowlist of action types (comma/space/semicolon-separated).",
    )
    parser.add_argument(
        "--recommendations-suppress-action-types",
        default="",
        help="Optional suppress list of action types (comma/space/semicolon-separated).",
    )
    parser.add_argument(
        "--recommendations-max-per-action-type",
        default="",
        help="Optional per-action caps, e.g. batch_input=5,reschedule=2",
    )
    parser.add_argument(
        "--recommendations-allow-processes",
        default="",
        help="Optional allowlist of process patterns to keep in recommendations.",
    )
    parser.add_argument(
        "--planner-allow",
        default="",
        help="Optional allowlist of plugin ids for planner auto mode (comma/space/semicolon-separated).",
    )
    parser.add_argument(
        "--planner-deny",
        default="",
        help="Optional denylist of plugin ids for planner auto mode (comma/space/semicolon-separated).",
    )
    parser.add_argument(
        "--allow-structural-not-applicable",
        action="store_true",
        help="Allow gauntlet execution when structural preflight predicts role/column blockers.",
    )
    parser.add_argument(
        "--known-issues-mode",
        choices=["on", "off"],
        default=env_known_mode,
        help="on=load known issues; off=disable known-issue loading/evaluation for this run.",
    )
    parser.add_argument(
        "--orchestrator-mode",
        choices=["legacy", "two_lane_strict"],
        default=str(os.environ.get("STAT_HARNESS_ORCHESTRATOR_MODE", "two_lane_strict") or "two_lane_strict"),
        help="legacy=mixed analysis DAG, two_lane_strict=decision lane first then explanation lane.",
    )
    args = parser.parse_args()
    _debug_stage("args_parsed")

    known_issues_mode = "off" if str(args.known_issues_mode).strip().lower() == "off" else "on"
    orchestrator_mode = (
        "legacy"
        if str(args.orchestrator_mode).strip().lower() == "legacy"
        else "two_lane_strict"
    )
    os.environ["STAT_HARNESS_KNOWN_ISSUES_MODE"] = known_issues_mode
    os.environ["STAT_HARNESS_INCLUDE_KNOWN_RECOMMENDATIONS"] = "0" if known_issues_mode == "off" else "1"
    os.environ["STAT_HARNESS_ORCHESTRATOR_MODE"] = orchestrator_mode
    os.environ.setdefault("STAT_HARNESS_CLI_PROGRESS", "1")
    # Default to reuse-cache for operator UX on large datasets. Still safe for "updated plugins"
    # because cache keys include plugin code hash + settings hash + dataset hash.
    os.environ.setdefault("STAT_HARNESS_REUSE_CACHE", "1")
    # Quick integrity pragma can be extremely slow on large state DBs; operators can
    # opt-in explicitly when needed, but full gauntlet runs should start immediately.
    os.environ.setdefault("STAT_HARNESS_STARTUP_INTEGRITY", "off")
    # Guard against runaway analysis plugins: fail closed per-plugin and continue.
    os.environ.setdefault("STAT_HARNESS_DEFAULT_PLUGIN_TIMEOUT_MS", "600000")

    ctx = get_tenant_context()
    _debug_stage("tenant_context_resolved")
    db_path = ctx.appdata_root / "state.sqlite"
    if not db_path.exists():
        raise SystemExit(f"Missing DB: {db_path}")

    requested = str(args.dataset_version_id or "").strip()
    if requested:
        _debug_stage("dataset_lookup_specific")
        dataset = _dataset_version_row(db_path, requested)
        if not dataset:
            raise SystemExit(f"Dataset version not found: {requested}")
    else:
        _debug_stage("dataset_lookup_latest")
        dataset = _latest_dataset_version_row(db_path)
        if not dataset:
            raise SystemExit("No dataset_version_id found. Upload data first.")

    dataset_version_id = str(dataset["dataset_version_id"])
    run_id = str(args.run_id or "").strip() or make_run_id()
    exclude_processes = _parse_exclude_processes(
        str(args.exclude_processes or os.environ.get("STAT_HARNESS_EXCLUDE_PROCESSES", ""))
    )
    if exclude_processes:
        os.environ["STAT_HARNESS_EXCLUDE_PROCESSES"] = ",".join(exclude_processes)
    if int(args.recommendations_top_n or 0) > 0:
        os.environ["STAT_HARNESS_DISCOVERY_TOP_N"] = str(int(args.recommendations_top_n))
    if float(args.recommendations_min_relevance or 0.0) > 0.0:
        os.environ["STAT_HARNESS_RECOMMENDATION_MIN_RELEVANCE"] = str(float(args.recommendations_min_relevance))
    if str(args.recommendations_allow_action_types or "").strip():
        os.environ["STAT_HARNESS_ALLOW_ACTION_TYPES"] = str(args.recommendations_allow_action_types).strip()
    if str(args.recommendations_suppress_action_types or "").strip():
        os.environ["STAT_HARNESS_SUPPRESS_ACTION_TYPES"] = str(args.recommendations_suppress_action_types).strip()
    if str(args.recommendations_max_per_action_type or "").strip():
        os.environ["STAT_HARNESS_MAX_PER_ACTION_TYPE"] = str(args.recommendations_max_per_action_type).strip()
    if str(args.recommendations_allow_processes or "").strip():
        os.environ["STAT_HARNESS_RECOMMENDATION_ALLOW_PROCESSES"] = str(args.recommendations_allow_processes).strip()
    # Print early so operators can attach watchers while the run is in progress.
    print(f"RUN_ID={run_id}", flush=True)
    row_count = None
    try:
        row_count = int(dataset.get("row_count") or 0)
    except (TypeError, ValueError):
        row_count = None

    resource_profile = str(
        os.environ.get("STAT_HARNESS_RESOURCE_PROFILE", "balanced")
    ).strip().lower()
    if resource_profile not in {"respectful", "balanced", "performance"}:
        resource_profile = "balanced"
    # Large datasets: cap parallelism unless explicitly overridden.
    if row_count is not None and row_count >= 200_000:
        if resource_profile == "respectful":
            os.environ.setdefault("STAT_HARNESS_MAX_WORKERS_ANALYSIS", "1")
        else:
            os.environ.setdefault("STAT_HARNESS_MAX_WORKERS_ANALYSIS", "2")

    plugin_ids: list[str]
    if args.plugin_set == "auto":
        plugin_ids = ["auto"]
    else:
        _debug_stage("discover_plugins_start")
        # Full harness run: execute every non-ingest plugin on the loaded dataset.
        # (Ingest is file-driven and is skipped for DB-only runs.)
        profiles = _discover_plugin_ids({"profile"})
        planners = _discover_plugin_ids({"planner"})
        transforms = _discover_plugin_ids({"transform"})
        analyses = _discover_plugin_ids({"analysis"})
        reports = _discover_plugin_ids({"report"})
        llm = _discover_plugin_ids({"llm"})
        plugin_ids = [*profiles, *planners, *transforms, *analyses, *reports, *llm]
        _debug_stage("discover_plugins_done")

    preflight_dir = ctx.tenant_root / "tmp" / "preflight" / run_id
    preflight = _structural_preflight(
        db_path=db_path,
        dataset_version_id=dataset_version_id,
        plugin_ids=plugin_ids,
        output_dir=preflight_dir,
    )
    print(
        "STRUCTURAL_PREFLIGHT="
        f"checked:{int(preflight.get('checked_count') or 0)},"
        f"blocking:{int(preflight.get('blocking_count') or 0)}",
        flush=True,
    )
    if int(preflight.get("blocking_count") or 0) > 0 and not bool(args.allow_structural_not_applicable):
        raise SystemExit(
            "Structural preflight failed; resolve missing roles/columns first "
            "(or use --allow-structural-not-applicable to override)."
        )

    planner_allow = _parse_exclude_processes(str(args.planner_allow or ""))
    planner_deny = _parse_exclude_processes(str(args.planner_deny or ""))
    run_settings: dict[str, Any] = {
        "exclude_processes": exclude_processes,
        "_system": {"orchestrator_mode": orchestrator_mode},
    }
    if planner_allow or planner_deny:
        planner_settings: dict[str, Any] = {}
        if planner_allow:
            planner_settings["allow"] = planner_allow
        if planner_deny:
            planner_settings["deny"] = planner_deny
        run_settings["planner_basic"] = planner_settings

    _debug_stage("pipeline_ctor_start")
    pipeline = Pipeline(ctx.appdata_root, Path("plugins"), tenant_id=ctx.tenant_id)
    _debug_stage("pipeline_ctor_done")
    _debug_stage("pipeline_run_start")
    run_id = pipeline.run(
        input_file=None,
        plugin_ids=plugin_ids,
        settings=run_settings,
        run_seed=int(args.run_seed),
        dataset_version_id=dataset_version_id,
        run_id=run_id,
        force=bool(args.force),
    )
    _debug_stage("pipeline_run_done")
    run_dir = ctx.tenant_root / "runs" / run_id
    preflight_report_path = preflight_dir / "structural_preflight.json"
    if preflight_report_path.exists():
        shutil.copy2(preflight_report_path, run_dir / "structural_preflight.json")
    report_path = run_dir / "report.json"
    if not report_path.exists():
        raise SystemExit(f"report.json not found for run: {run_id}")

    report = _load_json(report_path)
    recs = _extract_recommendations(report)
    known_checks = _extract_known_issue_checks(recs)
    route_map = _extract_ideaspace_route_map(run_dir)
    ideaspace_gap = _top_findings(report, "analysis_ideaspace_normative_gap", n=8)
    ideaspace_actions = _top_findings(report, "analysis_ideaspace_action_planner", n=12)
    runtime_trend = _runtime_trend(db_path, dataset_version_id, run_id)
    erp_baseline = _baseline_comparison(db_path, dataset)

    answers = {
        "dataset": dataset,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "report_path": str(report_path),
        "runtime_trend": runtime_trend,
        "erp_baseline_comparison": erp_baseline,
        "exclude_processes": exclude_processes,
        "planner_allow": planner_allow,
        "planner_deny": planner_deny,
        "recommendations": recs,
        "known_issue_checks": known_checks,
        "ideaspace_route_map": route_map,
        "ideaspace": {
            "normative_gap_findings": ideaspace_gap,
            "action_planner_findings": ideaspace_actions,
        },
    }
    _write_json(run_dir / "answers_summary.json", answers)
    _write_text(
        run_dir / "answers_recommendations.md",
        _render_recommendations_md(
            recs,
            known_checks,
            route_map=route_map,
            erp_baseline=erp_baseline,
        ),
    )
    _write_text(
        run_dir / "answers_recommendations_plain.md",
        _render_recommendations_plain_md(
            recs,
            known_checks,
            route_map=route_map,
            erp_baseline=erp_baseline,
        ),
    )

    print(f"DATASET_VERSION_ID={dataset_version_id}")
    print(f"KNOWN_ISSUES_MODE={known_issues_mode}")
    print(f"ORCHESTRATOR_MODE={orchestrator_mode}")
    print(f"ROWS={int(dataset.get('row_count') or 0)} COLS={int(dataset.get('column_count') or 0)}")
    print(f"RUN_ID={run_id}")
    if runtime_trend:
        current_minutes = runtime_trend.get("current_minutes")
        historical_count = int(runtime_trend.get("historical_count") or 0)
        avg_minutes = runtime_trend.get("historical_avg_minutes")
        std_minutes = runtime_trend.get("historical_stddev_minutes")
        delta_pct = runtime_trend.get("delta_vs_avg_percent")
        if isinstance(current_minutes, (int, float)):
            print(f"RUNTIME_MINUTES={float(current_minutes):.2f}")
        print(f"RUNTIME_HISTORY_N={historical_count}")
        if isinstance(avg_minutes, (int, float)):
            print(f"RUNTIME_AVG_MINUTES={float(avg_minutes):.2f}")
        if isinstance(std_minutes, (int, float)):
            print(f"RUNTIME_STDDEV_MINUTES={float(std_minutes):.2f}")
        if isinstance(delta_pct, (int, float)):
            print(f"RUNTIME_DELTA_VS_AVG_PCT={float(delta_pct):+.1f}")
    if isinstance(erp_baseline, dict):
        print(f"ERP_TYPE={str(erp_baseline.get('erp_type') or 'unknown')}")
        if erp_baseline.get("available"):
            print(
                "ERP_BASELINE_DATASET_VERSION_ID="
                + str(erp_baseline.get("baseline_dataset_version_id") or "")
            )
            print(
                "ERP_BASELINE_MATCH="
                + "same_erp="
                + str(bool((erp_baseline.get("baseline_match_basis") or {}).get("same_erp"))).lower()
                + ",same_schema_signature="
                + str(
                    bool(
                        (erp_baseline.get("baseline_match_basis") or {}).get(
                            "same_schema_signature"
                        )
                    )
                ).lower()
            )
            print(
                "ERP_BASELINE_ROW_DELTA="
                + str(int(erp_baseline.get("row_count_delta") or 0))
            )
            row_ratio = erp_baseline.get("row_count_ratio")
            if isinstance(row_ratio, (int, float)):
                print(f"ERP_BASELINE_ROW_RATIO={float(row_ratio):.6f}")
        else:
            print(
                "ERP_BASELINE_DATASET_VERSION_ID="
            )
            print("ERP_BASELINE_MATCH=none")
    print(str(run_dir / "report.md"))
    print(str(run_dir / "answers_recommendations.md"))
    print(str(run_dir / "answers_recommendations_plain.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
