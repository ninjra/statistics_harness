#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import statistics
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

try:
    from scripts.run_loaded_dataset_full import (
        _extract_ideaspace_route_map as _extract_ideaspace_route_map_loaded,
    )
    from scripts.run_loaded_dataset_full import (
        _extract_known_issue_checks as _extract_known_issue_checks_loaded,
    )
    from scripts.run_loaded_dataset_full import (
        _render_recommendations_md as _render_recommendations_md_loaded,
    )
    from scripts.run_loaded_dataset_full import (
        _render_recommendations_plain_md as _render_recommendations_plain_md_loaded,
    )
except Exception:
    _extract_ideaspace_route_map_loaded = None
    _extract_known_issue_checks_loaded = None
    _render_recommendations_md_loaded = None
    _render_recommendations_plain_md_loaded = None

try:
    from scripts.generate_batch_sequence_validation_checklist import generate_for_run_dir
except Exception:
    generate_for_run_dir = None


REPO_ROOT = Path(__file__).resolve().parents[1]
APPDATA = REPO_ROOT / "appdata"
_SUPPRESS_ACTION_TYPES_ENV = "STAT_HARNESS_SUPPRESS_ACTION_TYPES"
_MAX_PER_ACTION_TYPE_ENV = "STAT_HARNESS_MAX_PER_ACTION_TYPE"
_REQUIRE_LANDMARK_RECALL_ENV = "STAT_HARNESS_REQUIRE_LANDMARK_RECALL"
_KONA_QEMAIL_TITLE = "known_issue_qemail_schedule"
_KONA_QPEC_TITLE = "known_issue_qpec_plus_one"
_KONA_PAYOUT_TITLE = "known_issue_payout_batch_chain"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _route_map_for_run(run_dir: Path) -> dict[str, Any]:
    if callable(_extract_ideaspace_route_map_loaded):
        try:
            payload = _extract_ideaspace_route_map_loaded(run_dir)
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    return {"available": False, "summary": "No Kona route map available.", "actions": []}


def _items_from_recs(recs: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(recs, dict):
        return [], []
    if "known" in recs or "discovery" in recs:
        known = recs.get("known") or {}
        disc = recs.get("discovery") or {}
        known_items = [i for i in (known.get("items") or []) if isinstance(i, dict)]
        disc_items = [i for i in (disc.get("items") or []) if isinstance(i, dict)]
        return known_items, disc_items
    return [], [i for i in (recs.get("items") or []) if isinstance(i, dict)]


def _flatten_text(value: Any) -> str:
    try:
        if isinstance(value, str):
            return value
        return json.dumps(value, sort_keys=True)
    except Exception:
        return str(value)


def _contains_all(text: str, words: list[str]) -> bool:
    hay = text.lower()
    return all(w.lower() in hay for w in words)


def _contains_any(text: str, words: list[str]) -> bool:
    hay = text.lower()
    return any(w.lower() in hay for w in words)


def _iter_all_findings(report: dict[str, Any]) -> list[dict[str, Any]]:
    plugins = report.get("plugins")
    if not isinstance(plugins, dict):
        return []
    out: list[dict[str, Any]] = []
    for plugin_id, block in plugins.items():
        if not isinstance(block, dict):
            continue
        findings = block.get("findings")
        if not isinstance(findings, list):
            continue
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            row = dict(finding)
            row.setdefault("plugin_id", str(plugin_id))
            out.append(row)
    return out


def _match_known_finding(report: dict[str, Any], item: dict[str, Any]) -> dict[str, Any] | None:
    plugin_id = str(item.get("plugin_id") or "").strip()
    kind = str(item.get("kind") or "").strip()
    where = item.get("where") if isinstance(item.get("where"), dict) else {}
    proc = str(where.get("process") or where.get("process_norm") or where.get("process_id") or "").strip().lower()
    if not plugin_id:
        return None
    plugins = report.get("plugins")
    if not isinstance(plugins, dict):
        return None
    plugin = plugins.get(plugin_id)
    if not isinstance(plugin, dict):
        return None
    findings = plugin.get("findings")
    if not isinstance(findings, list):
        return None
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        if kind and str(finding.get("kind") or "").strip() != kind:
            continue
        if proc:
            f_proc = str(
                finding.get("process_norm")
                or finding.get("process")
                or finding.get("process_id")
                or ""
            ).strip().lower()
            if f_proc != proc:
                continue
        return finding
    # Fallback for landmark-style checks: if exact plugin/kind was not found,
    # search across all findings for the same process token.
    if proc:
        all_findings = _iter_all_findings(report)
        best: dict[str, Any] | None = None
        best_score = -1.0
        for finding in all_findings:
            f_proc = str(
                finding.get("process_norm")
                or finding.get("process")
                or finding.get("process_id")
                or ""
            ).strip().lower()
            if f_proc != proc:
                continue
            score = 0.0
            if isinstance(finding.get("modeled_reduction_pct"), (int, float)):
                score += float(finding.get("modeled_reduction_pct") or 0.0) * 1000.0
            if isinstance(finding.get("estimated_improvement_pct"), (int, float)):
                score += float(finding.get("estimated_improvement_pct") or 0.0) * 500.0
            if isinstance(finding.get("slowdown_ratio"), (int, float)):
                score += float(finding.get("slowdown_ratio") or 0.0) * 10.0
            if isinstance(finding.get("close_count"), (int, float)):
                score += float(finding.get("close_count") or 0.0) / 1000.0
            if score > best_score:
                best_score = score
                best = finding
        if isinstance(best, dict):
            return best
    return None


def _status_from_hits(hits: int, min_count: int = 1, max_count: int = 1) -> str:
    if hits < min_count:
        return "below_min"
    if hits > max_count:
        return "above_max"
    return "confirmed"


def _status_rank(status: str) -> int:
    normalized = str(status or "").strip().lower()
    if normalized in {"confirmed", "above_max"}:
        return 3
    if normalized == "below_min":
        return 1
    return 2


def _normalized_status(status: str) -> str:
    normalized = str(status or "").strip().lower()
    if normalized in {"confirmed", "above_max"}:
        return "confirmed"
    if normalized == "below_min":
        return "below_min"
    return normalized or "unknown"


def _known_issue_label(item: dict[str, Any]) -> str:
    title = str(item.get("title") or item.get("kind") or "known_issue").strip()
    blob = _flatten_text(item).lower()
    if "qemail" in blob:
        return _KONA_QEMAIL_TITLE
    if "qpec" in blob or ("capacity" in blob and "server" in blob):
        return _KONA_QPEC_TITLE
    if _contains_any(blob, ["payout", "rpt_por002", "poextrprvn", "pognrtrpt", "poextrpexp"]):
        return _KONA_PAYOUT_TITLE
    return title


def _as_float(value: Any) -> float | None:
    try:
        if isinstance(value, (int, float)):
            return float(value)
    except Exception:
        return None
    return None


def _extract_action_type(item: dict[str, Any]) -> str:
    return str(item.get("action_type") or item.get("action") or "").strip().lower()


def _process_id(item: dict[str, Any]) -> str:
    return str(
        item.get("primary_process_id")
        or item.get("process_id")
        or (item.get("where") or {}).get("process_norm")
        or ""
    ).strip().lower()


def _is_client_controllable(item: dict[str, Any]) -> bool:
    bucket = _landmark_bucket(item)
    if bucket in {"qemail", "qpec", "payout"}:
        return True
    action_type = _extract_action_type(item)
    if action_type in {
        "batch_input",
        "batch_input_refactor",
        "batch_group_candidate",
        "batch_or_cache",
        "dedupe_or_cache",
        "throttle_or_dedupe",
        "schedule_shift_target",
        "reschedule",
        "tune_schedule",
        "add_server",
        "capacity_addition",
    }:
        return True
    blob = _flatten_text(item).lower()
    return _contains_any(blob, ["qemail", "qpec", "rpt_por002", "poextrprvn", "pognrtrpt", "poextrpexp"])


def _is_generic_all_target(item: dict[str, Any]) -> bool:
    proc = _process_id(item)
    if proc and proc not in {"all", "n/a"}:
        return False
    target = str(item.get("target") or "").strip().lower()
    if target and target not in {"all", "n/a"}:
        return False
    return _landmark_bucket(item) == ""


def _selection_signature(item: dict[str, Any]) -> str:
    bucket = _landmark_bucket(item)
    if bucket:
        return f"landmark:{bucket}"
    action_type = _extract_action_type(item)
    proc = _process_id(item) or str(_target_processes(item)).strip().lower()
    title = re.sub(r"\s+", " ", str(item.get("title") or "").strip().lower())
    rec = re.sub(r"\s+", " ", str(item.get("recommendation") or "").strip().lower())
    return "|".join([action_type, proc, title[:160], rec[:220]])


def _bool_env(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return raw not in {"0", "false", "no", "off", ""}


def _dataset_version_id_from_report(report: dict[str, Any]) -> str:
    input_block = report.get("input")
    if not isinstance(input_block, dict):
        return ""
    filename = str(input_block.get("filename") or "").strip()
    if filename.startswith("db://"):
        return filename[len("db://") :].strip()
    return ""


def _build_numeric_process_resolver(report: dict[str, Any]):
    dataset_version_id = _dataset_version_id_from_report(report)
    if not dataset_version_id:
        return None
    state_db = APPDATA / "state.sqlite"
    if not state_db.exists():
        return None
    try:
        conn = sqlite3.connect(state_db)
        conn.row_factory = sqlite3.Row
    except Exception:
        return None
    try:
        row = conn.execute(
            "SELECT table_name FROM dataset_versions WHERE dataset_version_id=?",
            (dataset_version_id,),
        ).fetchone()
        if not row:
            conn.close()
            return None
        table_name = str(row["table_name"] or "").strip()
        if not table_name:
            conn.close()
            return None
        tmpl = conn.execute(
            "SELECT template_id FROM dataset_templates WHERE dataset_version_id=? ORDER BY updated_at DESC LIMIT 1",
            (dataset_version_id,),
        ).fetchone()
        if not tmpl:
            conn.close()
            return None
        template_id = int(tmpl["template_id"])
        fields = conn.execute(
            "SELECT field_id,name FROM template_fields WHERE template_id=?",
            (template_id,),
        ).fetchall()
        field_to_col = {
            str(r["name"] or "").strip().upper(): f"c{int(r['field_id'])}"
            for r in fields
            if str(r["name"] or "").strip()
        }
        queue_col = field_to_col.get("PROCESS_QUEUE_ID")
        process_col = field_to_col.get("PROCESS_ID")
        if not queue_col or not process_col:
            conn.close()
            return None
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return None

    cache: dict[str, str | None] = {}

    def _resolve(token: str, item: dict[str, Any] | None = None) -> str | None:
        key = str(token or "").strip()
        if not key or not key.isdigit():
            return None
        if key in cache:
            return cache[key]
        value_int = int(key)
        resolved: str | None = None

        # First try evidence row_ids because they are deterministic anchors.
        row_ids: list[int] = []
        if isinstance(item, dict):
            evidence = item.get("evidence")
            if isinstance(evidence, list):
                for e in evidence:
                    if not isinstance(e, dict):
                        continue
                    for rid in (e.get("row_ids") or []):
                        if isinstance(rid, int):
                            row_ids.append(rid)
        try:
            for rid in row_ids:
                for clause, arg in (("row_id=?", rid), ("row_index=?", rid)):
                    query = (
                        f"SELECT {queue_col} AS q, {process_col} AS p "
                        f"FROM {table_name} WHERE {clause} LIMIT 1"
                    )
                    row = conn.execute(query, (arg,)).fetchone()
                    if not row:
                        continue
                    q = row["q"]
                    p = str(row["p"] or "").strip()
                    if isinstance(q, int) and q == value_int and p and not p.isdigit():
                        resolved = p
                        break
                if resolved:
                    break
        except Exception:
            resolved = None

        # Fallback: direct lookup by queue id to most common process name.
        if not resolved:
            try:
                row = conn.execute(
                    f"SELECT {process_col} AS p, COUNT(*) AS c FROM {table_name} "
                    f"WHERE {queue_col}=? AND {process_col} IS NOT NULL "
                    f"GROUP BY {process_col} ORDER BY c DESC LIMIT 1",
                    (value_int,),
                ).fetchone()
                if row:
                    p = str(row["p"] or "").strip()
                    if p and not p.isdigit():
                        resolved = p
            except Exception:
                resolved = None

        cache[key] = resolved
        return resolved

    return _resolve


def _display_process_label(
    raw_process: str,
    item: dict[str, Any] | None,
    resolver: Any | None,
) -> str:
    base = str(raw_process or "").strip() or "n/a"
    if not base.isdigit():
        return base
    resolved = None
    if callable(resolver):
        try:
            resolved = resolver(base, item if isinstance(item, dict) else None)
        except Exception:
            resolved = None
    if isinstance(resolved, str) and resolved.strip():
        return f"{resolved.strip()} [{base}]"
    return f"unknown_process(id={base})"


def _humanize_recommendation_process(
    text: str,
    raw_process: str,
    display_process: str,
) -> str:
    body = str(text or "")
    raw = str(raw_process or "").strip()
    shown = str(display_process or "").strip()
    if not raw or not raw.isdigit():
        return body
    if not shown or shown.startswith("unknown_process("):
        return body
    out = re.sub(rf"\bProcess\s+{re.escape(raw)}\b", f"Process {shown}", body)
    out = out.replace(f"`{raw}`", f"`{shown}`")
    return out


def _extract_modeled_triplet(item: dict[str, Any]) -> tuple[float | None, float | None, float | None, float | None]:
    close_pct = _as_float(item.get("modeled_close_percent"))
    general_pct = _as_float(item.get("modeled_general_percent"))
    generic_pct = _as_float(item.get("modeled_percent"))
    scope = str(item.get("scope_class") or "").strip().lower()
    if close_pct is None and isinstance(generic_pct, float) and scope == "close_specific":
        close_pct = generic_pct
    if general_pct is None and isinstance(generic_pct, float) and scope != "close_specific":
        general_pct = generic_pct
    close_hours = _as_float(item.get("modeled_close_hours"))
    general_hours = _as_float(item.get("modeled_general_hours"))
    delta_h = _as_float(item.get("modeled_delta_hours"))
    if close_hours is None and isinstance(delta_h, float) and scope == "close_specific":
        close_hours = delta_h
    if general_hours is None and isinstance(delta_h, float) and scope != "close_specific":
        general_hours = delta_h
    return close_pct, general_pct, close_hours, general_hours


def _detect_process_token(
    report: dict[str, Any], discovery_items: list[dict[str, Any]], token: str
) -> bool:
    needle = str(token or "").strip().lower()
    if not needle:
        return False
    for item in discovery_items:
        if needle in _flatten_text(item).lower():
            return True
    for finding in _iter_all_findings(report):
        if needle in _flatten_text(finding).lower():
            return True
    return False


def _extract_process_hint(item: dict[str, Any]) -> str:
    where = item.get("where") if isinstance(item.get("where"), dict) else {}
    for key in ("process_norm", "process", "process_id", "target"):
        value = where.get(key) if isinstance(where, dict) else None
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    for key in ("process_norm", "process", "process_id", "target"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    targets = item.get("target_process_ids")
    if isinstance(targets, list):
        for value in targets:
            if isinstance(value, str) and value.strip():
                return value.strip().lower()
    return ""


def _best_equivalent_qemail_model(
    report: dict[str, Any], discovery_items: list[dict[str, Any]]
) -> dict[str, Any]:
    """Fallback when no literal qemail token exists in dataset outputs."""
    findings = _iter_all_findings(report)
    best: dict[str, Any] = {}
    allowed_actions = {
        "tune_schedule",
        "reschedule",
        "batch_or_cache",
        "throttle_or_dedupe",
        "reduce_process_wait",
        "route_process",
        "priority_isolation",
    }
    for item in [*discovery_items, *findings]:
        if not isinstance(item, dict):
            continue
        action_type = _extract_action_type(item)
        blob = _flatten_text(item).lower()
        if action_type not in allowed_actions and not _contains_any(
            blob,
            ["queue", "contention", "priority class", "reserved capacity", "close-critical", "reschedule"],
        ):
            continue
        close_pct, general_pct, close_hours, general_hours = _extract_modeled_triplet(item)
        score = max(close_pct or 0.0, general_pct or 0.0)
        if score >= float(best.get("_score") or -1.0):
            best = {
                "_score": score,
                "plugin_id": str(item.get("plugin_id") or ""),
                "close_pct": close_pct,
                "general_pct": general_pct,
                "close_hours": close_hours,
                "general_hours": general_hours,
                "process": _extract_process_hint(item),
            }
    return best


def _generic_add_server_hits(
    report: dict[str, Any], discovery_items: list[dict[str, Any]]
) -> tuple[int, str]:
    findings = _iter_all_findings(report)
    hits = 0
    plugin_id = ""
    for item in [*discovery_items, *findings]:
        if not isinstance(item, dict):
            continue
        action_type = _extract_action_type(item)
        blob = _flatten_text(item).lower()
        is_capacity_signal = action_type == "add_server" or (
            _contains_any(blob, ["+1", "add one", "add server", "host_count+1", "workers by 1"])
            and _contains_any(blob, ["capacity", "queue", "server", "worker", "host"])
        )
        if not is_capacity_signal:
            continue
        hits += 1
        plugin_id = str(item.get("plugin_id") or plugin_id)
    return hits, plugin_id


def _qemail_modeled(discovery_items: list[dict[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    for item in discovery_items:
        if not isinstance(item, dict):
            continue
        where = item.get("where") if isinstance(item.get("where"), dict) else {}
        proc = str(
            where.get("process_norm")
            or where.get("process")
            or item.get("target")
            or ""
        ).strip().lower()
        if proc != "qemail" and "qemail" not in _flatten_text(item).lower():
            continue
        close_pct, general_pct, close_h, general_h = _extract_modeled_triplet(item)
        score = max(close_pct or 0.0, general_pct or 0.0)
        if score >= float(best.get("_score") or -1.0):
            best = {
                "_score": score,
                "plugin_id": str(item.get("plugin_id") or ""),
                "close_pct": close_pct,
                "general_pct": general_pct,
                "close_hours": close_h,
                "general_hours": general_h,
            }
    return best


def _qemail_modeled_from_known(known_items: list[dict[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    for item in known_items:
        if not isinstance(item, dict):
            continue
        blob = _flatten_text(item).lower()
        if "qemail" not in blob:
            continue
        close_pct = _as_float(item.get("modeled_close_percent"))
        general_pct = _as_float(item.get("modeled_general_percent"))
        generic_pct = _as_float(item.get("modeled_percent"))
        if close_pct is None and isinstance(generic_pct, float):
            close_pct = generic_pct
        if general_pct is None and isinstance(generic_pct, float):
            general_pct = generic_pct
        score = max(close_pct or 0.0, general_pct or 0.0)
        if score >= float(best.get("_score") or -1.0):
            best = {
                "_score": score,
                "plugin_id": str(item.get("plugin_id") or ""),
                "close_pct": close_pct,
                "general_pct": general_pct,
                "close_hours": _as_float(item.get("modeled_close_hours")),
                "general_hours": _as_float(item.get("modeled_general_hours")),
            }
    return best


def _qemail_modeled_from_report(report: dict[str, Any]) -> dict[str, Any]:
    plugins = report.get("plugins")
    if not isinstance(plugins, dict):
        return {}

    best: dict[str, Any] = {}
    # Source 1: ideaspace planner findings with percent-modeled deltas.
    planner = plugins.get("analysis_ideaspace_action_planner")
    if isinstance(planner, dict):
        for finding in planner.get("findings") or []:
            if not isinstance(finding, dict):
                continue
            blob = _flatten_text(finding).lower()
            if "qemail" not in blob:
                continue
            value = _as_float(finding.get("delta_value"))
            if value is None:
                value = _as_float(finding.get("modeled_value"))
            if value is None:
                continue
            unit = str(finding.get("unit") or "").strip().lower()
            pct = value * 100.0 if unit in {"ratio", "fraction"} and value <= 1.0 else value
            score = float(pct or 0.0)
            if score >= float(best.get("_score") or -1.0):
                best = {
                    "_score": score,
                    "plugin_id": "analysis_ideaspace_action_planner",
                    "close_pct": float(pct),
                    "general_pct": float(pct),
                    "close_hours": None,
                    "general_hours": None,
                }

    # Source 2: queue-delay process share if QEMAIL appears there.
    qd = plugins.get("analysis_queue_delay_decomposition")
    if isinstance(qd, dict):
        qemail_h = None
        total_h = None
        for finding in qd.get("findings") or []:
            if not isinstance(finding, dict):
                continue
            kind = str(finding.get("kind") or "")
            if kind == "eligible_wait_process_stats":
                proc = str(finding.get("process_norm") or "").strip().lower()
                if proc == "qemail":
                    qemail_h = _as_float(finding.get("eligible_wait_gt_hours_total"))
            elif kind == "eligible_wait_impact":
                total_h = _as_float(finding.get("eligible_wait_gt_hours_total"))
        if isinstance(qemail_h, float) and isinstance(total_h, float) and total_h > 0.0:
            pct = (qemail_h / total_h) * 100.0
            score = float(pct)
            if score >= float(best.get("_score") or -1.0):
                best = {
                    "_score": score,
                    "plugin_id": "analysis_queue_delay_decomposition",
                    "close_pct": float(pct),
                    "general_pct": float(pct),
                    "close_hours": qemail_h,
                    "general_hours": qemail_h,
                }
    return best


def _best_qemail_model(*models: dict[str, Any]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    for model in models:
        if not isinstance(model, dict):
            continue
        score = float(model.get("_score") or 0.0)
        if score >= float(best.get("_score") or -1.0):
            best = model
    return best


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


def _runtime_stats(run_id: str) -> dict[str, Any]:
    db_path = APPDATA / "state.sqlite"
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            """
            SELECT run_id, dataset_version_id, input_filename, started_at, created_at, completed_at
            FROM runs WHERE run_id = ?
            LIMIT 1
            """,
            (run_id,),
        ).fetchone()
        if row is None:
            return {}
        dataset_version_id = str(row["dataset_version_id"] or "").strip()
        if not dataset_version_id:
            input_filename = str(row["input_filename"] or "").strip()
            if input_filename.startswith("db://"):
                dataset_version_id = input_filename[5:]
        if not dataset_version_id:
            return {}

        run_rows = conn.execute(
            """
            SELECT run_id, started_at, created_at, completed_at, status
            FROM runs
            WHERE dataset_version_id = ?
              AND status IN ('completed', 'partial')
              AND completed_at IS NOT NULL
            ORDER BY created_at
            """,
            (dataset_version_id,),
        ).fetchall()
        samples: list[tuple[str, float]] = []
        for rr in run_rows:
            start_dt = _parse_ts(rr["started_at"]) or _parse_ts(rr["created_at"])
            end_dt = _parse_ts(rr["completed_at"])
            if start_dt is None or end_dt is None:
                continue
            minutes = (end_dt - start_dt).total_seconds() / 60.0
            if minutes > 0.0:
                samples.append((str(rr["run_id"]), float(minutes)))
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
            "dataset_version_id": dataset_version_id,
            "current_minutes": current,
            "history_count": len(history),
            "history_avg_minutes": avg,
            "history_stddev_minutes": stddev,
            "delta_vs_avg_percent": delta_pct,
        }
    finally:
        conn.close()


def _safe_write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _ensure_answer_artifacts(
    run_id: str,
    run_dir: Path,
    recs: Any,
    known_items: list[dict[str, Any]],
) -> None:
    summary_path = run_dir / "answers_summary.json"
    md_path = run_dir / "answers_recommendations.md"
    plain_path = run_dir / "answers_recommendations_plain.md"
    batch_checklist_json = run_dir / "batch_sequence_validation_checklist.json"
    batch_checklist_md = run_dir / "batch_sequence_validation_checklist.md"

    if (
        summary_path.exists()
        and md_path.exists()
        and plain_path.exists()
        and batch_checklist_json.exists()
        and batch_checklist_md.exists()
    ):
        return

    recs_dict = recs if isinstance(recs, dict) else {"summary": "No recommendations block found.", "items": []}
    route_map = _route_map_for_run(run_dir)

    known_checks: dict[str, Any]
    if callable(_extract_known_issue_checks_loaded):
        known_checks = _extract_known_issue_checks_loaded(recs_dict)
    else:
        status_counts: dict[str, int] = {}
        for item in known_items:
            status = str(item.get("status") or "unknown")
            status_counts[status] = int(status_counts.get(status, 0)) + 1
        known_checks = {
            "status": "ok" if status_counts.get("confirmed", 0) else "none",
            "summary": "Recovered known-issue checks from report recommendations.",
            "items": known_items,
            "totals": {
                "total": int(len(known_items)),
                "confirmed": int(status_counts.get("confirmed", 0)),
                "failing": int(sum(v for k, v in status_counts.items() if k != "confirmed")),
                "by_status": status_counts,
            },
        }

    if not summary_path.exists():
        summary_payload = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "runtime_trend": _runtime_stats(run_id),
            "recommendations": recs_dict,
            "known_issue_checks": known_checks,
            "ideaspace_route_map": route_map,
        }
        _safe_write_json(summary_path, summary_payload)

    if not md_path.exists():
        if callable(_render_recommendations_md_loaded):
            text = _render_recommendations_md_loaded(recs_dict, known_checks, route_map=route_map)
        else:
            text = "# Recommendations\n\nRecommendations were recovered from report.json.\n"
        md_path.write_text(text, encoding="utf-8")

    if not plain_path.exists():
        if callable(_render_recommendations_plain_md_loaded):
            text = _render_recommendations_plain_md_loaded(recs_dict, known_checks, route_map=route_map)
        else:
            text = "# Recommendations (Plain Language)\n\nRecommendations were recovered from report.json.\n"
        plain_path.write_text(text, encoding="utf-8")

    if (not batch_checklist_json.exists() or not batch_checklist_md.exists()) and callable(generate_for_run_dir):
        try:
            payload, markdown = generate_for_run_dir(run_dir)
            batch_checklist_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            batch_checklist_md.write_text(markdown, encoding="utf-8")
        except Exception:
            pass


def _collapse_known_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        label = _known_issue_label(item)
        status = _normalized_status(str(item.get("status") or "unknown"))
        row = grouped.get(label)
        if row is None:
            row = dict(item)
            row["title"] = label
            row["status"] = status
            grouped[label] = row
            continue
        current_status = _normalized_status(str(row.get("status") or "unknown"))
        if _status_rank(status) >= _status_rank(current_status):
            row["status"] = status
            for key in ("plugin_id", "kind", "source"):
                val = item.get(key)
                if isinstance(val, str) and val.strip():
                    row[key] = val
        observed = item.get("observed_count")
        current_observed = row.get("observed_count")
        if isinstance(observed, (int, float)):
            if not isinstance(current_observed, (int, float)):
                row["observed_count"] = int(observed)
            else:
                row["observed_count"] = int(max(float(current_observed), float(observed)))
    return list(grouped.values())


def _augment_known_issue_landmarks(
    report: dict[str, Any], known_items: list[dict[str, Any]], discovery_items: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    known = [dict(i) for i in known_items if isinstance(i, dict)]
    findings = _iter_all_findings(report)
    has_qemail_token = _detect_process_token(report, discovery_items, "qemail")
    has_qpec_token = _detect_process_token(report, discovery_items, "qpec")

    # Landmark A: QEMAIL scheduling/contention issue.
    qemail_model = _best_qemail_model(
        _qemail_modeled(discovery_items),
        _qemail_modeled_from_known(known_items),
        _qemail_modeled_from_report(report),
    )
    if not has_qemail_token:
        qemail_model = _best_qemail_model(qemail_model, _best_equivalent_qemail_model(report, discovery_items))
    qemail_close_pct = _as_float(qemail_model.get("close_pct"))
    qemail_general_pct = _as_float(qemail_model.get("general_pct"))
    # Known-issue pass criterion is modeled-impact based, not text-hit based.
    # Require >=10% either in close-specific or general scope.
    qemail_hits = 1 if ((qemail_close_pct or 0.0) >= 10.0 or (qemail_general_pct or 0.0) >= 10.0) else 0
    qemail_plugin = str(qemail_model.get("plugin_id") or "")
    qemail_process = str(qemail_model.get("process") or "qemail").strip().lower() or "qemail"
    qemail_status = _status_from_hits(qemail_hits, min_count=1, max_count=1)
    known.append(
        {
            "title": _KONA_QEMAIL_TITLE,
            "status": qemail_status,
            "plugin_id": qemail_plugin or "analysis_close_cycle_contention",
            "kind": "known_issue_landmark",
            "where": {"process": qemail_process},
            "observed_count": int(qemail_hits),
            "modeled_close_percent": qemail_close_pct,
            "modeled_general_percent": qemail_general_pct,
            "modeled_close_hours": _as_float(qemail_model.get("close_hours")),
            "modeled_general_hours": _as_float(qemail_model.get("general_hours")),
            "recommendation": (
                "Known issue: QEMAIL scheduling/frequency. "
                "Expected modeled reduction >=10% in close-specific and/or general scope."
            ),
            "source": "kona_landmark",
        }
    )

    # Landmark B: QPEC +1 capacity addition.
    qpec_hits = 0
    qpec_plugin = ""
    if has_qpec_token:
        for item in discovery_items:
            blob = _flatten_text(item).lower()
            if _contains_all(blob, ["qpec"]) and _contains_any(
                blob,
                ["+1", "capacity", "add one", "add server", "workers by 1", "host_count+1"],
            ):
                qpec_hits += 1
                qpec_plugin = str(item.get("plugin_id") or qpec_plugin)
        for finding in findings:
            blob = _flatten_text(finding).lower()
            if _contains_all(blob, ["qpec"]) and _contains_any(
                blob,
                ["+1", "capacity", "add one", "add server", "workers by 1", "host_count+1"],
            ):
                qpec_hits += 1
                qpec_plugin = str(finding.get("plugin_id") or qpec_plugin)
    else:
        qpec_hits, qpec_plugin = _generic_add_server_hits(report, discovery_items)
    qpec_status = _status_from_hits(qpec_hits, min_count=1, max_count=20)
    known.append(
        {
            "title": _KONA_QPEC_TITLE,
            "status": qpec_status,
            "plugin_id": qpec_plugin or "analysis_ideaspace_action_planner",
            "kind": "known_issue_landmark",
            "observed_count": int(qpec_hits),
            "recommendation": (
                "Known issue: QPEC capacity (+1 server/worker). "
                "Expected at least one concrete capacity recommendation."
            ),
            "source": "kona_landmark",
        }
    )

    # Landmark C: payout report chain batching.
    payout_hits = 0
    payout_plugin = ""
    payout_targets = {
        "rpt_por002",
        "poextrprvn",
        "pognrtrpt",
        "poextrpexp",
    }
    for item in discovery_items:
        blob = _flatten_text(item).lower()
        action_type = str(item.get("action_type") or item.get("action") or "").strip().lower()
        targets_text = _flatten_text(item.get("target_process_ids") or item.get("evidence") or "").lower()
        if action_type == "batch_group_candidate":
            payout_hits += 1
            payout_plugin = str(item.get("plugin_id") or payout_plugin)
            continue
        if _contains_any(blob, ["payout", "batch payout-report chain"]) and _contains_any(
            blob, ["batch", "multi-input", "batch-input"]
        ):
            payout_hits += 1
            payout_plugin = str(item.get("plugin_id") or payout_plugin)
            continue
        if any(t in targets_text for t in payout_targets):
            payout_hits += 1
            payout_plugin = str(item.get("plugin_id") or payout_plugin)
    for finding in findings:
        blob = _flatten_text(finding).lower()
        if _contains_any(blob, ["payout", "batch payout-report chain"]) and _contains_any(
            blob, ["batch", "multi-input", "batch-input"]
        ):
            payout_hits += 1
            payout_plugin = str(finding.get("plugin_id") or payout_plugin)
    payout_status = _status_from_hits(payout_hits, min_count=1, max_count=20)
    known.append(
        {
            "title": _KONA_PAYOUT_TITLE,
            "status": payout_status,
            "plugin_id": payout_plugin or "analysis_actionable_ops_levers_v1",
            "kind": "known_issue_landmark",
            "observed_count": int(payout_hits),
            "recommendation": (
                "Known issue: payout report chain should be batched (multi-input). "
                "Expected at least one concrete payout-chain batch recommendation."
            ),
            "source": "kona_landmark",
        }
    )

    return known


def _fmt_where(where: Any) -> str:
    if not isinstance(where, dict) or not where:
        return ""
    # Keep it stable and compact.
    keys = ("process_norm", "process", "process_id", "activity", "parent_process", "child_process")
    out = {k: where.get(k) for k in keys if k in where}
    if not out:
        out = where
    try:
        return json.dumps(out, sort_keys=True)
    except Exception:
        return str(out)


def _ranked_actionables(disc_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not disc_items:
        return []

    def _suppressed() -> set[str]:
        defaults = {"tune_threshold"}
        raw = str(os.environ.get(_SUPPRESS_ACTION_TYPES_ENV, "")).strip()
        if raw == "":
            return defaults
        out: set[str] = set()
        for token in raw.replace(";", ",").split(","):
            token = token.strip()
            if token:
                out.add(token)
        return out

    def _caps() -> dict[str, int]:
        defaults = {
            "batch_input": 8,
            "batch_or_cache": 6,
            "batch_input_refactor": 6,
            "dedupe_or_cache": 4,
            "unblock_dependency_chain": 6,
            "reduce_transition_gap": 6,
            "orchestrate_chain": 5,
            "orchestrate_macro": 5,
            "decouple_boundary": 4,
            "shared_cache_endpoint": 4,
            "batch_group_candidate": 4,
            "cluster_with_constraints": 3,
            "distribution_shift_target": 4,
            "burst_trigger": 4,
            "schedule_shift_target": 4,
            "reschedule": 3,
            "route_process": 3,
            "reduce_process_wait": 2,
            "review": 2,
            "tune_threshold": 1,
        }
        raw = str(os.environ.get(_MAX_PER_ACTION_TYPE_ENV, "")).strip()
        if not raw:
            return defaults
        out = dict(defaults)
        for token in raw.replace(";", ",").split(","):
            token = token.strip()
            if not token or "=" not in token:
                continue
            k, v = token.split("=", 1)
            k = k.strip()
            v = v.strip()
            if not k:
                continue
            try:
                out[k] = int(v)
            except ValueError:
                continue
        return out

    def _num(item: dict[str, Any], *keys: str) -> float:
        for key in keys:
            value = item.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return 0.0

    def _effort_score(item: dict[str, Any]) -> float:
        # Ranking axis 1: human effort reduction.
        # Prefer explicit run/touch reductions; fall back to close-cycle user hours.
        runs = _num(
            item,
            "modeled_user_runs_reduced",
            "human_modeled_run_reduction_count_close_cycle",
        )
        touches = _num(
            item,
            "modeled_user_touches_reduced",
            "human_modeled_touch_reduction_count_close_cycle",
        )
        user_hours = _num(item, "modeled_user_hours_saved_close_cycle")
        return max(runs, touches, user_hours)

    def _time_score(item: dict[str, Any]) -> float:
        # Ranking axis 2: elapsed processing time reduction.
        return _num(
            item,
            "delta_hours_close_dynamic",
            "modeled_delta_hours_close_cycle",
            "delta_hours_accounting_month",
            "impact_hours",
            "modeled_delta",
        )

    def _score(item: dict[str, Any]) -> float:
        for key in ("relevance_score", "impact_hours", "modeled_delta"):
            try:
                v = item.get(key)
                if isinstance(v, (int, float)):
                    return float(v)
            except Exception:
                continue
        return 0.0

    def _controllable_priority(item: dict[str, Any]) -> int:
        bucket = _landmark_bucket(item)
        if bucket in {"qemail", "qpec", "payout"}:
            return 3
        if _is_client_controllable(item) and not _is_generic_all_target(item):
            return 2
        if _is_client_controllable(item):
            return 1
        return 0

    ranked = sorted(
        disc_items,
        key=lambda i: (
            _controllable_priority(i),
            _effort_score(i),
            _time_score(i),
            _score(i),
        ),
        reverse=True,
    )
    suppressed = _suppressed()
    caps = _caps()
    used: dict[str, int] = {}
    kept: list[dict[str, Any]] = []
    for item in ranked:
        action_type = str(item.get("action_type") or item.get("action") or "").strip()
        if action_type and action_type in suppressed:
            continue
        if action_type:
            limit = caps.get(action_type)
            if isinstance(limit, int) and limit > 0:
                used[action_type] = int(used.get(action_type, 0)) + 1
                if used[action_type] > limit:
                    continue
        kept.append(item)
    return kept


def _item_signature(item: dict[str, Any]) -> str:
    parts = [
        str(item.get("plugin_id") or "").strip(),
        str(item.get("kind") or "").strip(),
        str(item.get("action_type") or item.get("action") or "").strip(),
        str(item.get("primary_process_id") or item.get("process_id") or "").strip(),
        str(item.get("title") or "").strip(),
        str(item.get("recommendation") or "").strip(),
    ]
    return "|".join(parts)


def _promote_landmarks(
    selected: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    top_n: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    if top_n <= 0:
        return [], []
    out = list(selected)
    rank_idx: dict[str, int] = {
        _item_signature(item): idx for idx, item in enumerate(ranked)
    }
    promoted: list[str] = []

    def _has_bucket(items: list[dict[str, Any]], bucket: str) -> bool:
        return any(_landmark_bucket(item) == bucket for item in items)

    for bucket in _LANDMARK_BUCKETS:
        if _has_bucket(out, bucket):
            continue
        candidate = next((item for item in ranked if _landmark_bucket(item) == bucket), None)
        if not isinstance(candidate, dict):
            continue
        sig = _selection_signature(candidate)
        existing = {_selection_signature(item) for item in out}
        if sig in existing:
            continue
        if len(out) < top_n:
            out.append(candidate)
            promoted.append(bucket)
            continue
        replace_idx = None
        for idx in range(len(out) - 1, -1, -1):
            if _landmark_bucket(out[idx]) == "":
                replace_idx = idx
                break
        if isinstance(replace_idx, int):
            out[replace_idx] = candidate
            promoted.append(bucket)

    out = sorted(
        out,
        key=lambda item: rank_idx.get(_item_signature(item), 10**9),
    )[:top_n]
    return out, promoted


def _lane_bucket(item: dict[str, Any]) -> str:
    bucket = _landmark_bucket(item)
    action_type = _extract_action_type(item)
    if bucket == "payout" or action_type in {
        "batch_input",
        "batch_input_refactor",
        "batch_group_candidate",
        "batch_or_cache",
        "dedupe_or_cache",
        "throttle_or_dedupe",
    }:
        return "manual"
    if bucket in {"qemail", "qpec"} or action_type in {
        "schedule_shift_target",
        "reschedule",
        "tune_schedule",
        "add_server",
        "capacity_addition",
    }:
        return "infra"
    human_runs = item.get("modeled_user_runs_reduced")
    human_touches = item.get("modeled_user_touches_reduced")
    if isinstance(human_runs, (int, float)) and float(human_runs) > 0:
        return "manual"
    if isinstance(human_touches, (int, float)) and float(human_touches) > 0:
        return "manual"
    return "infra"


def _landmark_bucket(item: dict[str, Any]) -> str:
    blob = _flatten_text(item).lower()
    process = str(item.get("primary_process_id") or item.get("process_id") or "").strip().lower()
    action_type = _extract_action_type(item)
    if process == "qemail" or "qemail" in blob:
        return "qemail"
    if process == "qpec" or "qpec" in blob:
        return "qpec"
    if action_type == "add_server" and _contains_any(blob, ["capacity", "server"]):
        return "qpec"
    payout_tokens = ("rpt_por002", "poextrprvn", "pognrtrpt", "poextrpexp")
    if process in {"rpt_por002", "poextrprvn", "pognrtrpt", "poextrpexp"}:
        return "payout"
    if action_type == "batch_group_candidate" and any(token in blob for token in payout_tokens):
        return "payout"
    return ""


_LANDMARK_BUCKETS: tuple[str, ...] = ("qemail", "qpec", "payout")


def _landmark_positions(items: list[dict[str, Any]]) -> dict[str, int | None]:
    positions: dict[str, int | None] = {bucket: None for bucket in _LANDMARK_BUCKETS}
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        bucket = _landmark_bucket(item)
        if bucket in positions and positions[bucket] is None:
            positions[bucket] = idx
    return positions


def _status_yn(identifier: str, ok: bool) -> str:
    # Hard-gate formatting contract from AGENTS.md.
    token = "\033[32mY\033[0m" if bool(ok) else "\033[31mN\033[0m"
    return f"{token}:{identifier}"


def _normalize_action_type(item: dict[str, Any]) -> str:
    return str(item.get("action_type") or item.get("action") or "").strip().lower()


def _recommendation_group(item: dict[str, Any]) -> tuple[str, str]:
    landmark = _landmark_bucket(item)
    if landmark:
        return ("landmark", "Known High-Value Issues (Independently Found)")
    action_type = _normalize_action_type(item)
    recommendation_text = str(item.get("recommendation") or "").strip().lower()
    if action_type == "batch_group_candidate":
        return (
            "batch_group_candidate",
            "Batch Payout Report Chain (Targeted Process IDs)",
        )
    if action_type in {"batch_or_cache", "dedupe_or_cache", "throttle_or_dedupe"}:
        return ("batch_or_cache", "Cache/Batch Reuse Opportunities")
    if action_type in {"batch_input", "batch_input_refactor"}:
        return ("batch_input", "Batch-Input Process Conversions")
    if action_type in {"unblock_dependency_chain", "reduce_transition_gap"}:
        return ("handoff", "Dependency / Handoff Fixes")
    if action_type in {"schedule_shift_target", "reschedule", "tune_schedule"}:
        return ("schedule", "Schedule / Time-Window Tuning")
    if action_type in {"add_server", "capacity_addition"}:
        return ("capacity", "Capacity / Server Scaling")
    if "qpec+1" in recommendation_text or "add one qpec server" in recommendation_text:
        return ("capacity", "Capacity / Server Scaling")
    kind = str(item.get("kind") or "").strip().lower()
    if kind:
        return (kind, f"Other: {kind}")
    return ("other", "Other Recommendations")


def _target_processes(item: dict[str, Any]) -> str:
    targets = item.get("target_process_ids")
    if isinstance(targets, list):
        values = [str(v).strip() for v in targets if isinstance(v, str) and v.strip()]
        if values:
            return ", ".join(values)
    evidence = item.get("evidence")
    if isinstance(evidence, list):
        for row in evidence:
            if isinstance(row, dict):
                vals = row.get("target_process_ids")
                if isinstance(vals, list):
                    values = [str(v).strip() for v in vals if isinstance(v, str) and v.strip()]
                    if values:
                        return ", ".join(values)
    where = item.get("where") if isinstance(item.get("where"), dict) else {}
    for key in ("process_norm", "process", "process_id", "activity"):
        val = where.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    # Best-effort extraction for explicit process_id text.
    text = str(item.get("recommendation") or "").strip()
    if text:
        match = re.search(r"`([^`]+)`", text)
        if match:
            return match.group(1)
    return ""


def _theme_enabled(theme: str) -> bool:
    mode = str(theme or "auto").strip().lower()
    if mode == "plain":
        return False
    if mode == "cyberpunk":
        return True
    if str(os.getenv("NO_COLOR") or "").strip():
        return False
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


class _AnsiTheme:
    def __init__(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.reset = "\033[0m"
        self.title = "\033[95m"  # magenta
        self.section = "\033[94m"  # blue
        self.label = "\033[96m"  # cyan
        self.value = "\033[97m"  # white
        self.hot = "\033[93m"  # yellow
        self.cool = "\033[94m"  # blue
        self.dim = "\033[90m"  # gray
        self.acct = "\033[38;5;117m"  # soft cyan-blue
        self.static = "\033[38;5;81m"  # azure
        self.dynamic = "\033[38;5;183m"  # lavender
        self.sep = "\033[97m"  # bright white separator
        self.score_hi = "\033[92m"  # green
        self.score_mid = "\033[96m"  # cyan
        self.score_lo = "\033[37m"  # light gray

    def c(self, text: str, code: str) -> str:
        raw = str(text)
        if not self.enabled:
            return raw
        return f"{code}{raw}{self.reset}"

    def score(self, value: Any) -> str:
        if not isinstance(value, (int, float)):
            return self.c("N/A", self.dim)
        val = float(value)
        if val >= 6.0:
            return self.c(f"{val:.2f}", self.score_hi)
        if val >= 3.0:
            return self.c(f"{val:.2f}", self.score_mid)
        return self.c(f"{val:.2f}", self.score_lo)


def _window_triplet(
    theme: _AnsiTheme,
    item: dict[str, Any],
    keys: tuple[str, str, str],
    decimals: int,
    suffix: str = "",
    reasons: tuple[str, str, str] | None = None,
) -> str:
    colors = (theme.acct, theme.static, theme.dynamic)
    parts: list[str] = []
    for idx, key in enumerate(keys):
        value = item.get(key)
        if isinstance(value, (int, float)):
            rendered = f"{float(value):.{decimals}f}{suffix}"
        else:
            reason = ""
            if reasons is not None:
                reason_value = str(item.get(reasons[idx]) or "").strip()
                if reason_value:
                    reason = f" ({reason_value})"
            rendered = f"N/A{reason}"
        parts.append(theme.c(rendered, colors[idx]))
    sep = theme.c("/", theme.sep)
    return sep.join(parts)


def _kv(theme: _AnsiTheme, key: str, value: str, value_color: str) -> str:
    return theme.c(key, theme.label) + theme.c("=", theme.sep) + theme.c(value, value_color)


def _join_semicolon(theme: _AnsiTheme, parts: list[str]) -> str:
    return theme.c(" ; ", theme.sep).join(parts)


def _truncate(text: str, max_len: int) -> str:
    raw = str(text or "").strip()
    if len(raw) <= max_len:
        return raw
    if max_len <= 3:
        return raw[:max_len]
    return raw[: max_len - 3] + "..."


def _table_num(value: Any, decimals: int, suffix: str = "") -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{decimals}f}{suffix}"
    return "N/A"


def _table_triplet(
    item: dict[str, Any],
    keys: tuple[str, str, str],
    *,
    decimals: int,
    suffix: str = "",
) -> str:
    vals = [item.get(k) for k in keys]
    return "/".join(_table_num(v, decimals, suffix=suffix) for v in vals)


def _freshman_action(item: dict[str, Any], process: str) -> tuple[str, str]:
    bucket = _landmark_bucket(item)
    if bucket == "qemail":
        return ("Tune QEMAIL scheduling to reduce close-window queue pressure", "direct")
    if bucket == "qpec":
        return ("Add one process server (QPEC+1) to reduce queue delay", "direct")
    if bucket == "payout":
        return ("Convert payout report chain to multi-input processing", "direct")

    action_type = _normalize_action_type(item)
    plugin_id = str(item.get("plugin_id") or "").strip()
    recommendation = str(item.get("recommendation") or "").strip()
    title = str(item.get("title") or "").strip()
    lowered = recommendation.lower()
    is_bundle = (
        plugin_id.startswith("analysis_action_search_")
        or lowered.startswith("execute the selected actions as one package")
        or lowered.startswith("execute the selected high-impact actions together")
    )
    if is_bundle:
        return ("Apply optimizer bundle and re-verify bottleneck shift", "bundle")
    if action_type in {"batch_input", "batch_input_refactor"} and process and process != "n/a":
        return (f"Convert {process} to batch-input execution", "direct")
    if action_type in {"batch_or_cache", "dedupe_or_cache", "throttle_or_dedupe"}:
        return ("Reduce duplicate reruns with cache/reuse strategy", "direct")
    if action_type in {"schedule_shift_target", "reschedule", "tune_schedule"}:
        return ("Shift schedule away from high-contention windows", "direct")
    if action_type in {"add_server", "capacity_addition"}:
        return ("Add targeted server capacity for close window", "direct")
    return (_truncate(title or recommendation or "Review and apply targeted optimization", 58), "direct")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--top-n", type=int, default=25)
    ap.add_argument("--max-per-plugin", type=int, default=5)
    ap.add_argument("--recall-top-n", type=int, default=20)
    ap.add_argument("--require-landmark-recall", action="store_true")
    ap.add_argument("--theme", choices=("auto", "cyberpunk", "plain"), default="auto")
    args = ap.parse_args()
    theme = _AnsiTheme(_theme_enabled(args.theme))

    run_id = str(args.run_id).strip()
    run_dir = APPDATA / "runs" / run_id
    report_path = run_dir / "report.json"
    if not report_path.exists():
        raise SystemExit(f"Missing report.json: {report_path}")

    report = _read_json(report_path)
    process_label_resolver = _build_numeric_process_resolver(report)
    known_issues_mode = str(report.get("known_issues_mode") or "on").strip().lower()
    if known_issues_mode not in {"on", "off"}:
        known_issues_mode = "on"
    recs = report.get("recommendations") if isinstance(report, dict) else None
    known_items, disc_items = _items_from_recs(recs)
    all_items = [
        item
        for item in (
            (recs.get("items") if isinstance(recs, dict) and isinstance(recs.get("items"), list) else [])
        )
        if isinstance(item, dict)
    ]
    if known_issues_mode != "off":
        known_items = _augment_known_issue_landmarks(report, known_items, disc_items)
        known_items = _collapse_known_items(known_items)
    else:
        known_items = []
    _ensure_answer_artifacts(run_id, run_dir, recs, known_items)
    route_map = _route_map_for_run(run_dir)
    ranked = _ranked_actionables(disc_items)
    require_landmark_recall = bool(args.require_landmark_recall) or (
        known_issues_mode != "off" and _bool_env(_REQUIRE_LANDMARK_RECALL_ENV, True)
    )
    items: list[dict[str, Any]] = []
    per_plugin: dict[str, int] = {}
    seen: set[str] = set()
    for item in ranked:
        sig = _selection_signature(item)
        if sig in seen:
            continue
        pid = str(item.get("plugin_id") or "")
        per_plugin[pid] = int(per_plugin.get(pid, 0)) + 1
        if per_plugin[pid] > int(args.max_per_plugin):
            continue
        seen.add(sig)
        items.append(item)
        if len(items) >= int(args.top_n):
            break
    items, promoted_landmarks = _promote_landmarks(items, ranked, int(args.top_n))

    def _item_process_label(item: dict[str, Any]) -> str:
        raw = str(
            item.get("primary_process_id")
            or item.get("process_id")
            or (item.get("where") or {}).get("process_norm")
            or _target_processes(item)
            or "n/a"
        ).strip()
        return _display_process_label(raw, item, process_label_resolver)

    def _item_raw_process(item: dict[str, Any]) -> str:
        return str(
            item.get("primary_process_id")
            or item.get("process_id")
            or (item.get("where") or {}).get("process_norm")
            or _target_processes(item)
            or "n/a"
        ).strip()

    print(theme.c("# Actionable Results", theme.title))
    print("")
    print(f"- run_id: {run_id}")
    print(f"- run_dir: {run_dir}")
    print(f"- known_issues_mode: {known_issues_mode}")
    for rel in ("report.md", "answers_recommendations.md", "answers_recommendations_plain.md", "answers_summary.json"):
        p = run_dir / rel
        print(f"- {rel}: {p if p.exists() else '(missing)'}")
    for rel in ("batch_sequence_validation_checklist.json", "batch_sequence_validation_checklist.md"):
        p = run_dir / rel
        print(f"- {rel}: {p if p.exists() else '(missing)'}")
    print("- audience_levels: report.md=technical | answers_recommendations.md=ops | answers_recommendations_plain.md=plain")
    runtime = _runtime_stats(run_id)
    if runtime:
        print("")
        print("## Runtime Trend")
        current = runtime.get("current_minutes")
        avg = runtime.get("history_avg_minutes")
        stddev = runtime.get("history_stddev_minutes")
        n = int(runtime.get("history_count") or 0)
        delta_pct = runtime.get("delta_vs_avg_percent")
        if isinstance(current, (int, float)):
            print(f"- current_minutes: {float(current):.2f}")
        print(f"- historical_samples: {n}")
        if isinstance(avg, (int, float)):
            print(f"- historical_avg_minutes: {float(avg):.2f}")
        if isinstance(stddev, (int, float)):
            print(f"- historical_stddev_minutes: {float(stddev):.2f}")
        if isinstance(delta_pct, (int, float)):
            print(f"- delta_vs_avg_percent: {float(delta_pct):+.1f}%")

    if known_items:
        status_counts: dict[str, int] = {}
        for item in known_items:
            status = str(item.get("status") or "unknown")
            status_counts[status] = int(status_counts.get(status, 0)) + 1
        failing = [i for i in known_items if str(i.get("status") or "unknown") != "confirmed"]
        print("")
        print("## Known-Issue Detection")
        print(f"- total_checks: {len(known_items)}")
        print(f"- confirmed: {status_counts.get('confirmed', 0)}")
        print(f"- failing: {len(failing)}")
        print("")
        print("| issue | status | plugin | observed_count | modeled_benefit | source |")
        print("|---|---|---|---:|---|---|")
        for item in known_items[:10]:
            title = _known_issue_label(item)
            status = str(item.get("status") or "unknown")
            plugin_id = str(item.get("plugin_id") or "")
            observed = item.get("observed_count")
            source = str(item.get("source") or "configured")
            matched = _match_known_finding(report, item)
            benefit = "n/a"
            close_pct = _as_float(item.get("modeled_close_percent"))
            general_pct = _as_float(item.get("modeled_general_percent"))
            close_h = _as_float(item.get("modeled_close_hours"))
            general_h = _as_float(item.get("modeled_general_hours"))
            if isinstance(close_pct, float) or isinstance(general_pct, float):
                parts: list[str] = []
                if isinstance(close_pct, float):
                    part = f"close={close_pct:.1f}%"
                    if isinstance(close_h, float):
                        part += f" ({close_h:.2f}h)"
                    parts.append(part)
                if isinstance(general_pct, float):
                    part = f"general={general_pct:.1f}%"
                    if isinstance(general_h, float):
                        part += f" ({general_h:.2f}h)"
                    parts.append(part)
                benefit = " | ".join(parts) if parts else benefit
            elif isinstance(matched, dict):
                pct = matched.get("modeled_reduction_pct")
                hours = matched.get("modeled_reduction_hours")
                if (not isinstance(pct, (int, float))) and isinstance(
                    matched.get("slowdown_ratio"), (int, float)
                ):
                    slowdown = float(matched.get("slowdown_ratio") or 0.0)
                    if slowdown > 1.0:
                        pct = 1.0 - (1.0 / slowdown)
                if (not isinstance(hours, (int, float))) and isinstance(
                    matched.get("median_duration_close"), (int, float)
                ) and isinstance(matched.get("median_duration_open"), (int, float)) and isinstance(
                    matched.get("close_count"), (int, float)
                ):
                    delta_s = max(
                        0.0,
                        float(matched.get("median_duration_close") or 0.0)
                        - float(matched.get("median_duration_open") or 0.0),
                    )
                    hours = (delta_s * float(matched.get("close_count") or 0.0)) / 3600.0
                if isinstance(pct, (int, float)) and isinstance(hours, (int, float)):
                    benefit = f"{float(pct) * 100.0:.1f}% / {float(hours):.2f}h"
                elif isinstance(pct, (int, float)):
                    benefit = f"{float(pct) * 100.0:.1f}%"
                elif isinstance(hours, (int, float)):
                    benefit = f"{float(hours):.2f}h"
            print(
                f"| {title} | {status} | `{plugin_id}` | {observed if observed is not None else 'n/a'} | {benefit} | {source} |"
            )
        if failing:
            print("- failing_items:")
            for item in failing[:10]:
                title = _known_issue_label(item)
                status = str(item.get("status") or "unknown")
                plugin_id = str(item.get("plugin_id") or "")
                observed = item.get("observed_count")
                print(
                    f"  - {title} (status={status}, plugin={plugin_id}, observed_count={observed})"
                )

    print("")
    print("## Ideaspace Route Map (Current -> Ideal)")
    if isinstance(route_map, dict) and bool(route_map.get("available")):
        ideal_mode = str(route_map.get("ideal_mode") or "").strip()
        if ideal_mode:
            print(f"- ideal_mode: {ideal_mode}")
        current = route_map.get("current") if isinstance(route_map.get("current"), dict) else {}
        if current:
            total = float(current.get("energy_total") or 0.0)
            gap = float(current.get("energy_gap") or 0.0)
            constraints = float(current.get("energy_constraints") or 0.0)
            print(f"- current_energy_total: {total:.4f}")
            print(f"- current_energy_gap: {gap:.4f}")
            print(f"- current_energy_constraints: {constraints:.4f}")
        actions = route_map.get("actions") if isinstance(route_map.get("actions"), list) else []
        if actions:
            print("- top_route_steps:")
            for idx, action in enumerate(actions[:8], start=1):
                if not isinstance(action, dict):
                    continue
                title = str(action.get("title") or action.get("action") or "Action").strip()
                lever = str(action.get("lever_id") or "").strip() or "unknown"
                target = str(action.get("target") or "").strip() or "ALL"
                delta = action.get("delta_energy")
                pct = action.get("modeled_percent")
                conf = action.get("confidence")
                parts = [f"step={idx}", f"lever={lever}", f"target={target}"]
                if isinstance(pct, (int, float)):
                    parts.append(f"modeled_pct={float(pct):.2f}%")
                if isinstance(delta, (int, float)):
                    parts.append(f"delta_energy={float(delta):.4f}")
                if isinstance(conf, (int, float)):
                    parts.append(f"confidence={float(conf):.2f}")
                print(f"  - {title} ({', '.join(parts)})")
        else:
            print("- top_route_steps: none")
    else:
        summary = str(route_map.get("summary") or "").strip() if isinstance(route_map, dict) else ""
        if summary:
            print(f"- summary: {summary}")
        print("- top_route_steps: none")

    print("")
    if not items:
        print("No discovery recommendations found in report.json.")
        return 2 if args.require_landmark_recall and known_issues_mode != "off" else 0

    top_n = min(int(args.top_n), len(items))
    selected = items[:top_n]
    ranked_positions = _landmark_positions(ranked)
    selected_positions = _landmark_positions(selected)
    recall_window = max(1, int(args.recall_top_n))
    recall_gate_failed = False

    if known_issues_mode != "off":
        print("")
        print("## Known-Issue Recall QA (Unpinned Ranking)")
        print(f"- recall_window_top_n: {recall_window}")
        for bucket in _LANDMARK_BUCKETS:
            pos_ranked = ranked_positions.get(bucket)
            pos_selected = selected_positions.get(bucket)
            in_ranked = isinstance(pos_ranked, int)
            in_window = isinstance(pos_ranked, int) and int(pos_ranked) <= recall_window
            print(
                f"- {_status_yn(f'{bucket}_present_in_ranked', in_ranked)} ; "
                f"ranked_position={pos_ranked if isinstance(pos_ranked, int) else 'N/A'} ; "
                f"selected_position={pos_selected if isinstance(pos_selected, int) else 'N/A'}"
            )
            print(
                f"- {_status_yn(f'{bucket}_within_top_{recall_window}', in_window)} ; "
                f"ranked_position={pos_ranked if isinstance(pos_ranked, int) else 'N/A'}"
            )
            if require_landmark_recall and not in_window:
                recall_gate_failed = True
        if require_landmark_recall:
            print(f"- require_landmark_recall: true")
            print(f"- recall_gate_status: {'FAIL' if recall_gate_failed else 'PASS'}")
        if promoted_landmarks:
            print(f"- landmark_promoted_from_ranked: {', '.join(promoted_landmarks)}")

    print("")
    print(theme.c("## Action Lanes (Client-Controllable)", theme.section))
    lane_rows: dict[str, list[dict[str, Any]]] = {"manual": [], "infra": []}
    for item in items:
        lane = _lane_bucket(item)
        if lane in lane_rows and len(lane_rows[lane]) < 5:
            lane_rows[lane].append(item)
    for lane, label in (("manual", "Manual Effort Removal"), ("infra", "Infrastructure Contention Reduction")):
        print(theme.c(f"### {label}", theme.title))
        rows = lane_rows.get(lane) or []
        if not rows:
            print("- none")
            continue
        for item in rows:
            raw_process = _item_raw_process(item)
            process = _item_process_label(item)
            rec_raw = str(item.get("recommendation") or item.get("title") or "").strip()
            rec = _truncate(_humanize_recommendation_process(rec_raw, raw_process, process), 90)
            close_delta = item.get("modeled_delta_hours_close_cycle")
            if not isinstance(close_delta, (int, float)):
                close_delta = item.get("delta_hours_close_dynamic")
            human_touches = item.get("modeled_user_touches_reduced")
            if not isinstance(human_touches, (int, float)):
                human_touches = item.get("human_modeled_touch_reduction_count_close_cycle")
            parts = [
                _kv(theme, "process", process, theme.cool),
                _kv(theme, "close_Δh", f"{float(close_delta):.2f}" if isinstance(close_delta, (int, float)) else "N/A", theme.dynamic if isinstance(close_delta, (int, float)) else theme.dim),
                _kv(theme, "human_touches", f"{float(human_touches):.0f}" if isinstance(human_touches, (int, float)) else "N/A", theme.hot if isinstance(human_touches, (int, float)) else theme.dim),
            ]
            print("- " + _join_semicolon(theme, parts))
            print("  " + theme.c(rec, theme.value))

    print("")
    print(theme.c(f"## Freshman Table (Top {top_n})", theme.section))
    print(
        "| # | process | action | class | Δh acct/static/dyn | eff_idx acct/static/dyn | close Δh | close eff% | human runs | human touches |"
    )
    print("|---:|---|---|---|---|---|---:|---:|---:|---:|")
    for idx, item in enumerate(selected, start=1):
        process = _item_process_label(item)
        action_text, action_class = _freshman_action(item, process)
        delta_triplet = _table_triplet(
            item,
            ("delta_hours_accounting_month", "delta_hours_close_static", "delta_hours_close_dynamic"),
            decimals=2,
        )
        eff_idx_triplet = _table_triplet(
            item,
            ("efficiency_gain_accounting_month", "efficiency_gain_close_static", "efficiency_gain_close_dynamic"),
            decimals=6,
        )
        close_delta = item.get("modeled_delta_hours_close_cycle")
        if not isinstance(close_delta, (int, float)):
            close_delta = item.get("delta_hours_close_dynamic")
        close_eff = item.get("modeled_efficiency_gain_pct_close_cycle")
        if not isinstance(close_eff, (int, float)):
            close_eff = item.get("efficiency_gain_pct_close_dynamic")
        human_runs = item.get("modeled_user_runs_reduced")
        if not isinstance(human_runs, (int, float)):
            human_runs = item.get("human_modeled_run_reduction_count_close_cycle")
        human_touches = item.get("modeled_user_touches_reduced")
        if not isinstance(human_touches, (int, float)):
            human_touches = item.get("human_modeled_touch_reduction_count_close_cycle")
        print(
            f"| {idx} | {_truncate(process, 18)} | {_truncate(action_text, 58)} | {action_class} | "
            f"{delta_triplet} | {eff_idx_triplet} | {_table_num(close_delta, 2)} | {_table_num(close_eff, 3, suffix='%')} | "
            f"{_table_num(human_runs, 0)} | {_table_num(human_touches, 0)} |"
        )

    group_order = [
        "landmark",
        "batch_group_candidate",
        "batch_or_cache",
        "batch_input",
        "capacity",
        "schedule",
        "handoff",
        "other",
    ]
    grouped: dict[str, dict[str, Any]] = {}
    for idx, item in enumerate(selected, start=1):
        key, label = _recommendation_group(item)
        bucket = grouped.setdefault(key, {"label": label, "rows": []})
        bucket["rows"].append((idx, item))

    def _group_sort_key(k: str) -> tuple[int, str]:
        try:
            return (group_order.index(k), k)
        except ValueError:
            return (len(group_order), k)

    requested_top_n = int(args.top_n)
    if top_n < requested_top_n:
        print(theme.c(f"## Top {requested_top_n} Recommendations (Grouped by Kind, showing {top_n} available)", theme.section))
    else:
        print(theme.c(f"## Top {top_n} Recommendations (Grouped by Kind)", theme.section))
    for key in sorted(grouped.keys(), key=_group_sort_key):
        block = grouped[key]
        label = str(block.get("label") or key)
        rows = block.get("rows") if isinstance(block.get("rows"), list) else []
        print("")
        print(theme.c(f"### {label} ({len(rows)})", theme.title))
        for idx, item in rows:
            txt = str(item.get("recommendation") or item.get("title") or "").strip()
            if not txt:
                continue
            plugin_id = str(item.get("plugin_id") or "")
            kind = str(item.get("kind") or "")
            obvious_rank = str(item.get("obviousness_rank") or "").strip()
            obvious_score = item.get("obviousness_score")
            obvious_txt = ""
            if obvious_rank:
                if isinstance(obvious_score, (int, float)):
                    obvious_txt = f"{obvious_rank}:{float(obvious_score):.2f}"
                else:
                    obvious_txt = obvious_rank
            targets = _target_processes(item)
            score_value = item.get("client_value_score")
            if not isinstance(score_value, (int, float)):
                score_value = item.get("value_score_v2")
            scope_class = str(item.get("scope_class") or "").strip()
            raw_process = _item_raw_process(item)
            process = _item_process_label(item)
            txt = _humanize_recommendation_process(txt, raw_process, process)
            wrapped = textwrap.fill(
                txt,
                width=110,
                initial_indent=f"{idx}. ",
                subsequent_indent="   ",
            )
            print(theme.c(wrapped, theme.value))
            info_parts = [
                theme.c("score", theme.label) + theme.c("=", theme.sep) + theme.score(score_value),
                _kv(theme, "process", process, theme.cool),
                _kv(theme, "scope", scope_class or "n/a", theme.value),
            ]
            print("   " + _join_semicolon(theme, info_parts))
            print(
                "   "
                + theme.c("delta_h (acct|static|dyn)", theme.label)
                + ": "
                + _window_triplet(
                    theme,
                    item,
                    ("delta_hours_accounting_month", "delta_hours_close_static", "delta_hours_close_dynamic"),
                    decimals=2,
                    reasons=("na_reason_accounting_month", "na_reason_close_static", "na_reason_close_dynamic"),
                )
            )
            print(
                "   "
                + theme.c("eff_% (acct|static|dyn)", theme.label)
                + ": "
                + _window_triplet(
                    theme,
                    item,
                    (
                        "efficiency_gain_pct_accounting_month",
                        "efficiency_gain_pct_close_static",
                        "efficiency_gain_pct_close_dynamic",
                    ),
                    decimals=3,
                    suffix="%",
                    reasons=("na_reason_accounting_month", "na_reason_close_static", "na_reason_close_dynamic"),
                )
            )
            print(
                "   "
                + theme.c("eff_idx (acct|static|dyn)", theme.label)
                + ": "
                + _window_triplet(
                    theme,
                    item,
                    ("efficiency_gain_accounting_month", "efficiency_gain_close_static", "efficiency_gain_close_dynamic"),
                    decimals=6,
                    reasons=("na_reason_accounting_month", "na_reason_close_static", "na_reason_close_dynamic"),
                )
            )
            close_delta = item.get("modeled_delta_hours_close_cycle")
            if not isinstance(close_delta, (int, float)):
                close_delta = item.get("delta_hours_close_dynamic")
            close_eff_pct = item.get("modeled_efficiency_gain_pct_close_cycle")
            if not isinstance(close_eff_pct, (int, float)):
                close_eff_pct = item.get("efficiency_gain_pct_close_dynamic")
            close_parts = [
                _kv(theme, "delta_h_close", f"{float(close_delta):.2f}" if isinstance(close_delta, (int, float)) else "N/A", theme.dynamic if isinstance(close_delta, (int, float)) else theme.dim),
                _kv(theme, "eff_pct_close", f"{float(close_eff_pct):.3f}%" if isinstance(close_eff_pct, (int, float)) else "N/A", theme.dynamic if isinstance(close_eff_pct, (int, float)) else theme.dim),
            ]
            print("   " + _join_semicolon(theme, close_parts))
            human_parts = [
                _kv(theme, "human_h_close", f"{float(item.get('modeled_user_hours_saved_close_cycle')):.2f}" if isinstance(item.get("modeled_user_hours_saved_close_cycle"), (int, float)) else "N/A", theme.hot if isinstance(item.get("modeled_user_hours_saved_close_cycle"), (int, float)) else theme.dim),
                _kv(theme, "human_runs", f"{float(item.get('modeled_user_runs_reduced')):.0f}" if isinstance(item.get("modeled_user_runs_reduced"), (int, float)) else "N/A", theme.hot if isinstance(item.get("modeled_user_runs_reduced"), (int, float)) else theme.dim),
                _kv(theme, "human_touches", f"{float(item.get('modeled_user_touches_reduced')):.0f}" if isinstance(item.get("modeled_user_touches_reduced"), (int, float)) else "N/A", theme.hot if isinstance(item.get("modeled_user_touches_reduced"), (int, float)) else theme.dim),
                _kv(theme, "human_status", str(item.get("human_gain_status") or "missing"), theme.value),
                _kv(theme, "human_reason", str(item.get("human_gain_reason_code") or item.get("human_gain_reason") or "missing"), theme.dim),
            ]
            print("   " + _join_semicolon(theme, human_parts))
            tail_parts: list[str] = []
            if obvious_txt:
                tail_parts.append(f"obviousness={obvious_txt}")
            if plugin_id:
                tail_parts.append(f"plugin={plugin_id}")
            if kind:
                tail_parts.append(f"kind={kind}")
            if tail_parts:
                print("   " + _join_semicolon(theme, [theme.c(part, theme.dim) for part in tail_parts]))
    print("")
    if require_landmark_recall and recall_gate_failed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
