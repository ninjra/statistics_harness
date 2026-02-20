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
        close_pct = _as_float(item.get("modeled_close_percent"))
        general_pct = _as_float(item.get("modeled_general_percent"))
        generic_pct = _as_float(item.get("modeled_percent"))
        scope = str(item.get("scope_class") or "").strip().lower()
        if close_pct is None and isinstance(generic_pct, float) and scope == "close_specific":
            close_pct = generic_pct
        if general_pct is None and isinstance(generic_pct, float) and scope != "close_specific":
            general_pct = generic_pct
        close_h = _as_float(item.get("modeled_delta_hours")) if scope == "close_specific" else None
        general_h = _as_float(item.get("modeled_delta_hours")) if scope != "close_specific" else None
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

    # Landmark A: QEMAIL scheduling/contention issue.
    qemail_model = _best_qemail_model(
        _qemail_modeled(discovery_items),
        _qemail_modeled_from_known(known_items),
        _qemail_modeled_from_report(report),
    )
    qemail_close_pct = _as_float(qemail_model.get("close_pct"))
    qemail_general_pct = _as_float(qemail_model.get("general_pct"))
    # Known-issue pass criterion is modeled-impact based, not text-hit based.
    # Require >=10% either in close-specific or general scope.
    qemail_hits = 1 if ((qemail_close_pct or 0.0) >= 10.0 or (qemail_general_pct or 0.0) >= 10.0) else 0
    qemail_plugin = str(qemail_model.get("plugin_id") or "")
    qemail_status = _status_from_hits(qemail_hits, min_count=1, max_count=1)
    known.append(
        {
            "title": _KONA_QEMAIL_TITLE,
            "status": qemail_status,
            "plugin_id": qemail_plugin or "analysis_close_cycle_contention",
            "kind": "known_issue_landmark",
            "where": {"process": "qemail"},
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

    def _priority(item: dict[str, Any]) -> int:
        plugin_id = str(item.get("plugin_id") or "")
        kind = str(item.get("kind") or "")
        action_type = str(item.get("action_type") or item.get("action") or "")
        if plugin_id == "analysis_actionable_ops_levers_v1" or kind == "actionable_ops_lever":
            return 6
        if plugin_id.startswith("analysis_ideaspace_"):
            return 5
        if "sequence" in plugin_id or "bottleneck" in plugin_id or "conformance" in plugin_id:
            return 4
        if plugin_id == "analysis_upload_linkage":
            return 3
        if action_type and action_type not in ("review", "tune_threshold"):
            return 2
        if plugin_id in ("analysis_queue_delay_decomposition", "analysis_busy_period_segmentation_v2"):
            return 1
        return 0

    def _score(item: dict[str, Any]) -> float:
        for key in ("relevance_score", "impact_hours", "modeled_delta"):
            try:
                v = item.get(key)
                if isinstance(v, (int, float)):
                    return float(v)
            except Exception:
                continue
        return 0.0

    ranked = sorted(disc_items, key=lambda i: (_priority(i), _score(i)), reverse=True)
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


def _normalize_action_type(item: dict[str, Any]) -> str:
    return str(item.get("action_type") or item.get("action") or "").strip().lower()


def _recommendation_group(item: dict[str, Any]) -> tuple[str, str]:
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--top-n", type=int, default=25)
    ap.add_argument("--max-per-plugin", type=int, default=5)
    ap.add_argument("--theme", choices=("auto", "cyberpunk", "plain"), default="auto")
    args = ap.parse_args()
    theme = _AnsiTheme(_theme_enabled(args.theme))

    run_id = str(args.run_id).strip()
    run_dir = APPDATA / "runs" / run_id
    report_path = run_dir / "report.json"
    if not report_path.exists():
        raise SystemExit(f"Missing report.json: {report_path}")

    report = _read_json(report_path)
    known_issues_mode = str(report.get("known_issues_mode") or "on").strip().lower()
    if known_issues_mode not in {"on", "off"}:
        known_issues_mode = "on"
    recs = report.get("recommendations") if isinstance(report, dict) else None
    known_items, disc_items = _items_from_recs(recs)
    if known_issues_mode != "off":
        known_items = _augment_known_issue_landmarks(report, known_items, disc_items)
        known_items = _collapse_known_items(known_items)
    else:
        known_items = []
    _ensure_answer_artifacts(run_id, run_dir, recs, known_items)
    route_map = _route_map_for_run(run_dir)
    ranked = _ranked_actionables(disc_items)
    items: list[dict[str, Any]] = []
    per_plugin: dict[str, int] = {}
    for item in ranked:
        pid = str(item.get("plugin_id") or "")
        per_plugin[pid] = int(per_plugin.get(pid, 0)) + 1
        if per_plugin[pid] > int(args.max_per_plugin):
            continue
        items.append(item)
        if len(items) >= int(args.top_n):
            break

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
        return 0

    top_n = min(int(args.top_n), len(items))
    selected = items[:top_n]
    group_order = [
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
            process = str(
                item.get("primary_process_id")
                or item.get("process_id")
                or (item.get("where") or {}).get("process_norm")
                or targets
                or "n/a"
            ).strip()
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
