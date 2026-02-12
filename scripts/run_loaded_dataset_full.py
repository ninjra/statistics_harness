from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from statistic_harness.core.pipeline import Pipeline
from statistic_harness.core.tenancy import get_tenant_context
from statistic_harness.core.utils import make_run_id


REPO_ROOT = Path(__file__).resolve().parents[1]


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
            SELECT dataset_version_id, dataset_id, created_at, table_name, row_count, column_count, data_hash
            FROM dataset_versions
            ORDER BY row_count DESC, created_at DESC
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
            SELECT dataset_version_id, dataset_id, created_at, table_name, row_count, column_count, data_hash
            FROM dataset_versions
            WHERE dataset_version_id = ?
            LIMIT 1
            """,
            (dataset_version_id,),
        ).fetchone()
        return dict(row) if row else None
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
    recs: dict[str, Any], known_checks: dict[str, Any], route_map: dict[str, Any] | None = None, max_items: int = 40
) -> str:
    summary = str(recs.get("summary") or "").strip()
    lines = []
    lines.append("# Recommendations (From report.json)")
    if summary:
        lines.append("")
        lines.append(summary)

    def _as_items(block: Any) -> list[dict[str, Any]]:
        if isinstance(block, dict) and isinstance(block.get("items"), list):
            return [i for i in block["items"] if isinstance(i, dict)]
        if isinstance(block, list):
            return [i for i in block if isinstance(i, dict)]
        return []

    known = _as_items(recs.get("known"))
    discovery = _as_items(recs.get("discovery"))
    flat = _as_items(recs.get("items"))

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
    value = item.get("modeled_delta_hours")
    if not isinstance(value, (int, float)) or float(value) <= 0:
        value = item.get("impact_hours")
    if not isinstance(value, (int, float)) or float(value) <= 0:
        value = item.get("modeled_delta")
    if isinstance(value, (int, float)) and float(value) > 0:
        return f"about {float(value):.2f} hours"
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
        target_list = ", ".join(
            [str(t).strip() for t in targets if isinstance(t, str) and t.strip()]
        ) if isinstance(targets, list) else process
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


def _render_recommendations_plain_md(
    recs: dict[str, Any], known_checks: dict[str, Any], route_map: dict[str, Any] | None = None, max_items: int = 30
) -> str:
    lines: list[str] = []
    lines.append("# Recommendations (Plain Language)")
    lines.append("")
    lines.append("This version is written for easier reading (high school to early college level).")

    def _as_items(block: Any) -> list[dict[str, Any]]:
        if isinstance(block, dict) and isinstance(block.get("items"), list):
            return [i for i in block["items"] if isinstance(i, dict)]
        if isinstance(block, list):
            return [i for i in block if isinstance(i, dict)]
        return []

    known = _as_items(recs.get("known"))
    discovery = _as_items(recs.get("discovery"))
    flat = _as_items(recs.get("items"))
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
        lines.append("")
        lines.append(f"## {title}")
        for idx, item in enumerate(items[:max_items], start=1):
            change, why = _plain_recommendation_fields(item)
            lines.append(f"{idx}. What to change: {change}")
            lines.append(f"   Why this matters: {why}")
            lines.append(f"   Expected benefit: {_plain_hours(item)}")
            modeled_pct = item.get("modeled_percent")
            if isinstance(modeled_pct, (int, float)):
                lines.append(f"   Modeled improvement percent: {float(modeled_pct):.2f}%")
            else:
                reason = str(item.get("not_modeled_reason") or "").strip()
                if reason:
                    lines.append(f"   Modeled improvement percent: not available ({reason})")
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
    args = parser.parse_args()

    # User-facing completeness: include known-issue recommendations in report synthesis.
    os.environ.setdefault("STAT_HARNESS_INCLUDE_KNOWN_RECOMMENDATIONS", "1")
    os.environ.setdefault("STAT_HARNESS_CLI_PROGRESS", "1")
    # Default to reuse-cache for operator UX on large datasets. Still safe for "updated plugins"
    # because cache keys include plugin code hash + settings hash + dataset hash.
    os.environ.setdefault("STAT_HARNESS_REUSE_CACHE", "1")

    ctx = get_tenant_context()
    db_path = ctx.appdata_root / "state.sqlite"
    if not db_path.exists():
        raise SystemExit(f"Missing DB: {db_path}")

    requested = str(args.dataset_version_id or "").strip()
    if requested:
        dataset = _dataset_version_row(db_path, requested)
        if not dataset:
            raise SystemExit(f"Dataset version not found: {requested}")
    else:
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

    # Large datasets: cap parallelism unless explicitly overridden.
    if row_count is not None and row_count >= 200_000:
        os.environ.setdefault("STAT_HARNESS_MAX_WORKERS_ANALYSIS", "2")

    plugin_ids: list[str]
    if args.plugin_set == "auto":
        plugin_ids = ["auto"]
    else:
        # Full harness run: execute every non-ingest plugin on the loaded dataset.
        # (Ingest is file-driven and is skipped for DB-only runs.)
        profiles = _discover_plugin_ids({"profile"})
        planners = _discover_plugin_ids({"planner"})
        transforms = _discover_plugin_ids({"transform"})
        analyses = _discover_plugin_ids({"analysis"})
        reports = _discover_plugin_ids({"report"})
        llm = _discover_plugin_ids({"llm"})
        plugin_ids = [*profiles, *planners, *transforms, *analyses, *reports, *llm]

    pipeline = Pipeline(ctx.appdata_root, Path("plugins"), tenant_id=ctx.tenant_id)
    run_id = pipeline.run(
        input_file=None,
        plugin_ids=plugin_ids,
        settings={"exclude_processes": exclude_processes},
        run_seed=int(args.run_seed),
        dataset_version_id=dataset_version_id,
        run_id=run_id,
        force=bool(args.force),
    )
    run_dir = ctx.tenant_root / "runs" / run_id
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

    answers = {
        "dataset": dataset,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "report_path": str(report_path),
        "runtime_trend": runtime_trend,
        "exclude_processes": exclude_processes,
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
        _render_recommendations_md(recs, known_checks, route_map=route_map),
    )
    _write_text(
        run_dir / "answers_recommendations_plain.md",
        _render_recommendations_plain_md(recs, known_checks, route_map=route_map),
    )

    print(f"DATASET_VERSION_ID={dataset_version_id}")
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
    print(str(run_dir / "report.md"))
    print(str(run_dir / "answers_recommendations.md"))
    print(str(run_dir / "answers_recommendations_plain.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
