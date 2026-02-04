from __future__ import annotations

import csv
import json
import re
import os
from pathlib import Path
from typing import Any

from jsonschema import validate
import yaml

from .stat_controls import confidence_from_p


def _matches_expected(
    item: dict[str, Any],
    where: dict[str, Any] | None,
    contains: dict[str, Any] | None,
) -> bool:
    if where:
        for key, expected in where.items():
            actual = item.get(key)
            if actual != expected:
                return False
    if contains:
        for key, expected in contains.items():
            actual = item.get(key)
            if isinstance(actual, str):
                if str(expected) not in actual:
                    return False
            elif isinstance(actual, (list, tuple, set)):
                if isinstance(expected, (list, tuple, set)):
                    if not set(expected).issubset(set(actual)):
                        return False
                else:
                    if expected not in actual:
                        return False
            else:
                return False
    return True


def _collect_findings_for_plugin(
    report: dict[str, Any], plugin_id: str | None, kind: str | None = None
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    plugins = report.get("plugins", {}) or {}
    for pid, plugin in plugins.items():
        if plugin_id and pid != plugin_id:
            continue
        for item in plugin.get("findings", []) or []:
            if kind and item.get("kind") != kind:
                continue
            if isinstance(item, dict):
                findings.append(item)
    return findings


def _process_hint(where: dict[str, Any] | None) -> str:
    if not where:
        return ""
    for key in ("process", "process_norm", "process_name", "activity"):
        value = where.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _recommendation_text(status: str, label: str, process_hint: str) -> str:
    suffix = f" (process {process_hint})" if process_hint else ""
    if status == "confirmed":
        return f"Act on {label}{suffix}."
    if status == "over_limit":
        return f"Investigate excess occurrences of {label}{suffix}."
    if status in {"missing", "below_min"}:
        return f"Missing evidence for {label}{suffix}; check inputs and re-run."
    return f"Review {label}{suffix}."


def _capacity_scale_recommendation(
    kind: str | None, matched: list[dict[str, Any]], label: str, process_hint: str
) -> tuple[str | None, dict[str, Any]]:
    if not kind or not matched:
        return None, {}
    suffix = f" (process {process_hint})" if process_hint else ""
    item = matched[0]
    meta: dict[str, Any] = {"action": None, "modeled_delta": None}
    if kind == "capacity_scale_model":
        base = item.get("eligible_wait_gt_hours_without_target")
        modeled = item.get("eligible_wait_gt_hours_modeled")
        scale = item.get("scale_factor")
        if isinstance(base, (int, float)) and isinstance(modeled, (int, float)):
            delta = float(base) - float(modeled)
            scale_text = f" scale_factor={scale:.3f}" if isinstance(scale, (int, float)) else ""
            meta = {"action": "add_one_server", "modeled_delta": delta}
            return (
                f"Add one server{suffix}: modeled >threshold eligible-wait drops "
                f"from {float(base):.3f}h to {float(modeled):.3f}h (Δ {delta:.3f}h){scale_text}."
            ), meta
    if kind == "capacity_scaling":
        base = item.get("baseline_wait_hours")
        modeled = item.get("modeled_wait_hours")
        reduction = item.get("reduction_hours")
        scale = item.get("scale_factor")
        if isinstance(base, (int, float)) and isinstance(modeled, (int, float)):
            reduction_val = reduction if isinstance(reduction, (int, float)) else float(base) - float(modeled)
            scale_text = f" scale_factor={scale:.3f}" if isinstance(scale, (int, float)) else ""
            meta = {"action": "add_one_server", "modeled_delta": float(reduction_val)}
            return (
                f"Add one server{suffix}: eligible-wait drops from "
                f"{float(base):.3f}h to {float(modeled):.3f}h (Δ {float(reduction_val):.3f}h){scale_text}."
            ), meta
    if kind == "close_cycle_capacity_impact":
        effect = item.get("effect")
        decision = str(item.get("decision") or "")
        if isinstance(effect, (int, float)) and decision == "detected":
            pct = abs(float(effect)) * 100.0
            meta = {"action": "add_one_server", "modeled_delta": None}
            return f"Add one server{suffix}: median close-cycle improves by ~{pct:.1f}% (measured).", meta
    return None, {}


def _dedupe_recommendations(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, float], dict[str, Any]] = {}
    passthrough: list[dict[str, Any]] = []
    for item in items:
        action = item.get("action")
        delta = item.get("modeled_delta")
        if not action or delta is None:
            passthrough.append(item)
            continue
        try:
            delta_key = round(float(delta), 3)
        except (TypeError, ValueError):
            passthrough.append(item)
            continue
        key = (str(action), delta_key)
        if key not in merged:
            merged_item = dict(item)
            merged_item["merged_titles"] = [item.get("title") or ""]
            merged[key] = merged_item
            continue
        current = merged[key]
        current["merged_titles"].append(item.get("title") or "")
        current["observed_count"] = int(current.get("observed_count") or 0) + int(
            item.get("observed_count") or 0
        )
        if isinstance(current.get("evidence"), list) and isinstance(
            item.get("evidence"), list
        ):
            merged_evidence = current["evidence"] + item["evidence"]
            current["evidence"] = merged_evidence[:3]
    return list(merged.values()) + passthrough


def _build_recommendations(report: dict[str, Any]) -> dict[str, Any]:
    known = report.get("known_issues")
    if not isinstance(known, dict):
        return {
            "status": "no_known_issues",
            "summary": "No known issues attached; recommendations not generated.",
            "items": [],
        }
    expected = known.get("expected_findings") or []
    if not isinstance(expected, list) or not expected:
        return {
            "status": "no_expected_findings",
            "summary": "Known issues attached but no expected findings provided.",
            "items": [],
        }

    items: list[dict[str, Any]] = []
    for issue in expected:
        if not isinstance(issue, dict):
            continue
        plugin_id = issue.get("plugin_id")
        kind = issue.get("kind")
        where = issue.get("where") if isinstance(issue.get("where"), dict) else None
        contains = (
            issue.get("contains") if isinstance(issue.get("contains"), dict) else None
        )
        min_count = issue.get("min_count")
        max_count = issue.get("max_count")
        title = issue.get("title") or issue.get("description") or ""
        if not title:
            label = f"{plugin_id or 'any'}:{kind or 'finding'}"
        else:
            label = title

        findings = _collect_findings_for_plugin(report, plugin_id, kind)
        matched = [f for f in findings if _matches_expected(f, where, contains)]
        count = len(matched)

        status = "confirmed"
        if count == 0:
            status = "missing"
        if min_count is not None:
            try:
                if count < int(min_count):
                    status = "below_min"
            except (TypeError, ValueError):
                pass
        if max_count is not None:
            try:
                if count > int(max_count):
                    status = "over_limit"
            except (TypeError, ValueError):
                pass

        process_hint = _process_hint(where)
        recommendation, meta = _capacity_scale_recommendation(
            kind, matched, label, process_hint
        )
        if not recommendation:
            recommendation = _recommendation_text(status, label, process_hint)
            meta = {"action": None, "modeled_delta": None}

        evidence: list[dict[str, Any]] = []
        for item in matched[:3]:
            snippet: dict[str, Any] = {"kind": item.get("kind")}
            for key in ("feature", "pair", "row_index", "index", "score", "metric"):
                if key in item:
                    snippet[key] = item.get(key)
            evidence.append(snippet)

        items.append(
            {
                "title": label,
                "status": status,
                "recommendation": recommendation,
                "plugin_id": plugin_id,
                "kind": kind,
                "where": where,
                "contains": contains,
                "expected": {"min_count": min_count, "max_count": max_count},
                "observed_count": count,
                "evidence": evidence,
                "action": meta.get("action"),
                "modeled_delta": meta.get("modeled_delta"),
            }
        )
    deduped = _dedupe_recommendations(items)
    return {
        "status": "ok",
        "summary": f"Generated {len(deduped)} recommendation(s) from known issues.",
        "items": deduped,
    }


def _build_executive_summary(report: dict[str, Any]) -> list[str]:
    plugins = report.get("plugins", {}) or {}
    queue_plugin = plugins.get("analysis_queue_delay_decomposition")
    if not isinstance(queue_plugin, dict):
        return []
    findings = queue_plugin.get("findings") or []
    qemail_stats = None
    impact = None
    scale = None
    for item in findings:
        if not isinstance(item, dict):
            continue
        kind = item.get("kind")
        if kind == "eligible_wait_process_stats" and item.get("process_norm") == "qemail":
            qemail_stats = item
        elif kind == "eligible_wait_impact":
            impact = item
        elif kind == "capacity_scale_model":
            scale = item

    lines: list[str] = []
    if qemail_stats and impact:
        q_gt = qemail_stats.get("eligible_wait_gt_hours_total")
        total_gt = impact.get("eligible_wait_gt_hours_total")
        runs_total = qemail_stats.get("runs_total")
        if isinstance(q_gt, (int, float)) and isinstance(total_gt, (int, float)) and total_gt:
            share = (float(q_gt) / float(total_gt)) * 100.0
            runs_text = f" across {int(runs_total):,} runs" if isinstance(runs_total, (int, float)) else ""
            lines.append(
                "QEMAIL is a major close-cycle drag: "
                f"{float(q_gt):.2f}h of >threshold eligible wait out of "
                f"{float(total_gt):.2f}h total ({share:.1f}%).{runs_text}"
            )

    if scale:
        base = scale.get("eligible_wait_gt_hours_without_target")
        modeled = scale.get("eligible_wait_gt_hours_modeled")
        if isinstance(base, (int, float)) and isinstance(modeled, (int, float)) and base:
            delta = float(base) - float(modeled)
            pct = (delta / float(base)) * 100.0 if base else 0.0
            lines.append(
                "QPEC+1 recommended (modeled): "
                f">threshold eligible wait drops from {float(base):.2f}h to "
                f"{float(modeled):.2f}h (Δ {delta:.2f}h, {pct:.1f}%)."
            )

    return lines


def _evaluate_known_issues(report: dict[str, Any]) -> list[dict[str, Any]]:
    known = report.get("known_issues")
    if not isinstance(known, dict):
        return []
    expected = known.get("expected_findings") or []
    if not isinstance(expected, list) or not expected:
        return []
    evaluations: list[dict[str, Any]] = []
    for issue in expected:
        if not isinstance(issue, dict):
            continue
        plugin_id = issue.get("plugin_id")
        kind = issue.get("kind")
        where = issue.get("where") if isinstance(issue.get("where"), dict) else None
        contains = issue.get("contains") if isinstance(issue.get("contains"), dict) else None
        min_count = issue.get("min_count")
        max_count = issue.get("max_count")
        title = issue.get("title") or issue.get("description") or ""
        label = title or f"{plugin_id or 'any'}:{kind or 'finding'}"

        findings = _collect_findings_for_plugin(report, plugin_id, kind)
        matched = [f for f in findings if _matches_expected(f, where, contains)]
        count = len(matched)

        status = "confirmed"
        if count == 0:
            status = "missing"
        if min_count is not None:
            try:
                if count < int(min_count):
                    status = "below_min"
            except (TypeError, ValueError):
                pass
        if max_count is not None:
            try:
                if count > int(max_count):
                    status = "over_limit"
            except (TypeError, ValueError):
                pass

        evaluations.append(
            {
                "issue": issue,
                "label": label,
                "plugin_id": plugin_id,
                "kind": kind,
                "where": where,
                "contains": contains,
                "min_count": min_count,
                "max_count": max_count,
                "matched": matched,
                "count": count,
                "status": status,
                "process_hint": _process_hint(where),
            }
        )
    return evaluations


def _format_issue_value(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if isinstance(value, int):
            return f"{value}"
        if float(value).is_integer():
            return f"{int(value)}"
        return f"{float(value):.{digits}f}"
    return str(value)


def _artifact_paths(report: dict[str, Any], plugin_id: str | None) -> list[str]:
    if not plugin_id:
        return []
    plugin = report.get("plugins", {}).get(plugin_id) or {}
    artifacts = plugin.get("artifacts") or []
    paths = []
    for artifact in artifacts:
        if isinstance(artifact, dict):
            path = artifact.get("path")
            if isinstance(path, str) and path:
                paths.append(path)
    return paths


def _metric_spec(kind: str | None) -> dict[str, Any]:
    return {
        "eligible_wait_process_stats": {
            "name": "Eligible wait > threshold (hours)",
            "definition": "Total hours of eligible wait above the threshold for the process.",
            "baseline_field": "eligible_wait_gt_hours_total",
            "observed_field": "eligible_wait_gt_hours_total",
            "denominator_field": "runs_total",
        },
        "eligible_wait_impact": {
            "name": "Eligible wait > threshold (hours)",
            "definition": "Total hours of eligible wait above the threshold (all runs).",
            "baseline_field": "eligible_wait_gt_hours_total",
            "observed_field": "eligible_wait_gt_hours_without_target",
            "denominator_field": "runs_total",
        },
        "capacity_scale_model": {
            "name": "Eligible wait > threshold (hours)",
            "definition": "Modeled eligible wait above the threshold after capacity scaling.",
            "baseline_field": "eligible_wait_gt_hours_without_target",
            "observed_field": "eligible_wait_gt_hours_modeled",
            "denominator_field": "runs_total",
        },
        "capacity_scaling": {
            "name": "Eligible wait (hours)",
            "definition": "Modeled eligible wait hours after capacity scaling.",
            "baseline_field": "baseline_wait_hours",
            "observed_field": "modeled_wait_hours",
            "denominator_field": "rows",
        },
        "close_cycle_capacity_model": {
            "name": "Close-cycle median duration (seconds)",
            "definition": "Modeled median close-cycle duration under added capacity.",
            "baseline_field": "baseline_median_sec",
            "observed_field": "modeled_median_sec",
            "denominator_field": "bucket_count",
        },
        "close_cycle_capacity_impact": {
            "name": "Close-cycle median duration (seconds)",
            "definition": "Measured close-cycle median duration shift.",
            "baseline_field": "baseline_median_sec",
            "observed_field": "modeled_median_sec",
            "denominator_field": "bucket_count",
        },
        "close_cycle_revenue_compression": {
            "name": "Close-cycle span (days)",
            "definition": "Median close-cycle span by revenue month.",
            "baseline_field": "baseline_span_days_median",
            "observed_field": "baseline_span_days_median",
            "denominator_field": "months",
        },
    }.get(kind or "", {"name": "Finding count", "definition": "Count of matching findings."})


def _denominator_text(item: dict[str, Any], report: dict[str, Any], spec: dict[str, Any]) -> str:
    field = spec.get("denominator_field")
    value = item.get(field) if isinstance(field, str) else None
    parts: list[str] = []
    if value is not None:
        parts.append(f"{field}={_format_issue_value(value)}")
    if "close_cycle_start_day" in item and "close_cycle_end_day" in item:
        parts.append(
            f"close_window=day{item.get('close_cycle_start_day')}-day{item.get('close_cycle_end_day')}"
        )
    if "close_window_mode" in item:
        parts.append(f"close_window_mode={item.get('close_window_mode')}")
    if not parts:
        rows = report.get("input", {}).get("rows")
        if rows is not None:
            parts.append(f"rows={rows}")
    return ", ".join(parts) if parts else "n/a"


def _impact_hours(item: dict[str, Any]) -> float:
    for field in (
        "eligible_wait_gt_hours_total",
        "eligible_wait_hours_total",
        "baseline_wait_hours",
        "reduction_hours",
    ):
        value = item.get(field)
        if isinstance(value, (int, float)):
            return float(value)
    baseline_days = item.get("baseline_span_days_median")
    if isinstance(baseline_days, (int, float)):
        return float(baseline_days) * 24.0
    baseline_sec = item.get("baseline_median_sec")
    if isinstance(baseline_sec, (int, float)):
        return float(baseline_sec) / 3600.0
    return 0.0


def _confidence_weight(item: dict[str, Any], issue: dict[str, Any]) -> float:
    for key in ("confidence_weight", "confidence"):
        value = issue.get(key) if isinstance(issue, dict) else None
        if isinstance(value, (int, float)):
            return max(0.0, min(1.0, float(value)))
    value = item.get("confidence")
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    p_value = item.get("p_value")
    if isinstance(p_value, (int, float)):
        return confidence_from_p(float(p_value))
    return 0.5


def _controllability_weight(kind: str | None, issue: dict[str, Any]) -> float:
    override = issue.get("controllability_weight") if isinstance(issue, dict) else None
    if isinstance(override, (int, float)):
        return max(0.0, min(1.0, float(override)))
    mapping = {
        "capacity_scale_model": 0.9,
        "capacity_scaling": 0.9,
        "close_cycle_capacity_model": 0.8,
        "close_cycle_capacity_impact": 0.8,
        "eligible_wait_process_stats": 0.7,
        "eligible_wait_impact": 0.7,
        "close_cycle_revenue_compression": 0.6,
    }
    return mapping.get(kind or "", 0.5)


def _issue_cards(report: dict[str, Any]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for evaluation in _evaluate_known_issues(report):
        matched = evaluation.get("matched") or []
        item = matched[0] if matched else {}
        spec = _metric_spec(evaluation.get("kind"))
        metric_name = spec.get("name", "Metric")
        definition = spec.get("definition", "")
        baseline_field = spec.get("baseline_field")
        observed_field = spec.get("observed_field")
        baseline_val = item.get(baseline_field) if baseline_field else None
        observed_val = item.get(observed_field) if observed_field else None
        target_val = None
        if evaluation.get("kind") == "capacity_scale_model" and isinstance(
            baseline_val, (int, float)
        ):
            scale = item.get("scale_factor")
            if isinstance(scale, (int, float)):
                target_val = float(baseline_val) * float(scale)
        if evaluation.get("kind") == "capacity_scaling" and isinstance(
            baseline_val, (int, float)
        ):
            scale = item.get("scale_factor")
            if isinstance(scale, (int, float)):
                target_val = float(baseline_val) / float(scale)
        if evaluation.get("kind") == "close_cycle_capacity_model":
            target_reduction = item.get("target_reduction")
            if isinstance(target_reduction, (int, float)) and isinstance(
                baseline_val, (int, float)
            ):
                target_val = float(baseline_val) * (1.0 + float(target_reduction))
        if evaluation.get("kind") == "close_cycle_revenue_compression":
            target_val = item.get("target_days")

        status = evaluation.get("status")
        pass_fail_reason = "REVIEW"
        if status == "confirmed":
            pass_fail_reason = "PASS: expected evidence present."
        elif status == "missing":
            pass_fail_reason = "FAIL: no matching findings."
        elif status == "below_min":
            pass_fail_reason = "FAIL: observed below minimum threshold."
        elif status == "over_limit":
            pass_fail_reason = "FAIL: observed above maximum threshold."

        denominator = _denominator_text(item, report, spec)
        artifacts = _artifact_paths(report, evaluation.get("plugin_id"))
        impact_hours = _impact_hours(item)
        confidence_weight = _confidence_weight(item, evaluation.get("issue") or {})
        controllability_weight = _controllability_weight(
            evaluation.get("kind"), evaluation.get("issue") or {}
        )
        relevance_score = impact_hours * confidence_weight * controllability_weight

        cards.append(
            {
                "title": evaluation.get("label"),
                "plugin_id": evaluation.get("plugin_id"),
                "kind": evaluation.get("kind"),
                "metric_name": metric_name,
                "definition": definition,
                "denominator": denominator,
        "baseline": _format_issue_value(baseline_val),
        "target_threshold": _format_issue_value(target_val),
        "observed": _format_issue_value(observed_val),
                "status": status,
                "reason": pass_fail_reason,
                "artifact_paths": artifacts,
                "impact_hours": impact_hours,
                "confidence_weight": confidence_weight,
                "controllability_weight": controllability_weight,
                "relevance_score": relevance_score,
            }
        )
    return cards


def _waterfall_summary(report: dict[str, Any]) -> dict[str, Any] | None:
    plugins = report.get("plugins", {}) or {}
    queue_plugin = plugins.get("analysis_queue_delay_decomposition")
    if not isinstance(queue_plugin, dict):
        return None
    findings = queue_plugin.get("findings") or []
    impact = None
    qemail = None
    scale = None
    for item in findings:
        if not isinstance(item, dict):
            continue
        kind = item.get("kind")
        if kind == "eligible_wait_impact":
            impact = item
        elif kind == "capacity_scale_model":
            scale = item
        elif kind == "eligible_wait_process_stats" and item.get("process_norm") == "qemail":
            qemail = item
    if impact is None:
        return None
    if qemail is None:
        candidates = [
            f
            for f in findings
            if isinstance(f, dict) and f.get("kind") == "eligible_wait_process_stats"
        ]
        if candidates:
            qemail = max(
                candidates,
                key=lambda f: float(f.get("eligible_wait_gt_hours_total") or 0.0),
            )
    total = impact.get("eligible_wait_gt_hours_total")
    qemail_val = qemail.get("eligible_wait_gt_hours_total") if qemail else None
    remainder = impact.get("eligible_wait_gt_hours_without_target")
    modeled = scale.get("eligible_wait_gt_hours_modeled") if scale else None
    return {
        "total": total,
        "qemail": qemail_val,
        "remainder": remainder,
        "modeled": modeled,
        "scale_factor": scale.get("scale_factor") if scale else None,
    }


def _load_artifact_json(run_dir: Path, path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    target = run_dir / path
    if not target.exists():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _queue_delay_results(report: dict[str, Any], run_dir: Path) -> dict[str, Any] | None:
    plugin = report.get("plugins", {}).get("analysis_queue_delay_decomposition")
    if not isinstance(plugin, dict):
        return None
    artifacts = plugin.get("artifacts") or []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        path = artifact.get("path")
        if isinstance(path, str) and path.endswith("results.json"):
            payload = _load_artifact_json(run_dir, path)
            if payload:
                return payload
    return None


def _recommendation_merge_key(item: dict[str, Any]) -> tuple[str, str]:
    text = str(item.get("recommendation") or "").strip()
    title = str(item.get("title") or "").strip()
    plugin_id = str(item.get("plugin_id") or "")
    where = item.get("where") if isinstance(item.get("where"), dict) else None
    contains = item.get("contains") if isinstance(item.get("contains"), dict) else None
    where_key = json_dumps(where) if where else ""
    contains_key = json_dumps(contains) if contains else ""
    delta = item.get("modeled_delta")
    delta_text = _format_issue_value(delta) if delta is not None else ""

    text_lower = text.lower()
    title_lower = title.lower()
    if text_lower.startswith("add one server") or "3rd server" in title_lower or "3rd server" in text_lower or "third server" in title_lower or "third server" in text_lower:
        base = "capacity_add_server"
    elif "close-cycle capacity" in title_lower or "close-cycle capacity" in text_lower:
        base = "close_cycle_capacity"
    elif text_lower.startswith("act on "):
        base = f"act_on:{plugin_id}:{where_key}:{contains_key}"
    elif text:
        base = text_lower
    else:
        base = title_lower
    return base, delta_text


def _dedupe_recommendations_text(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for item in items:
        key = _recommendation_merge_key(item)
        if key not in merged:
            merged_item = dict(item)
            merged_item["merged_titles"] = [item.get("title") or ""]
            merged[key] = merged_item
            continue
        current = merged[key]
        current["merged_titles"].append(item.get("title") or "")
    return list(merged.values())


def _write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in headers})


def _collapse_findings(
    findings: list[Any], max_examples: int = 10
) -> dict[str, Any]:
    if not isinstance(findings, list) or not findings:
        return {"count": 0, "unique_count": 0, "top_examples": []}
    deduped: dict[str, dict[str, Any]] = {}
    for item in findings:
        key = json_dumps(item) if isinstance(item, (dict, list)) else str(item)
        if key not in deduped:
            deduped[key] = {"count": 0, "example": item}
        deduped[key]["count"] += 1
    ordered = sorted(
        deduped.values(), key=lambda entry: entry["count"], reverse=True
    )
    total = len(findings)
    return {
        "count": total,
        "unique_count": len(deduped),
        "top_examples": ordered[:max_examples],
    }


def _plugin_summary_rows(report: dict[str, Any]) -> tuple[list[tuple[str, int, str]], list[tuple[str, int, str]]]:
    rows: list[tuple[str, int, str]] = []
    plugins = report.get("plugins", {}) or {}
    for plugin_id, data in plugins.items():
        if not isinstance(data, dict):
            continue
        findings = data.get("findings") or []
        summary = (data.get("summary") or "").strip()
        rows.append((plugin_id, len(findings), summary))
    rows.sort()
    yes_rows = [row for row in rows if row[1] > 0]
    no_rows = [row for row in rows if row[1] == 0]
    return yes_rows, no_rows


def _format_plugin_table(rows: list[tuple[str, int, str]]) -> list[str]:
    lines = ["| Plugin | Findings | One-line summary |", "|---|---:|---|"]
    for plugin_id, count, summary in rows:
        lines.append(f"| `{plugin_id}` | {count} | {summary} |")
    return lines


def _load_known_issues_fallback(run_dir: Path) -> dict[str, Any] | None:
    known_dir = run_dir.parent.parent / "known_issues"
    if not known_dir.exists():
        return None
    payloads: list[dict[str, Any]] = []
    for path in sorted(known_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if data.get("expected_findings"):
            payloads.append(data)
    if not payloads:
        return None
    expected: list[dict[str, Any]] = []
    strict_values: list[bool] = []
    notes: list[str] = []
    for data in payloads:
        strict_values.append(bool(data.get("strict", False)))
        note = str(data.get("notes") or "").strip()
        if note:
            notes.append(note)
        expected.extend(data.get("expected_findings") or [])
    note_text = "Fallback merged from appdata/known_issues"
    if notes:
        note_text = f"{note_text} | " + " | ".join(notes)
    return {
        "scope_type": "fallback",
        "scope_value": "appdata/known_issues",
        "strict": all(strict_values) if strict_values else False,
        "notes": note_text,
        "natural_language": [],
        "expected_findings": expected,
    }

from .storage import Storage
from .utils import json_dumps, now_iso, read_json, write_json


def build_report(
    storage: Storage, run_id: str, run_dir: Path, schema_path: Path
) -> dict[str, Any]:
    run_row = storage.fetch_run(run_id)
    if not run_row or not run_row.get("dataset_version_id"):
        raise ValueError("Run dataset version not found")
    upload_row = (
        storage.fetch_upload(run_row["upload_id"])
        if run_row.get("upload_id")
        else None
    )
    from .dataset_io import DatasetAccessor

    accessor = DatasetAccessor(storage, run_row["dataset_version_id"])
    info = accessor.info()

    plugin_rows = storage.fetch_plugin_results(run_id)
    plugins: dict[str, Any] = {}

    def _ensure_measurement(findings: list[Any]) -> list[Any]:
        for item in findings:
            if isinstance(item, dict) and "measurement_type" not in item:
                item["measurement_type"] = "measured"
        return findings

    def _canonicalize_payload(payload: Any) -> Any:
        try:
            return json.loads(json_dumps(payload))
        except TypeError:
            return payload

    def _sort_payload_list(items: list[Any]) -> list[Any]:
        try:
            return sorted(items, key=lambda item: json_dumps(item))
        except Exception:
            return items

    for row in sorted(plugin_rows, key=lambda item: item["plugin_id"]):
        findings = json.loads(row["findings_json"])
        if isinstance(findings, list):
            findings = _ensure_measurement(findings)
            findings = _sort_payload_list(findings)
        artifacts = json.loads(row["artifacts_json"])
        if isinstance(artifacts, list):
            artifacts = _sort_payload_list(artifacts)
        budget = None
        if "budget_json" in row.keys() and row.get("budget_json"):
            try:
                budget = json.loads(row["budget_json"])
            except json.JSONDecodeError:
                budget = None
        if not isinstance(budget, dict):
            budget = {
                "row_limit": None,
                "sampled": False,
                "time_limit_ms": None,
                "cpu_limit_ms": None,
            }
        plugins[row["plugin_id"]] = {
            "status": row["status"],
            "summary": row["summary"],
            "metrics": _canonicalize_payload(json.loads(row["metrics_json"])),
            "findings": findings,
            "artifacts": artifacts,
            "budget": _canonicalize_payload(budget),
            "error": json.loads(row["error_json"]) if row["error_json"] else None,
        }

    dataset_version = storage.get_dataset_version(run_row["dataset_version_id"])
    dataset_context = storage.get_dataset_version_context(run_row["dataset_version_id"])
    project_row = None
    if dataset_context and dataset_context.get("project_id"):
        project_row = storage.fetch_project(dataset_context["project_id"])
    dataset_template = storage.fetch_dataset_template(run_row["dataset_version_id"])
    raw_format = None
    raw_format_id = None
    if dataset_version:
        raw_format_id = dataset_version.get("raw_format_id")
    if raw_format_id:
        with storage.connection() as conn:
            cur = conn.execute(
                """
                SELECT format_id, fingerprint, name, created_at
                FROM raw_formats
                WHERE format_id = ?
                """,
                (raw_format_id,),
            )
            row = cur.fetchone()
            raw_format = dict(row) if row else None
    if raw_format:
        raw_format = {
            "format_id": int(raw_format.get("format_id") or raw_format_id or 0),
            "fingerprint": raw_format.get("fingerprint") or "",
            "name": raw_format.get("name") or "",
            "created_at": raw_format.get("created_at") or "",
        }

    mapping = None
    if dataset_template and dataset_template.get("mapping_json"):
        try:
            mapping = json.loads(dataset_template["mapping_json"])
        except json.JSONDecodeError:
            mapping = {}

    dataset_block = {
        "dataset_version_id": run_row.get("dataset_version_id") or "unknown",
    }
    if dataset_context:
        if dataset_context.get("project_id"):
            dataset_block["project_id"] = dataset_context["project_id"]
        if dataset_context.get("dataset_id"):
            dataset_block["dataset_id"] = dataset_context["dataset_id"]
        if dataset_context.get("table_name"):
            dataset_block["table_name"] = dataset_context["table_name"]
    if dataset_version:
        if dataset_version.get("data_hash"):
            dataset_block["data_hash"] = dataset_version["data_hash"]
        if dataset_version.get("row_count") is not None:
            dataset_block["row_count"] = int(dataset_version["row_count"])
        if dataset_version.get("column_count") is not None:
            dataset_block["column_count"] = int(dataset_version["column_count"])
        if dataset_version.get("raw_format_id"):
            dataset_block["raw_format_id"] = int(dataset_version["raw_format_id"])

    def _string_or_empty(value: Any) -> str:
        return value if isinstance(value, str) else ""

    lineage_plugins: dict[str, Any] = {}
    for row in plugin_rows:
        lineage_plugins[row["plugin_id"]] = {
            "plugin_version": _string_or_empty(row.get("plugin_version")),
            "code_hash": _string_or_empty(row.get("code_hash")),
            "settings_hash": _string_or_empty(row.get("settings_hash")),
            "dataset_hash": _string_or_empty(row.get("dataset_hash")),
            "executed_at": _string_or_empty(row.get("executed_at")),
            "status": _string_or_empty(row.get("status")),
            "summary": _string_or_empty(row.get("summary")),
        }

    template_block = None
    if dataset_template:
        template_block = {
            "template_id": int(dataset_template["template_id"]),
            "table_name": dataset_template.get("table_name") or "",
            "status": dataset_template.get("status") or "",
            "mapping_hash": dataset_template.get("mapping_hash") or "",
            "mapping": mapping if isinstance(mapping, dict) else {},
        }
        if dataset_template.get("template_name"):
            template_block["template_name"] = dataset_template["template_name"]
        if dataset_template.get("template_version"):
            template_block["template_version"] = dataset_template["template_version"]

    known_block = None
    known_scope_type = ""
    known_scope_value = ""
    if project_row and project_row.get("erp_type"):
        known_scope_type = "erp_type"
        known_scope_value = str(project_row.get("erp_type") or "unknown").strip() or "unknown"
        known_block = storage.fetch_known_issues(known_scope_value, known_scope_type)
    if not known_block and upload_row and upload_row.get("sha256"):
        known_scope_type = "sha256"
        known_scope_value = str(upload_row.get("sha256") or "")
        if known_scope_value:
            known_block = storage.fetch_known_issues(known_scope_value, known_scope_type)
    if not known_block and dataset_block.get("data_hash"):
        data_hash = str(dataset_block.get("data_hash") or "")
        if re.fullmatch(r"[a-f0-9]{64}", data_hash):
            known_scope_type = "sha256"
            known_scope_value = data_hash
            known_block = storage.fetch_known_issues(known_scope_value, known_scope_type)

    known_payload = None
    if known_block:
        known_payload = {
            "scope_type": known_block.get("scope_type") or known_scope_type,
            "scope_value": known_block.get("scope_value") or known_scope_value,
            "strict": bool(known_block.get("strict", True)),
            "notes": known_block.get("notes") or "",
            "natural_language": known_block.get("natural_language") or [],
            "expected_findings": known_block.get("expected_findings") or [],
        }

    report = {
        "run_id": run_id,
        "created_at": now_iso(),
        "status": "completed",
        "input": {
            "filename": run_row.get("input_filename") or "unknown",
            **info,
        },
        "lineage": {
            "run": {
                "run_id": run_id,
                "created_at": run_row.get("created_at") or "",
                "status": run_row.get("status") or "",
                "run_seed": int(run_row.get("run_seed") or 0),
            },
            "input": {
                "upload_id": run_row.get("upload_id") or "",
                "filename": run_row.get("input_filename") or "unknown",
                "canonical_path": run_row.get("canonical_path") or "",
                "input_hash": run_row.get("input_hash") or "",
                "sha256": upload_row.get("sha256") if upload_row else "",
                "size_bytes": int(upload_row.get("size_bytes") or 0)
                if upload_row
                else 0,
            },
            "dataset": dataset_block,
            "raw_format": raw_format,
            "template": template_block,
            "plugins": lineage_plugins,
        },
        "plugins": plugins,
    }
    if not known_payload:
        known_payload = _load_known_issues_fallback(run_dir)
    if known_payload:
        report["known_issues"] = known_payload
    report["recommendations"] = _build_recommendations(report)
    evaluation_path = run_dir / "evaluation.json"
    if evaluation_path.exists():
        try:
            report["evaluation"] = json.loads(evaluation_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report["evaluation"] = None

    schema = read_json(schema_path)
    validate(instance=report, schema=schema)
    return report


def write_report(report: dict[str, Any], run_dir: Path) -> None:
    report_path = run_dir / "report.json"
    write_json(report_path, report)

    lines = ["# Statistic Harness Report", ""]
    lines.append("## Decision")
    lines.append("")
    lines.append(f"Run ID: {report.get('run_id')}")
    lines.append(f"Created: {report.get('created_at')}")
    lines.append(f"Status: {report.get('status')}")
    lines.append(f"Rows: {report.get('input', {}).get('rows')}")
    lines.append(f"Cols: {report.get('input', {}).get('cols')}")
    lines.append("")

    exec_summary = _build_executive_summary(report)
    lines.append("### Executive Summary")
    if exec_summary:
        for entry in exec_summary:
            lines.append(f"- {entry}")
    else:
        lines.append("No executive summary available.")
    lines.append("")

    cards = _issue_cards(report)
    cards_sorted = sorted(cards, key=lambda c: c.get("relevance_score", 0.0), reverse=True)
    top_n = int(os.environ.get("STAT_HARNESS_DECISION_TOP_N", "5"))
    decision_cards = cards_sorted[:top_n] if top_n > 0 else []

    lines.append("### Decision Items")
    if decision_cards:
        lines.append("| Issue | Score | Status | Reason |")
        lines.append("|---|---:|---|---|")
        for card in decision_cards:
            title = card.get("title") or "Issue"
            score = _format_issue_value(card.get("relevance_score"), digits=3)
            status = card.get("status") or "unknown"
            reason = card.get("reason") or ""
            lines.append(f"| {title} | {score} | {status} | {reason} |")
    else:
        lines.append("No decision items available.")
    lines.append("")

    lines.append("### Waterfall Summary")
    waterfall = _waterfall_summary(report)
    if waterfall and all(
        isinstance(waterfall.get(key), (int, float)) for key in ("total", "qemail", "remainder")
    ):
        total = float(waterfall["total"])
        qemail = float(waterfall["qemail"])
        remainder = float(waterfall["remainder"])
        modeled = waterfall.get("modeled")
        scale = waterfall.get("scale_factor")
        total_text = _format_issue_value(total)
        qemail_text = _format_issue_value(qemail)
        remainder_text = _format_issue_value(remainder)
        lines.append(f"Total over-threshold eligible wait: {total_text}h")
        lines.append(f"QEMAIL contribution: {qemail_text}h")
        lines.append(
            f"Remainder without QEMAIL: {remainder_text}h = {total_text}h - {qemail_text}h"
        )
        if isinstance(modeled, (int, float)):
            modeled_text = _format_issue_value(float(modeled))
            if isinstance(scale, (int, float)):
                scale_text = _format_issue_value(float(scale))
                lines.append(
                    f"Modeled after add one server: {modeled_text}h = {remainder_text}h × {scale_text}"
                )
            else:
                lines.append(f"Modeled after add one server: {modeled_text}h")
    else:
        lines.append("No waterfall summary available.")
    lines.append("")

    recommendations = report.get("recommendations")
    lines.append("### Recommendations")
    if isinstance(recommendations, dict):
        summary = recommendations.get("summary") or ""
        if summary:
            lines.append(summary)
        items = recommendations.get("items") or []
        if items:
            items = _dedupe_recommendations_text(
                [item for item in items if isinstance(item, dict)]
            )
            lines.append("| Status | Recommendation |")
            lines.append("|---|---|")
            for item in items:
                status = item.get("status") or "unknown"
                rec = item.get("recommendation") or "Recommendation"
                merged_titles = [
                    title
                    for title in (item.get("merged_titles") or [])
                    if isinstance(title, str) and title
                ]
                if len(merged_titles) > 1:
                    merged_text = "; ".join(dict.fromkeys(merged_titles))
                    rec = f"{rec} (covers: {merged_text})"
                lines.append(f"| {status} | {rec} |")
        else:
            lines.append("No recommendations available.")
    else:
        lines.append("No recommendations available.")
    lines.append("")

    if decision_cards:
        lines.append("### Issue Cards (Top)")
        for card in decision_cards:
            lines.append(f"#### {card.get('title')}")
            lines.append(
                f"Metric: {card.get('metric_name')} — {card.get('definition')}"
            )
            lines.append(f"Denominator: {card.get('denominator')}")
            lines.append(f"Baseline: {card.get('baseline')}")
            lines.append(f"Target Threshold: {card.get('target_threshold')}")
            lines.append(f"Observed: {card.get('observed')}")
            lines.append(f"Decision: {card.get('reason')}")
            artifacts = ", ".join(card.get("artifact_paths") or []) or "n/a"
            lines.append(f"Evidence Artifacts: {artifacts}")
            lines.append("")

    lines.append("## Appendix")
    lines.append("")
    lines.append("### Issue Cards (Full)")
    if cards:
        for card in cards:
            lines.append(f"#### {card.get('title')}")
            lines.append(
                f"Metric: {card.get('metric_name')} — {card.get('definition')}"
            )
            lines.append(f"Denominator: {card.get('denominator')}")
            lines.append(f"Baseline: {card.get('baseline')}")
            lines.append(f"Target Threshold: {card.get('target_threshold')}")
            lines.append(f"Observed: {card.get('observed')}")
            lines.append(f"Decision: {card.get('reason')}")
            artifacts = ", ".join(card.get("artifact_paths") or []) or "n/a"
            lines.append(f"Evidence Artifacts: {artifacts}")
            lines.append("")
    else:
        lines.append("No issue cards available.")
        lines.append("")

    lines.append("### Plugin Dumps")
    for plugin_id in sorted(report.get("plugins", {}).keys()):
        data = report["plugins"][plugin_id]
        lines.append(f"#### {plugin_id}")
        lines.append("```json")
        lines.append(json_dumps(data))
        lines.append("```")
        lines.append("")
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    _write_business_summary(report, run_dir)
    _write_engineering_summary(report, run_dir)
    _write_appendix_raw(report, run_dir)
    _write_slide_kit(report, run_dir)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (int, bool)):
        return str(value)
    if isinstance(value, list):
        preview = ", ".join(str(v) for v in value[:5])
        if len(value) > 5:
            preview += ", ..."
        return f"[{preview}]"
    if isinstance(value, dict):
        return "{...}"
    text = str(value)
    if len(text) > 80:
        return text[:77] + "..."
    return text


def _format_metrics(metrics: dict[str, Any]) -> list[str]:
    if not isinstance(metrics, dict):
        return []
    items: list[tuple[str, Any]] = []
    for key, value in metrics.items():
        if isinstance(value, (int, float, str, bool)):
            items.append((key, value))
    items = sorted(items, key=lambda item: item[0])
    return [f"{key}: {_format_value(value)}" for key, value in items]


def _format_findings(findings: list[Any]) -> list[str]:
    rendered: list[str] = []
    for finding in findings:
        if not isinstance(finding, dict):
            rendered.append(_format_value(finding))
            continue
        kind = finding.get("kind", "finding")
        measurement = finding.get("measurement_type", "measured")
        parts = [f"kind={kind}", f"measurement={measurement}"]
        key_fields = [
            "role",
            "column",
            "process",
            "process_norm",
            "process_id",
            "process_name",
            "module",
            "module_cd",
            "user",
            "user_id",
            "dimension",
            "key",
            "sequence",
            "host",
            "feature",
            "metric",
        ]
        for field in key_fields:
            if field in finding and finding[field] not in (None, ""):
                parts.append(f"{field}={_format_value(finding[field])}")
        numeric_fields = [
            key
            for key, value in finding.items()
            if isinstance(value, (int, float))
            and key not in {"row_index"}
            and (
                key.endswith("_sec")
                or key.endswith("_hours")
                or key.endswith("_count")
                or key.endswith("_runs")
                or key.endswith("_ratio")
                or key.endswith("_pct")
                or key in {"p50", "p95", "p99", "mean", "min", "max", "score"}
            )
        ]
        for key in sorted(numeric_fields)[:6]:
            parts.append(f"{key}={_format_value(finding[key])}")
        evidence = finding.get("evidence")
        if isinstance(evidence, dict):
            row_ids = evidence.get("row_ids")
            col_ids = evidence.get("column_ids")
            if isinstance(row_ids, list) and row_ids:
                parts.append(f"rows={len(row_ids)}")
            if isinstance(col_ids, list) and col_ids:
                parts.append(f"cols={len(col_ids)}")
            query = evidence.get("query")
            if query:
                parts.append(f"query={_format_value(query)}")
        rendered.append(", ".join(parts))
    return rendered


def _matches_expected(
    item: dict[str, Any], where: dict[str, Any] | None, contains: dict[str, Any] | None
) -> bool:
    if where:
        for key, expected in where.items():
            if item.get(key) != expected:
                return False
    if contains:
        for key, expected in contains.items():
            actual = item.get(key)
            if isinstance(actual, str):
                if str(expected) not in actual:
                    return False
            elif isinstance(actual, (list, tuple, set)):
                if isinstance(expected, (list, tuple, set)):
                    if not set(expected).issubset(set(actual)):
                        return False
                else:
                    if expected not in actual:
                        return False
            else:
                return False
    return True


def _format_known_issue_checks(
    report: dict[str, Any], expected: list[dict[str, Any]]
) -> list[str]:
    items: list[str] = []
    for entry in expected:
        if not isinstance(entry, dict):
            continue
        plugin_id = entry.get("plugin_id")
        kind = entry.get("kind")
        if not kind:
            continue
        where = entry.get("where") or {}
        contains = entry.get("contains") or {}
        min_count = int(entry.get("min_count", 1))
        max_count = entry.get("max_count")
        candidates = []
        for pid, plugin in report.get("plugins", {}).items():
            if plugin_id and pid != plugin_id:
                continue
            for item in plugin.get("findings", []):
                if item.get("kind") == kind:
                    candidates.append(item)
        matches = [item for item in candidates if _matches_expected(item, where, contains)]
        status = "PASS" if len(matches) >= min_count and (max_count is None or len(matches) <= int(max_count)) else "FAIL"
        detail = f"{len(matches)} match(es)"
        title = entry.get("title") or entry.get("description") or ""
        if title:
            title = title.strip()
        context = f"{kind} ({plugin_id or '*'})"
        if title:
            context = f"{title} :: {context}"
        items.append(f"{status} - {context} - {detail}")
    return items


def _metric_unit(metric_name: str, item: dict[str, Any] | None = None) -> str:
    name = metric_name.lower()
    if "hours" in name:
        return "hours"
    if "seconds" in name or "sec" in name:
        return "seconds"
    if "days" in name:
        return "days"
    if item:
        for key in item.keys():
            if key.endswith("_hours"):
                return "hours"
            if key.endswith("_sec"):
                return "seconds"
            if key.endswith("_days"):
                return "days"
    return "count"


def _scenario_rows_from_item(item: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(item, dict):
        return []
    baseline = None
    modeled = None
    unit = None
    if "eligible_wait_gt_hours_without_target" in item and "eligible_wait_gt_hours_modeled" in item:
        baseline = item.get("eligible_wait_gt_hours_without_target")
        modeled = item.get("eligible_wait_gt_hours_modeled")
        unit = "hours"
    elif "baseline_wait_hours" in item and "modeled_wait_hours" in item:
        baseline = item.get("baseline_wait_hours")
        modeled = item.get("modeled_wait_hours")
        unit = "hours"
    elif "baseline_median_sec" in item and "modeled_median_sec" in item:
        baseline = item.get("baseline_median_sec")
        modeled = item.get("modeled_median_sec")
        unit = "seconds"
    elif "baseline_span_days_median" in item and "target_days" in item:
        baseline = item.get("baseline_span_days_median")
        modeled = item.get("target_days")
        unit = "days"

    if not isinstance(baseline, (int, float)) or not isinstance(modeled, (int, float)):
        return []
    delta = float(modeled) - float(baseline)
    return [
        {"scenario": "current", "value": float(baseline), "delta": 0.0, "unit": unit},
        {"scenario": "modeled", "value": float(modeled), "delta": delta, "unit": unit},
    ]


def _busy_periods_info(report: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    results = _queue_delay_results(report, run_dir)
    if not results:
        return {
            "busy_periods": [],
            "threshold_seconds": None,
            "bucket": None,
            "basis": None,
            "artifact_path": None,
        }
    summary = results.get("summary") or {}
    busy_periods = results.get("busy_periods") or []
    artifact_path = None
    plugin = report.get("plugins", {}).get("analysis_queue_delay_decomposition") or {}
    for artifact in plugin.get("artifacts", []) or []:
        if isinstance(artifact, dict) and str(artifact.get("path", "")).endswith("results.json"):
            artifact_path = artifact.get("path")
            break
    return {
        "busy_periods": busy_periods,
        "threshold_seconds": summary.get("wait_threshold_seconds"),
        "bucket": summary.get("busy_period_bucket"),
        "basis": summary.get("busy_period_basis"),
        "artifact_path": artifact_path,
        "summary": summary,
    }


def _top_process_contributors(report: dict[str, Any], limit: int = 10) -> list[dict[str, Any]]:
    plugin = report.get("plugins", {}).get("analysis_queue_delay_decomposition")
    if not isinstance(plugin, dict):
        return []
    findings = plugin.get("findings") or []
    process_stats = [
        f for f in findings if isinstance(f, dict) and f.get("kind") == "eligible_wait_process_stats"
    ]
    impact = next(
        (f for f in findings if isinstance(f, dict) and f.get("kind") == "eligible_wait_impact"),
        None,
    )
    total = impact.get("eligible_wait_gt_hours_total") if isinstance(impact, dict) else None
    rows: list[dict[str, Any]] = []
    for item in process_stats:
        value = item.get("eligible_wait_gt_hours_total")
        if not isinstance(value, (int, float)):
            continue
        share = float(value) / float(total) if isinstance(total, (int, float)) and total else None
        rows.append(
            {
                "process": item.get("process") or item.get("process_norm"),
                "eligible_wait_gt_hours_total": float(value),
                "runs_total": int(item.get("runs_total") or 0),
                "share_of_total": share,
            }
        )
    rows.sort(key=lambda r: r["eligible_wait_gt_hours_total"], reverse=True)
    return rows[:limit]


def _qemail_profile(report: dict[str, Any]) -> dict[str, Any] | None:
    plugin = report.get("plugins", {}).get("analysis_queue_delay_decomposition")
    if not isinstance(plugin, dict):
        return None
    findings = plugin.get("findings") or []
    for item in findings:
        if (
            isinstance(item, dict)
            and item.get("kind") == "eligible_wait_process_stats"
            and item.get("process_norm") == "qemail"
        ):
            return {
                "process": item.get("process"),
                "eligible_wait_gt_hours_total": item.get("eligible_wait_gt_hours_total"),
                "eligible_wait_hours_total": item.get("eligible_wait_hours_total"),
                "runs_total": item.get("runs_total"),
                "runs_close": item.get("runs_close"),
                "runs_open": item.get("runs_open"),
            }
    return None


def _write_business_summary(report: dict[str, Any], run_dir: Path) -> None:
    info = _busy_periods_info(report, run_dir)
    busy_periods = info.get("busy_periods") or []
    threshold = info.get("threshold_seconds")
    summary = info.get("summary") or {}
    close_start = summary.get("close_cycle_start_day")
    close_end = summary.get("close_cycle_end_day")
    artifact_path = info.get("artifact_path") or "n/a"
    busy_periods_sorted = sorted(
        busy_periods,
        key=lambda b: float(b.get("wait_to_start_hours_total") or 0.0),
        reverse=True,
    )
    top_busy = busy_periods_sorted[:10]
    total_busy_hours = sum(
        float(item.get("wait_to_start_hours_total") or 0.0) for item in busy_periods_sorted
    )

    lines = ["# Business Summary", ""]
    threshold_text = f"{int(threshold)}s" if isinstance(threshold, (int, float)) else "n/a"
    if isinstance(close_start, int) and isinstance(close_end, int):
        close_window = f"days {close_start}-{close_end} of month"
    else:
        close_window = "all days"
    lines.append(
        "Primary KPI: busy-period wait-to-start hours, defined as total wait-to-start hours "
        "in hourly buckets (by queue timestamp) within the close-cycle window "
        f"({close_window}) where wait-to-start exceeds {threshold_text}."
    )
    lines.append(
        "Metric context: unit=hours; population=standalone, non-excluded rows in close-cycle window "
        "with queue/start timestamps; "
        f"threshold=wait-to-start > {threshold_text}; filters=close_cycle=true; busy_period_bucket=hour."
    )
    lines.append("")

    lines.append("## Busy Periods (Top 10)")
    if top_busy:
        lines.append(
            "Metric context: unit=hours; population=standalone, non-excluded rows in close-cycle window "
            f"with queue/start timestamps; threshold=wait-to-start > {threshold_text}; "
            "filters=close_cycle=true; busy_period_bucket=hour."
        )
        lines.append("| Period Start | Period End | Wait-to-Start Hours | Rows | After-hours | Weekend |")
        lines.append("|---|---|---:|---:|---|---|")
        for row in top_busy:
            lines.append(
                "| {start} | {end} | {hours:.3f} | {rows} | {after} | {weekend} |".format(
                    start=row.get("period_start"),
                    end=row.get("period_end"),
                    hours=float(row.get("wait_to_start_hours_total") or 0.0),
                    rows=int(row.get("rows_total") or 0),
                    after="yes" if row.get("after_hours") else "no",
                    weekend="yes" if row.get("weekend") else "no",
                )
            )
        weekday_count = sum(1 for row in top_busy if not row.get("weekend"))
        after_hours_count = sum(1 for row in top_busy if row.get("after_hours"))
        lines.append(
            f"Note: {weekday_count}/{len(top_busy)} periods are weekdays; "
            f"{after_hours_count}/{len(top_busy)} occur after-hours."
        )
        lines.append(f"Full detail: {artifact_path}")
    else:
        lines.append("No busy period data available.")
    lines.append("")

    cards = _issue_cards(report)
    card_by_title = {card.get("title"): card for card in cards}
    evaluations = _evaluate_known_issues(report)
    eval_by_title = {entry.get("label"): entry for entry in evaluations}
    recommendations = report.get("recommendations", {}).get("items") or []
    recs = [item for item in recommendations if isinstance(item, dict)]
    recs = _dedupe_recommendations_text(recs)
    recs_sorted = sorted(
        recs,
        key=lambda item: card_by_title.get(item.get("title"), {}).get("relevance_score", 0.0),
        reverse=True,
    )
    recs_sorted = recs_sorted[:3]

    if recs_sorted:
        lines.append("## Top Recommendations")
    for idx, rec in enumerate(recs_sorted, start=1):
        title = rec.get("title") or f"Recommendation {idx}"
        merged_titles = [
            text
            for text in (rec.get("merged_titles") or [])
            if isinstance(text, str) and text
        ]
        card = card_by_title.get(title) or {}
        evaluation = eval_by_title.get(title) or {}
        matched = evaluation.get("matched") or []
        item = matched[0] if matched else {}
        metric_name = card.get("metric_name") or "Metric"
        unit = _metric_unit(metric_name, item if isinstance(item, dict) else None)
        denominator = card.get("denominator") or "n/a"
        filter_text = ""
        if evaluation.get("where"):
            filter_text = f"where={evaluation.get('where')}"
        if evaluation.get("contains"):
            filter_text = f"{filter_text} contains={evaluation.get('contains')}".strip()
        if not filter_text:
            filter_text = f"threshold={threshold_text}"
        measurement_type = (
            item.get("measurement_type") if isinstance(item, dict) else None
        ) or "measured"
        confidence_tag = "Modeled" if measurement_type == "modeled" else "Measured"

        lines.append(f"### {title}")
        lines.append(f"Problem: {title}")
        if len(merged_titles) > 1:
            merged_text = "; ".join(dict.fromkeys(merged_titles))
            lines.append(f"Also covers: {merged_text}")
        lines.append(
            "Evidence: "
            f"baseline={card.get('baseline')}, observed={card.get('observed')}, "
            f"target={card.get('target_threshold')}."
        )
        lines.append(f"Action: {rec.get('recommendation')}")
        lines.append(
            f"Metric context: unit={unit}; population={denominator}; threshold/filters={filter_text}."
        )
        scenario_rows = _scenario_rows_from_item(item) if isinstance(item, dict) else []
        if not scenario_rows:
            scenario_rows = [
                {
                    "scenario": "current",
                    "value": float(total_busy_hours),
                    "delta": 0.0,
                    "unit": "hours",
                }
            ]
            if isinstance(rec.get("modeled_delta"), (int, float)):
                modeled_value = float(total_busy_hours) - float(rec["modeled_delta"])
                scenario_rows.append(
                    {
                        "scenario": "modeled",
                        "value": modeled_value,
                        "delta": modeled_value - float(total_busy_hours),
                        "unit": "hours",
                    }
                )
        lines.append("Expected impact (current vs scenarios):")
        lines.append(
            f"Impact metric context: unit={unit}; population={denominator}; threshold/filters={filter_text}."
        )
        lines.append("| Scenario | Value | Delta | Unit |")
        lines.append("|---|---:|---:|---|")
        for row in scenario_rows[:10]:
            lines.append(
                "| {scenario} | {value:.3f} | {delta:.3f} | {unit} |".format(
                    scenario=row["scenario"],
                    value=float(row["value"]),
                    delta=float(row["delta"]),
                    unit=row.get("unit") or unit,
                )
            )
        lines.append(f"Confidence: {confidence_tag}")
        artifact_paths = card.get("artifact_paths") or []
        artifact_text = ", ".join(artifact_paths) if artifact_paths else "n/a"
        query = item.get("query") if isinstance(item, dict) else None
        filters = query or filter_text
        plugin_id = rec.get("plugin_id") or card.get("plugin_id") or "n/a"
        lines.append(
            f"How to validate: re-run `{plugin_id}` and confirm `{metric_name}` in {artifact_text}; "
            f"filters: {filters}."
        )
        lines.append("")

    (run_dir / "business_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _write_engineering_summary(report: dict[str, Any], run_dir: Path) -> None:
    lines = ["# Engineering Summary", ""]
    lines.append("## Glossary")
    lines.append("- eligible_wait: time from eligible timestamp to start timestamp.")
    lines.append("- threshold: per-run wait threshold in seconds for defining over-threshold wait.")
    lines.append(
        "- busy period: hourly bucket (by queue timestamp) within the close-cycle window "
        "where wait-to-start exceeds the threshold."
    )
    lines.append("- close cycle window: day-of-month window used to tag close-cycle activity.")
    lines.append("")

    cards = _issue_cards(report)
    card_by_title = {card.get("title"): card for card in cards}
    evaluations = _evaluate_known_issues(report)
    eval_by_title = {entry.get("label"): entry for entry in evaluations}
    recommendations = report.get("recommendations", {}).get("items") or []
    recs = _dedupe_recommendations_text(
        [item for item in recommendations if isinstance(item, dict)]
    )
    recs = sorted(
        recs,
        key=lambda item: card_by_title.get(item.get("title"), {}).get("relevance_score", 0.0),
        reverse=True,
    )[:3]
    merged_notes: list[str] = []

    lines.append("## Traceability")
    if recs:
        lines.append("| Claim | Plugin | Kind | Measurement | Artifact Path | Query/Filter | Metric Context |")
        lines.append("|---|---|---|---|---|---|---|")
        for rec in recs:
            title = rec.get("title") or "Recommendation"
            merged_titles = [
                text
                for text in (rec.get("merged_titles") or [])
                if isinstance(text, str) and text
            ]
            claim_text = title
            if len(merged_titles) > 1:
                claim_text = f"{title} (+{len(merged_titles) - 1} merged)"
                merged_text = "; ".join(dict.fromkeys(merged_titles))
                merged_notes.append(f"{title}: {merged_text}")
            evaluation = eval_by_title.get(title) or {}
            matched = evaluation.get("matched") or []
            item = matched[0] if matched else {}
            plugin_id = rec.get("plugin_id") or evaluation.get("plugin_id") or "n/a"
            kind = evaluation.get("kind") or rec.get("kind") or "n/a"
            measurement = item.get("measurement_type") if isinstance(item, dict) else "n/a"
            card = card_by_title.get(title) or {}
            artifact_paths = card.get("artifact_paths") or []
            artifact_text = ", ".join(artifact_paths) if artifact_paths else "n/a"
            query = item.get("query") if isinstance(item, dict) else None
            filters = query or evaluation.get("where") or evaluation.get("contains") or "n/a"
            metric_name = card.get("metric_name") or "Metric"
            unit = _metric_unit(metric_name, item if isinstance(item, dict) else None)
            denominator = card.get("denominator") or "n/a"
            lines.append(
                f"| {claim_text} | `{plugin_id}` | {kind} | {measurement} | {artifact_text} | {filters} | "
                f"unit={unit}; population={denominator}; threshold/filters={filters} |"
            )
    else:
        lines.append("No traceability data available.")
    lines.append("")
    if merged_notes:
        lines.append("Merged claims:")
        for note in merged_notes:
            lines.append(f"- {note}")
        lines.append("")

    lines.append("## Failing Checks")
    failing = [entry for entry in evaluations if entry.get("status") != "confirmed"]
    if not failing:
        lines.append("No failing checks.")
    else:
        for entry in failing[:10]:
            predicate = (
                f"plugin={entry.get('plugin_id')} kind={entry.get('kind')} "
                f"where={entry.get('where')} contains={entry.get('contains')} "
                f"min={entry.get('min_count')} max={entry.get('max_count')}"
            )
            matched = entry.get("matched") or []
            item = matched[0] if matched else {}
            computed = {
                "observed_count": entry.get("count"),
                "baseline": item.get("baseline_wait_hours")
                or item.get("eligible_wait_gt_hours_total")
                or item.get("baseline_median_sec"),
                "observed": item.get("modeled_wait_hours")
                or item.get("eligible_wait_gt_hours_modeled")
                or item.get("modeled_median_sec"),
            }
            card = card_by_title.get(entry.get("label")) or {}
            metric_name = card.get("metric_name") or entry.get("metric") or "Metric"
            unit = _metric_unit(metric_name, item if isinstance(item, dict) else None)
            denominator = card.get("denominator") or entry.get("denominator") or "n/a"
            filter_text = entry.get("where") or entry.get("contains") or "n/a"
            lines.append(f"- Predicate: {predicate}")
            lines.append(f"  - Computed: {computed}")
            lines.append(
                f"  - Metric context: unit={unit}; population={denominator}; "
                f"threshold/filters={filter_text}."
            )
            lines.append(f"  - Reason: {entry.get('status')}")
    (run_dir / "engineering_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _write_appendix_raw(report: dict[str, Any], run_dir: Path) -> None:
    lines = ["# Appendix (Raw)", ""]
    for plugin_id in sorted(report.get("plugins", {}).keys()):
        data = report["plugins"][plugin_id]
        trimmed = dict(data)
        findings = data.get("findings") or []
        summary = _collapse_findings(findings, max_examples=10)
        trimmed["findings_summary"] = summary
        trimmed["findings_full_path"] = f"report.json -> plugins.{plugin_id}.findings"
        if summary.get("count", 0) > 10:
            trimmed["findings"] = []
        else:
            trimmed["findings"] = summary.get("top_examples") or []
        lines.append(f"## {plugin_id}")
        lines.append("```json")
        lines.append(json_dumps(trimmed))
        lines.append("```")
        lines.append("")
    (run_dir / "appendix_raw.md").write_text("\n".join(lines), encoding="utf-8")


def _write_slide_kit(report: dict[str, Any], run_dir: Path) -> None:
    kit_dir = run_dir / "slide_kit"
    kit_dir.mkdir(parents=True, exist_ok=True)
    info = _busy_periods_info(report, run_dir)
    busy_periods = info.get("busy_periods") or []
    threshold = info.get("threshold_seconds")
    threshold_val = int(threshold) if isinstance(threshold, (int, float)) else ""
    busy_periods_sorted = sorted(
        busy_periods,
        key=lambda b: float(b.get("wait_to_start_hours_total") or 0.0),
        reverse=True,
    )
    busy_periods_top = busy_periods_sorted[:10]
    busy_rows = []
    for row in busy_periods_top:
        busy_rows.append(
            {
                "period_start": row.get("period_start"),
                "period_end": row.get("period_end"),
                "wait_to_start_hours_total": float(row.get("wait_to_start_hours_total") or 0.0),
                "rows_total": int(row.get("rows_total") or 0),
                "rows_over_threshold": int(row.get("rows_over_threshold") or 0),
                "weekday": row.get("weekday"),
                "weekend": row.get("weekend"),
                "after_hours": row.get("after_hours"),
                "threshold_seconds": threshold_val,
            }
        )
    _write_csv(
        kit_dir / "busy_periods.csv",
        busy_rows,
        [
            "period_start",
            "period_end",
            "wait_to_start_hours_total",
            "rows_total",
            "rows_over_threshold",
            "weekday",
            "weekend",
            "after_hours",
            "threshold_seconds",
        ],
    )

    total_busy_hours = sum(
        float(item.get("wait_to_start_hours_total") or 0.0) for item in busy_periods_sorted
    )
    scenarios = [
        {
            "scenario": "current",
            "kpi_hours": total_busy_hours,
            "delta_hours": 0.0,
            "confidence": "Measured",
            "scale_factor": "",
            "unit": "hours",
            "population": "standalone_non_excluded",
            "threshold_seconds": threshold_val,
        }
    ]
    scale_factor = None
    plugin = report.get("plugins", {}).get("analysis_queue_delay_decomposition")
    if isinstance(plugin, dict):
        for item in plugin.get("findings") or []:
            if isinstance(item, dict) and item.get("kind") == "capacity_scale_model":
                scale_factor = item.get("scale_factor")
                break
    if isinstance(scale_factor, (int, float)) and total_busy_hours:
        modeled = float(total_busy_hours) * float(scale_factor)
        scenarios.append(
            {
                "scenario": "add_one_server",
                "kpi_hours": modeled,
                "delta_hours": modeled - float(total_busy_hours),
                "confidence": "Modeled",
                "scale_factor": float(scale_factor),
                "unit": "hours",
                "population": "standalone_non_excluded",
                "threshold_seconds": threshold_val,
            }
        )
    _write_csv(
        kit_dir / "scenario_summary.csv",
        scenarios,
        [
            "scenario",
            "kpi_hours",
            "delta_hours",
            "confidence",
            "scale_factor",
            "unit",
            "population",
            "threshold_seconds",
        ],
    )

    contributors = _top_process_contributors(report, limit=10)
    contrib_rows = []
    for row in contributors:
        contrib_rows.append(
            {
                "process": row.get("process"),
                "eligible_wait_gt_hours_total": row.get("eligible_wait_gt_hours_total"),
                "runs_total": row.get("runs_total"),
                "share_of_total": row.get("share_of_total"),
                "unit": "hours",
                "population": "standalone_non_excluded",
                "threshold_seconds": threshold_val,
            }
        )
    _write_csv(
        kit_dir / "top_process_contributors.csv",
        contrib_rows,
        [
            "process",
            "eligible_wait_gt_hours_total",
            "runs_total",
            "share_of_total",
            "unit",
            "population",
            "threshold_seconds",
        ],
    )

    qemail = _qemail_profile(report)
    qemail_rows = []
    if qemail:
        qemail_rows.append(
            {
                "process": qemail.get("process"),
                "eligible_wait_gt_hours_total": qemail.get("eligible_wait_gt_hours_total"),
                "eligible_wait_hours_total": qemail.get("eligible_wait_hours_total"),
                "runs_total": qemail.get("runs_total"),
                "runs_close": qemail.get("runs_close"),
                "runs_open": qemail.get("runs_open"),
                "unit": "hours",
                "population": "standalone_non_excluded",
                "threshold_seconds": threshold_val,
            }
        )
    _write_csv(
        kit_dir / "qemail_profile.csv",
        qemail_rows,
        [
            "process",
            "eligible_wait_gt_hours_total",
            "eligible_wait_hours_total",
            "runs_total",
            "runs_close",
            "runs_open",
            "unit",
            "population",
            "threshold_seconds",
        ],
    )
