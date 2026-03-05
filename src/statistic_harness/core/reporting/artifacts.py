from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import _include_capacity_recommendations
from .known_issues import _evaluate_known_issues, _item_process_norm
from .scoring import _confidence_weight, _controllability_weight
from .text import _format_issue_value


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
        "close_cycle_duration_shift": {
            "name": "Close-cycle median duration (seconds)",
            "definition": "Median close-cycle duration vs open-window duration for the process.",
            "baseline_field": "median_open",
            "observed_field": "median_close",
            "denominator_field": "close_count",
        },
        "upload_bkrvnu_linkage": {
            "name": "BKRVNU upload linkage coverage (%)",
            "definition": "Share of BKRVNU rows that can be matched to an XLSX upload by user + time window.",
            "baseline_field": "matched_user_pct",
            "observed_field": "matched_user_pct",
            "denominator_field": "bkrvnu_rows",
        },
        "process_counterfactual": {
            "name": "Over-threshold wait-to-start (hours)",
            "definition": "Modeled process wait above threshold after reducing to baseline quantile.",
            "baseline_field": "baseline_over_threshold_hours",
            "observed_field": "modeled_over_threshold_hours",
            "denominator_field": "runs_count",
        },
        "sequence_bottleneck": {
            "name": "Transition over-threshold wait (hours)",
            "definition": "Modeled transition gap above threshold after reducing to baseline quantile.",
            "baseline_field": "baseline_over_threshold_hours",
            "observed_field": "modeled_over_threshold_hours",
            "denominator_field": "transition_count",
        },
        "user_host_savings": {
            "name": "Over-threshold wait-to-start (hours)",
            "definition": "Modeled savings from rebalancing user/host to baseline wait level.",
            "baseline_field": "baseline_over_threshold_hours",
            "observed_field": "modeled_over_threshold_hours",
            "denominator_field": "runs_count",
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
        "delta_value",
        "delta_hours",
        "modeled_delta",
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


def _item_evidence_row(item: dict[str, Any]) -> dict[str, Any]:
    evidence = item.get("evidence")
    if isinstance(evidence, list):
        for row in evidence:
            if isinstance(row, dict):
                return row
    if isinstance(evidence, dict):
        return evidence
    return {}


def _target_process_ids_for_item(item: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for bucket in (item, _item_evidence_row(item)):
        raw = bucket.get("target_process_ids") if isinstance(bucket, dict) else None
        if isinstance(raw, list):
            for value in raw:
                if isinstance(value, str) and value.strip():
                    out.append(value.strip().lower())
    process_norm = _item_process_norm(item)
    if process_norm:
        out.append(process_norm)
    deduped: list[str] = []
    seen: set[str] = set()
    for token in out:
        if not token or token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped


def _scope_type_for_item(item: dict[str, Any]) -> str:
    targets = _target_process_ids_for_item(item)
    return "grouped_explicit" if len(targets) > 1 else "single_process"


def _primary_process_for_item(item: dict[str, Any]) -> str:
    targets = _target_process_ids_for_item(item)
    return targets[0] if targets else _item_process_norm(item)


def _scope_class_for_item(item: dict[str, Any]) -> str:
    plugin_id = str(item.get("plugin_id") or "").strip().lower()
    kind = str(item.get("kind") or "").strip().lower()
    action_type = str(item.get("action_type") or item.get("action") or "").strip().lower()
    text = str(item.get("recommendation") or "").strip().lower()
    process_norm = _item_process_norm(item)

    if plugin_id.startswith("analysis_close_cycle_"):
        return "close_specific"
    if kind.startswith("close_cycle_") or "spillover" in kind:
        return "close_specific"
    if "close-cycle" in text or "month-end" in text or "eom" in text:
        return "close_specific"
    if process_norm in {"qemail", "qpec"} and action_type in {"tune_schedule", "add_server"}:
        return "close_specific"
    return "general"


def _metric_unit(metric_name: str, item: dict[str, Any] | None = None) -> str:
    name = metric_name.lower()
    if "%" in name or "percent" in name or "pct" in name:
        return "percent"
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
            if key.endswith("_pct"):
                return "percent"
    return "count"


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

    if scale and _include_capacity_recommendations():
        base = scale.get("eligible_wait_gt_hours_without_target")
        modeled = scale.get("eligible_wait_gt_hours_modeled")
        if isinstance(base, (int, float)) and isinstance(modeled, (int, float)) and base:
            delta = float(base) - float(modeled)
            pct = (delta / float(base)) * 100.0 if base else 0.0
            lines.append(
                "QPEC+1 recommended (modeled): "
                f">threshold eligible wait drops from {float(base):.2f}h to "
                f"{float(modeled):.2f}h (\u0394 {delta:.2f}h, {pct:.1f}%)."
            )

    return lines
