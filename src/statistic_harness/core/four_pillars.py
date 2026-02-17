from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


def _clamp_0_4(value: float) -> float:
    return max(0.0, min(4.0, float(value)))


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _score_piecewise(value: float, bands: list[tuple[float, float]]) -> float:
    for threshold, score in bands:
        if value <= threshold:
            return _clamp_0_4(score)
    return 0.0


def _parse_iso(value: Any) -> datetime | None:
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


def _runtime_minutes_from_run(run_row: dict[str, Any] | None) -> float | None:
    if not isinstance(run_row, dict):
        return None
    start = _parse_iso(run_row.get("started_at")) or _parse_iso(run_row.get("created_at"))
    end = _parse_iso(run_row.get("completed_at"))
    if start is None or end is None:
        return None
    minutes = (end - start).total_seconds() / 60.0
    return float(minutes) if minutes > 0 else None


@dataclass(frozen=True)
class PillarScore:
    score: float
    components: dict[str, float]
    rationale: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "score_0_4": _clamp_0_4(self.score),
            "components": {k: _clamp_0_4(v) for k, v in self.components.items()},
            "rationale": list(self.rationale),
        }


def _plugin_status_counts(report: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    plugins = report.get("plugins")
    if not isinstance(plugins, dict):
        return counts
    for _pid, payload in plugins.items():
        if not isinstance(payload, dict):
            continue
        status = str(payload.get("status") or "unknown").strip().lower() or "unknown"
        counts[status] = int(counts.get(status, 0)) + 1
    return counts


def _finding_stats(report: dict[str, Any]) -> dict[str, float]:
    total_findings = 0
    measured_tagged = 0
    with_evidence = 0
    plugins = report.get("plugins")
    if not isinstance(plugins, dict):
        return {
            "total_findings": 0.0,
            "measurement_tag_coverage": 0.0,
            "evidence_coverage": 0.0,
        }
    for _pid, payload in plugins.items():
        if not isinstance(payload, dict):
            continue
        findings = payload.get("findings")
        if not isinstance(findings, list):
            continue
        for item in findings:
            if not isinstance(item, dict):
                continue
            total_findings += 1
            if isinstance(item.get("measurement_type"), str):
                measured_tagged += 1
            evidence = item.get("evidence")
            if isinstance(evidence, dict) and len(evidence) > 0:
                with_evidence += 1
    if total_findings <= 0:
        return {
            "total_findings": 0.0,
            "measurement_tag_coverage": 0.0,
            "evidence_coverage": 0.0,
        }
    return {
        "total_findings": float(total_findings),
        "measurement_tag_coverage": float(measured_tagged) / float(total_findings),
        "evidence_coverage": float(with_evidence) / float(total_findings),
    }


def _known_issue_quality(report: dict[str, Any]) -> tuple[float, int]:
    recs = report.get("recommendations")
    if not isinstance(recs, dict):
        return 0.0, 0
    known_block = recs.get("known")
    if not isinstance(known_block, dict):
        return 0.0, 0
    items = known_block.get("items")
    if not isinstance(items, list) or not items:
        return 0.0, 0
    total = 0
    confirmed = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        total += 1
        status = str(item.get("status") or "").strip().lower()
        if status == "confirmed":
            confirmed += 1
    if total <= 0:
        return 0.0, 0
    return float(confirmed) / float(total), total


def _modeled_coverage(report: dict[str, Any]) -> float:
    recs = report.get("recommendations")
    if not isinstance(recs, dict):
        return 0.0
    discovery_block = recs.get("discovery")
    if isinstance(discovery_block, dict):
        items = discovery_block.get("items")
    else:
        items = recs.get("items")
    if not isinstance(items, list) or not items:
        return 0.0
    total = 0
    modeled = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        total += 1
        if isinstance(item.get("modeled_percent"), (int, float)):
            modeled += 1
    if total <= 0:
        return 0.0
    return float(modeled) / float(total)


def _traceability_signal(report: dict[str, Any]) -> float:
    # Use multiple independent report signals so this is not tied to a single plugin.
    plugins = report.get("plugins")
    if not isinstance(plugins, dict):
        return 0.0
    checks: list[bool] = []
    for pid in (
        "analysis_traceability_manifest_v2",
        "report_decision_bundle_v2",
        "analysis_recommendation_dedupe_v2",
    ):
        payload = plugins.get(pid)
        checks.append(isinstance(payload, dict) and str(payload.get("status") or "") in {"ok", "na"})
    return float(sum(1 for v in checks if v)) / float(len(checks)) if checks else 0.0


def _reproducibility_signal(report: dict[str, Any]) -> float:
    lineage = report.get("lineage")
    if not isinstance(lineage, dict):
        return 0.0
    run = lineage.get("run")
    plugins = lineage.get("plugins")
    run_fp = str((run or {}).get("run_fingerprint") or "").strip() if isinstance(run, dict) else ""
    has_run_fp = 1.0 if run_fp else 0.0
    plugin_cov = 0.0
    if isinstance(plugins, dict) and plugins:
        total = 0
        with_fp = 0
        for _pid, entry in plugins.items():
            if not isinstance(entry, dict):
                continue
            total += 1
            if str(entry.get("execution_fingerprint") or "").strip():
                with_fp += 1
        if total > 0:
            plugin_cov = float(with_fp) / float(total)
    return (has_run_fp * 0.4) + (plugin_cov * 0.6)


def _security_signals(report: dict[str, Any]) -> tuple[int, int]:
    plugins = report.get("plugins")
    if not isinstance(plugins, dict):
        return 0, 0
    violation_count = 0
    fail_closed_count = 0
    for _pid, payload in plugins.items():
        if not isinstance(payload, dict):
            continue
        status = str(payload.get("status") or "").strip().lower()
        summary = str(payload.get("summary") or "").strip().lower()
        findings = payload.get("findings")
        if isinstance(findings, list):
            for item in findings:
                if not isinstance(item, dict):
                    continue
                kind = str(item.get("kind") or "").lower()
                if "security" in kind or "policy" in kind or "pii" in kind:
                    violation_count += 1
        # "File read denied" and similar are evidence of fail-closed policy in action.
        if status in {"error", "na"} and (
            "file read denied" in summary
            or "permission denied" in summary
            or "sandbox" in summary
            or "policy" in summary
        ):
            fail_closed_count += 1
    return violation_count, fail_closed_count


def _performant_score(report: dict[str, Any], run_row: dict[str, Any] | None = None) -> PillarScore:
    runtime_minutes = _runtime_minutes_from_run(run_row)
    hotspots = report.get("hotspots") if isinstance(report.get("hotspots"), dict) else {}
    top_rss = hotspots.get("top_by_max_rss_kb") if isinstance(hotspots, dict) else None
    max_rss_kb = 0.0
    if isinstance(top_rss, list):
        for item in top_rss:
            if not isinstance(item, dict):
                continue
            value = _safe_float(item.get("max_rss_kb"))
            if isinstance(value, float):
                max_rss_kb = max(max_rss_kb, value)

    status_counts = _plugin_status_counts(report)
    total_plugins = max(1, sum(status_counts.values()))
    ok_weighted = (
        float(status_counts.get("ok", 0))
        + float(status_counts.get("na", 0)) * 0.8
        + float(status_counts.get("skipped", 0)) * 0.8
        + float(status_counts.get("degraded", 0)) * 0.4
    ) / float(total_plugins)

    runtime_component = 2.5
    if isinstance(runtime_minutes, float):
        runtime_component = _score_piecewise(
            runtime_minutes,
            [
                (15.0, 4.0),
                (30.0, 3.2),
                (60.0, 2.4),
                (120.0, 1.5),
            ],
        )
    rss_component = _score_piecewise(
        max_rss_kb,
        [
            (512_000.0, 4.0),   # <= 0.5GB
            (1_048_576.0, 3.3), # <= 1.0GB
            (2_097_152.0, 2.4), # <= 2.0GB
            (4_194_304.0, 1.4), # <= 4.0GB
        ],
    )
    reliability_component = _clamp_0_4(ok_weighted * 4.0)

    score = (runtime_component * 0.35) + (rss_component * 0.25) + (reliability_component * 0.40)
    rationale = [
        f"Runtime component={runtime_component:.2f} (minutes={runtime_minutes if runtime_minutes is not None else 'unknown'})",
        f"Memory component={rss_component:.2f} (max_rss_kb={max_rss_kb:.0f})",
        f"Execution reliability component={reliability_component:.2f}",
    ]
    return PillarScore(
        score=score,
        components={
            "runtime": runtime_component,
            "memory": rss_component,
            "execution_reliability": reliability_component,
        },
        rationale=rationale,
    )


def _accurate_score(report: dict[str, Any]) -> PillarScore:
    known_ratio, known_total = _known_issue_quality(report)
    modeled_cov = _modeled_coverage(report)
    status_counts = _plugin_status_counts(report)
    total_plugins = max(1, sum(status_counts.values()))
    quality_ratio = (
        float(status_counts.get("ok", 0))
        + float(status_counts.get("na", 0)) * 0.85
        + float(status_counts.get("skipped", 0)) * 0.85
        + float(status_counts.get("degraded", 0)) * 0.45
    ) / float(total_plugins)

    known_component = _clamp_0_4(known_ratio * 4.0)
    modeled_component = _clamp_0_4(modeled_cov * 4.0)
    quality_component = _clamp_0_4(quality_ratio * 4.0)
    score = (known_component * 0.45) + (modeled_component * 0.25) + (quality_component * 0.30)
    rationale = [
        f"Known-issue confirmation={known_ratio:.2f} across {known_total} checks",
        f"Modeled recommendation coverage={modeled_cov:.2f}",
        f"Plugin-quality ratio={quality_ratio:.2f}",
    ]
    return PillarScore(
        score=score,
        components={
            "known_issue_confirmation": known_component,
            "modeled_coverage": modeled_component,
            "execution_quality": quality_component,
        },
        rationale=rationale,
    )


def _secure_score(report: dict[str, Any]) -> PillarScore:
    violation_count, fail_closed_count = _security_signals(report)
    base = 3.6
    # Hard penalties for explicit violations.
    penalty = min(3.0, float(violation_count) * 0.4)
    # Minor positive for demonstrated fail-closed behavior.
    fail_closed_bonus = min(0.3, float(fail_closed_count) * 0.05)
    score = _clamp_0_4(base - penalty + fail_closed_bonus)
    rationale = [
        f"Security/policy violation findings={violation_count}",
        f"Fail-closed guard activations={fail_closed_count}",
    ]
    return PillarScore(
        score=score,
        components={
            "policy_violation_penalty": _clamp_0_4(4.0 - penalty),
            "fail_closed_behavior": _clamp_0_4(3.0 + fail_closed_bonus),
        },
        rationale=rationale,
    )


def _citable_score(report: dict[str, Any]) -> PillarScore:
    finding_stats = _finding_stats(report)
    tag_cov = float(finding_stats.get("measurement_tag_coverage") or 0.0)
    evidence_cov = float(finding_stats.get("evidence_coverage") or 0.0)
    traceability = _traceability_signal(report)
    reproducibility = _reproducibility_signal(report)

    measurement_component = _clamp_0_4(tag_cov * 4.0)
    evidence_component = _clamp_0_4(evidence_cov * 4.0)
    trace_component = _clamp_0_4(traceability * 4.0)
    repro_component = _clamp_0_4(reproducibility * 4.0)
    score = (
        measurement_component * 0.25
        + evidence_component * 0.30
        + trace_component * 0.25
        + repro_component * 0.20
    )
    rationale = [
        f"Measurement tagging coverage={tag_cov:.2f}",
        f"Evidence coverage={evidence_cov:.2f}",
        f"Traceability signal={traceability:.2f}",
        f"Reproducibility signal={reproducibility:.2f}",
    ]
    return PillarScore(
        score=score,
        components={
            "measurement_tags": measurement_component,
            "evidence_quality": evidence_component,
            "traceability": trace_component,
            "reproducibility": repro_component,
        },
        rationale=rationale,
    )


def build_four_pillars_scorecard(
    report: dict[str, Any],
    *,
    run_row: dict[str, Any] | None = None,
    balance_max_spread: float = 1.0,
    min_floor: float = 2.5,
    degradation_tolerance: float = 0.2,
) -> dict[str, Any]:
    performant = _performant_score(report, run_row=run_row)
    accurate = _accurate_score(report)
    secure = _secure_score(report)
    citable = _citable_score(report)

    pillars = {
        "performant": performant.as_dict(),
        "accurate": accurate.as_dict(),
        "secure": secure.as_dict(),
        "citable": citable.as_dict(),
    }
    values = [pillars[p]["score_0_4"] for p in ("performant", "accurate", "secure", "citable")]
    min_v = min(values) if values else 0.0
    max_v = max(values) if values else 0.0
    spread = max_v - min_v
    mean_v = sum(values) / float(len(values)) if values else 0.0

    spread_over = max(0.0, spread - float(balance_max_spread))
    balance_penalty = min(2.0, spread_over * 1.5)
    balanced_score = _clamp_0_4(mean_v - balance_penalty)
    balance_index = _clamp_0_4(4.0 - (spread * 2.0))

    vetoes: list[dict[str, Any]] = []
    if min_v < float(min_floor):
        vetoes.append(
            {
                "code": "pillar_floor_breach",
                "message": f"At least one pillar is below floor {float(min_floor):.2f}",
                "minimum_pillar": float(min_v),
            }
        )
    if spread > float(balance_max_spread):
        vetoes.append(
            {
                "code": "pillar_imbalance",
                "message": f"Pillar spread {spread:.2f} exceeds max spread {float(balance_max_spread):.2f}",
                "spread": float(spread),
            }
        )
    # Explicit tradeoff warning: high one-side gains with low minimum pillar is not allowed.
    if max_v >= 3.5 and min_v <= (float(min_floor) - float(degradation_tolerance)):
        vetoes.append(
            {
                "code": "tradeoff_rejected",
                "message": "High score in one pillar cannot compensate for weak pillar performance.",
                "max_pillar": float(max_v),
                "min_pillar": float(min_v),
            }
        )

    if vetoes:
        status = "vetoed_imbalanced"
    elif min_v >= 3.5 and spread <= 0.5:
        status = "balanced_optimal"
    elif spread <= float(balance_max_spread):
        status = "balanced"
    else:
        status = "imbalanced"

    return {
        "scale": "0.0-4.0",
        "pillars": pillars,
        "balance": {
            "objective": "max-min with spread penalty",
            "min_pillar": float(min_v),
            "max_pillar": float(max_v),
            "spread": float(spread),
            "mean_score": float(mean_v),
            "balance_penalty": float(balance_penalty),
            "balance_index_0_4": float(balance_index),
            "balanced_score_0_4": float(balanced_score),
            "constraints": {
                "min_floor": float(min_floor),
                "max_spread": float(balance_max_spread),
                "degradation_tolerance": float(degradation_tolerance),
            },
            "vetoes": vetoes,
            "status": status,
        },
        "summary": {
            "overall_0_4": float(balanced_score),
            "status": status,
            "vetoed": bool(vetoes),
        },
        "notes": [
            "Scores are derived from full run functionality and telemetry, not only Kona/ideaspace signals.",
            "Balance policy rejects one-pillar gains that materially degrade another pillar.",
        ],
    }
