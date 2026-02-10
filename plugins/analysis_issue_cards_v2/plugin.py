from __future__ import annotations

from typing import Any

from statistic_harness.core.report import (
    _evaluate_known_issues,
    _metric_spec,
    _metric_unit,
    _denominator_text,
    _format_issue_value,
)
from statistic_harness.core.report_v2_utils import (
    artifact_paths_for_plugin,
    build_minimal_report,
    claim_id,
    find_artifact_path,
    load_artifact_json,
)
from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import json_dumps, write_json


def _predicate_text(min_count: Any, max_count: Any, where: dict[str, Any] | None) -> str:
    parts: list[str] = []
    if min_count is not None:
        parts.append(f"count >= {min_count}")
    if max_count is not None:
        parts.append(f"count <= {max_count}")
    if where:
        parts.append(f"where={json_dumps(where)}")
    return " and ".join(parts) if parts else "count >= 1"


def _failure_reason(status: str) -> str:
    if status == "confirmed":
        return "PASS: expected evidence present."
    if status == "missing":
        return "FAIL: no matching findings."
    if status == "below_min":
        return "FAIL: observed below minimum threshold."
    if status == "over_limit":
        return "FAIL: observed above maximum threshold."
    return "REVIEW"


def _match_recommendations(
    recommendations: list[dict[str, Any]], target: str
) -> list[dict[str, Any]]:
    if not target:
        return []
    matches = []
    target_lower = target.lower()
    for item in recommendations:
        if str(item.get("target") or "").lower() == target_lower:
            matches.append(item)
    return matches


class Plugin:
    def run(self, ctx) -> PluginResult:
        report = build_minimal_report(
            ctx.storage,
            ctx.run_id,
            ctx.run_dir,
            project_id=ctx.project_id,
            dataset_version_id=ctx.dataset_version_id,
        )
        evaluations = _evaluate_known_issues(report)
        if not evaluations:
            return PluginResult(
                "ok", "No known issues to evaluate", {}, [], [], None
            )

        plugins = report.get("plugins") or {}
        rec_path = find_artifact_path(
            plugins, "analysis_recommendation_dedupe_v2", "recommendations.json"
        )
        rec_payload = load_artifact_json(ctx.run_dir, rec_path) if rec_path else None
        recs = []
        if isinstance(rec_payload, dict):
            recs = rec_payload.get("summary_recommendations") or []

        cards: list[dict[str, Any]] = []
        failures: list[str] = []
        for evaluation in evaluations:
            issue = evaluation.get("issue") or {}
            matched = evaluation.get("matched") or []
            item = matched[0] if matched else {}
            kind = evaluation.get("kind")
            spec = _metric_spec(kind)
            metric_name = spec.get("name", "Metric")
            definition = spec.get("definition", "")
            baseline_field = spec.get("baseline_field")
            observed_field = spec.get("observed_field")
            baseline_val = item.get(baseline_field) if baseline_field else None
            observed_val = item.get(observed_field) if observed_field else None
            target_expression = issue.get("target_expression")
            if not target_expression:
                target_expression = _predicate_text(
                    evaluation.get("min_count"),
                    evaluation.get("max_count"),
                    evaluation.get("where") if isinstance(evaluation.get("where"), dict) else None,
                )

            status = evaluation.get("status") or "unknown"
            decision = "PASS" if status == "confirmed" else "FAIL"
            failure_reason = _failure_reason(status)

            denominator = _denominator_text(item, report, spec)
            artifacts = artifact_paths_for_plugin(plugins, evaluation.get("plugin_id"))
            target_process = evaluation.get("process_hint") or ""
            recommended_actions = _match_recommendations(recs, target_process)

            predicate = _predicate_text(
                evaluation.get("min_count"),
                evaluation.get("max_count"),
                evaluation.get("where") if isinstance(evaluation.get("where"), dict) else None,
            )
            computed_values = {
                "count": evaluation.get("count"),
                "baseline": baseline_val,
                "observed": observed_val,
            }
            target_values = {
                "min_count": evaluation.get("min_count"),
                "max_count": evaluation.get("max_count"),
            }
            if decision == "FAIL" and (not predicate or computed_values.get("count") is None):
                failures.append(str(evaluation.get("label") or "issue"))

            cards.append(
                {
                    "issue_id": claim_id(
                        f"{evaluation.get('label')}:{evaluation.get('plugin_id')}:{kind}"
                    ),
                    "title": evaluation.get("label"),
                    "scope": evaluation.get("where") or evaluation.get("contains") or {},
                    "metric_name": metric_name,
                    "metric_definition": definition,
                    "unit": _metric_unit(metric_name, item),
                    "denominator": denominator,
                    "baseline_value": baseline_val,
                    "target_expression": target_expression,
                    "observed_value": observed_val,
                    "decision": decision,
                    "failure_reason": failure_reason if decision == "FAIL" else "",
                    "evidence": {
                        "plugin": evaluation.get("plugin_id"),
                        "kind": kind,
                        "measurement_type": item.get("measurement_type"),
                        "artifact_path": artifacts,
                        "query_or_grouping": {
                            "where": evaluation.get("where"),
                            "contains": evaluation.get("contains"),
                        },
                    },
                    "recommended_actions": recommended_actions,
                    "predicate_text": predicate,
                    "computed_values": computed_values,
                    "target_values": target_values,
                }
            )

        if failures:
            raise ValueError(
                "Failing checks missing predicate/computed values: "
                + ", ".join(failures)
            )

        artifacts_dir = ctx.artifacts_dir("analysis_issue_cards_v2")
        out_path = ctx.run_dir / "slide_kit" / "issue_cards.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(out_path, {"issue_cards": cards})

        md_path = artifacts_dir / "issue_cards.md"
        lines = ["# Issue Cards", ""]
        for card in cards:
            lines.append(f"## {card.get('title')}")
            lines.append(f"- Decision: {card.get('decision')}")
            lines.append(f"- Metric: {card.get('metric_name')} ({card.get('unit')})")
            lines.append(f"- Baseline: {_format_issue_value(card.get('baseline_value'))}")
            lines.append(f"- Observed: {_format_issue_value(card.get('observed_value'))}")
            lines.append(f"- Target: {card.get('target_expression')}")
            lines.append(f"- Reason: {card.get('failure_reason') or 'PASS'}")
            lines.append("")
        ctx.write_text(md_path, "\n".join(lines))

        artifacts = [
            PluginArtifact(
                path=str(out_path.relative_to(ctx.run_dir)),
                type="json",
                description="Issue cards (json)",
            ),
            PluginArtifact(
                path=str(md_path.relative_to(ctx.run_dir)),
                type="markdown",
                description="Issue cards (markdown)",
            ),
        ]

        metrics = {"issue_cards": len(cards)}
        findings = [
            {
                "kind": "issue_cards_summary",
                "issue_cards": len(cards),
                "measurement_type": "measured",
            }
        ]

        return PluginResult(
            "ok",
            f"Built {len(cards)} issue cards",
            metrics,
            findings,
            artifacts,
            None,
        )
