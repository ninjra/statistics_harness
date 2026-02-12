from __future__ import annotations

from typing import Any

import csv
from pathlib import Path

from statistic_harness.core.report import _collapse_findings
from statistic_harness.core.report_v2_utils import (
    build_minimal_report,
    compute_artifact_manifest,
    ensure_modeled_fields,
    load_artifact_json,
    load_plugin_payloads,
)
from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import json_dumps, now_iso, write_json


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader]


def _format_table(headers: list[str], rows: list[dict[str, Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")).strip() for h in headers) + " |")
    return lines


def _busy_period_note(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No busy periods detected."
    weekend = sum(1 for row in rows if str(row.get("weekend")).lower() == "true")
    after_hours = sum(1 for row in rows if str(row.get("after_hours")).lower() == "true")
    total = len(rows)
    weekend_pct = (weekend / total * 100.0) if total else 0.0
    after_pct = (after_hours / total * 100.0) if total else 0.0
    return (
        f"Busy periods are {weekend_pct:.0f}% weekend and {after_pct:.0f}% after-hours based on top {total}."
    )


def _record_contract_failure(
    failures: list[dict[str, Any]],
    *,
    failure_class: str,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "failure_class": failure_class,
        "code": code,
        "message": message,
    }
    if isinstance(details, dict) and details:
        payload["details"] = details
    failures.append(payload)


class Plugin:
    def run(self, ctx) -> PluginResult:
        report = build_minimal_report(
            ctx.storage,
            ctx.run_id,
            ctx.run_dir,
            project_id=ctx.project_id,
            dataset_version_id=ctx.dataset_version_id,
        )
        plugins = report.get("plugins") or {}

        modeled_errors = ensure_modeled_fields(plugins)
        has_modeled_gaps = bool(modeled_errors)
        degraded = False

        issue_cards_path = ctx.run_dir / "slide_kit" / "issue_cards.json"
        trace_path = ctx.run_dir / "slide_kit" / "traceability_manifest.json"
        busy_csv_path = ctx.run_dir / "slide_kit" / "busy_periods.csv"
        scenario_csv_path = ctx.run_dir / "slide_kit" / "scenario_summary.csv"
        process_csv_path = ctx.run_dir / "slide_kit" / "top_process_contributors.csv"
        waterfall_csv_path = ctx.run_dir / "slide_kit" / "waterfall_summary.csv"
        busy_def_path = ctx.run_dir / "artifacts" / "analysis_busy_period_segmentation_v2" / "definition.json"
        recs_path = ctx.run_dir / "artifacts" / "analysis_recommendation_dedupe_v2" / "recommendations.json"

        issue_cards = load_artifact_json(ctx.run_dir, str(issue_cards_path.relative_to(ctx.run_dir)))
        trace_payload = load_artifact_json(ctx.run_dir, str(trace_path.relative_to(ctx.run_dir)))
        busy_rows = _read_csv(busy_csv_path)
        scenario_rows = _read_csv(scenario_csv_path)
        process_rows = _read_csv(process_csv_path)
        waterfall_rows = _read_csv(waterfall_csv_path)
        busy_def = load_artifact_json(ctx.run_dir, str(busy_def_path.relative_to(ctx.run_dir)))
        recs_payload = load_artifact_json(ctx.run_dir, str(recs_path.relative_to(ctx.run_dir)))

        contract_failures: list[dict[str, Any]] = []

        required_inputs = [
            ("scenario_summary_csv", scenario_csv_path),
            ("top_process_contributors_csv", process_csv_path),
            ("busy_periods_csv", busy_csv_path),
        ]
        for check_name, path in required_inputs:
            if not path.exists():
                _record_contract_failure(
                    contract_failures,
                    failure_class="runtime",
                    code="required_input_missing",
                    message=f"Missing required input: {check_name}",
                    details={"path": str(path.relative_to(ctx.run_dir))},
                )

        trace_claims = (trace_payload or {}).get("claims", [])
        if not isinstance(trace_claims, list):
            _record_contract_failure(
                contract_failures,
                failure_class="traceability",
                code="traceability_manifest_malformed",
                message="Traceability manifest claims payload must be a list.",
                details={"path": str(trace_path.relative_to(ctx.run_dir))},
            )
            trace_claims = []
        claims = {c.get("claim_id"): c for c in trace_claims if isinstance(c, dict)}

        used_claim_ids: list[str] = []
        missing_claim_ids: set[str] = set()
        for row in scenario_rows + process_rows + busy_rows[:10]:
            cid = row.get("claim_id")
            if cid:
                used_claim_ids.append(cid)
                if cid not in claims:
                    missing_claim_ids.add(str(cid))
        trace_manifest_present = bool(trace_path.exists() and trace_claims)
        synthesized_traceability = False
        if missing_claim_ids and trace_manifest_present:
            _record_contract_failure(
                contract_failures,
                failure_class="mapping",
                code="missing_claim_mapping",
                message="One or more claim_id values do not exist in traceability_manifest claims.",
                details={"claim_ids": sorted(missing_claim_ids)},
            )
        elif missing_claim_ids:
            # Fail-closed for reporting: when no traceability manifest exists, synthesize
            # local claim stubs so business/engineering bundles are still explainable.
            synthesized_traceability = True
            for cid in sorted(missing_claim_ids):
                claims[cid] = {
                    "claim_id": cid,
                    "summary_text": "Synthesized claim mapping (traceability manifest unavailable).",
                    "source": {
                        "plugin": "report_decision_bundle_v2",
                        "kind": "synthesized_traceability",
                        "measurement_type": "unknown",
                        "artifact_path": "",
                        "query_or_grouping": "",
                    },
                }
            missing_claim_ids = set()
        degraded = bool(contract_failures)

        enable_redaction = bool(ctx.settings.get("enable_redaction", False))

        busy_threshold = (
            busy_def.get("wait_threshold_seconds") if isinstance(busy_def, dict) else None
        )
        gap_tolerance = (
            busy_def.get("gap_tolerance_seconds") if isinstance(busy_def, dict) else None
        )
        if busy_threshold is None:
            busy_threshold = 60
        if gap_tolerance is None:
            gap_tolerance = 60
        population_text = "over-threshold runs"
        if report.get("input", {}).get("rows"):
            population_text += f" (rows={report.get('input', {}).get('rows')})"

        business_lines: list[str] = []
        business_lines.append("# Business Summary")
        business_lines.append("")
        business_lines.append("Primary KPI: Busy-period over-threshold wait-to-start hours (BP_OT_WTS_HOURS)")
        business_lines.append("Unit: hours")
        business_lines.append(
            f"Threshold: {busy_threshold}s (over_threshold_wait_sec = max(wait_to_start_sec - threshold_sec, 0))"
        )
        business_lines.append(f"Population: {population_text}")
        business_lines.append("")
        business_lines.append("## Scenario Summary")
        business_lines.append(
            f"Unit: hours. Population: {population_text}. Threshold: {busy_threshold}s."
        )
        business_lines.extend(
            _format_table(
                [
                    "scenario_id",
                    "scenario_name",
                    "bp_over_threshold_wait_hours",
                    "delta_hours_vs_current",
                    "delta_percent_vs_current",
                    "evidence_type",
                    "claim_id",
                ],
                scenario_rows[:6],
            )
        )
        business_lines.append("Full detail: slide_kit/scenario_summary.csv")
        business_lines.append("")
        business_lines.append("## Busy Periods (Top 10)")
        business_lines.append(
            f"Unit: hours. Population: {population_text}. Threshold: {busy_threshold}s. Gap tolerance: {gap_tolerance}s."
        )
        business_lines.extend(
            _format_table(
                [
                    "busy_period_id",
                    "start_ts",
                    "end_ts",
                    "total_over_threshold_wait_hours",
                    "runs_over_threshold_count",
                    "top_process_id",
                    "claim_id",
                ],
                busy_rows[:10],
            )
        )
        business_lines.append("Full detail: slide_kit/busy_periods.csv")
        business_lines.append(_busy_period_note(busy_rows[:10]))
        business_lines.append("")
        business_lines.append("## Top Process Contributors (Top 5)")
        business_lines.append(
            f"Unit: hours. Population: {population_text}. Threshold: {busy_threshold}s."
        )
        business_lines.extend(
            _format_table(
                [
                    "process_id",
                    "bp_over_threshold_wait_hours",
                    "share_percent",
                    "claim_id",
                ],
                process_rows[:5],
            )
        )
        business_lines.append("Full detail: slide_kit/top_process_contributors.csv")
        business_lines.append("")
        business_lines.append("## Recommendations (Top 10)")
        summary_recs = []
        if isinstance(recs_payload, dict):
            summary_recs = recs_payload.get("summary_recommendations") or []
        for idx, rec in enumerate(summary_recs[:10], start=1):
            target = rec.get("target") or ""
            evidence = rec.get("evidence") or []
            delta_hours = rec.get("delta_hours")
            business_lines.append(f"### Recommendation {idx}")
            business_lines.append(f"Problem: {target or 'Reduce over-threshold wait.'}")
            business_lines.append(f"Evidence: {json_dumps(evidence)[:200]}")
            business_lines.append(f"Action: {rec.get('action_type')}")
            business_lines.append("Expected impact:")
            business_lines.extend(
                _format_table(
                    ["current_hours", "modeled_hours", "delta_hours"],
                    [
                        {
                            "current_hours": scenario_rows[0].get("bp_over_threshold_wait_hours")
                            if scenario_rows
                            else "",
                            "modeled_hours": (
                                scenario_rows[1].get("bp_over_threshold_wait_hours")
                                if len(scenario_rows) > 1
                                else ""
                            ),
                            "delta_hours": delta_hours,
                        }
                    ],
                )
            )
            business_lines.append(f"Confidence: {rec.get('confidence_tag')}")
            business_lines.append(
                f"How to validate: {'; '.join(rec.get('validation_steps') or [])}"
            )
            business_lines.append("")
        if summary_recs:
            business_lines.append(
                "Full detail: artifacts/analysis_recommendation_dedupe_v2/recommendations.json"
            )
            business_lines.append("")

        business_path = ctx.run_dir / "business_summary.md"
        ctx.write_text(business_path, "\n".join(business_lines))

        engineering_lines: list[str] = []
        engineering_lines.append("# Engineering Summary")
        engineering_lines.append("")
        if has_modeled_gaps:
            engineering_lines.append("## Modeled Finding Field Gaps")
            engineering_lines.append(
                "Some modeled findings are missing preferred normalization fields."
            )
            engineering_lines.append(
                "The bundle was still generated, and these gaps were recorded for follow-up."
            )
            engineering_lines.append("")
            engineering_lines.extend([f"- {e}" for e in modeled_errors[:50]])
            if len(modeled_errors) > 50:
                engineering_lines.append(f"- ... and {len(modeled_errors) - 50} more")
            engineering_lines.append("")

        engineering_lines.append("## Glossary")
        engineering_lines.append("- eligible_wait: time between eligible/queued and start.")
        engineering_lines.append("- threshold: wait-to-start seconds above which delay is counted.")
        engineering_lines.append("- busy period: merged intervals of over-threshold waiting.")
        engineering_lines.append("- close cycle window: configured close period for month end.")
        engineering_lines.append("")

        engineering_lines.append("## Issue Cards")
        cards = (issue_cards or {}).get("issue_cards") if isinstance(issue_cards, dict) else []
        if cards:
            engineering_lines.extend(
                _format_table(
                    [
                        "issue_id",
                        "title",
                        "decision",
                        "metric_name",
                        "baseline_value",
                        "observed_value",
                        "target_expression",
                        "failure_reason",
                    ],
                    cards,
                )
            )
            engineering_lines.append("Full detail: slide_kit/issue_cards.json")
        else:
            engineering_lines.append("No issue cards available.")
        engineering_lines.append("")

        engineering_lines.append("## Checks")
        if cards:
            for card in cards:
                engineering_lines.append(f"- {card.get('title')}")
                engineering_lines.append(
                    f"Predicate: {card.get('predicate_text')}; "
                    f"Computed: {json_dumps(card.get('computed_values'))}; "
                    f"Target: {json_dumps(card.get('target_values'))}; "
                    f"Reason: {card.get('failure_reason')}"
                )
        else:
            engineering_lines.append("No checks available.")
        engineering_lines.append("")

        engineering_lines.append("## Contract Checks")
        if contract_failures:
            engineering_lines.append(
                "Report contract checks found violations. Decision bundle was emitted in degraded mode."
            )
            engineering_lines.extend(
                _format_table(
                    ["failure_class", "code", "message", "details"],
                    [
                        {
                            "failure_class": row.get("failure_class"),
                            "code": row.get("code"),
                            "message": row.get("message"),
                            "details": json_dumps(row.get("details") or {}),
                        }
                        for row in contract_failures
                    ],
                )
            )
        else:
            engineering_lines.append("All report contract checks passed.")
        engineering_lines.append("")

        engineering_lines.append("## Traceability")
        if synthesized_traceability:
            engineering_lines.append(
                "Traceability manifest was unavailable; synthesized claim mappings were used for referenced claim IDs."
            )
            engineering_lines.append("")
        trace_rows = []
        for cid in used_claim_ids:
            claim = claims.get(cid) or {}
            source = claim.get("source") or {}
            trace_rows.append(
                {
                    "claim_id": cid,
                    "summary_text": claim.get("summary_text") or ("MISSING claim mapping" if cid in missing_claim_ids else ""),
                    "plugin": source.get("plugin") or ("<missing>" if cid in missing_claim_ids else ""),
                    "kind": source.get("kind") or ("mapping" if cid in missing_claim_ids else ""),
                    "measurement_type": source.get("measurement_type") or ("unknown" if cid in missing_claim_ids else ""),
                    "artifact_path": source.get("artifact_path") or ("<missing>" if cid in missing_claim_ids else ""),
                    "query_or_grouping": source.get("query_or_grouping") or "",
                }
            )
        if trace_rows:
            engineering_lines.extend(
                _format_table(
                    [
                        "claim_id",
                        "summary_text",
                        "plugin",
                        "kind",
                        "measurement_type",
                        "artifact_path",
                        "query_or_grouping",
                    ],
                    trace_rows,
                )
            )
            engineering_lines.append("Full detail: slide_kit/traceability_manifest.json")
        else:
            engineering_lines.append("No traceability claims referenced.")

        engineering_path = ctx.run_dir / "engineering_summary.md"
        ctx.write_text(engineering_path, "\n".join(engineering_lines))

        appendix_lines: list[str] = []
        appendix_lines.append("# Appendix (Raw)")
        appendix_lines.append("")
        payloads = load_plugin_payloads(ctx.storage, ctx.run_id)
        for plugin_id in sorted(payloads.keys()):
            payload = payloads[plugin_id]
            appendix_lines.append(f"## {plugin_id}")
            appendix_lines.append(f"Status: {payload.get('status')}")
            appendix_lines.append(f"Summary: {payload.get('summary')}")
            appendix_lines.append(f"Metrics: {json_dumps(payload.get('metrics'))}")
            collapsed = _collapse_findings(payload.get("findings") or [])
            appendix_lines.append(
                f"Findings: count={collapsed.get('count')} unique={collapsed.get('unique_count')}"
            )
            appendix_lines.append(
                f"Top examples: {json_dumps(collapsed.get('top_examples'))}"
            )
            artifacts = payload.get("artifacts") or []
            appendix_lines.append(f"Artifacts: {json_dumps(artifacts)}")
            appendix_lines.append("")

        appendix_path = ctx.run_dir / "appendix_raw.md"
        ctx.write_text(appendix_path, "\n".join(appendix_lines))

        contract_path = ctx.run_dir / "slide_kit" / "report_contract_checks.json"
        write_json(
            contract_path,
            {
                "schema_version": "v1",
                "status": "failed" if contract_failures else "passed",
                "failure_count": int(len(contract_failures)),
                "failures": contract_failures,
            },
        )

        manifest_entries = [
            {
                "path": str(business_path.relative_to(ctx.run_dir)),
                "source_plugins": ["report_decision_bundle_v2"],
                "created_at_utc": now_iso(),
            },
            {
                "path": str(engineering_path.relative_to(ctx.run_dir)),
                "source_plugins": ["report_decision_bundle_v2"],
                "created_at_utc": now_iso(),
            },
            {
                "path": str(appendix_path.relative_to(ctx.run_dir)),
                "source_plugins": ["report_decision_bundle_v2"],
                "created_at_utc": now_iso(),
            },
            {
                "path": str(busy_csv_path.relative_to(ctx.run_dir)),
                "source_plugins": ["analysis_busy_period_segmentation_v2"],
                "created_at_utc": now_iso(),
            },
            {
                "path": str(waterfall_csv_path.relative_to(ctx.run_dir)),
                "source_plugins": ["analysis_waterfall_summary_v2"],
                "created_at_utc": now_iso(),
            },
            {
                "path": str(scenario_csv_path.relative_to(ctx.run_dir)),
                "source_plugins": ["report_slide_kit_emitter_v2"],
                "created_at_utc": now_iso(),
            },
            {
                "path": str(process_csv_path.relative_to(ctx.run_dir)),
                "source_plugins": ["report_slide_kit_emitter_v2"],
                "created_at_utc": now_iso(),
            },
            {
                "path": str(issue_cards_path.relative_to(ctx.run_dir)),
                "source_plugins": ["analysis_issue_cards_v2"],
                "created_at_utc": now_iso(),
            },
            {
                "path": str(trace_path.relative_to(ctx.run_dir)),
                "source_plugins": ["analysis_traceability_manifest_v2"],
                "created_at_utc": now_iso(),
            },
            {
                "path": str(contract_path.relative_to(ctx.run_dir)),
                "source_plugins": ["report_decision_bundle_v2"],
                "created_at_utc": now_iso(),
            },
        ]
        manifest = compute_artifact_manifest(ctx.run_dir, manifest_entries)
        manifest_path = ctx.run_dir / "slide_kit" / "artifacts_manifest.json"
        write_json(manifest_path, {"artifacts": manifest})

        artifacts = [
            PluginArtifact(
                path=str(business_path.relative_to(ctx.run_dir)),
                type="markdown",
                description="Business summary",
            ),
            PluginArtifact(
                path=str(engineering_path.relative_to(ctx.run_dir)),
                type="markdown",
                description="Engineering summary",
            ),
            PluginArtifact(
                path=str(appendix_path.relative_to(ctx.run_dir)),
                type="markdown",
                description="Appendix raw",
            ),
            PluginArtifact(
                path=str(manifest_path.relative_to(ctx.run_dir)),
                type="json",
                description="Slide kit artifact manifest",
            ),
            PluginArtifact(
                path=str(contract_path.relative_to(ctx.run_dir)),
                type="json",
                description="Report contract checks",
            ),
        ]

        metrics = {
            "business_summary": str(business_path.relative_to(ctx.run_dir)),
            "engineering_summary": str(engineering_path.relative_to(ctx.run_dir)),
            "appendix_raw": str(appendix_path.relative_to(ctx.run_dir)),
            "contract_failure_count": int(len(contract_failures)),
        }
        findings = [
            {
                "kind": "report_bundle_v2",
                "summary_paths": metrics,
                "measurement_type": "measured",
            }
        ]
        if has_modeled_gaps:
            findings.append(
                {
                    "kind": "modeled_field_gaps",
                    "missing": modeled_errors[:100],
                    "measurement_type": "measured",
                }
            )
        if synthesized_traceability:
            findings.append(
                {
                    "kind": "traceability_synthesized",
                    "measurement_type": "measured",
                    "summary": "Traceability manifest missing; synthesized local claim mappings for referenced IDs.",
                }
            )
        if contract_failures:
            findings.append(
                {
                    "kind": "report_contract_violation",
                    "measurement_type": "measured",
                    "failure_count": int(len(contract_failures)),
                    "failures": contract_failures[:100],
                }
            )

        return PluginResult(
            "degraded" if degraded else "ok",
            (
                "Decision report bundle generated (contract gaps)"
                if degraded
                else (
                    "Decision report bundle generated (modeled gaps noted)"
                    if has_modeled_gaps
                    else "Decision report bundle generated"
                )
            ),
            metrics,
            findings,
            artifacts,
            None,
        )
