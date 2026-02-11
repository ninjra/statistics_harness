from __future__ import annotations

from typing import Any

from statistic_harness.core.report import _build_recommendations
from statistic_harness.core.report_v2_utils import (
    build_minimal_report,
    filter_excluded_processes,
    load_known_issues,
)
from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json


def _infer_action_type(item: dict[str, Any]) -> str:
    action = item.get("action")
    if isinstance(action, str) and action.strip():
        return action.strip()
    text = str(item.get("recommendation") or item.get("title") or "").lower()
    if (
        "add one server" in text
        or "add server" in text
        or "qpec+1" in text
        or "qpec(+1" in text
        or "qpec +1" in text
    ):
        return "add_server"
    if "remove" in text and "process" in text:
        return "remove_process"
    if (
        "schedule" in text
        or "reschedule" in text
        or "frequency" in text
        or ("every " in text and "min" in text)
    ):
        return "tune_schedule"
    return "review"


def _infer_target(item: dict[str, Any]) -> str:
    for bucket in (item.get("where"), item.get("contains")):
        if isinstance(bucket, dict):
            for key in ("process", "process_norm", "process_name", "process_id", "activity"):
                value = bucket.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
                if isinstance(value, (list, tuple, set)):
                    for entry in value:
                        if isinstance(entry, str) and entry.strip():
                            return entry.strip()
    return ""


def _infer_scenario_id(action_type: str, target: str) -> str:
    if action_type == "add_server":
        return "add_1_server"
    if action_type == "remove_process":
        return f"remove_{target or 'process'}"
    if action_type == "tune_schedule":
        return f"schedule_{target or 'process'}"
    return action_type or "review"


def _delta_signature(item: dict[str, Any]) -> int | None:
    delta = item.get("modeled_delta")
    if delta is None:
        delta = item.get("delta_hours") or item.get("delta_value")
    if isinstance(delta, (int, float)):
        return int(round(float(delta) * 3600.0))
    return None


def _confidence_tag(item: dict[str, Any]) -> str:
    measurement = str(item.get("measurement_type") or "").lower()
    if measurement == "modeled":
        return "MODELED"
    if measurement == "measured":
        return "MEASURED"
    return "MIXED"


def _validation_steps(target: str) -> list[str]:
    if target:
        return [
            f"Apply change to {target}.",
            "Re-run the harness on the same dataset scope.",
            "Confirm BP_OT_WTS_HOURS and busy periods improve vs baseline.",
        ]
    return [
        "Apply the change.",
        "Re-run the harness on the same dataset scope.",
        "Confirm BP_OT_WTS_HOURS improves vs baseline.",
    ]


class Plugin:
    def run(self, ctx) -> PluginResult:
        override = ctx.settings.get("recommendations_override")
        if isinstance(override, list):
            items = [item for item in override if isinstance(item, dict)]
        else:
            report = build_minimal_report(
                ctx.storage,
                ctx.run_id,
                ctx.run_dir,
                project_id=ctx.project_id,
                dataset_version_id=ctx.dataset_version_id,
            )
            recommendations = _build_recommendations(report, ctx.storage, run_dir=ctx.run_dir)
            items = []
            if isinstance(recommendations, dict):
                if "known" in recommendations or "discovery" in recommendations:
                    for block in ("known", "discovery"):
                        block_items = recommendations.get(block, {}).get("items") or []
                        items.extend(
                            [item for item in block_items if isinstance(item, dict)]
                        )
                else:
                    items.extend(
                        [
                            item
                            for item in (recommendations.get("items") or [])
                            if isinstance(item, dict)
                        ]
                    )

        known_payload = load_known_issues(
            ctx.storage,
            ctx.run_id,
            ctx.run_dir,
            project_id=ctx.project_id,
            dataset_version_id=ctx.dataset_version_id,
        )

        # "Known-only" recommendation sets are valid and should not fail the run.
        # Many datasets are clean enough that only known-issue checks trigger.
        discovery_count = sum(
            1 for item in items if (item.get("category") or "discovery") == "discovery"
        )
        known_count = sum(
            1 for item in items if (item.get("category") or "discovery") != "discovery"
        )
        has_discovery = discovery_count > 0

        deduped: dict[tuple[str, str, str, int | None], dict[str, Any]] = {}
        conflicts: list[str] = []
        for item in items:
            action_type = (
                item.get("action_type")
                or item.get("action")
                or _infer_action_type(item)
            )
            target = item.get("target") or _infer_target(item)
            scenario_id = item.get("scenario_id") or _infer_scenario_id(
                action_type, target
            )
            delta_sig = item.get("delta_signature")
            if delta_sig is None:
                delta_sig = _delta_signature(item)
            key = (action_type, target, scenario_id, delta_sig)
            if key in deduped:
                current = deduped[key]
                # Defensive: even though delta_signature is part of the key, preserve a record
                # if upstream payloads are inconsistent.
                if delta_sig != current.get("delta_signature"):
                    conflicts.append(f"{action_type}:{target}:{scenario_id} delta mismatch")
                current["titles"].append(item.get("title") or "")
                current["recommendations"].append(item.get("recommendation") or "")
                current["evidence"] += item.get("evidence") or []
                current["observed_count"] += int(item.get("observed_count") or 0)
                if item.get("category") == "discovery":
                    current["category"] = "discovery"
                continue
            deduped[key] = {
                "action_type": action_type,
                "target": target,
                "scenario_id": scenario_id,
                "delta_signature": delta_sig,
                "delta_hours": (float(delta_sig) / 3600.0) if delta_sig is not None else None,
                "titles": [item.get("title") or ""],
                "recommendations": [item.get("recommendation") or ""],
                "evidence": list(item.get("evidence") or []),
                "observed_count": int(item.get("observed_count") or 0),
                "confidence_tag": _confidence_tag(item),
                "validation_steps": _validation_steps(target),
                "relevance_score": item.get("relevance_score"),
                "category": item.get("category") or "discovery",
            }

        merged = list(deduped.values())
        merged = filter_excluded_processes(merged, known_payload)

        merged = sorted(
            merged,
            key=lambda row: float(row.get("relevance_score") or 0.0),
            reverse=True,
        )
        max_summary = int(ctx.settings.get("max_summary", 10))
        summary_recommendations = merged[:max_summary]
        if has_discovery and summary_recommendations:
            has_discovery_summary = any(
                rec.get("category") == "discovery" for rec in summary_recommendations
            )
            if not has_discovery_summary:
                for rec in merged:
                    if rec.get("category") == "discovery":
                        summary_recommendations[-1] = rec
                        break

        artifacts_dir = ctx.artifacts_dir("analysis_recommendation_dedupe_v2")
        out_path = artifacts_dir / "recommendations.json"
        write_json(
            out_path,
            {
                "has_discovery": has_discovery,
                "counts": {
                    "source_items_total": int(len(items)),
                    "source_items_discovery": int(discovery_count),
                    "source_items_known": int(known_count),
                    "deduped_total": int(len(merged)),
                },
                "conflicts": conflicts,
                "summary_recommendations": summary_recommendations,
                "recommendations": merged,
            },
        )

        artifacts = [
            PluginArtifact(
                path=str(out_path.relative_to(ctx.run_dir)),
                type="json",
                description="Deduplicated recommendations",
            )
        ]

        metrics = {
            "recommendations": len(merged),
            "has_discovery": int(has_discovery),
            "discovery_items": int(discovery_count),
            "known_items": int(known_count),
            "conflicts": int(len(conflicts)),
        }
        findings = [
            {
                "kind": "recommendation_dedupe_summary",
                "recommendations": len(merged),
                "has_discovery": bool(has_discovery),
                "discovery_items": int(discovery_count),
                "known_items": int(known_count),
                "conflicts": conflicts[:10],
                "measurement_type": "measured",
            }
        ]

        return PluginResult(
            "ok",
            f"Deduplicated {len(merged)} recommendations",
            metrics,
            findings,
            artifacts,
            None,
        )
