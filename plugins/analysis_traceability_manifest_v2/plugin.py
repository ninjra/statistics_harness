from __future__ import annotations

from typing import Any

from statistic_harness.core.report_v2_utils import (
    claim_id,
    find_artifact_path,
    load_artifact_json,
    load_plugin_payloads,
)
from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json


def _hours(sec: float) -> float:
    return round(sec / 3600.0, 4)


class Plugin:
    def run(self, ctx) -> PluginResult:
        plugins = load_plugin_payloads(ctx.storage, ctx.run_id)
        busy_path = find_artifact_path(
            plugins, "analysis_busy_period_segmentation_v2", "busy_periods.json"
        )
        waterfall_path = find_artifact_path(
            plugins, "analysis_waterfall_summary_v2", "waterfall_summary.json"
        )

        busy_payload = load_artifact_json(ctx.run_dir, busy_path)
        waterfall_payload = load_artifact_json(ctx.run_dir, waterfall_path)

        claims: list[dict[str, Any]] = []

        total_over_sec = 0.0
        busy_periods = []
        if busy_payload:
            busy_periods = busy_payload.get("busy_periods") or []
            total_over_sec = sum(
                float(row.get("total_over_threshold_wait_sec") or 0.0)
                for row in busy_periods
                if isinstance(row, dict)
            )

        claims.append(
            {
                "claim_id": claim_id("kpi_bp_ot_wts_hours_total"),
                "label": "MEASURED",
                "summary_text": "Busy-period over-threshold wait-to-start hours",
                "value": _hours(total_over_sec),
                "unit": "hours",
                "population_scope": "over-threshold runs",
                "source": {
                    "plugin": "analysis_busy_period_segmentation_v2",
                    "kind": "busy_period_summary",
                    "measurement_type": "measured",
                    "artifact_path": busy_path,
                    "query_or_grouping": "sum(total_over_threshold_wait_sec)",
                    "row_keys": [],
                },
                "render_targets": ["business_summary", "engineering_summary", "slide_kit"],
            }
        )

        if busy_periods:
            top_periods = sorted(
                busy_periods,
                key=lambda row: float(row.get("total_over_threshold_wait_sec") or 0.0),
                reverse=True,
            )[:10]
            for row in top_periods:
                period_id = row.get("busy_period_id") or ""
                value_sec = float(row.get("total_over_threshold_wait_sec") or 0.0)
                claims.append(
                    {
                        "claim_id": claim_id(f"busy_period:{period_id}:total"),
                        "label": "MEASURED",
                        "summary_text": f"Busy period {period_id} total over-threshold wait",
                        "value": _hours(value_sec),
                        "unit": "hours",
                        "population_scope": "over-threshold runs",
                        "source": {
                            "plugin": "analysis_busy_period_segmentation_v2",
                            "kind": "busy_period",
                            "measurement_type": "measured",
                            "artifact_path": busy_path,
                            "query_or_grouping": "total_over_threshold_wait_sec",
                            "row_keys": [period_id],
                        },
                        "render_targets": ["business_summary", "slide_kit"],
                    }
                )

            process_totals: dict[str, float] = {}
            for row in busy_periods:
                if not isinstance(row, dict):
                    continue
                per_process = row.get("per_process_over_threshold_wait_sec") or {}
                if not isinstance(per_process, dict):
                    continue
                for key, value in per_process.items():
                    try:
                        process_totals[str(key)] = process_totals.get(str(key), 0.0) + float(
                            value
                        )
                    except (TypeError, ValueError):
                        continue
            top_processes = sorted(
                process_totals.items(), key=lambda item: item[1], reverse=True
            )[:10]
            for process_id, value_sec in top_processes:
                claims.append(
                    {
                        "claim_id": claim_id(f"process:{process_id}:total"),
                        "label": "MEASURED",
                        "summary_text": f"Process {process_id} over-threshold wait",
                        "value": _hours(value_sec),
                        "unit": "hours",
                        "population_scope": "over-threshold runs",
                        "source": {
                            "plugin": "analysis_busy_period_segmentation_v2",
                            "kind": "process_over_threshold_wait",
                            "measurement_type": "measured",
                            "artifact_path": busy_path,
                            "query_or_grouping": process_id,
                            "row_keys": [process_id],
                        },
                        "render_targets": ["business_summary", "slide_kit"],
                    }
                )

        if waterfall_payload:
            rows = waterfall_payload.get("rows") or []
            for row in rows:
                row_id = row.get("row_id") or ""
                value_hours = row.get("value_hours")
                claims.append(
                    {
                        "claim_id": claim_id(f"waterfall:{row_id}"),
                        "label": "MEASURED" if row_id.startswith("total") else "MODELED",
                        "summary_text": f"Waterfall {row_id}",
                        "value": value_hours,
                        "unit": "hours",
                        "population_scope": "busy-period wait",
                        "source": {
                            "plugin": "analysis_waterfall_summary_v2",
                            "kind": "waterfall_row",
                            "measurement_type": "measured"
                            if row_id.startswith("total")
                            else "modeled",
                            "artifact_path": waterfall_path,
                            "query_or_grouping": row_id,
                            "row_keys": [row_id],
                        },
                        "render_targets": ["business_summary", "slide_kit"],
                    }
                )

        out_path = ctx.run_dir / "slide_kit" / "traceability_manifest.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(out_path, {"claims": claims})

        artifacts = [
            PluginArtifact(
                path=str(out_path.relative_to(ctx.run_dir)),
                type="json",
                description="Traceability manifest",
            )
        ]

        metrics = {"claims": len(claims)}
        findings = [
            {
                "kind": "traceability_manifest",
                "claims": len(claims),
                "measurement_type": "measured",
            }
        ]

        return PluginResult(
            "ok",
            f"Built {len(claims)} traceability claims",
            metrics,
            findings,
            artifacts,
            None,
        )
