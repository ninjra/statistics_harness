from __future__ import annotations

from typing import Any

import csv

from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json
from statistic_harness.core.report_v2_utils import (
    load_plugin_payloads,
    find_artifact_path,
    load_artifact_json,
)


def _seconds(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _hours_from_seconds(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value) / 3600.0


class Plugin:
    def run(self, ctx) -> PluginResult:
        plugins = load_plugin_payloads(ctx.storage, ctx.run_id)
        busy_path = find_artifact_path(
            plugins, "analysis_busy_period_segmentation_v2", "busy_periods.json"
        )
        busy_payload = load_artifact_json(ctx.run_dir, busy_path)
        if not busy_payload:
            return PluginResult(
                "skipped",
                "Missing busy period segmentation output",
                {},
                [],
                [],
                None,
            )

        busy_periods = busy_payload.get("busy_periods") or []
        total_over_sec = sum(
            float(row.get("total_over_threshold_wait_sec") or 0.0)
            for row in busy_periods
            if isinstance(row, dict)
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

        top_driver_id = None
        top_driver_sec = 0.0
        if process_totals:
            top_driver_id, top_driver_sec = max(
                process_totals.items(), key=lambda item: item[1]
            )

        remainder_sec = total_over_sec - top_driver_sec

        queue_path = find_artifact_path(
            plugins, "analysis_queue_delay_decomposition", "results.json"
        )
        queue_payload = load_artifact_json(ctx.run_dir, queue_path)
        modeled_baseline_sec = None
        modeled_value_sec = None
        baseline_source = None
        baseline_population = None
        reconciliation_diff_sec = None
        if queue_payload:
            findings = queue_payload.get("findings") or []
            for item in findings:
                if not isinstance(item, dict):
                    continue
                if item.get("kind") == "capacity_scale_model":
                    base_hours = item.get("eligible_wait_gt_hours_without_target")
                    modeled_hours = item.get("eligible_wait_gt_hours_modeled")
                    if isinstance(base_hours, (int, float)) and isinstance(
                        modeled_hours, (int, float)
                    ):
                        modeled_baseline_sec = float(base_hours) * 3600.0
                        modeled_value_sec = float(modeled_hours) * 3600.0
                        baseline_source = "analysis_queue_delay_decomposition"
                        baseline_population = "eligible_wait_gt_hours_without_target"
                        reconciliation_diff_sec = modeled_baseline_sec - remainder_sec
                        break

        rows: list[dict[str, Any]] = []
        rows.append(
            {
                "row_id": "total_bp_over_threshold_wait",
                "label": "Total busy-period over-threshold wait",
                "value_sec": total_over_sec,
                "value_hours": round(total_over_sec / 3600.0, 2),
                "delta_sec": 0.0,
                "delta_hours": 0.0,
                "driver_id": "",
                "notes": "",
            }
        )
        rows.append(
            {
                "row_id": "top_driver_over_threshold_wait",
                "label": "Top driver over-threshold wait",
                "value_sec": top_driver_sec,
                "value_hours": round(top_driver_sec / 3600.0, 2),
                "delta_sec": 0.0,
                "delta_hours": 0.0,
                "driver_id": top_driver_id or "",
                "notes": "",
            }
        )
        rows.append(
            {
                "row_id": "remainder_without_top_driver",
                "label": "Remainder without top driver",
                "value_sec": remainder_sec,
                "value_hours": round(remainder_sec / 3600.0, 2),
                "delta_sec": 0.0,
                "delta_hours": 0.0,
                "driver_id": top_driver_id or "",
                "notes": "",
            }
        )

        if modeled_value_sec is not None:
            delta_sec = modeled_value_sec - remainder_sec
            rows.append(
                {
                    "row_id": "modeled_remainder_after_add_one_server",
                    "label": "Modeled remainder after add-one-server",
                    "value_sec": modeled_value_sec,
                    "value_hours": round(modeled_value_sec / 3600.0, 2),
                    "delta_sec": delta_sec,
                    "delta_hours": round(delta_sec / 3600.0, 2),
                    "driver_id": top_driver_id or "",
                    "notes": "modeled",
                }
            )

        slide_path = ctx.run_dir / "slide_kit" / "waterfall_summary.csv"
        slide_path.parent.mkdir(parents=True, exist_ok=True)
        headers = [
            "row_id",
            "label",
            "value_sec",
            "value_hours",
            "delta_sec",
            "delta_hours",
            "driver_id",
            "notes",
        ]
        with slide_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in headers})

        artifacts_dir = ctx.artifacts_dir("analysis_waterfall_summary_v2")
        out_path = artifacts_dir / "waterfall_summary.json"
        write_json(
            out_path,
            {
                "rows": rows,
                "total_over_threshold_wait_sec": total_over_sec,
                "top_driver_id": top_driver_id,
                "top_driver_over_threshold_wait_sec": top_driver_sec,
                "remainder_without_top_driver_sec": remainder_sec,
                "modeled_remainder_after_capacity_sec": modeled_value_sec,
                "baseline_source": baseline_source,
                "baseline_population": baseline_population,
                "reconciliation_diff_sec": reconciliation_diff_sec,
            },
        )

        metrics = {
            "total_over_threshold_wait_hours": total_over_sec / 3600.0,
            "top_driver_id": top_driver_id,
            "remainder_without_top_driver_hours": remainder_sec / 3600.0,
            "modeled_remainder_after_capacity_hours": _hours_from_seconds(
                modeled_value_sec
            ),
        }

        findings = [
            {
                "kind": "waterfall_summary",
                "total_over_threshold_wait_hours": metrics[
                    "total_over_threshold_wait_hours"
                ],
                "top_driver_id": top_driver_id,
                "measurement_type": "measured",
            }
        ]

        artifacts = [
            PluginArtifact(
                path=str(out_path.relative_to(ctx.run_dir)),
                type="json",
                description="Waterfall summary (json)",
            ),
            PluginArtifact(
                path=str(slide_path.relative_to(ctx.run_dir)),
                type="csv",
                description="Waterfall summary (csv)",
            ),
        ]

        return PluginResult(
            "ok",
            "Built waterfall summary",
            metrics,
            findings,
            artifacts,
            None,
        )
