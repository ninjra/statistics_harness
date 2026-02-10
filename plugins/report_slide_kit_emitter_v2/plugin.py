from __future__ import annotations

from typing import Any

import csv

from statistic_harness.core.report_v2_utils import (
    claim_id,
    find_artifact_path,
    load_artifact_json,
    load_plugin_payloads,
)
from statistic_harness.core.types import PluginArtifact, PluginResult


def _build_claim_lookup(payload: dict[str, Any] | None) -> dict[str, str]:
    lookup: dict[str, str] = {}
    if not isinstance(payload, dict):
        return lookup
    for claim in payload.get("claims") or []:
        if not isinstance(claim, dict):
            continue
        cid = claim.get("claim_id")
        summary = claim.get("summary_text")
        if isinstance(cid, str) and isinstance(summary, str):
            lookup[summary] = cid
    return lookup


class Plugin:
    def run(self, ctx) -> PluginResult:
        plugins = load_plugin_payloads(ctx.storage, ctx.run_id)
        busy_path = find_artifact_path(
            plugins, "analysis_busy_period_segmentation_v2", "busy_periods.json"
        )
        waterfall_path = find_artifact_path(
            plugins, "analysis_waterfall_summary_v2", "waterfall_summary.json"
        )
        trace_path = find_artifact_path(
            plugins, "analysis_traceability_manifest_v2", "traceability_manifest.json"
        )

        busy_payload = load_artifact_json(ctx.run_dir, busy_path)
        waterfall_payload = load_artifact_json(ctx.run_dir, waterfall_path)
        trace_payload = load_artifact_json(ctx.run_dir, trace_path)
        claim_lookup = _build_claim_lookup(trace_payload)

        total_over_sec = 0.0
        top_driver = None
        top_driver_sec = 0.0
        modeled_remainder_sec = None
        if isinstance(waterfall_payload, dict):
            total_over_sec = float(waterfall_payload.get("total_over_threshold_wait_sec") or 0.0)
            top_driver = waterfall_payload.get("top_driver_id")
            top_driver_sec = float(waterfall_payload.get("top_driver_over_threshold_wait_sec") or 0.0)
            modeled_remainder_sec = waterfall_payload.get("modeled_remainder_after_capacity_sec")

        # ---- Scenario summary ----
        scenario_rows: list[dict[str, Any]] = []
        current_hours = round(total_over_sec / 3600.0, 2)
        scenario_rows.append(
            {
                "scenario_id": "current",
                "scenario_name": "Current",
                "evidence_type": "MEASURED",
                "bp_over_threshold_wait_hours": current_hours,
                "delta_hours_vs_current": 0.0,
                "delta_percent_vs_current": 0.0,
                "claim_id": claim_id("kpi_bp_ot_wts_hours_total"),
            }
        )

        if modeled_remainder_sec is not None:
            modeled_total_sec = top_driver_sec + float(modeled_remainder_sec)
            modeled_hours = round(modeled_total_sec / 3600.0, 2)
            delta_hours = modeled_hours - current_hours
            delta_pct = (delta_hours / current_hours * 100.0) if current_hours else 0.0
            scenario_rows.append(
                {
                    "scenario_id": "add_1_server",
                    "scenario_name": "Add 1 server",
                    "evidence_type": "MODELED",
                    "bp_over_threshold_wait_hours": modeled_hours,
                    "delta_hours_vs_current": round(delta_hours, 2),
                    "delta_percent_vs_current": round(delta_pct, 2),
                    "notes": "Modeled add-one-server on remainder",
                    "claim_id": claim_id("waterfall:modeled_remainder_after_add_one_server"),
                }
            )

        scenario_path = ctx.run_dir / "slide_kit" / "scenario_summary.csv"
        scenario_path.parent.mkdir(parents=True, exist_ok=True)
        scenario_headers = [
            "scenario_id",
            "scenario_name",
            "evidence_type",
            "bp_over_threshold_wait_hours",
            "delta_hours_vs_current",
            "delta_percent_vs_current",
            "claim_id",
        ]
        with scenario_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=scenario_headers)
            writer.writeheader()
            for row in scenario_rows:
                writer.writerow({key: row.get(key, "") for key in scenario_headers})

        # ---- Busy periods table ----
        busy_rows: list[dict[str, Any]] = []
        if isinstance(busy_payload, dict):
            busy_periods = busy_payload.get("busy_periods") or []
            if isinstance(busy_periods, list):
                # Sort by total over-threshold wait, descending.
                sorted_periods = []
                for row in busy_periods:
                    if not isinstance(row, dict):
                        continue
                    try:
                        total_sec = float(row.get("total_over_threshold_wait_sec") or 0.0)
                    except (TypeError, ValueError):
                        total_sec = 0.0
                    sorted_periods.append((total_sec, row))
                sorted_periods.sort(key=lambda t: t[0], reverse=True)
                for total_sec, row in sorted_periods[:10]:
                    top_proc = row.get("top_process_by_wait")
                    top_proc_id = ""
                    if isinstance(top_proc, dict):
                        top_proc_id = str(top_proc.get("id") or "")
                    elif isinstance(top_proc, str):
                        top_proc_id = top_proc
                    busy_rows.append(
                        {
                            "busy_period_id": row.get("busy_period_id") or "",
                            "start_ts": row.get("start_ts") or "",
                            "end_ts": row.get("end_ts") or "",
                            "total_over_threshold_wait_hours": round(total_sec / 3600.0, 2),
                            "runs_over_threshold_count": int(row.get("runs_over_threshold_count") or 0),
                            "top_process_id": top_proc_id,
                            "claim_id": claim_id(f"busy_period:{row.get('busy_period_id') or ''}"),
                        }
                    )

        busy_csv_path = ctx.run_dir / "slide_kit" / "busy_periods.csv"
        busy_csv_path.parent.mkdir(parents=True, exist_ok=True)
        busy_headers = [
            "busy_period_id",
            "start_ts",
            "end_ts",
            "total_over_threshold_wait_hours",
            "runs_over_threshold_count",
            "top_process_id",
            "claim_id",
        ]
        with busy_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=busy_headers)
            writer.writeheader()
            for row in busy_rows:
                writer.writerow({key: row.get(key, "") for key in busy_headers})

        # ---- Top process contributors ----
        process_rows: list[dict[str, Any]] = []
        if isinstance(busy_payload, dict):
            busy_periods = busy_payload.get("busy_periods") or []
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
            total_sec = sum(process_totals.values()) or 0.0
            for process_id, value_sec in sorted(
                process_totals.items(), key=lambda item: item[1], reverse=True
            )[:10]:
                hours = round(value_sec / 3600.0, 2)
                share = (value_sec / total_sec * 100.0) if total_sec else 0.0
                process_rows.append(
                    {
                        "process_id": process_id,
                        "bp_over_threshold_wait_hours": hours,
                        "share_percent": round(share, 2),
                        "claim_id": claim_lookup.get(
                            f"Process {process_id} over-threshold wait",
                            claim_id(f"process:{process_id}:bp_ot_wts_hours_total"),
                        ),
                    }
                )

        process_path = ctx.run_dir / "slide_kit" / "top_process_contributors.csv"
        process_path.parent.mkdir(parents=True, exist_ok=True)
        process_headers = [
            "process_id",
            "bp_over_threshold_wait_hours",
            "share_percent",
            "claim_id",
        ]
        with process_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=process_headers)
            writer.writeheader()
            for row in process_rows:
                writer.writerow({key: row.get(key, "") for key in process_headers})

        artifacts = [
            PluginArtifact(
                path=str(scenario_path.relative_to(ctx.run_dir)),
                type="csv",
                description="Scenario summary table",
            ),
            PluginArtifact(
                path=str(busy_csv_path.relative_to(ctx.run_dir)),
                type="csv",
                description="Busy periods table",
            ),
            PluginArtifact(
                path=str(process_path.relative_to(ctx.run_dir)),
                type="csv",
                description="Top process contributors",
            ),
        ]

        metrics = {"scenarios": len(scenario_rows), "busy_periods": len(busy_rows), "processes": len(process_rows)}
        findings = [
            {
                "kind": "slide_kit_summary",
                "scenarios": len(scenario_rows),
                "busy_periods": len(busy_rows),
                "processes": len(process_rows),
                "measurement_type": "measured",
            }
        ]

        return PluginResult(
            "ok",
            "Slide kit tables emitted",
            metrics,
            findings,
            artifacts,
            None,
        )
