from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

from conftest import make_context
from plugins.report_decision_bundle_v2.plugin import Plugin


def _write_csv(path: Path, headers: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_report_decision_bundle_degrades_with_contract_mapping_failure(run_dir: Path) -> None:
    df = pd.DataFrame({"x": [1, 2, 3]})
    ctx = make_context(run_dir, df, settings={})

    slide_kit = run_dir / "slide_kit"
    slide_kit.mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts" / "analysis_busy_period_segmentation_v2").mkdir(parents=True, exist_ok=True)

    # Required report inputs.
    _write_csv(
        slide_kit / "scenario_summary.csv",
        [
            "scenario_id",
            "scenario_name",
            "bp_over_threshold_wait_hours",
            "delta_hours_vs_current",
            "delta_percent_vs_current",
            "evidence_type",
            "claim_id",
        ],
        [
            {
                "scenario_id": "s1",
                "scenario_name": "current",
                "bp_over_threshold_wait_hours": "10.0",
                "delta_hours_vs_current": "0.0",
                "delta_percent_vs_current": "0.0",
                "evidence_type": "measured",
                "claim_id": "claim_missing",
            }
        ],
    )
    _write_csv(
        slide_kit / "top_process_contributors.csv",
        ["process_id", "bp_over_threshold_wait_hours", "share_percent", "claim_id"],
        [{"process_id": "proc_a", "bp_over_threshold_wait_hours": "5.0", "share_percent": "50", "claim_id": "claim_missing"}],
    )
    _write_csv(
        slide_kit / "busy_periods.csv",
        [
            "busy_period_id",
            "start_ts",
            "end_ts",
            "total_over_threshold_wait_hours",
            "runs_over_threshold_count",
            "top_process_id",
            "claim_id",
            "weekend",
            "after_hours",
        ],
        [
            {
                "busy_period_id": "bp1",
                "start_ts": "2026-01-01T00:00:00Z",
                "end_ts": "2026-01-01T01:00:00Z",
                "total_over_threshold_wait_hours": "1.0",
                "runs_over_threshold_count": "2",
                "top_process_id": "proc_a",
                "claim_id": "claim_missing",
                "weekend": "false",
                "after_hours": "false",
            }
        ],
    )
    _write_csv(slide_kit / "waterfall_summary.csv", ["kpi", "current", "target"], [{"kpi": "x", "current": "1", "target": "0"}])

    (slide_kit / "traceability_manifest.json").write_text(
        json.dumps({"claims": [{"claim_id": "claim_other", "summary_text": "other", "source": {"plugin": "analysis_x"}}]}),
        encoding="utf-8",
    )
    (slide_kit / "issue_cards.json").write_text(json.dumps({"issue_cards": []}), encoding="utf-8")
    (run_dir / "artifacts" / "analysis_busy_period_segmentation_v2" / "definition.json").write_text(
        json.dumps({"wait_threshold_seconds": 60, "gap_tolerance_seconds": 60}),
        encoding="utf-8",
    )

    result = Plugin().run(ctx)
    assert result.status == "degraded"
    assert any(str(f.get("kind") or "") == "report_contract_violation" for f in result.findings)

    contract_path = run_dir / "slide_kit" / "report_contract_checks.json"
    assert contract_path.exists()
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    assert payload.get("status") == "failed"
    failures = payload.get("failures") or []
    assert any(str(item.get("code") or "") == "missing_claim_mapping" for item in failures if isinstance(item, dict))


def test_report_decision_bundle_synthesizes_traceability_when_manifest_missing(run_dir: Path) -> None:
    df = pd.DataFrame({"x": [1, 2, 3]})
    ctx = make_context(run_dir, df, settings={})

    slide_kit = run_dir / "slide_kit"
    slide_kit.mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts" / "analysis_busy_period_segmentation_v2").mkdir(parents=True, exist_ok=True)

    _write_csv(
        slide_kit / "scenario_summary.csv",
        [
            "scenario_id",
            "scenario_name",
            "bp_over_threshold_wait_hours",
            "delta_hours_vs_current",
            "delta_percent_vs_current",
            "evidence_type",
            "claim_id",
        ],
        [
            {
                "scenario_id": "s1",
                "scenario_name": "current",
                "bp_over_threshold_wait_hours": "10.0",
                "delta_hours_vs_current": "0.0",
                "delta_percent_vs_current": "0.0",
                "evidence_type": "measured",
                "claim_id": "claim_missing",
            }
        ],
    )
    _write_csv(
        slide_kit / "top_process_contributors.csv",
        ["process_id", "bp_over_threshold_wait_hours", "share_percent", "claim_id"],
        [{"process_id": "proc_a", "bp_over_threshold_wait_hours": "5.0", "share_percent": "50", "claim_id": "claim_missing"}],
    )
    _write_csv(
        slide_kit / "busy_periods.csv",
        [
            "busy_period_id",
            "start_ts",
            "end_ts",
            "total_over_threshold_wait_hours",
            "runs_over_threshold_count",
            "top_process_id",
            "claim_id",
            "weekend",
            "after_hours",
        ],
        [
            {
                "busy_period_id": "bp1",
                "start_ts": "2026-01-01T00:00:00Z",
                "end_ts": "2026-01-01T01:00:00Z",
                "total_over_threshold_wait_hours": "1.0",
                "runs_over_threshold_count": "2",
                "top_process_id": "proc_a",
                "claim_id": "claim_missing",
                "weekend": "false",
                "after_hours": "false",
            }
        ],
    )
    _write_csv(slide_kit / "waterfall_summary.csv", ["kpi", "current", "target"], [{"kpi": "x", "current": "1", "target": "0"}])

    # Deliberately omit traceability_manifest.json; plugin should synthesize claim mappings.
    (slide_kit / "issue_cards.json").write_text(json.dumps({"issue_cards": []}), encoding="utf-8")
    (run_dir / "artifacts" / "analysis_busy_period_segmentation_v2" / "definition.json").write_text(
        json.dumps({"wait_threshold_seconds": 60, "gap_tolerance_seconds": 60}),
        encoding="utf-8",
    )

    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert any(str(f.get("kind") or "") == "traceability_synthesized" for f in result.findings)
    assert not any(str(f.get("kind") or "") == "report_contract_violation" for f in result.findings)

    contract_path = run_dir / "slide_kit" / "report_contract_checks.json"
    assert contract_path.exists()
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    assert payload.get("status") == "passed"
