from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_batch_sequence_validation_checklist import build_checklist_rows, generate_for_run_dir


def _report_payload() -> dict[str, object]:
    return {
        "run_id": "run_123",
        "recommendations": {
            "discovery": {
                "items": [
                    {
                        "plugin_id": "analysis_actionable_ops_levers_v1",
                        "kind": "actionable_ops_lever",
                        "action_type": "batch_group_candidate",
                        "key": "payout_id",
                        "best_close_month": "2026-01",
                        "target_process_ids": ["rpt_por002", "poextrprvn", "pognrtrpt"],
                        "modeled_delta_hours": 3.25,
                        "validation_steps": ["Step A", "Step B"],
                        "recommendation": "Convert payout chain to batch input.",
                    }
                ]
            }
        },
    }


def test_build_checklist_rows_extracts_batch_groups() -> None:
    rows = build_checklist_rows(_report_payload())
    assert len(rows) == 1
    row = rows[0]
    assert row["plugin_id"] == "analysis_actionable_ops_levers_v1"
    assert row["close_month"] == "2026-01"
    assert row["key"] == "payout_id"
    assert row["target_process_ids"] == ["rpt_por002", "poextrprvn", "pognrtrpt"]
    assert row["target_process_count"] == 3
    assert row["modeled_delta_hours"] == 3.25
    assert str(row["sequence_id"]).startswith("batchseq_")


def test_generate_for_run_dir_writes_payload_and_markdown(tmp_path: Path) -> None:
    run_dir = tmp_path / "appdata" / "runs" / "run_123"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "report.json").write_text(json.dumps(_report_payload()), encoding="utf-8")

    payload, markdown = generate_for_run_dir(run_dir)

    assert payload["run_id"] == "run_123"
    assert payload["sequence_count"] == 1
    assert "Batch Sequence Validation Checklist" in markdown
    assert "rpt_por002, poextrprvn, pognrtrpt" in markdown
