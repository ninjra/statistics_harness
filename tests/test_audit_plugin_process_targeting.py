from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_plugin_process_targeting import audit_process_targeting


def test_audit_plugin_process_targeting_detects_missing_process_for_direct_action(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "recommendations": {
            "items": [
                {"plugin_id": "p1", "action_type": "batch_input", "process_norm": ""},
            ]
        }
    }
    (run_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
    payload = audit_process_targeting(run_dir)
    assert payload["ok"] is False
    assert payload["violation_count"] == 1


def test_audit_plugin_process_targeting_passes_for_specific_process(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "recommendations": {
            "items": [
                {"plugin_id": "p1", "action_type": "batch_input", "process_norm": "rpt_por002"},
            ]
        }
    }
    (run_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
    payload = audit_process_targeting(run_dir)
    assert payload["ok"] is True

