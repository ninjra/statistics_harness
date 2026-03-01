from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_plugin_targeting_windows import audit_targeting_windows


def test_audit_plugin_targeting_windows_detects_missing_fields(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "recommendations": {
            "items": [
                {"plugin_id": "p1", "action_type": "batch_input"},
            ]
        }
    }
    (run_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
    payload = audit_targeting_windows(run_dir)
    assert payload["ok"] is False
    assert payload["missing_window_metric_count"] == 1


def test_audit_plugin_targeting_windows_passes_with_window_metrics(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "recommendations": {
            "items": [
                {
                    "plugin_id": "p1",
                    "action_type": "batch_input",
                    "delta_hours_close_dynamic": 1.2,
                },
            ]
        }
    }
    (run_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
    payload = audit_targeting_windows(run_dir)
    assert payload["ok"] is True

