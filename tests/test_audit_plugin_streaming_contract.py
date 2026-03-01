from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_plugin_streaming_contract import audit_streaming_contract


def test_audit_plugin_streaming_contract_flags_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "r1"
    (run_dir / "artifacts" / "p1").mkdir(parents=True, exist_ok=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"plugins": [{"plugin_id": "p1"}]}),
        encoding="utf-8",
    )
    (run_dir / "artifacts" / "p1" / "runtime_access.json").write_text(
        json.dumps({"data_access": {"iter_batches_calls": 0, "dataset_loader_calls": 1, "dataset_loader_unbounded_calls": 1}}),
        encoding="utf-8",
    )
    matrix = {
        "plugins": [
            {
                "plugin_id": "p1",
                "uses_dataset_iter_batches": True,
                "uses_dataset_loader": False,
                "dataset_loader_mode": "none",
            }
        ]
    }
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    payload = audit_streaming_contract(run_dir=run_dir, matrix_path=matrix_path)
    assert payload["ok"] is False
    assert payload["mismatch_count"] == 1


def test_audit_plugin_streaming_contract_passes_match(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "r1"
    (run_dir / "artifacts" / "p1").mkdir(parents=True, exist_ok=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"plugins": [{"plugin_id": "p1"}]}),
        encoding="utf-8",
    )
    (run_dir / "artifacts" / "p1" / "runtime_access.json").write_text(
        json.dumps({"data_access": {"iter_batches_calls": 2, "dataset_loader_calls": 0}}),
        encoding="utf-8",
    )
    matrix = {
        "plugins": [
            {
                "plugin_id": "p1",
                "uses_dataset_iter_batches": True,
                "uses_dataset_loader": False,
                "dataset_loader_mode": "none",
            }
        ]
    }
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    payload = audit_streaming_contract(run_dir=run_dir, matrix_path=matrix_path)
    assert payload["ok"] is True

