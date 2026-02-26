from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from jsonschema import validate


def _init_root_with_manifest(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    run_dir = root / "appdata" / "runs" / "run_beta"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "plugin_run_manifest.v1",
        "generated_at": "2026-02-26T00:00:00+00:00",
        "run_id": "run_beta",
        "run_seed": 777,
        "requested_run_seed": 777,
        "dataset_hash": "dataset_hash_beta",
        "dataset_version_id": "dataset_beta",
        "run_status": "completed",
        "run_manifest_path": None,
        "plugin_registry_hash": "registry_hash_beta",
        "plugin_count": 4,
        "plugins": [
            {
                "plugin_id": "plugin_actionable",
                "declared": True,
                "executed": True,
                "has_result": True,
                "execution_count": 1,
                "execution_status": "ok",
                "result_status": "ok",
                "plugin_type": "analysis",
                "plugin_version": "1",
                "entrypoint": "run.py:run",
                "code_hash": "c1",
                "settings_hash": "s1",
                "execution_fingerprint": "f1",
                "dataset_hash": "dataset_hash_beta",
                "latest_execution_id": 1,
                "latest_result_id": 1
            },
            {
                "plugin_id": "plugin_low",
                "declared": True,
                "executed": True,
                "has_result": True,
                "execution_count": 1,
                "execution_status": "ok",
                "result_status": "ok",
                "plugin_type": "analysis",
                "plugin_version": "1",
                "entrypoint": "run.py:run",
                "code_hash": "c2",
                "settings_hash": "s2",
                "execution_fingerprint": "f2",
                "dataset_hash": "dataset_hash_beta",
                "latest_execution_id": 2,
                "latest_result_id": 2
            },
            {
                "plugin_id": "plugin_marker_absent",
                "declared": True,
                "executed": True,
                "has_result": True,
                "execution_count": 1,
                "execution_status": "na",
                "result_status": "na",
                "plugin_type": "analysis",
                "plugin_version": "1",
                "entrypoint": "run.py:run",
                "code_hash": "c3",
                "settings_hash": "s3",
                "execution_fingerprint": "f3",
                "dataset_hash": "dataset_hash_beta",
                "latest_execution_id": 3,
                "latest_result_id": 3
            },
            {
                "plugin_id": "plugin_fail",
                "declared": True,
                "executed": True,
                "has_result": True,
                "execution_count": 1,
                "execution_status": "skipped",
                "result_status": "skipped",
                "plugin_type": "analysis",
                "plugin_version": "1",
                "entrypoint": "run.py:run",
                "code_hash": "c4",
                "settings_hash": "s4",
                "execution_fingerprint": "f4",
                "dataset_hash": "dataset_hash_beta",
                "latest_execution_id": 4,
                "latest_result_id": 4
            }
        ]
    }
    report = {
        "plugins": {
            "plugin_actionable": {"status": "ok", "findings": []},
            "plugin_low": {"status": "ok", "findings": []},
            "plugin_marker_absent": {"status": "na", "findings": []},
            "plugin_fail": {"status": "skipped", "findings": []}
        },
        "recommendations": {
            "items": [
                {
                    "plugin_id": "plugin_actionable",
                    "delta_hours_close_dynamic": 1.7,
                    "efficiency_gain_pct_close_dynamic": 12.1
                },
                {
                    "plugin_id": "plugin_low",
                    "delta_hours_close_dynamic": 0.45,
                    "efficiency_gain_pct_close_dynamic": 2.2
                }
            ],
            "explanations": {
                "items": [
                    {
                        "plugin_id": "plugin_marker_absent",
                        "reason_code": "NO_STATISTICAL_SIGNAL"
                    }
                ]
            }
        }
    }
    (run_dir / "plugin_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return root


def test_classify_plugin_validity_state_mapping(tmp_path: Path) -> None:
    root = _init_root_with_manifest(tmp_path)
    out_path = root / "appdata" / "runs" / "run_beta" / "audit" / "plugin_validity_contract.json"
    cmd = [
        sys.executable,
        "scripts/classify_plugin_validity.py",
        "--root",
        str(root),
        "--run-id",
        "run_beta",
        "--out-json",
        str(out_path),
    ]
    proc = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    schema = json.loads(Path("docs/plugin_validation_contract.schema.json").read_text(encoding="utf-8"))
    validate(instance=payload, schema=schema)

    by_id = {row["plugin_id"]: row for row in payload["plugins"]}
    assert by_id["plugin_actionable"]["state"] == "PASS_ACTIONABLE"
    assert by_id["plugin_low"]["state"] == "PASS_VALID_LOW_SIGNAL"
    assert by_id["plugin_marker_absent"]["state"] == "PASS_VALID_MARKER_ABSENT"
    assert by_id["plugin_fail"]["state"] == "FAIL_LOGIC"


def test_classify_plugin_validity_strict_fails_on_fail_states(tmp_path: Path) -> None:
    root = _init_root_with_manifest(tmp_path)
    cmd = [
        sys.executable,
        "scripts/classify_plugin_validity.py",
        "--root",
        str(root),
        "--run-id",
        "run_beta",
        "--strict",
    ]
    proc = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
