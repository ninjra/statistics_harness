from __future__ import annotations

import json
from pathlib import Path

from statistic_harness.core.evaluator import evaluate_report


def test_evaluator_harness_passes_on_synthetic_report(tmp_path):
    # Keep this test focused on the evaluator wrapper module, not the full pipeline.
    report = {
        "run_id": "r1",
        "created_at": "now",
        "status": "completed",
        "input": {"filename": "x", "rows": 1, "cols": 1, "inferred_types": {}},
        "lineage": {
            "run": {"run_id": "r1", "created_at": "now", "status": "completed", "run_seed": 0},
            "input": {"upload_id": "", "filename": "x", "canonical_path": "", "input_hash": "", "sha256": "", "size_bytes": 0},
            "dataset": {"dataset_version_id": "dv"},
            "raw_format": None,
            "template": None,
            "plugins": {},
        },
        "plugins": {
            "analysis_gaussian_knockoffs": {
                "status": "ok",
                "summary": "",
                "metrics": {},
                "findings": [
                    {"kind": "feature_discovery", "feature": "x1", "score": 1.0, "selected": True},
                    {"kind": "feature_discovery", "feature": "x2", "score": 0.8, "selected": True},
                ],
                "artifacts": [],
                "error": None,
            },
            "analysis_bocpd_gaussian": {
                "status": "ok",
                "summary": "",
                "metrics": {},
                "findings": [{"kind": "changepoint", "index": 10, "prob": 0.9}],
                "artifacts": [],
                "error": None,
            },
            "analysis_gaussian_copula_shift": {
                "status": "ok",
                "summary": "",
                "metrics": {},
                "findings": [{"kind": "dependence_shift", "pair": ["a", "b"], "delta": 0.5, "p_value": 0.1}],
                "artifacts": [],
                "error": None,
            },
            "analysis_conformal_feature_prediction": {
                "status": "ok",
                "summary": "",
                "metrics": {},
                "findings": [{"kind": "anomaly", "column": "y", "row_index": 7, "score": 1.0, "lower": 0.0, "upper": 2.0}],
                "artifacts": [],
                "error": None,
            },
        },
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    ok, messages = evaluate_report(report_path, Path("tests/fixtures/ground_truth_synth.yaml"))
    assert ok, messages
