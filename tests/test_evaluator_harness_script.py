from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluator_harness import main as evaluator_main


def test_evaluator_harness_writes_payload(tmp_path: Path, monkeypatch) -> None:
    report = {
        "plugins": {
            "p1": {
                "findings": [
                    {"kind": "feature_discovery", "feature": "f1"},
                ]
            }
        }
    }
    truth = {"strict": False, "features": ["f1"]}
    report_path = tmp_path / "report.json"
    gt_path = tmp_path / "truth.yaml"
    out_path = tmp_path / "eval.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    gt_path.write_text("strict: false\nfeatures:\n  - f1\n", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluator_harness.py",
            "--report-json",
            str(report_path),
            "--ground-truth",
            str(gt_path),
            "--out-json",
            str(out_path),
            "--strict",
        ],
    )
    rc = evaluator_main()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["ok"] is True

