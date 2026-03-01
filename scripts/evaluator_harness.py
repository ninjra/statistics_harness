#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from statistic_harness.core.evaluation import evaluate_report


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate report.json against ground_truth.yaml.")
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--out-json", default="")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    report_path = Path(str(args.report_json))
    if not report_path.is_absolute():
        report_path = ROOT / report_path
    gt_path = Path(str(args.ground_truth))
    if not gt_path.is_absolute():
        gt_path = ROOT / gt_path

    ok, messages = evaluate_report(report_path, gt_path)
    payload: dict[str, Any] = {
        "schema_version": "evaluator_harness.v1",
        "report_json": str(report_path),
        "ground_truth": str(gt_path),
        "ok": bool(ok),
        "messages": list(messages),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    out_json = str(args.out_json).strip()
    if out_json:
        out_path = Path(out_json)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if bool(args.strict) and not bool(ok):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

