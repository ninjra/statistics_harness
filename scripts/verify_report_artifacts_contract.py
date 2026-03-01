#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from jsonschema import validate


ROOT = Path(__file__).resolve().parents[1]


def verify_report_artifacts(run_dir: Path) -> dict[str, Any]:
    report_json = run_dir / "report.json"
    report_md = run_dir / "report.md"
    schema_path = ROOT / "docs" / "report.schema.json"
    violations: list[str] = []

    if not report_json.exists() or report_json.stat().st_size <= 0:
        violations.append("REPORT_JSON_MISSING_OR_EMPTY")
    if not report_md.exists() or report_md.stat().st_size <= 0:
        violations.append("REPORT_MD_MISSING_OR_EMPTY")

    if not violations:
        try:
            payload = json.loads(report_json.read_text(encoding="utf-8"))
        except Exception:
            violations.append("REPORT_JSON_INVALID_JSON")
            payload = None
        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
        except Exception:
            violations.append("REPORT_SCHEMA_UNREADABLE")
            schema = None
        if isinstance(payload, dict) and isinstance(schema, dict):
            try:
                validate(instance=payload, schema=schema)
            except Exception:
                violations.append("REPORT_JSON_SCHEMA_INVALID")

    return {
        "schema_version": "report_artifacts_contract.v1",
        "run_dir": str(run_dir),
        "ok": len(violations) == 0,
        "violations": violations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify report.md/report.json contract for a run.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-root", default=str(ROOT / "appdata" / "runs"))
    parser.add_argument("--out-json", default="")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    runs_root = Path(str(args.runs_root))
    run_dir = runs_root / str(args.run_id).strip()
    payload = verify_report_artifacts(run_dir)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    out_json = str(args.out_json).strip()
    if out_json:
        out_path = Path(out_json)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if bool(args.strict) and not bool(payload.get("ok")):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

