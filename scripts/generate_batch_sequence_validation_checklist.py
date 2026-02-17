#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_text(value: Any) -> str:
    return str(value).strip() if isinstance(value, str) else ""


def _target_process_ids(item: dict[str, Any]) -> list[str]:
    targets = item.get("target_process_ids")
    if isinstance(targets, list):
        values = [str(v).strip() for v in targets if str(v).strip()]
        if values:
            return values
    evidence = item.get("evidence")
    if isinstance(evidence, list):
        for row in evidence:
            if not isinstance(row, dict):
                continue
            vals = row.get("target_process_ids")
            if isinstance(vals, list):
                values = [str(v).strip() for v in vals if str(v).strip()]
                if values:
                    return values
    if isinstance(evidence, dict):
        vals = evidence.get("target_process_ids")
        if isinstance(vals, list):
            values = [str(v).strip() for v in vals if str(v).strip()]
            if values:
                return values
    return []


def _first_evidence(item: dict[str, Any]) -> dict[str, Any]:
    evidence = item.get("evidence")
    if isinstance(evidence, dict):
        return evidence
    if isinstance(evidence, list):
        for row in evidence:
            if isinstance(row, dict):
                return row
    return {}


def _sequence_id(item: dict[str, Any], evidence: dict[str, Any], targets: list[str]) -> str:
    explicit = _safe_text(item.get("sequence_id")) or _safe_text(evidence.get("sequence_id"))
    if explicit:
        return explicit
    close_month = _safe_text(item.get("best_close_month")) or _safe_text(evidence.get("close_month")) or "unknown"
    key = _safe_text(item.get("key")) or _safe_text(evidence.get("key")) or "unknown"
    payload = f"{close_month}|{key}|{'|'.join(targets)}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"batchseq_{digest}"


def _as_float(value: Any) -> float | None:
    try:
        if isinstance(value, (int, float)):
            return float(value)
    except Exception:
        return None
    return None


def _validation_steps(item: dict[str, Any]) -> list[str]:
    out: list[str] = []
    raw = item.get("validation_steps")
    if isinstance(raw, list):
        for step in raw:
            text = _safe_text(step)
            if text:
                out.append(text)
    if out:
        return out
    return [
        "Deploy batched input for listed process IDs using the detected key.",
        "Re-run the same close-month cohort and compare job-launch counts and queue wait.",
        "Confirm modeled delta tracks in both report.md and report.json recommendations.",
    ]


def build_checklist_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    recommendations = report.get("recommendations") if isinstance(report, dict) else {}
    if not isinstance(recommendations, dict):
        return []
    discovery = recommendations.get("discovery") if isinstance(recommendations.get("discovery"), dict) else {}
    items = discovery.get("items") if isinstance(discovery.get("items"), list) else []
    rows: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        action_type = _safe_text(item.get("action_type")) or _safe_text(item.get("action"))
        if action_type != "batch_group_candidate":
            continue
        evidence = _first_evidence(item)
        targets = _target_process_ids(item)
        if not targets:
            continue
        sequence_id = _sequence_id(item, evidence, targets)
        close_month = _safe_text(item.get("best_close_month")) or _safe_text(evidence.get("close_month")) or "unknown"
        key = _safe_text(item.get("key")) or _safe_text(evidence.get("key")) or "unknown"
        modeled_delta_hours = _as_float(item.get("modeled_delta_hours"))
        if modeled_delta_hours is None:
            modeled_delta_hours = _as_float(item.get("impact_hours"))
        if modeled_delta_hours is None:
            modeled_delta_hours = _as_float(item.get("modeled_delta"))
        rows.append(
            {
                "sequence_id": sequence_id,
                "plugin_id": _safe_text(item.get("plugin_id")) or "analysis_actionable_ops_levers_v1",
                "close_month": close_month,
                "key": key,
                "target_process_ids": targets,
                "target_process_count": len(targets),
                "modeled_delta_hours": modeled_delta_hours,
                "validation_steps": _validation_steps(item),
                "recommendation": _safe_text(item.get("recommendation")) or _safe_text(item.get("title")),
            }
        )
    return sorted(rows, key=lambda row: (str(row.get("close_month") or ""), str(row.get("sequence_id") or "")))


def _render_markdown(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Batch Sequence Validation Checklist")
    lines.append("")
    if not rows:
        lines.append("- No `batch_group_candidate` discovery recommendations were found.")
        lines.append("")
        return "\n".join(lines)
    lines.append(f"- sequence_count: {len(rows)}")
    lines.append("")
    for idx, row in enumerate(rows, start=1):
        sequence_id = str(row.get("sequence_id") or "")
        close_month = str(row.get("close_month") or "unknown")
        key = str(row.get("key") or "unknown")
        targets = row.get("target_process_ids") if isinstance(row.get("target_process_ids"), list) else []
        delta = row.get("modeled_delta_hours")
        delta_text = f"{float(delta):.2f}h" if isinstance(delta, (int, float)) else "n/a"
        lines.append(f"## {idx}. {sequence_id}")
        lines.append("")
        lines.append(f"- close_month: `{close_month}`")
        lines.append(f"- key: `{key}`")
        lines.append(f"- modeled_delta_hours: {delta_text}")
        lines.append(f"- target_process_ids: {', '.join([str(v) for v in targets])}")
        lines.append("- validation_steps:")
        steps = row.get("validation_steps") if isinstance(row.get("validation_steps"), list) else []
        for step in steps:
            lines.append(f"  - {str(step)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def generate_for_run_dir(run_dir: Path) -> tuple[dict[str, Any], str]:
    report_path = run_dir / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing report.json: {report_path}")
    report = _read_json(report_path)
    rows = build_checklist_rows(report if isinstance(report, dict) else {})
    payload = {
        "run_id": str(report.get("run_id") or run_dir.name) if isinstance(report, dict) else run_dir.name,
        "generated_from": str(report_path),
        "sequence_count": len(rows),
        "sequences": rows,
    }
    return payload, _render_markdown(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--appdata-root", default="appdata")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    args = parser.parse_args()

    run_id = str(args.run_id).strip()
    if not run_id:
        raise SystemExit("--run-id is required")
    appdata = Path(str(args.appdata_root))
    run_dir = appdata / "runs" / run_id
    if not run_dir.exists():
        raise SystemExit(f"Missing run directory: {run_dir}")

    payload, markdown = generate_for_run_dir(run_dir)
    out_json = Path(str(args.out_json)).resolve() if str(args.out_json).strip() else (run_dir / "batch_sequence_validation_checklist.json").resolve()
    out_md = Path(str(args.out_md)).resolve() if str(args.out_md).strip() else (run_dir / "batch_sequence_validation_checklist.md").resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text(markdown, encoding="utf-8")
    print(f"out_json={out_json}")
    print(f"out_md={out_md}")
    print(f"sequence_count={payload.get('sequence_count')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
