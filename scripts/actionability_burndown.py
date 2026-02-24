#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from scripts.audit_plugin_actionability import audit_run
except ModuleNotFoundError:  # pragma: no cover
    from audit_plugin_actionability import audit_run


ROOT = Path(__file__).resolve().parents[1]


def _build_payload(run_id: str, *, recompute: bool) -> dict[str, Any]:
    audit = audit_run(run_id, recompute=recompute)
    rows = audit.get("plugins") if isinstance(audit.get("plugins"), list) else []
    unresolved = [
        row
        for row in rows
        if isinstance(row, dict)
        and str(row.get("actionability_state") or "").strip().lower() != "actionable"
    ]
    by_reason: Counter[str] = Counter()
    by_lane: Counter[str] = Counter()
    lane_samples: dict[str, list[str]] = defaultdict(list)
    reason_samples: dict[str, list[str]] = defaultdict(list)

    for row in unresolved:
        plugin_id = str(row.get("plugin_id") or "").strip()
        reason = str(row.get("reason_code") or "UNSPECIFIED").strip() or "UNSPECIFIED"
        lane = str(row.get("next_step_lane_id") or "UNMAPPED").strip() or "UNMAPPED"
        by_reason[reason] += 1
        by_lane[lane] += 1
        if plugin_id and len(lane_samples[lane]) < 12:
            lane_samples[lane].append(plugin_id)
        if plugin_id and len(reason_samples[reason]) < 12:
            reason_samples[reason].append(plugin_id)

    unresolved_sorted = sorted(
        unresolved,
        key=lambda row: (
            str(row.get("next_step_lane_id") or "zzzz"),
            str(row.get("reason_code") or "zzzz"),
            -int(row.get("finding_count") or 0),
            str(row.get("plugin_id") or "zzzz"),
        ),
    )

    payload = {
        "schema_version": "actionability_burndown.v1",
        "run_id": run_id,
        "recomputed_recommendations": bool(recompute),
        "unresolved_count": int(len(unresolved_sorted)),
        "reason_counts": {k: int(v) for k, v in by_reason.most_common()},
        "lane_counts": {k: int(v) for k, v in by_lane.most_common()},
        "lane_samples": {k: v for k, v in sorted(lane_samples.items())},
        "reason_samples": {k: v for k, v in sorted(reason_samples.items())},
        "unresolved_plugins": unresolved_sorted,
    }
    return payload


def _render_md(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Actionability Burndown")
    lines.append("")
    lines.append(f"- run_id: {payload.get('run_id')}")
    lines.append(f"- unresolved_count: {int(payload.get('unresolved_count') or 0)}")
    lines.append("")
    lines.append("## By Lane")
    for lane, count in (payload.get("lane_counts") or {}).items():
        sample = ", ".join((payload.get("lane_samples") or {}).get(lane, [])[:5])
        sample_txt = f" ({sample})" if sample else ""
        lines.append(f"- {lane}: {int(count)}{sample_txt}")
    lines.append("")
    lines.append("## By Reason")
    for reason, count in (payload.get("reason_counts") or {}).items():
        sample = ", ".join((payload.get("reason_samples") or {}).get(reason, [])[:5])
        sample_txt = f" ({sample})" if sample else ""
        lines.append(f"- {reason}: {int(count)}{sample_txt}")
    lines.append("")
    lines.append("## Unresolved Plugins")
    lines.append("")
    lines.append("| plugin_id | lane | reason | finding_count | next_step |")
    lines.append("| --- | --- | --- | ---: | --- |")
    for row in payload.get("unresolved_plugins") or []:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("plugin_id") or ""),
                    str(row.get("next_step_lane_id") or ""),
                    str(row.get("reason_code") or ""),
                    str(int(row.get("finding_count") or 0)),
                    str(row.get("recommended_next_step") or ""),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a deterministic unresolved-actionability burndown for a run."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--recompute-recommendations", action="store_true")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    payload = _build_payload(
        str(args.run_id).strip(),
        recompute=bool(args.recompute_recommendations),
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"

    out_json = str(args.out_json).strip()
    if out_json:
        out_path = Path(out_json)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered, encoding="utf-8")

    out_md = str(args.out_md).strip()
    if out_md:
        out_path = Path(out_md)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(_render_md(payload), encoding="utf-8")

    print(rendered, end="")
    if bool(args.strict) and int(payload.get("unresolved_count") or 0) > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
