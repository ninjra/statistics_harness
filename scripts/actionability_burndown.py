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


def _count_map_delta(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> dict[str, dict[str, int]]:
    before_map = before if isinstance(before, dict) else {}
    after_map = after if isinstance(after, dict) else {}
    keys = sorted(set(before_map.keys()) | set(after_map.keys()))
    out: dict[str, dict[str, int]] = {}
    for key in keys:
        try:
            before_n = int(before_map.get(key) or 0)
        except (TypeError, ValueError):
            before_n = 0
        try:
            after_n = int(after_map.get(key) or 0)
        except (TypeError, ValueError):
            after_n = 0
        out[str(key)] = {
            "before": before_n,
            "after": after_n,
            "delta": int(after_n - before_n),
        }
    return out


def _plugin_ids(payload: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for row in payload.get("unresolved_plugins") or []:
        if not isinstance(row, dict):
            continue
        plugin_id = str(row.get("plugin_id") or "").strip()
        if plugin_id:
            out.add(plugin_id)
    return out


def _comparison_payload(before_payload: dict[str, Any], after_payload: dict[str, Any]) -> dict[str, Any]:
    before_ids = _plugin_ids(before_payload)
    after_ids = _plugin_ids(after_payload)
    return {
        "before_run_id": str(before_payload.get("run_id") or ""),
        "after_run_id": str(after_payload.get("run_id") or ""),
        "unresolved_count_before": int(before_payload.get("unresolved_count") or 0),
        "unresolved_count_after": int(after_payload.get("unresolved_count") or 0),
        "unresolved_count_delta": int(
            int(after_payload.get("unresolved_count") or 0)
            - int(before_payload.get("unresolved_count") or 0)
        ),
        "reason_counts": _count_map_delta(
            before_payload.get("reason_counts"),
            after_payload.get("reason_counts"),
        ),
        "lane_counts": _count_map_delta(
            before_payload.get("lane_counts"),
            after_payload.get("lane_counts"),
        ),
        "newly_unresolved_plugins": sorted(after_ids - before_ids),
        "resolved_plugins": sorted(before_ids - after_ids),
        "unchanged_unresolved_plugins": sorted(before_ids & after_ids),
    }


def _render_md(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Actionability Burndown")
    lines.append("")
    lines.append(f"- run_id: {payload.get('run_id')}")
    lines.append(f"- unresolved_count: {int(payload.get('unresolved_count') or 0)}")
    before_run_id = str(payload.get("before_run_id") or "").strip()
    if before_run_id:
        lines.append(f"- before_run_id: {before_run_id}")
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
    comparison = payload.get("comparison") if isinstance(payload.get("comparison"), dict) else {}
    if comparison:
        lines.append("## Delta Vs Before")
        lines.append("")
        lines.append(
            f"- unresolved_count_before: {int(comparison.get('unresolved_count_before') or 0)}"
        )
        lines.append(
            f"- unresolved_count_after: {int(comparison.get('unresolved_count_after') or 0)}"
        )
        lines.append(
            f"- unresolved_count_delta: {int(comparison.get('unresolved_count_delta') or 0)}"
        )
        lines.append("")
        lines.append("### Reason Delta")
        for reason, row in (comparison.get("reason_counts") or {}).items():
            if not isinstance(row, dict):
                continue
            lines.append(
                f"- {reason}: before={int(row.get('before') or 0)}"
                f" after={int(row.get('after') or 0)}"
                f" delta={int(row.get('delta') or 0)}"
            )
        lines.append("")
        lines.append("### Lane Delta")
        for lane, row in (comparison.get("lane_counts") or {}).items():
            if not isinstance(row, dict):
                continue
            lines.append(
                f"- {lane}: before={int(row.get('before') or 0)}"
                f" after={int(row.get('after') or 0)}"
                f" delta={int(row.get('delta') or 0)}"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a deterministic unresolved-actionability burndown for a run."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--before-run-id", default="")
    parser.add_argument("--recompute-recommendations", action="store_true")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    payload = _build_payload(
        str(args.run_id).strip(),
        recompute=bool(args.recompute_recommendations),
    )
    before_run_id = str(args.before_run_id).strip()
    if before_run_id:
        before_payload = _build_payload(
            before_run_id,
            recompute=bool(args.recompute_recommendations),
        )
        payload["before_run_id"] = before_run_id
        payload["comparison"] = _comparison_payload(before_payload, payload)
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
