from __future__ import annotations

import json
from typing import Any

from statistic_harness.core.process_filters import process_is_excluded
from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json


_OPERATIONAL_ACTIONS = [
    "isolate capacity for this process during close-cycle peak hours",
    "apply aged-first prioritization with strict dependency ordering",
    "pre-stage prerequisites to remove avoidable queue latency",
    "batch low-variance work and add retry backoff to reduce contention",
]


def _confidence_tag(gap_pct: float, runs: int, windows: int) -> str:
    if gap_pct >= 0.5 and runs >= 80 and windows >= 2:
        return "high"
    if gap_pct >= 0.25 and runs >= 40 and windows >= 2:
        return "medium"
    return "low"


class Plugin:
    def run(self, ctx) -> PluginResult:
        rows = ctx.storage.fetch_plugin_results(ctx.run_id)
        gaps: list[dict[str, Any]] = []
        for row in rows:
            if row.get("plugin_id") != "analysis_ideaspace_normative_gap":
                continue
            try:
                findings = json.loads(row.get("findings_json") or "[]")
            except json.JSONDecodeError:
                findings = []
            for item in findings:
                if isinstance(item, dict) and item.get("kind") == "ideaspace_gap":
                    gaps.append(item)

        if not gaps:
            return PluginResult("skipped", "No normative gaps available", {}, [], [], None)

        min_gap_sec = float(ctx.settings.get("min_gap_sec") or 180.0)
        min_runs = int(ctx.settings.get("min_runs") or 20)
        min_gap_pct = float(ctx.settings.get("min_gap_pct") or 0.1)
        max_actions = int(ctx.settings.get("max_actions") or 5)
        exclude_tokens = ctx.settings.get("exclude_processes") or []

        candidates = []
        for gap in gaps:
            process_id = gap.get("process_id")
            runs = int(gap.get("runs") or 0)
            windows = int(gap.get("windows") or 0)
            gap_sec = float(gap.get("gap_sec") or 0.0)
            gap_pct = float(gap.get("gap_pct") or 0.0)
            if process_is_excluded(process_id, exclude_tokens):
                continue
            if runs < min_runs or windows < 2:
                continue
            if gap_sec < min_gap_sec or gap_pct < min_gap_pct:
                continue
            candidates.append(gap)

        candidates.sort(
            key=lambda g: (float(g.get("gap_sec") or 0), float(g.get("gap_pct") or 0), int(g.get("runs") or 0)),
            reverse=True,
        )

        findings: list[dict[str, Any]] = []
        for idx, gap in enumerate(candidates[:max_actions], start=1):
            process_id = str(gap.get("process_id") or "")
            gap_sec = float(gap.get("gap_sec") or 0.0)
            runs = int(gap.get("runs") or 0)
            windows = int(gap.get("windows") or 0)
            gap_pct = float(gap.get("gap_pct") or 0.0)
            action = _OPERATIONAL_ACTIONS[(idx - 1) % len(_OPERATIONAL_ACTIONS)]
            # Conservative no-speculation modeled reduction from observed gap.
            delta_sec = round(gap_sec * 0.25, 2)
            delta_hours = round((delta_sec * runs) / 3600.0, 2)
            conf = _confidence_tag(gap_pct, runs, windows)
            findings.append(
                {
                    "kind": "ideaspace_action",
                    "measurement_type": "modeled",
                    "action_type": "operational_procedure",
                    "target": process_id,
                    "process_id": process_id,
                    "scenario_id": f"ideaspace-operational-{idx}",
                    "estimated_delta_seconds": delta_sec,
                    "estimated_delta_hours_total": delta_hours,
                    "estimated_delta_pct": round(gap_pct * 25.0, 4),
                    "confidence_tag": conf,
                    "validation_steps": [
                        "Run a controlled before/after for at least 2 monthly windows",
                        "Compare p50 and p90 eligible->start latency for the target process_id",
                        "Rollback if p90 regresses by >5% versus baseline",
                    ],
                    "evidence": {
                        "runs": runs,
                        "windows": windows,
                        "median_wait_sec": float(gap.get("median_wait_sec") or 0.0),
                        "ideal_wait_sec": float(gap.get("ideal_wait_sec") or 0.0),
                        "gap_sec": gap_sec,
                        "gap_pct": gap_pct,
                    },
                    "recommendation": (
                        f"Target process {process_id}: {action}. "
                        f"Estimated reduction {delta_sec:.0f}s/run (~{delta_hours:.1f}h total), "
                        f"supported by {runs} runs across {windows} windows with {gap_sec:.0f}s measured gap."
                    ),
                    "references": ["docs/ideaspace.md#action-planner", "ideaspace:action-planner"],
                }
            )

        if not findings:
            findings = [
                {
                    "kind": "ideaspace_action",
                    "measurement_type": "not_applicable",
                    "reason": "No candidate passed minimum N, >=2-window stability, and effect-size gates.",
                    "references": ["docs/ideaspace.md#action-planner"],
                }
            ]
            status = "skipped"
            summary = "No actions cleared no-speculation gates"
        else:
            status = "ok"
            summary = f"Generated {len(findings)} actionable ideaspace recommendation(s)"

        artifacts_dir = ctx.artifacts_dir("analysis_ideaspace_action_planner")
        out_path = artifacts_dir / "actions.json"
        write_json(out_path, {"actions": findings})
        artifacts = [
            PluginArtifact(
                path=str(out_path.relative_to(ctx.run_dir)),
                type="json",
                description="Ideaspace action recommendations",
            )
        ]
        return PluginResult(
            status,
            summary,
            {"actions": len([f for f in findings if f.get("process_id")])},
            findings,
            artifacts,
            None,
        )
