from __future__ import annotations

import json

from statistic_harness.core.process_filters import process_is_excluded
from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json


ACTION_BY_SCORE = [
    "isolate queue lane and pin capacity during close-cycle windows",
    "prioritize aged work first with strict dependency ordering",
    "batch small jobs and apply retry backoff to avoid contention spikes",
]


class Plugin:
    def run(self, ctx) -> PluginResult:
        rows = ctx.storage.fetch_plugin_results(ctx.run_id)
        gaps: list[dict] = []
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

        min_gap_sec = float(ctx.settings.get("min_gap_sec") or 180)
        min_runs = int(ctx.settings.get("min_runs") or 20)
        max_actions = int(ctx.settings.get("max_actions") or 5)
        exclude_tokens = ctx.settings.get("exclude_processes") or []

        candidates = [
            g for g in gaps
            if float(g.get("gap_sec") or 0) >= min_gap_sec
            and int(g.get("runs") or 0) >= min_runs
            and not process_is_excluded(g.get("process_id"), exclude_tokens)
        ]
        candidates.sort(key=lambda g: (float(g.get("gap_sec") or 0), float(g.get("gap_pct") or 0)), reverse=True)

        findings: list[dict] = []
        for idx, gap in enumerate(candidates[:max_actions], start=1):
            process_id = str(gap.get("process_id") or "")
            action = ACTION_BY_SCORE[(idx - 1) % len(ACTION_BY_SCORE)]
            delta_sec = float(gap.get("gap_sec") or 0) * 0.35
            delta_hours = delta_sec * float(gap.get("runs") or 0) / 3600.0
            findings.append(
                {
                    "kind": "ideaspace_action",
                    "measurement_type": "modeled",
                    "action_type": "operational_procedure",
                    "target": process_id,
                    "process_id": process_id,
                    "scenario_id": f"ideaspace-{idx}",
                    "estimated_delta_seconds": round(delta_sec, 2),
                    "estimated_delta_hours_total": round(delta_hours, 2),
                    "estimated_delta_pct": round(float(gap.get("gap_pct") or 0) * 35.0, 4),
                    "confidence_tag": "medium" if float(gap.get("gap_pct") or 0) < 0.4 else "high",
                    "validation_steps": [
                        "A/B the queue policy for two close-cycle windows",
                        "Track p50/p90 eligible->start by process_id and rollback if regression >5%",
                    ],
                    "evidence": {
                        "runs": int(gap.get("runs") or 0),
                        "median_wait_sec": float(gap.get("median_wait_sec") or 0),
                        "ideal_wait_sec": float(gap.get("ideal_wait_sec") or 0),
                        "gap_sec": float(gap.get("gap_sec") or 0),
                    },
                    "recommendation": f"For process {process_id}, {action}; estimated median eligible->start reduction {delta_sec:.0f}s/run (~{delta_hours:.1f}h total).",
                    "references": ["ideaspace:action-planner"],
                }
            )

        if not findings:
            findings = [{"kind": "ideaspace_action", "measurement_type": "not_applicable", "reason": "No candidate passed no-speculation gates."}]
            status = "skipped"
            summary = "No actions cleared gates"
        else:
            status = "ok"
            summary = f"Generated {len(findings)} actionable ideaspace recommendation(s)"

        artifacts_dir = ctx.artifacts_dir("analysis_ideaspace_action_planner")
        out = artifacts_dir / "actions.json"
        write_json(out, {"actions": findings})
        artifacts = [PluginArtifact(path=str(out.relative_to(ctx.run_dir)), type="json", description="Ideaspace action recommendations")]
        return PluginResult(status, summary, {"actions": len([f for f in findings if f.get('process_id')])}, findings, artifacts, None)
