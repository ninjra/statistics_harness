from __future__ import annotations

import json

from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json


class Plugin:
    def run(self, ctx) -> PluginResult:
        run_id = ctx.run_id
        rows = ctx.storage.fetch_plugin_results(run_id)

        missing_measurement = 0
        modeled_missing_assumption = 0
        checked = 0
        violations = []

        for row in rows:
            plugin_id = row.get("plugin_id") or "unknown"
            if plugin_id == "analysis_determinism_discipline":
                continue
            findings = []
            try:
                findings = json.loads(row.get("findings_json") or "[]")
            except json.JSONDecodeError:
                findings = []
            if not isinstance(findings, list):
                continue
            for item in findings:
                if not isinstance(item, dict):
                    continue
                checked += 1
                measurement_type = item.get("measurement_type")
                if not measurement_type:
                    missing_measurement += 1
                    violations.append(
                        {
                            "kind": "determinism_violation",
                            "plugin_id": plugin_id,
                            "issue": "missing_measurement_type",
                            "measurement_type": "error",
                        }
                    )
                    continue
                assumptions = item.get("assumptions")
                if measurement_type == "modeled" and (
                    not isinstance(assumptions, list)
                    or not assumptions
                    or not all(isinstance(a, str) and a.strip() for a in assumptions)
                ):
                    modeled_missing_assumption += 1
                    violations.append(
                        {
                            "kind": "determinism_violation",
                            "plugin_id": plugin_id,
                            "issue": "modeled_missing_assumption",
                            "measurement_type": "error",
                        }
                    )

        summary = {
            "kind": "determinism_discipline_summary",
            "checked_findings": checked,
            "missing_measurement_type": missing_measurement,
            "modeled_missing_assumption": modeled_missing_assumption,
            "violations": len(violations),
            "measurement_type": "measured",
        }
        findings_out = [summary, *violations]

        artifacts_dir = ctx.artifacts_dir("analysis_determinism_discipline")
        out_path = artifacts_dir / "determinism_checks.json"
        write_json(out_path, {"summary": summary, "violations": violations})
        artifacts = [
            PluginArtifact(
                path=str(out_path.relative_to(ctx.run_dir)),
                type="json",
                description="Determinism discipline checks",
            )
        ]

        return PluginResult(
            "ok",
            "Checked measurement_type discipline",
            {
                "checked_findings": checked,
                "missing_measurement_type": missing_measurement,
                "modeled_missing_assumption": modeled_missing_assumption,
            },
            findings_out,
            artifacts,
            None,
        )
