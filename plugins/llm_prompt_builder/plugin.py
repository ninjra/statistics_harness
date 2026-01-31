from __future__ import annotations

import json

from statistic_harness.core.types import PluginArtifact, PluginResult


class Plugin:
    def run(self, ctx) -> PluginResult:
        report_path = ctx.run_dir / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        artifacts_dir = ctx.artifacts_dir("llm_prompt_builder")
        prompt = "# Analysis Summary\n\n" + json.dumps(report, indent=2)
        brief = "# Brief\n\nSummary of findings."
        prompt_path = artifacts_dir / "prompt.md"
        brief_path = artifacts_dir / "brief.md"
        prompt_path.write_text(prompt, encoding="utf-8")
        brief_path.write_text(brief, encoding="utf-8")
        artifacts = [
            PluginArtifact(path=str(prompt_path.relative_to(ctx.run_dir)), type="markdown", description="Prompt"),
            PluginArtifact(path=str(brief_path.relative_to(ctx.run_dir)), type="markdown", description="Brief"),
        ]
        return PluginResult("ok", "Built LLM prompt", {}, [], artifacts, None)
