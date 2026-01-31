from __future__ import annotations

from pathlib import Path

from statistic_harness.core.report import build_report, write_report
from statistic_harness.core.types import PluginArtifact, PluginResult


class Plugin:
    def run(self, ctx) -> PluginResult:
        schema_path = Path("docs/report.schema.json")
        report = build_report(ctx.storage, ctx.run_id, ctx.run_dir, schema_path)
        write_report(report, ctx.run_dir)
        artifacts = [
            PluginArtifact(path="report.json", type="json", description="Report JSON"),
            PluginArtifact(path="report.md", type="markdown", description="Report Markdown"),
        ]
        return PluginResult("ok", "Report generated", {}, [], artifacts, None)
