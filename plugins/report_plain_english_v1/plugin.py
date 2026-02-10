from __future__ import annotations

from statistic_harness.core.plain_report import build_plain_report
from statistic_harness.core.types import PluginArtifact, PluginResult


class Plugin:
    def run(self, ctx) -> PluginResult:
        report_path = ctx.run_dir / "report.json"
        if not report_path.exists():
            return PluginResult("error", "Missing report.json", {}, [], [], None)

        text = build_plain_report(report_path)
        out_path = ctx.run_dir / "plain_report.md"
        out_path.write_text(text, encoding="utf-8")
        artifacts = [
            PluginArtifact(path="plain_report.md", type="markdown", description="Plain-English report"),
        ]
        return PluginResult("ok", "Wrote plain_report.md", {}, [], artifacts, None)

