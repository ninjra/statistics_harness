from __future__ import annotations

from pathlib import Path

from statistic_harness.core.payout_report import build_payout_report
from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json


class Plugin:
    def run(self, ctx) -> PluginResult:
        # Report plugins run late; keep logic local and deterministic.
        df = ctx.dataset_loader()
        if df is None or df.empty:
            return PluginResult(
                status="skipped",
                summary="Empty dataset",
                metrics={"rows_seen": 0, "rows_used": 0, "cols_used": 0},
                findings=[],
                artifacts=[],
                error=None,
            )

        regex = str(ctx.settings.get("payout_process_regex") or "")
        report = build_payout_report(df, payout_process_regex=regex) if regex else build_payout_report(df)

        artifacts_dir = ctx.artifacts_dir("report_payout_report_v1")
        json_path = artifacts_dir / "payout_report.json"
        csv_path = artifacts_dir / "payout_report.csv"

        # JSON (canonical)
        write_json(json_path, report)
        # CSV (human / spreadsheet friendly)
        rows = []
        overall = dict(report.get("metrics") or {})
        rows.append({"scope": "overall", **overall})
        for row in report.get("per_source") or []:
            if isinstance(row, dict):
                rows.append({"scope": "source", **row})
        import pandas as pd

        pd.DataFrame(rows).to_csv(csv_path, index=False)

        artifacts = [
            PluginArtifact(path=str(json_path.relative_to(ctx.run_dir)), type="json", description="payout_report.json"),
            PluginArtifact(path=str(csv_path.relative_to(ctx.run_dir)), type="csv", description="payout_report.csv"),
        ]
        findings = [
            {
                "kind": "payout_report",
                "summary": str(report.get("summary") or ""),
                "metrics": report.get("metrics") or {},
                "measurement_type": "measured",
            }
        ]
        return PluginResult(
            status="ok",
            summary="Generated payout report artifacts",
            metrics={"rows_seen": int(len(df)), "rows_used": int(len(df)), "cols_used": int(len(df.columns))},
            findings=findings,
            artifacts=artifacts,
            error=None,
        )

