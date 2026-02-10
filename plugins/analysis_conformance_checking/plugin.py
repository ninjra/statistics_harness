from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from statistic_harness.core.stat_plugins import (
    BudgetTimer,
    merge_config,
    stable_id,
)
from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json


DEFAULTS = {
    "conformance": {
        "max_variants": 20,
        "min_unexpected_count": 1,
        "baseline_coverage": 0.9,
    }
}


def _infer_eventlog_columns(df: pd.DataFrame, roles: dict[str, str]) -> tuple[str | None, str | None, str | None]:
    case_col = None
    activity_col = None
    timestamp_col = None
    lower_names = {col: str(col).lower() for col in df.columns}
    for col in df.columns:
        if roles.get(col) == "id":
            case_col = case_col or col
        if roles.get(col) == "timestamp":
            timestamp_col = timestamp_col or col
    for col in df.columns:
        lname = lower_names[col]
        if case_col is None and (lname.endswith("id") or "case" in lname or "session" in lname):
            case_col = col
        if activity_col is None and (
            "activity" in lname
            or "event" in lname
            or "step" in lname
            or "process" in lname
            or "action" in lname
            or "task" in lname
        ):
            activity_col = col
        if timestamp_col is None and ("time" in lname or "date" in lname):
            timestamp_col = col
    return case_col, activity_col, timestamp_col


class Plugin:
    def run(self, ctx) -> PluginResult:
        config = merge_config(ctx.settings)
        config["conformance"] = {**DEFAULTS["conformance"], **config.get("conformance", {})}
        timer = BudgetTimer(config.get("time_budget_ms"))

        df = ctx.dataset_loader()
        if df.empty:
            return PluginResult("skipped", "Empty dataset", {}, [], [], None)

        roles = {}
        if ctx.dataset_version_id:
            columns_meta = ctx.storage.fetch_dataset_columns(ctx.dataset_version_id)
            roles = {col["original_name"]: (col.get("role") or "") for col in columns_meta}

        case_col, activity_col, timestamp_col = _infer_eventlog_columns(df, roles)
        if case_col is None or activity_col is None:
            return PluginResult("skipped", "No event log columns detected", {}, [], [], None)

        work = df.copy().reset_index().rename(columns={"index": "row_index"})
        sort_cols = [case_col]
        if timestamp_col and timestamp_col in work.columns:
            sort_cols.append(timestamp_col)
        else:
            sort_cols.append("row_index")
        work = work.sort_values(sort_cols)

        transitions: dict[tuple[str, str], int] = defaultdict(int)
        for _, group in work.groupby(case_col, sort=False):
            if timer.exceeded():
                break
            activities = [str(x) for x in group[activity_col].tolist()]
            for a, b in zip(activities, activities[1:]):
                transitions[(a, b)] += 1

        if not transitions:
            return PluginResult("skipped", "No transitions detected", {}, [], [], None)

        ordered = sorted(transitions.items(), key=lambda item: (-item[1], item[0]))
        max_variants = int(config["conformance"].get("max_variants", 20))
        baseline_coverage = float(config["conformance"].get("baseline_coverage", 0.9))
        total = sum(transitions.values())
        baseline: set[tuple[str, str]] = set()
        covered = 0
        for pair, count in ordered:
            if len(baseline) >= max_variants:
                break
            baseline.add(pair)
            covered += count
            if total and (covered / total) >= baseline_coverage:
                break

        unexpected = [
            (pair, count)
            for pair, count in ordered
            if pair not in baseline
            and count >= int(config["conformance"].get("min_unexpected_count", 5))
        ]

        baseline_count = sum(count for pair, count in transitions.items() if pair in baseline)
        conformance_rate = baseline_count / total if total else 0.0

        findings: list[dict[str, Any]] = []
        for (a, b), count in unexpected[: int(config.get("max_findings", 30))]:
            findings.append(
                {
                    "id": stable_id(f"{a}->{b}:{count}"),
                    "severity": "warn",
                    "confidence": 0.6,
                    "title": "Unexpected transition detected",
                    "what": f"Transition {a} -> {b} not in baseline flow.",
                    "why": "Observed transition deviates from dominant process variants.",
                    "evidence": {"metrics": {"count": count, "share": count / total if total else 0.0}},
                    "where": {"from": a, "to": b},
                    "recommendation": "Inspect event ordering or update expected flow.",
                    "measurement_type": "measured",
                    "references": [
                        {
                            "title": "Process Mining Conformance Checking",
                            "url": "https://doi.org/10.1007/978-3-642-19345-3",
                            "doi": "10.1007/978-3-642-19345-3",
                        }
                    ],
                }
            )

        artifacts_dir = ctx.artifacts_dir("analysis_conformance_checking")
        out_path = artifacts_dir / "conformance.json"
        write_json(
            out_path,
            {
                "transitions": [{"from": a, "to": b, "count": count} for (a, b), count in ordered],
                "baseline": [{"from": a, "to": b} for (a, b) in baseline],
                "conformance_rate": conformance_rate,
            },
        )
        artifacts = [
            PluginArtifact(
                path=str(out_path.relative_to(ctx.run_dir)),
                type="json",
                description="Conformance summary",
            )
        ]

        metrics = {
            "transitions": len(transitions),
            "conformance_rate": conformance_rate,
            "references": [
                {
                    "title": "Process Mining Conformance Checking",
                    "url": "https://doi.org/10.1007/978-3-642-19345-3",
                    "doi": "10.1007/978-3-642-19345-3",
                }
            ],
        }

        summary = f"Conformance rate {conformance_rate:.2f}, {len(findings)} unexpected transitions."

        return PluginResult(
            "ok",
            summary,
            metrics,
            findings,
            artifacts,
            None,
        )
