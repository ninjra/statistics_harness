from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

import pandas as pd

from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json


def _pick_column(df: pd.DataFrame, preferred: str | None, candidates: list[str]) -> str | None:
    if preferred and preferred in df.columns:
        return preferred
    col_map = {str(c).lower(): str(c) for c in df.columns}
    for cand in candidates:
        if cand in col_map:
            return col_map[cand]
    return None


class Plugin:
    def run(self, ctx) -> PluginResult:
        loader = getattr(ctx, "dataset_loader", None)
        if not callable(loader):
            return PluginResult("skipped", "dataset_loader unavailable", {}, [], [], None)
        df = loader()
        if df is None or len(df) == 0:
            return PluginResult("skipped", "Empty dataset", {"rows": 0}, [], [], None)

        case_col = _pick_column(
            df,
            ctx.settings.get("case_column"),
            ["case_id", "case", "trace_id", "session_id"],
        )
        activity_col = _pick_column(
            df,
            ctx.settings.get("activity_column"),
            ["activity", "event", "process", "step", "task"],
        )
        time_col = _pick_column(
            df,
            ctx.settings.get("time_column"),
            ["timestamp", "ts", "start_ts", "event_time", "datetime"],
        )
        if not case_col or not activity_col:
            return PluginResult(
                "skipped",
                "Missing case/activity columns for process conformance",
                {"rows": int(len(df))},
                [],
                [],
                None,
            )

        data = df[[case_col, activity_col] + ([time_col] if time_col else [])].copy()
        data[activity_col] = data[activity_col].astype(str)
        if time_col:
            data["_t"] = pd.to_datetime(data[time_col], errors="coerce")
            data = data.sort_values([case_col, "_t"], kind="mergesort")
        else:
            data = data.sort_values([case_col], kind="mergesort")

        variants: Counter[str] = Counter()
        for _, group in data.groupby(case_col, sort=False):
            acts = [str(x) for x in group[activity_col].tolist() if str(x)]
            if acts:
                variants[" > ".join(acts)] += 1

        if not variants:
            return PluginResult("skipped", "No valid traces for conformance", {"rows": int(len(data))}, [], [], None)

        expected_path = [str(x) for x in (ctx.settings.get("expected_path") or []) if isinstance(x, str) and str(x).strip()]
        expected_variant = " > ".join(expected_path) if expected_path else None
        total_cases = int(sum(variants.values()))
        top_variant, top_count = sorted(variants.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        if expected_variant:
            conforming_cases = int(variants.get(expected_variant, 0))
            baseline_variant = expected_variant
        else:
            conforming_cases = int(top_count)
            baseline_variant = top_variant
        conformance_score = float(conforming_cases / max(1, total_cases))

        findings = [
            {
                "kind": "conformance",
                "baseline_variant": baseline_variant,
                "conformance_score": round(conformance_score, 6),
                "total_cases": total_cases,
            }
        ]
        artifact_payload = {
            "schema_version": "process_mining.v1",
            "plugin_id": "process_mining_conformance_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "nodes": [],
            "edges": [],
            "variants": [
                {"variant": variant, "count": int(count)}
                for variant, count in sorted(variants.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
            ],
            "baseline_variant": baseline_variant,
            "conformance_score": round(conformance_score, 6),
        }
        out_dir = ctx.artifacts_dir("process_mining_conformance_v1")
        out_path = out_dir / "conformance.json"
        write_json(out_path, artifact_payload)
        artifacts = [
            PluginArtifact(
                path=str(out_path.relative_to(ctx.run_dir)),
                type="json",
                description="process conformance report",
            )
        ]
        return PluginResult(
            status="ok",
            summary="Computed deterministic process conformance score",
            metrics={
                "rows": int(len(data)),
                "cases": total_cases,
                "variants": int(len(variants)),
                "conformance_score": round(conformance_score, 6),
            },
            findings=findings,
            artifacts=artifacts,
            references=[],
            debug={"case_column": case_col, "activity_column": activity_col, "time_column": time_col},
        )
