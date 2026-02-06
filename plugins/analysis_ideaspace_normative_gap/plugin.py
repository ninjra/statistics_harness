from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from statistic_harness.core.column_inference import choose_timestamp_column
from statistic_harness.core.process_filters import normalize_process, process_is_excluded
from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json


def _pick_column(preferred: str | None, columns: list[str], needles: list[str]) -> str | None:
    if preferred and preferred in columns:
        return preferred
    for col in columns:
        name = col.lower()
        if any(token in name for token in needles):
            return col
    return None


class Plugin:
    def run(self, ctx) -> PluginResult:
        df = ctx.dataset_loader()
        if df.empty:
            return PluginResult("skipped", "Empty dataset", {}, [], [], None)

        cols = list(df.columns)
        process_col = _pick_column(ctx.settings.get("process_column"), cols, ["process", "activity", "step", "event"])
        queue_col = _pick_column(ctx.settings.get("queue_column"), cols, ["queue", "queued", "enqueue"])
        eligible_col = _pick_column(ctx.settings.get("eligible_column"), cols, ["eligible", "ready", "available"])
        start_col = _pick_column(ctx.settings.get("start_column"), cols, ["start", "begin"])

        if not process_col or not start_col or (not queue_col and not eligible_col):
            return PluginResult("skipped", "Ideaspace prerequisites unavailable", {}, [{"kind": "ideaspace_gap", "measurement_type": "not_applicable", "reason": "Missing process/start/(queue|eligible) columns."}], [], None)

        process = df[process_col].map(normalize_process)
        start_ts = pd.to_datetime(df[start_col], errors="coerce", utc=False)
        base_series = df[eligible_col] if eligible_col else df[choose_timestamp_column(df, [queue_col]) or queue_col]
        eligible_ts = pd.to_datetime(base_series, errors="coerce", utc=False)
        work = pd.DataFrame({"process": process, "start_ts": start_ts, "eligible_ts": eligible_ts})
        work = work.loc[work["process"] != ""]
        work = work.loc[work["start_ts"].notna() & work["eligible_ts"].notna()].copy()
        if work.empty:
            return PluginResult("skipped", "No valid timestamp rows", {}, [], [], None)

        work["wait_seconds"] = (work["start_ts"] - work["eligible_ts"]).dt.total_seconds()
        work = work.loc[work["wait_seconds"].between(0, 3600 * 72)].copy()
        if work.empty:
            return PluginResult("skipped", "No valid wait durations", {}, [], [], None)

        exclude_tokens = ctx.settings.get("exclude_processes") or []
        work = work.loc[~work["process"].map(lambda p: process_is_excluded(p, exclude_tokens))]
        if work.empty:
            return PluginResult("skipped", "All processes excluded", {}, [], [], None)

        grouped = work.groupby("process", as_index=False).agg(
            runs=("wait_seconds", "size"),
            median_wait_sec=("wait_seconds", "median"),
            p90_wait_sec=("wait_seconds", lambda s: float(np.percentile(s, 90))),
        )
        min_samples = int(ctx.settings.get("min_samples") or 15)
        grouped = grouped.loc[grouped["runs"] >= min_samples].copy()
        if grouped.empty:
            return PluginResult("skipped", "No process met minimum sample gate", {}, [], [], None)

        grouped = grouped.sort_values(["median_wait_sec", "runs", "process"], ascending=[True, False, True]).reset_index(drop=True)
        ideal = grouped.iloc[0]
        grouped["ideal_wait_sec"] = float(ideal["median_wait_sec"])
        grouped["gap_sec"] = (grouped["median_wait_sec"] - grouped["ideal_wait_sec"]).clip(lower=0.0)
        grouped["gap_pct"] = grouped["gap_sec"] / grouped["median_wait_sec"].replace(0, np.nan)
        grouped["gap_pct"] = grouped["gap_pct"].fillna(0.0)

        max_processes = int(ctx.settings.get("max_processes") or 50)
        grouped = grouped.head(max_processes)
        findings: list[dict[str, Any]] = []
        for row in grouped.to_dict(orient="records"):
            findings.append({
                "kind": "ideaspace_gap",
                "measurement_type": "measured",
                "process_id": row["process"],
                "runs": int(row["runs"]),
                "median_wait_sec": float(row["median_wait_sec"]),
                "p90_wait_sec": float(row["p90_wait_sec"]),
                "ideal_wait_sec": float(row["ideal_wait_sec"]),
                "gap_sec": float(row["gap_sec"]),
                "gap_pct": float(row["gap_pct"]),
                "tiers_used": ["process", "window"],
                "references": ["ideaspace:normative-gap"],
            })

        artifacts_dir = ctx.artifacts_dir("analysis_ideaspace_normative_gap")
        out = artifacts_dir / "normative_gap.json"
        write_json(out, {"ideal_process": ideal["process"], "rows": findings})
        artifacts = [PluginArtifact(path=str(out.relative_to(ctx.run_dir)), type="json", description="Ideaspace normative gap table")]
        return PluginResult("ok", f"Built ideaspace gaps for {len(findings)} process(es)", {"entities": len(findings), "ideal_process": ideal["process"]}, findings, artifacts, None)
