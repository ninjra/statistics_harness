from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from statistic_harness.core.column_inference import choose_timestamp_column
from statistic_harness.core.process_filters import normalize_process, process_is_excluded
from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json


@dataclass(frozen=True)
class _Columns:
    process: str
    start: str
    eligible_or_queue: str


def _pick_column(preferred: str | None, columns: list[str], needles: list[str]) -> str | None:
    if preferred and preferred in columns:
        return preferred
    for col in columns:
        lowered = col.lower()
        if any(token in lowered for token in needles):
            return col
    return None


def _resolve_columns(df: pd.DataFrame, settings: dict[str, Any]) -> _Columns | None:
    columns = list(df.columns)
    process_col = _pick_column(
        settings.get("process_column"), columns, ["process", "activity", "step", "event"]
    )
    start_col = _pick_column(settings.get("start_column"), columns, ["start", "begin"])
    eligible_col = _pick_column(
        settings.get("eligible_column"), columns, ["eligible", "ready", "available"]
    )
    queue_col = _pick_column(settings.get("queue_column"), columns, ["queue", "queued", "enqueue"])
    if not process_col or not start_col:
        return None
    basis = eligible_col or choose_timestamp_column(df, [queue_col] if queue_col else []) or queue_col
    if not basis:
        return None
    return _Columns(process=process_col, start=start_col, eligible_or_queue=basis)


def _deterministic_sample(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame
    return frame.sample(n=max_rows, random_state=seed).sort_index()


class Plugin:
    def run(self, ctx) -> PluginResult:
        df = ctx.dataset_loader()
        if df.empty:
            return PluginResult("skipped", "Empty dataset", {}, [], [], None)

        cols = _resolve_columns(df, ctx.settings)
        if not cols:
            return PluginResult(
                "skipped",
                "Ideaspace prerequisites unavailable",
                {},
                [
                    {
                        "kind": "ideaspace_gap",
                        "measurement_type": "not_applicable",
                        "reason": "Missing process/start/(queue|eligible) columns.",
                    }
                ],
                [],
                None,
            )

        max_rows = int(ctx.settings.get("max_rows") or 150_000)
        sampled_df = _deterministic_sample(df, max_rows=max_rows, seed=int(ctx.run_seed))
        process = sampled_df[cols.process].map(normalize_process)
        start_ts = pd.to_datetime(sampled_df[cols.start], errors="coerce", utc=False)
        eligible_ts = pd.to_datetime(sampled_df[cols.eligible_or_queue], errors="coerce", utc=False)
        work = pd.DataFrame(
            {
                "process": process,
                "start_ts": start_ts,
                "eligible_ts": eligible_ts,
            }
        )
        work = work.loc[work["process"] != ""]
        work = work.loc[work["start_ts"].notna() & work["eligible_ts"].notna()].copy()
        if work.empty:
            return PluginResult("skipped", "No valid timestamp rows", {}, [], [], None)

        work["wait_seconds"] = (work["start_ts"] - work["eligible_ts"]).dt.total_seconds()
        work = work.loc[work["wait_seconds"].between(0, 3600 * 72)].copy()
        if work.empty:
            return PluginResult("skipped", "No valid wait durations", {}, [], [], None)

        exclude_tokens = ctx.settings.get("exclude_processes") or []
        work = work.loc[~work["process"].map(lambda pid: process_is_excluded(pid, exclude_tokens))].copy()
        if work.empty:
            return PluginResult("skipped", "All processes excluded", {}, [], [], None)

        # Deterministic monthly window slices for stability checks.
        work["window_id"] = work["eligible_ts"].dt.to_period("M").astype(str)
        grouped = work.groupby("process", as_index=False).agg(
            runs=("wait_seconds", "size"),
            median_wait_sec=("wait_seconds", "median"),
            p90_wait_sec=("wait_seconds", lambda s: float(np.percentile(s, 90))),
            windows=("window_id", "nunique"),
        )

        min_samples = int(ctx.settings.get("min_samples") or 15)
        grouped = grouped.loc[grouped["runs"] >= min_samples].copy()
        if grouped.empty:
            return PluginResult("skipped", "No process met minimum sample gate", {}, [], [], None)

        frontier = grouped.sort_values(
            ["median_wait_sec", "p90_wait_sec", "runs", "process"],
            ascending=[True, True, False, True],
        ).reset_index(drop=True)
        ideal = frontier.iloc[0]
        grouped["ideal_wait_sec"] = float(ideal["median_wait_sec"])
        grouped["gap_sec"] = (grouped["median_wait_sec"] - grouped["ideal_wait_sec"]).clip(lower=0.0)
        grouped["gap_pct"] = (
            grouped["gap_sec"] / grouped["median_wait_sec"].replace(0.0, np.nan)
        ).fillna(0.0)

        max_processes = int(ctx.settings.get("max_processes") or 50)
        grouped = grouped.sort_values(
            ["gap_sec", "runs", "process"], ascending=[False, False, True]
        ).head(max_processes)

        findings: list[dict[str, Any]] = []
        for row in grouped.to_dict(orient="records"):
            findings.append(
                {
                    "kind": "ideaspace_gap",
                    "measurement_type": "measured",
                    "process_id": row["process"],
                    "runs": int(row["runs"]),
                    "windows": int(row["windows"]),
                    "median_wait_sec": float(row["median_wait_sec"]),
                    "p90_wait_sec": float(row["p90_wait_sec"]),
                    "ideal_wait_sec": float(row["ideal_wait_sec"]),
                    "gap_sec": float(row["gap_sec"]),
                    "gap_pct": float(row["gap_pct"]),
                    "tiers_used": ["process", "window_month"],
                    "columns_chosen": {
                        "process": cols.process,
                        "start": cols.start,
                        "eligible_or_queue": cols.eligible_or_queue,
                    },
                    "references": ["docs/ideaspace.md#normative-gap", "ideaspace:normative-gap"],
                }
            )

        artifacts_dir = ctx.artifacts_dir("analysis_ideaspace_normative_gap")
        out_path = artifacts_dir / "normative_gap.json"
        write_json(
            out_path,
            {
                "ideal_process": str(ideal["process"]),
                "sampled_rows": int(len(sampled_df)),
                "input_rows": int(len(df)),
                "rows": findings,
            },
        )
        artifacts = [
            PluginArtifact(
                path=str(out_path.relative_to(ctx.run_dir)),
                type="json",
                description="Ideaspace normative gap table",
            )
        ]
        return PluginResult(
            "ok",
            f"Built ideaspace gaps for {len(findings)} process(es)",
            {
                "entities": len(findings),
                "ideal_process": str(ideal["process"]),
                "sampled": len(sampled_df) != len(df),
            },
            findings,
            artifacts,
            None,
        )
