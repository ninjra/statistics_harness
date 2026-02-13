from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from statistic_harness.core.stat_plugins import (
    BudgetTimer,
    merge_config,
    infer_columns,
    deterministic_sample,
    robust_center_scale,
    stable_id,
)
from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json


DEFAULTS = {
    "pelt": {
        "max_points": 20000,
        "penalty_beta": 2.0,
        "min_segment_size": 50,
        "max_changepoints": 20,
    }
}


def _standardize_matrix(matrix: np.ndarray) -> np.ndarray:
    standardized = np.zeros_like(matrix, dtype=float)
    for idx in range(matrix.shape[1]):
        center, scale = robust_center_scale(matrix[:, idx])
        if scale <= 0 or not np.isfinite(scale):
            scale = 1.0
        standardized[:, idx] = (matrix[:, idx] - center) / scale
    return standardized


def _pelt(series: np.ndarray, penalty: float, min_size: int) -> list[int]:
    n = series.shape[0]
    if n < 2 * min_size:
        return []
    prefix = np.zeros(n + 1)
    prefix_sq = np.zeros(n + 1)
    prefix[1:] = np.cumsum(series)
    prefix_sq[1:] = np.cumsum(series ** 2)

    def segment_cost(start: int, end: int) -> float:
        seg_len = end - start
        if seg_len <= 0:
            return 0.0
        seg_sum = prefix[end] - prefix[start]
        seg_sq = prefix_sq[end] - prefix_sq[start]
        mean = seg_sum / seg_len
        return seg_sq - 2 * mean * seg_sum + seg_len * mean * mean

    F = np.full(n + 1, np.inf)
    F[0] = -penalty
    cps: dict[int, list[int]] = {0: []}
    R = [0]
    for t in range(min_size, n + 1):
        if not R:
            R = [t - min_size]
        candidates = [s for s in R if t - s >= min_size]
        costs = []
        for s in candidates:
            costs.append(F[s] + segment_cost(s, t) + penalty)
        if not costs:
            continue
        best_idx = int(np.argmin(costs))
        best_s = candidates[best_idx]
        F[t] = costs[best_idx]
        cps[t] = cps.get(best_s, []) + [best_s]
        R = [
            s
            for s in candidates
            if F[s] + segment_cost(s, t) + penalty <= F[t] + penalty
        ]
        R.append(t - min_size + 1)
    changepoints = cps.get(n, [])
    changepoints = [cp for cp in changepoints if cp not in (0, n)]
    return sorted(set(changepoints))


class Plugin:
    def run(self, ctx) -> PluginResult:
        config = merge_config(ctx.settings)
        config["pelt"] = {**DEFAULTS["pelt"], **config.get("pelt", {})}
        timer = BudgetTimer(config.get("time_budget_ms"))

        df = ctx.dataset_loader()
        if df.empty:
            return PluginResult("skipped", "Empty dataset", {}, [], [], None)

        df, sample_meta = deterministic_sample(df, config.get("max_rows"), seed=config.get("seed", 1337))
        inferred = infer_columns(df, config)
        value_cols = inferred.get("value_columns") or []
        time_col = inferred.get("time_column")

        if len(value_cols) < 2:
            return PluginResult("skipped", "Not enough numeric columns", {}, [], [], None)

        if time_col and time_col in df.columns:
            df = df.sort_values(time_col)

        max_points = int(config["pelt"].get("max_points", 20000))
        if df.shape[0] > max_points:
            df = df.iloc[np.linspace(0, df.shape[0] - 1, max_points, dtype=int)]

        matrix = df[value_cols].dropna().to_numpy(dtype=float)
        if matrix.shape[0] < int(config["pelt"].get("min_segment_size", 50)) * 2:
            return PluginResult("skipped", "Not enough rows for changepoint detection", {}, [], [], None)

        standardized = _standardize_matrix(matrix)
        u, s, vt = np.linalg.svd(standardized, full_matrices=False)
        first_pc = u[:, 0] * s[0]
        n = first_pc.shape[0]
        beta = float(config["pelt"].get("penalty_beta", 2.0))
        penalty = beta * np.log(max(n, 2)) * standardized.shape[1]
        min_seg = int(config["pelt"].get("min_segment_size", 50))
        changepoints = _pelt(first_pc, penalty, min_seg)
        changepoints = changepoints[: int(config["pelt"].get("max_changepoints", 20))]

        findings: list[dict[str, Any]] = []
        for cp in changepoints:
            if timer.exceeded():
                break
            pre = standardized[max(0, cp - min_seg) : cp]
            post = standardized[cp : min(cp + min_seg, standardized.shape[0])]
            if pre.size == 0 or post.size == 0:
                continue
            diff = np.abs(pre.mean(axis=0) - post.mean(axis=0))
            top_idx = np.argsort(diff)[::-1][:3]
            top_cols = [value_cols[i] for i in top_idx]
            finding = {
                "id": stable_id(f"cp:{cp}:{top_cols}"),
                "severity": "warn",
                "confidence": 0.6,
                "title": f"Changepoint detected at index {cp}",
                "what": "Segment boundary detected in multivariate stream.",
                "why": "PELT segmentation identified regime shift.",
                "evidence": {
                    "metrics": {
                        "changepoint_index": int(cp),
                        "top_columns": top_cols,
                        "penalty": float(penalty),
                    }
                },
                "where": {"index": int(cp)},
                "recommendation": "Inspect upstream changes around this time window.",
                "measurement_type": "measured",
                "references": [
                    {
                        "title": "Killick et al. (2012) PELT",
                        "url": "https://doi.org/10.1080/01621459.2012.737745",
                        "doi": "10.1080/01621459.2012.737745",
                    }
                ],
            }
            findings.append(finding)

        artifacts_dir = ctx.artifacts_dir("analysis_multivariate_changepoint_pelt")
        out_path = artifacts_dir / "changepoints.json"
        write_json(out_path, {"changepoints": changepoints, "penalty": penalty})
        artifacts = [
            PluginArtifact(
                path=str(out_path.relative_to(ctx.run_dir)),
                type="json",
                description="Changepoint summary",
            )
        ]

        metrics = {
            "rows_seen": int(sample_meta.get("rows_total", len(df))),
            "rows_used": int(sample_meta.get("rows_used", len(df))),
            "cols_used": len(value_cols),
            "references": [
                {
                    "title": "Killick et al. (2012) PELT",
                    "url": "https://doi.org/10.1080/01621459.2012.737745",
                    "doi": "10.1080/01621459.2012.737745",
                }
            ],
        }

        summary = f"Detected {len(findings)} changepoints." if findings else "No changepoints detected."

        return PluginResult(
            "ok",
            summary,
            metrics,
            findings,
            artifacts,
            None,
        )
