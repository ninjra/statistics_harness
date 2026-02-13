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
    "robust_pca": {
        "components": "auto",
        "threshold_sigma": 3.0,
    }
}


def _standardize(matrix: np.ndarray) -> np.ndarray:
    out = np.zeros_like(matrix, dtype=float)
    for idx in range(matrix.shape[1]):
        center, scale = robust_center_scale(matrix[:, idx])
        if scale <= 0 or not np.isfinite(scale):
            scale = 1.0
        out[:, idx] = (matrix[:, idx] - center) / scale
    return out


class Plugin:
    def run(self, ctx) -> PluginResult:
        config = merge_config(ctx.settings)
        config["robust_pca"] = {**DEFAULTS["robust_pca"], **config.get("robust_pca", {})}
        timer = BudgetTimer(config.get("time_budget_ms"))

        df = ctx.dataset_loader()
        if df.empty:
            return PluginResult("skipped", "Empty dataset", {}, [], [], None)

        df, sample_meta = deterministic_sample(df, config.get("max_rows"), seed=config.get("seed", 1337))
        inferred = infer_columns(df, config)
        value_cols = inferred.get("value_columns") or []

        if len(value_cols) < 2:
            return PluginResult("skipped", "Not enough numeric columns", {}, [], [], None)

        matrix = df[value_cols].dropna().to_numpy(dtype=float)
        if matrix.size == 0:
            return PluginResult("skipped", "No numeric data after filtering", {}, [], [], None)

        standardized = _standardize(matrix)
        components_setting = config["robust_pca"].get("components", "auto")
        if components_setting == "auto":
            components = max(1, min(3, standardized.shape[1] - 1))
        else:
            try:
                components = int(components_setting)
            except (TypeError, ValueError):
                components = max(1, min(3, standardized.shape[1] - 1))
        u, s, vt = np.linalg.svd(standardized, full_matrices=False)
        k = max(1, min(components, vt.shape[0] - 1))
        recon = (u[:, :k] * s[:k]) @ vt[:k, :]
        residuals = standardized - recon
        residual_norm = np.sqrt((residuals**2).sum(axis=1))
        scores = standardized @ vt[:k, :].T
        score_norm = np.sqrt((scores**2).sum(axis=1))

        threshold_sigma = float(config["robust_pca"].get("threshold_sigma", 3.0))
        residual_center, residual_scale = robust_center_scale(residual_norm)
        score_center, score_scale = robust_center_scale(score_norm)
        residual_threshold = residual_center + threshold_sigma * residual_scale
        score_threshold = score_center + threshold_sigma * score_scale

        outlier_mask = (residual_norm > residual_threshold) | (score_norm > score_threshold)
        outlier_idx = np.where(outlier_mask)[0]
        outlier_idx = outlier_idx[: int(config.get("max_findings", 30))]

        findings: list[dict[str, Any]] = []
        for idx in outlier_idx:
            if timer.exceeded():
                break
            contrib = np.abs(residuals[idx])
            top_idx = np.argsort(contrib)[::-1][:3]
            top_cols = [value_cols[i] for i in top_idx]
            residual_score = float(residual_norm[idx] / (residual_threshold + 1e-9))
            score_score = float(score_norm[idx] / (score_threshold + 1e-9))
            findings.append(
                {
                    "id": stable_id(f"rpca:{idx}:{top_cols}"),
                    "severity": "warn",
                    "confidence": min(1.0, max(residual_score, score_score)),
                    "title": "Sparse outlier detected",
                    "what": "Row has large residual after low-rank reconstruction.",
                    "why": "Robust PCA residual norm exceeds threshold.",
                    "evidence": {
                        "metrics": {
                            "residual_norm": float(residual_norm[idx]),
                            "residual_threshold": float(residual_threshold),
                            "score_norm": float(score_norm[idx]),
                            "score_threshold": float(score_threshold),
                            "top_columns": top_cols,
                        }
                    },
                    "where": {"row_index": int(idx)},
                    "recommendation": "Inspect row for unusual metric combination.",
                    "measurement_type": "measured",
                    "references": [
                        {
                            "title": "Principal Component Pursuit (Candes et al., 2011)",
                            "url": "https://doi.org/10.1145/1970392.1970395",
                            "doi": "10.1145/1970392.1970395",
                        }
                    ],
                }
            )

        artifacts_dir = ctx.artifacts_dir("analysis_robust_pca_sparse_outliers")
        out_path = artifacts_dir / "rpca_outliers.json"
        write_json(
            out_path,
            {
                "residual_threshold": float(residual_threshold),
                "score_threshold": float(score_threshold),
                "outliers": outlier_idx.tolist(),
            },
        )
        artifacts = [
            PluginArtifact(
                path=str(out_path.relative_to(ctx.run_dir)),
                type="json",
                description="Robust PCA outliers",
            )
        ]

        metrics = {
            "rows_seen": int(sample_meta.get("rows_total", len(df))),
            "rows_used": int(sample_meta.get("rows_used", len(df))),
            "cols_used": len(value_cols),
            "references": [
                {
                    "title": "Principal Component Pursuit (Candes et al., 2011)",
                    "url": "https://doi.org/10.1145/1970392.1970395",
                    "doi": "10.1145/1970392.1970395",
                }
            ],
        }

        summary = f"Detected {len(findings)} sparse outliers." if findings else "No sparse outliers detected."

        return PluginResult(
            "ok",
            summary,
            metrics,
            findings,
            artifacts,
            None,
        )
