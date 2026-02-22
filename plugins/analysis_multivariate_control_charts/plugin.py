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
    "mv_control": {
        "min_points": 200,
        "baseline_fraction": 0.3,
        "shrinkage": "diag",
        "pca_components": "auto",
        "mewma_lambda": 0.2,
        "threshold_quantile": 0.995,
    }
}


def _group_row_indices(df: pd.DataFrame, group_cols: list[str], max_groups: int) -> list[tuple[str, pd.Index]]:
    slices: list[tuple[str, pd.Index]] = [("ALL", df.index)]
    for col in group_cols:
        if col not in df.columns:
            continue
        counts = df[col].value_counts(dropna=False)
        for value in counts.index[:max_groups]:
            label = f"{col}={value}"
            row_idx = df.index[df[col].eq(value)]
            if row_idx.empty:
                continue
            slices.append((label, row_idx))
    return slices


def _standardize_matrix(matrix: np.ndarray) -> tuple[np.ndarray, list[float], list[float]]:
    centers = []
    scales = []
    standardized = np.zeros_like(matrix, dtype=float)
    for idx in range(matrix.shape[1]):
        col = matrix[:, idx]
        center, scale = robust_center_scale(col)
        if scale <= 0 or not np.isfinite(scale):
            scale = 1.0
        centers.append(float(center))
        scales.append(float(scale))
        standardized[:, idx] = (col - center) / scale
    return standardized, centers, scales


def _pca_residuals(matrix: np.ndarray, components: int) -> np.ndarray:
    if matrix.size == 0:
        return np.array([])
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    k = max(1, min(components, vt.shape[0]))
    recon = (u[:, :k] * s[:k]) @ vt[:k, :]
    residuals = matrix - recon
    return np.sqrt((residuals**2).sum(axis=1))


class Plugin:
    def run(self, ctx) -> PluginResult:
        config = merge_config(ctx.settings)
        config["mv_control"] = {**DEFAULTS["mv_control"], **config.get("mv_control", {})}
        timer = BudgetTimer(config.get("time_budget_ms"))

        df = ctx.dataset_loader()
        if df.empty:
            return PluginResult("skipped", "Empty dataset", {}, [], [], None)

        df, sample_meta = deterministic_sample(df, config.get("max_rows"), seed=config.get("seed", 1337))
        inferred = infer_columns(df, config)
        value_cols = inferred.get("value_columns") or []
        time_col = inferred.get("time_column")
        group_cols = inferred.get("group_by") or []

        if len(value_cols) < 2:
            return PluginResult("skipped", "Not enough numeric columns", {}, [], [], None)

        if time_col and time_col in df.columns:
            df = df.sort_values(time_col)

        settings = config["mv_control"]
        min_points = int(settings.get("min_points", 200))
        max_findings = int(config.get("max_findings", 30))
        max_groups = int(config.get("max_groups", 30))
        threshold_quantile = float(settings.get("threshold_quantile", 0.995))
        mewma_lambda = float(settings.get("mewma_lambda", 0.2))

        findings: list[dict[str, Any]] = []
        artifacts_payload: list[dict[str, Any]] = []

        for label, row_idx in _group_row_indices(df, group_cols, max_groups):
            if timer.exceeded():
                break
            matrix = df.loc[row_idx, value_cols].dropna().to_numpy(dtype=float)
            if matrix.shape[0] < min_points:
                continue
            standardized, centers, scales = _standardize_matrix(matrix)
            t2 = (standardized**2).sum(axis=1)
            t2_threshold = np.quantile(t2, threshold_quantile)

            z_t = np.zeros_like(standardized)
            z_t[0] = standardized[0]
            for idx in range(1, standardized.shape[0]):
                z_t[idx] = mewma_lambda * standardized[idx] + (1.0 - mewma_lambda) * z_t[idx - 1]
            mewma_stat = (z_t**2).sum(axis=1)
            mewma_threshold = np.quantile(mewma_stat, threshold_quantile)

            components_setting = settings.get("pca_components", "auto")
            if components_setting == "auto":
                components = min(5, standardized.shape[1])
            else:
                try:
                    components = int(components_setting)
                except (TypeError, ValueError):
                    components = min(5, standardized.shape[1])
            residuals = _pca_residuals(standardized, components)
            residual_threshold = np.quantile(residuals, threshold_quantile)

            alarm_idx = int(np.argmax(t2))
            if t2[alarm_idx] < t2_threshold and mewma_stat[alarm_idx] < mewma_threshold and residuals[alarm_idx] < residual_threshold:
                continue

            contributions = np.abs(standardized[alarm_idx])
            top_idx = np.argsort(contributions)[::-1][:3]
            top_columns = [value_cols[i] for i in top_idx]

            findings.append(
                {
                    "id": stable_id(f"{label}:{alarm_idx}:{top_columns}"),
                    "severity": "warn",
                    "confidence": min(1.0, float(t2[alarm_idx] / (t2_threshold + 1e-9))),
                    "title": f"Multivariate drift detected ({label})",
                    "what": "Multivariate control chart statistic exceeded threshold.",
                    "why": "T²/MEWMA/PCA residuals indicate coordinated shift.",
                    "evidence": {
                        "metrics": {
                            "t2": float(t2[alarm_idx]),
                            "t2_threshold": float(t2_threshold),
                            "mewma": float(mewma_stat[alarm_idx]),
                            "mewma_threshold": float(mewma_threshold),
                            "residual": float(residuals[alarm_idx]),
                            "residual_threshold": float(residual_threshold),
                            "top_columns": top_columns,
                        }
                    },
                    "where": {"group": label},
                    "recommendation": "Inspect top contributing metrics for coordinated shift.",
                    "measurement_type": "measured",
                    "references": [
                        {
                            "title": "Hotelling T² / MEWMA control charts",
                            "url": "https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc44.htm",
                            "doi": "",
                        }
                    ],
                }
            )
            artifacts_payload.append(
                {
                    "group": label,
                    "t2_threshold": float(t2_threshold),
                    "mewma_threshold": float(mewma_threshold),
                    "residual_threshold": float(residual_threshold),
                    "top_columns": top_columns,
                }
            )
            if len(findings) >= max_findings:
                break

        artifacts_dir = ctx.artifacts_dir("analysis_multivariate_control_charts")
        out_path = artifacts_dir / "mv_control.json"
        write_json(out_path, {"summary": artifacts_payload})
        artifacts = [
            PluginArtifact(
                path=str(out_path.relative_to(ctx.run_dir)),
                type="json",
                description="Multivariate control chart summary",
            )
        ]

        metrics = {
            "rows_seen": int(sample_meta.get("rows_total", len(df))),
            "rows_used": int(sample_meta.get("rows_used", len(df))),
            "cols_used": len(value_cols),
            "references": [
                {
                    "title": "Hotelling T² / MEWMA control charts",
                    "url": "https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc44.htm",
                    "doi": "",
                }
            ],
        }

        summary = f"Detected {len(findings)} multivariate shifts." if findings else "No multivariate shifts detected."

        return PluginResult(
            "ok",
            summary,
            metrics,
            findings,
            artifacts,
            None,
        )
