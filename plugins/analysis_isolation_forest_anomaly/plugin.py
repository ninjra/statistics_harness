from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from statistic_harness.core.stat_plugins import (
    BudgetTimer,
    merge_config,
    infer_columns,
    deterministic_sample,
    robust_zscores,
    stable_id,
)
from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json


DEFAULTS = {
    "iforest": {
        "enabled": True,
        "sample_size": 10000,
        "contamination": "auto",
        "n_estimators": 200,
        "score_top_k": 50,
        "fallback": "robust_z",
    }
}


def _fit_iforest(X: np.ndarray, seed: int, cfg: dict[str, Any]) -> np.ndarray | None:
    try:
        from sklearn.ensemble import IsolationForest  # type: ignore

        model = IsolationForest(
            n_estimators=int(cfg.get("n_estimators", 200)),
            contamination=cfg.get("contamination", "auto"),
            random_state=seed,
        )
        model.fit(X)
        scores = -model.decision_function(X)
        return scores
    except Exception:
        return None


class Plugin:
    def run(self, ctx) -> PluginResult:
        config = merge_config(ctx.settings)
        config["iforest"] = {**DEFAULTS["iforest"], **config.get("iforest", {})}
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

        cfg = config["iforest"]
        sample_size = int(cfg.get("sample_size", 10000))
        if matrix.shape[0] > sample_size:
            idx = np.linspace(0, matrix.shape[0] - 1, sample_size, dtype=int)
            sample = matrix[idx]
        else:
            sample = matrix

        scores = None
        if cfg.get("enabled", True):
            scores = _fit_iforest(sample, int(config.get("seed", 1337)), cfg)

        if scores is None:
            z = np.abs(robust_zscores(sample))
            scores = z.max(axis=1)

        top_k = int(cfg.get("score_top_k", 50))
        top_idx = np.argsort(scores)[::-1][:top_k]

        findings: list[dict[str, Any]] = []
        for rank, idx in enumerate(top_idx[: int(config.get("max_findings", 30))], start=1):
            if timer.exceeded():
                break
            row = sample[idx]
            z = np.abs(robust_zscores(row))
            contrib_idx = np.argsort(z)[::-1][:3]
            top_cols = [value_cols[i] for i in contrib_idx]
            findings.append(
                {
                    "id": stable_id(f"iforest:{idx}:{top_cols}"),
                    "severity": "warn",
                    "confidence": min(1.0, float(scores[idx]) / (float(scores[top_idx[0]]) + 1e-9)),
                    "title": f"Anomalous row #{rank}",
                    "what": "Row exhibits unusual multivariate pattern.",
                    "why": "IsolationForest/robust z-score flagged extreme deviation.",
                    "evidence": {
                        "metrics": {
                            "score": float(scores[idx]),
                            "top_columns": top_cols,
                        }
                    },
                    "where": {"row_index": int(idx)},
                    "recommendation": "Inspect raw row for unusual parameter combinations.",
                    "measurement_type": "measured",
                    "references": [
                        {
                            "title": "Isolation Forest (Liu et al., 2008)",
                            "url": "https://doi.org/10.1109/ICDM.2008.17",
                            "doi": "10.1109/ICDM.2008.17",
                        }
                    ],
                }
            )

        artifacts_dir = ctx.artifacts_dir("analysis_isolation_forest_anomaly")
        out_path = artifacts_dir / "anomalies.json"
        write_json(out_path, {"scores": scores.tolist(), "top_indices": top_idx.tolist()})
        artifacts = [
            PluginArtifact(
                path=str(out_path.relative_to(ctx.run_dir)),
                type="json",
                description="Isolation forest anomaly scores",
            )
        ]

        metrics = {
            "rows_seen": int(sample_meta.get("rows_total", len(df))),
            "rows_used": int(sample_meta.get("rows_used", len(df))),
            "cols_used": len(value_cols),
            "references": [
                {
                    "title": "Isolation Forest (Liu et al., 2008)",
                    "url": "https://doi.org/10.1109/ICDM.2008.17",
                    "doi": "10.1109/ICDM.2008.17",
                }
            ],
        }

        summary = f"Detected {len(findings)} anomalies." if findings else "No anomalies detected."

        return PluginResult(
            "ok",
            summary,
            metrics,
            findings,
            artifacts,
            None,
        )
