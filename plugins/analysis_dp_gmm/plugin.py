from __future__ import annotations

import numpy as np
import pandas as pd

from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json

try:
    from sklearn.mixture import BayesianGaussianMixture

    HAS_SKLEARN = True
except Exception:  # pragma: no cover
    BayesianGaussianMixture = None
    HAS_SKLEARN = False


class Plugin:
    def run(self, ctx) -> PluginResult:
        df = ctx.dataset_loader()
        numeric = df.select_dtypes(include="number")
        if numeric.empty:
            return PluginResult("skipped", "No numeric columns", {}, [], [], None)

        seed = int(getattr(ctx, "run_seed", 0) or 0)
        values = numeric.to_numpy(dtype=float)
        values = values[~np.isnan(values).any(axis=1)]
        if values.shape[0] < 5:
            return PluginResult("skipped", "Too few rows after NaN removal", {}, [], [], None)

        if not HAS_SKLEARN:
            return PluginResult("degraded", "sklearn not available for DP-GMM", {}, [], [], None)

        # True Dirichlet Process Gaussian Mixture Model via sklearn
        max_components = min(10, max(2, values.shape[0] // 10))
        dpgmm = BayesianGaussianMixture(
            n_components=max_components,
            covariance_type="full",
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=1.0 / max_components,
            max_iter=200,
            random_state=seed,
            n_init=1,
        )
        dpgmm.fit(values)
        labels = dpgmm.predict(values)

        # Identify active components (weight > 0.01)
        active_mask = dpgmm.weights_ > 0.01
        active_ids = np.where(active_mask)[0]
        n_active = int(active_ids.size)

        assignments = pd.DataFrame({"cluster_id": labels.tolist()})

        clusters = []
        findings = []
        for cid in sorted(set(labels)):
            cluster_points = values[labels == cid]
            cluster_info = {
                "cluster_id": int(cid),
                "size": int(cluster_points.shape[0]),
                "mean": cluster_points.mean(axis=0).tolist(),
                "weight": float(dpgmm.weights_[cid]),
            }
            clusters.append(cluster_info)
            findings.append({
                "id": f"analysis_dp_gmm:cluster:{cid}",
                "severity": "info",
                "confidence": float(dpgmm.weights_[cid]),
                "title": f"DP-GMM cluster {cid} ({cluster_points.shape[0]} points)",
                "what": f"Dirichlet Process GMM identified cluster {cid} with {cluster_points.shape[0]} points and weight {dpgmm.weights_[cid]:.3f}",
                "why": "Cluster represents a distinct data subpopulation identified by Bayesian nonparametric mixture modeling.",
                "kind": "cluster",
                "cluster_id": int(cid),
                "size": int(cluster_points.shape[0]),
                "weight": float(dpgmm.weights_[cid]),
            })

        artifacts_dir = ctx.artifacts_dir("analysis_dp_gmm")
        assignments_path = artifacts_dir / "assignments.csv"
        assignments.to_csv(assignments_path, index=False)
        summary_path = artifacts_dir / "summary.json"
        write_json(summary_path, {"clusters": clusters, "n_active_components": n_active})
        artifacts = [
            PluginArtifact(
                path=str(assignments_path.relative_to(ctx.run_dir)),
                type="csv",
                description="Cluster assignments",
            ),
            PluginArtifact(
                path=str(summary_path.relative_to(ctx.run_dir)),
                type="json",
                description="Cluster summary",
            ),
        ]
        return PluginResult(
            "ok",
            f"DP-GMM identified {n_active} active components from {max_components} candidates",
            {"clusters": n_active, "total_components": max_components, "bic_lower_bound": float(dpgmm.lower_bound_)},
            findings,
            artifacts,
            None,
        )
