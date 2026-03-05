from __future__ import annotations
import logging
import traceback
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from statistic_harness.core.types import PluginResult, PluginArtifact, PluginError

logger = logging.getLogger(__name__)


class Plugin:
    def run(self, ctx) -> PluginResult:
        try:
            df = ctx.dataset_loader()
            if df.empty:
                return PluginResult("skipped", "Empty dataset", {}, [], [], None)

            settings = ctx.settings or {}
            group_col = settings.get("group_column")
            value_col = settings.get("value_column")
            bandwidth = float(settings.get("bandwidth", 0.5))

            # Auto-detect: group = first categorical, value = first numeric
            if not group_col:
                for c in df.columns:
                    if df[c].dtype == object or str(df[c].dtype) == "category":
                        if 2 <= df[c].dropna().nunique() <= 50:
                            group_col = c
                            break
            if not value_col:
                for c in df.columns:
                    if c == group_col:
                        continue
                    if pd.api.types.is_numeric_dtype(df[c]):
                        value_col = c
                        break

            if not group_col or not value_col:
                return PluginResult(
                    "skipped",
                    "Could not identify a group column and a numeric value column",
                    {}, [], [], None,
                )

            work = df[[group_col, value_col]].dropna()
            groups = work[group_col].unique()
            if len(groups) < 2:
                return PluginResult("skipped", f"Need >=2 groups, found {len(groups)}", {}, [], [], None)
            if len(work) < 10:
                return PluginResult("skipped", f"Insufficient rows ({len(work)})", {}, [], [], None)

            # Fit KDE per group
            kdes = {}
            group_data = {}
            for g in groups:
                vals = work.loc[work[group_col] == g, value_col].values.reshape(-1, 1)
                if len(vals) < 3:
                    continue
                kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
                kde.fit(vals)
                kdes[g] = kde
                group_data[g] = vals

            if len(kdes) < 2:
                return PluginResult("skipped", "Need at least 2 groups with >=3 observations each", {}, [], [], None)

            # Compute KL divergence between each pair of group densities
            # Use shared evaluation grid
            all_vals = work[value_col].values
            grid_min = float(all_vals.min()) - 3 * bandwidth
            grid_max = float(all_vals.max()) + 3 * bandwidth
            grid = np.linspace(grid_min, grid_max, 500).reshape(-1, 1)

            log_densities = {}
            for g, kde in kdes.items():
                log_densities[g] = kde.score_samples(grid)

            group_list = sorted(kdes.keys(), key=str)
            kl_pairs = []
            for i, g1 in enumerate(group_list):
                for g2 in group_list[i + 1:]:
                    p = np.exp(log_densities[g1])
                    q = np.exp(log_densities[g2])
                    # Add small epsilon to avoid log(0)
                    eps = 1e-10
                    p = p + eps
                    q = q + eps
                    # Normalize
                    p = p / p.sum()
                    q = q / q.sum()
                    kl_pq = float(np.sum(p * np.log(p / q)))
                    kl_qp = float(np.sum(q * np.log(q / p)))
                    kl_pairs.append({
                        "group_a": str(g1),
                        "group_b": str(g2),
                        "kl_a_to_b": round(kl_pq, 6),
                        "kl_b_to_a": round(kl_qp, 6),
                        "symmetric_kl": round((kl_pq + kl_qp) / 2, 6),
                    })

            # Sort by symmetric KL descending
            kl_pairs.sort(key=lambda x: x["symmetric_kl"], reverse=True)
            max_kl = kl_pairs[0]["symmetric_kl"] if kl_pairs else 0.0

            findings = [{
                "kind": "distribution",
                "measurement_type": "measured",
                "group_column": group_col,
                "value_column": value_col,
                "n_groups": len(kdes),
                "kl_divergences": kl_pairs,
                "max_symmetric_kl": round(max_kl, 6),
                "bandwidth": bandwidth,
                "method": "Conditional Density Estimation (KDE + KL divergence)",
            }]

            return PluginResult(
                status="ok",
                summary=(
                    f"Conditional density estimation: {len(kdes)} groups, "
                    f"max symmetric KL={max_kl:.4f}"
                ),
                metrics={
                    "n_observations": len(work),
                    "n_groups": len(kdes),
                    "max_symmetric_kl": round(max_kl, 6),
                    "n_pairs": len(kl_pairs),
                },
                findings=findings,
                artifacts=[],
                error=None,
            )
        except Exception as e:
            logger.error(f"Conditional density estimation failed: {e}", exc_info=True)
            return PluginResult(
                status="error",
                summary=f"Conditional density estimation failed: {e}",
                metrics={},
                findings=[],
                artifacts=[],
                error=PluginError(type=type(e).__name__, message=str(e), traceback=traceback.format_exc()),
            )
