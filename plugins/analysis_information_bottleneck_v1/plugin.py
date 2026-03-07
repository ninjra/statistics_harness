from __future__ import annotations
import logging
import traceback
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
from statistic_harness.core.types import PluginResult, PluginArtifact, PluginError

logger = logging.getLogger(__name__)


class Plugin:
    def run(self, ctx) -> PluginResult:
        try:
            df = ctx.dataset_loader()
            if df.empty:
                return PluginResult("na", "Empty dataset", {}, [], [], None)

            settings = ctx.settings or {}
            target_col = settings.get("target_column")
            max_k = int(settings.get("max_k", 10))

            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if len(numeric_cols) < 2:
                return PluginResult(
                    "na",
                    f"Need at least 2 numeric columns, found {len(numeric_cols)}",
                    {}, [], [], None,
                )

            if not target_col:
                target_col = numeric_cols[-1]

            feature_cols = [c for c in numeric_cols if c != target_col]
            if not feature_cols:
                return PluginResult("na", "No feature columns after excluding target", {}, [], [], None)

            work = df[feature_cols + [target_col]].dropna()
            if len(work) < 10:
                return PluginResult("na", f"Insufficient rows ({len(work)})", {}, [], [], None)

            X = work[feature_cols].values
            y = work[target_col].values

            # Discretize target for MI computation
            n_bins = min(10, len(np.unique(y)))
            if n_bins < 2:
                return PluginResult("na", "Target column has fewer than 2 unique values", {}, [], [], None)

            kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
            y_discrete = kbd.fit_transform(y.reshape(-1, 1)).ravel().astype(int)

            # Sweep k from 2 to max_k, compute MI between cluster labels and target
            max_k = min(max_k, len(work) - 1)
            if max_k < 2:
                max_k = 2

            curve = []
            for k in range(2, max_k + 1):
                km = KMeans(n_clusters=k, random_state=ctx.run_seed, n_init=3, max_iter=100)
                labels = km.fit_predict(X)
                mi = mutual_info_score(y_discrete, labels)

                # Distortion = mean squared distance to cluster center
                distortion = float(km.inertia_ / len(X))
                curve.append({
                    "k": k,
                    "mutual_information": round(float(mi), 6),
                    "distortion": round(distortion, 6),
                })

            # Find the k with maximum MI (best compression)
            best_point = max(curve, key=lambda p: p["mutual_information"])
            best_k = best_point["k"]
            best_mi = best_point["mutual_information"]

            findings = [{
                "kind": "cluster",
                "measurement_type": "measured",
                "target_column": target_col,
                "feature_columns": feature_cols,
                "best_k": best_k,
                "best_mutual_information": round(best_mi, 6),
                "information_distortion_curve": curve,
                "method": "Information Bottleneck (KMeans + MI)",
            }]

            return PluginResult(
                status="ok",
                summary=(
                    f"Information bottleneck: best k={best_k} "
                    f"(MI={best_mi:.4f}), {len(feature_cols)} features"
                ),
                metrics={
                    "n_observations": len(work),
                    "n_features": len(feature_cols),
                    "best_k": best_k,
                    "best_mutual_information": round(best_mi, 6),
                },
                findings=findings,
                artifacts=[],
                error=None,
            )
        except Exception as e:
            logger.error(f"Information bottleneck failed: {e}", exc_info=True)
            return PluginResult(
                status="error",
                summary=f"Information bottleneck failed: {e}",
                metrics={},
                findings=[],
                artifacts=[],
                error=PluginError(type=type(e).__name__, message=str(e), traceback=traceback.format_exc()),
            )
