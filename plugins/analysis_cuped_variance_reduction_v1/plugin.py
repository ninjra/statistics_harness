from __future__ import annotations
import logging
import traceback
import numpy as np
import pandas as pd
from statistic_harness.core.types import PluginResult, PluginArtifact, PluginError

logger = logging.getLogger(__name__)


class Plugin:
    def run(self, ctx) -> PluginResult:
        try:
            df = ctx.dataset_loader()
            if df.empty:
                return PluginResult("skipped", "Empty dataset", {}, [], [], None)

            settings = ctx.settings or {}
            pre_col = settings.get("pre_metric")
            post_col = settings.get("post_metric")

            # Auto-detect if not specified: look for numeric column pairs
            if not pre_col or not post_col:
                numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                if len(numeric_cols) < 2:
                    return PluginResult(
                        "skipped",
                        "Need at least 2 numeric columns for CUPED (pre_metric, post_metric)",
                        {}, [], [], None,
                    )
                # Convention: first numeric as pre, second as post
                if not pre_col:
                    pre_col = numeric_cols[0]
                if not post_col:
                    post_col = numeric_cols[1] if numeric_cols[1] != pre_col else numeric_cols[0]
                    if post_col == pre_col:
                        return PluginResult(
                            "skipped",
                            "pre_metric and post_metric must be different columns",
                            {}, [], [], None,
                        )

            if pre_col not in df.columns or post_col not in df.columns:
                return PluginResult("skipped", f"Required columns missing: pre={pre_col}, post={post_col}", {}, [], [], None)

            work = df[[pre_col, post_col]].dropna()
            if len(work) < 10:
                return PluginResult("skipped", f"Insufficient rows ({len(work)})", {}, [], [], None)

            pre = work[pre_col].values.astype(float)
            post = work[post_col].values.astype(float)

            # CUPED: theta = Cov(post, pre) / Var(pre)
            var_pre = float(np.var(pre, ddof=1))
            if var_pre < 1e-15:
                return PluginResult("skipped", "Pre-metric has zero variance", {}, [], [], None)

            cov_post_pre = float(np.cov(post, pre, ddof=1)[0, 1])
            theta = cov_post_pre / var_pre

            # Adjusted metric: post - theta * (pre - E[pre])
            mean_pre = float(np.mean(pre))
            adjusted = post - theta * (pre - mean_pre)

            var_post = float(np.var(post, ddof=1))
            var_adjusted = float(np.var(adjusted, ddof=1))
            variance_reduction = 1.0 - (var_adjusted / var_post) if var_post > 1e-15 else 0.0

            findings = [{
                "kind": "distribution",
                "measurement_type": "measured",
                "pre_metric": pre_col,
                "post_metric": post_col,
                "theta": round(theta, 6),
                "variance_pre": round(var_pre, 6),
                "variance_post": round(var_post, 6),
                "variance_adjusted": round(var_adjusted, 6),
                "variance_reduction_ratio": round(variance_reduction, 6),
                "mean_pre": round(mean_pre, 6),
                "mean_post": round(float(np.mean(post)), 6),
                "mean_adjusted": round(float(np.mean(adjusted)), 6),
                "method": "CUPED",
            }]

            return PluginResult(
                status="ok",
                summary=(
                    f"CUPED variance reduction: {variance_reduction:.1%} reduction "
                    f"(theta={theta:.4f}, var_post={var_post:.4f} -> var_adj={var_adjusted:.4f})"
                ),
                metrics={
                    "n_observations": len(work),
                    "pre_metric": pre_col,
                    "post_metric": post_col,
                    "theta": round(theta, 6),
                    "variance_reduction_ratio": round(variance_reduction, 6),
                    "method": "CUPED (Controlled-experiment Using Pre-Experiment Data)",
                },
                findings=findings,
                artifacts=[],
                error=None,
            )
        except Exception as e:
            logger.error(f"CUPED variance reduction failed: {e}", exc_info=True)
            return PluginResult(
                status="error",
                summary=f"CUPED variance reduction failed: {e}",
                metrics={},
                findings=[],
                artifacts=[],
                error=PluginError(type=type(e).__name__, message=str(e), traceback=traceback.format_exc()),
            )
