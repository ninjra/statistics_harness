from __future__ import annotations

import logging
import traceback

import numpy as np
import pandas as pd

from statistic_harness.core.types import PluginArtifact, PluginError, PluginResult

logger = logging.getLogger(__name__)


def _find_compositional_cols(df: pd.DataFrame, tol: float = 0.05) -> list[str]:
    """Find groups of columns whose rows sum approximately to 1."""
    numeric_cols = [
        c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
    ]
    if len(numeric_cols) < 2:
        return []

    # Check if all numeric columns together are compositional
    sub = df[numeric_cols].dropna()
    if sub.empty:
        return []

    row_sums = sub.sum(axis=1)
    if ((row_sums - 1.0).abs() < tol).mean() > 0.8:
        return numeric_cols

    return []


class Plugin:
    def run(self, ctx) -> PluginResult:
        try:
            df = ctx.dataset_loader()
            if df.empty:
                return PluginResult("na", "Empty dataset", {}, [], [], None)

            comp_cols = ctx.settings.get("compositional_columns")
            if comp_cols:
                comp_cols = [c for c in comp_cols if c in df.columns]
            else:
                tol = float(ctx.settings.get("sum_tolerance", 0.05))
                comp_cols = _find_compositional_cols(df, tol)

            if not comp_cols or len(comp_cols) < 2:
                return PluginResult(
                    "na",
                    "No compositional columns detected",
                    {},
                    [],
                    [],
                    None,
                )

            work = df[comp_cols].dropna()
            if len(work) < 5:
                return PluginResult(
                    "na",
                    f"Insufficient rows ({len(work)})",
                    {},
                    [],
                    [],
                    None,
                )

            # Replace zeros with small value for log-ratio
            work = work.clip(lower=1e-10)

            from compositional import transform_clr

            clr_values = transform_clr(work.values)
            clr_df = pd.DataFrame(clr_values, columns=comp_cols, index=work.index)

            # Identify components deviating most from geometric mean (CLR=0)
            mean_clr = clr_df.mean()
            std_clr = clr_df.std()
            abs_mean = mean_clr.abs().sort_values(ascending=False)

            deviations = []
            for col in abs_mean.index:
                deviations.append(
                    {
                        "component": col,
                        "mean_clr": round(float(mean_clr[col]), 6),
                        "std_clr": round(float(std_clr[col]), 6),
                        "abs_mean_clr": round(float(abs_mean[col]), 6),
                    }
                )

            findings = [
                {
                    "kind": "distribution",
                    "measurement_type": "measured",
                    "method": "CLR (Centered Log-Ratio) Transform",
                    "compositional_columns": comp_cols,
                    "n_components": len(comp_cols),
                    "n_observations": len(work),
                    "component_deviations": deviations,
                    "most_deviant_component": deviations[0]["component"]
                    if deviations
                    else None,
                }
            ]

            return PluginResult(
                status="ok",
                summary=(
                    f"CLR analysis on {len(comp_cols)} components: "
                    f"most deviant is '{deviations[0]['component']}' "
                    f"(mean CLR={deviations[0]['mean_clr']:.4f})"
                    if deviations
                    else "CLR analysis completed"
                ),
                metrics={
                    "n_components": len(comp_cols),
                    "n_observations": len(work),
                    "most_deviant_component": deviations[0]["component"]
                    if deviations
                    else None,
                    "max_abs_mean_clr": round(float(abs_mean.iloc[0]), 6)
                    if len(abs_mean) > 0
                    else 0.0,
                },
                findings=findings,
                artifacts=[],
                error=None,
            )
        except Exception as e:
            logger.error(
                f"Compositional log-ratio analysis failed: {e}", exc_info=True
            )
            return PluginResult(
                "error",
                f"Compositional log-ratio analysis failed: {e}",
                {},
                [],
                [],
                PluginError(
                    type=type(e).__name__,
                    message=str(e),
                    traceback=traceback.format_exc(),
                ),
            )
