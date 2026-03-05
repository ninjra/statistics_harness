from __future__ import annotations

import logging
import traceback

import numpy as np
import pandas as pd

from statistic_harness.core.types import PluginArtifact, PluginError, PluginResult

logger = logging.getLogger(__name__)


class Plugin:
    def run(self, ctx) -> PluginResult:
        try:
            df = ctx.dataset_loader()
            if df.empty:
                return PluginResult("skipped", "Empty dataset", {}, [], [], None)

            numeric_cols = [
                c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
            ]
            if len(numeric_cols) < 3:
                return PluginResult(
                    "skipped",
                    "Need at least 3 numeric columns",
                    {},
                    [],
                    [],
                    None,
                )

            outcome_col = ctx.settings.get("outcome_column")
            endogenous_col = ctx.settings.get("endogenous_column")
            instrument_col = ctx.settings.get("instrument_column")
            exogenous_cols = ctx.settings.get("exogenous_columns")

            if not outcome_col or not endogenous_col or not instrument_col:
                return PluginResult(
                    "skipped",
                    "Requires outcome_column, endogenous_column, and instrument_column in settings",
                    {},
                    [],
                    [],
                    None,
                )

            required = [outcome_col, endogenous_col, instrument_col]
            for col in required:
                if col not in df.columns:
                    return PluginResult(
                        "skipped",
                        f"Column '{col}' not found",
                        {},
                        [],
                        [],
                        None,
                    )

            if exogenous_cols:
                exogenous_cols = [c for c in exogenous_cols if c in df.columns]
            else:
                exogenous_cols = [
                    c
                    for c in numeric_cols
                    if c not in (outcome_col, endogenous_col, instrument_col)
                ]

            all_cols = [outcome_col, endogenous_col, instrument_col] + exogenous_cols
            work = df[all_cols].dropna()
            if len(work) < 30:
                return PluginResult(
                    "skipped",
                    f"Insufficient rows ({len(work)})",
                    {},
                    [],
                    [],
                    None,
                )

            from linearmodels.iv import IV2SLS

            dependent = work[outcome_col]
            endog_vars = work[[endogenous_col]]
            instruments = work[[instrument_col]]

            if exogenous_cols:
                exog = work[exogenous_cols]
            else:
                exog = None

            model = IV2SLS(dependent, exog, endog_vars, instruments)
            result = model.fit(cov_type="robust")

            coef = float(result.params[endogenous_col])
            se = float(result.std_errors[endogenous_col])
            pvalue = float(result.pvalues[endogenous_col])
            ci = result.conf_int()
            ci_low = float(ci.loc[endogenous_col, "lower"])
            ci_high = float(ci.loc[endogenous_col, "upper"])

            findings = [
                {
                    "kind": "causal",
                    "measurement_type": "measured",
                    "method": "IV-2SLS (Two-Stage Least Squares)",
                    "outcome_column": outcome_col,
                    "endogenous_column": endogenous_col,
                    "instrument_column": instrument_col,
                    "n_exogenous": len(exogenous_cols),
                    "coefficient": round(coef, 6),
                    "std_error": round(se, 6),
                    "p_value": round(pvalue, 6),
                    "ci_95_lower": round(ci_low, 6),
                    "ci_95_upper": round(ci_high, 6),
                    "significant": pvalue < 0.05,
                    "n_observations": len(work),
                    "interpretation": (
                        f"IV-2SLS coefficient: {coef:+.4f} "
                        f"(p={pvalue:.4f}, 95% CI [{ci_low:.4f}, {ci_high:.4f}])"
                    ),
                }
            ]

            return PluginResult(
                status="ok",
                summary=f"IV-2SLS: {endogenous_col} -> {outcome_col}, coef={coef:+.4f} (p={pvalue:.4f})",
                metrics={
                    "coefficient": round(coef, 6),
                    "std_error": round(se, 6),
                    "p_value": round(pvalue, 6),
                    "n_observations": len(work),
                    "r_squared": round(float(result.rsquared), 6),
                },
                findings=findings,
                artifacts=[],
                error=None,
            )
        except Exception as e:
            logger.error(f"IV-2SLS analysis failed: {e}", exc_info=True)
            return PluginResult(
                "error",
                f"IV-2SLS analysis failed: {e}",
                {},
                [],
                [],
                PluginError(
                    type=type(e).__name__,
                    message=str(e),
                    traceback=traceback.format_exc(),
                ),
            )
