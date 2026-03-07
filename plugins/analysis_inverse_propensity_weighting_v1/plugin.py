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
                return PluginResult("na", "Empty dataset", {}, [], [], None)

            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if len(numeric_cols) < 3:
                return PluginResult("na", "Need at least 3 numeric columns", {}, [], [], None)

            treatment_col = ctx.settings.get("treatment_column")
            outcome_col = ctx.settings.get("outcome_column")

            binary_cols = [c for c in numeric_cols if df[c].dropna().isin([0, 1, 0.0, 1.0]).all()]
            non_binary = [c for c in numeric_cols if c not in binary_cols]

            if not treatment_col:
                if not binary_cols:
                    return PluginResult("na", "No binary treatment column found", {}, [], [], None)
                treatment_col = binary_cols[0]
            if not outcome_col:
                if not non_binary:
                    return PluginResult("na", "No continuous outcome column found", {}, [], [], None)
                outcome_col = non_binary[0]

            covariate_cols = [c for c in numeric_cols if c not in (treatment_col, outcome_col)]
            if len(covariate_cols) < 1:
                return PluginResult("na", "Need at least 1 covariate", {}, [], [], None)

            needed = [outcome_col, treatment_col] + covariate_cols
            work = df[needed].dropna()
            if len(work) < 50:
                return PluginResult("na", f"Insufficient rows ({len(work)})", {}, [], [], None)

            max_rows = int(ctx.budget.get("row_limit") or 5000)
            rng = np.random.RandomState(ctx.run_seed)
            if len(work) > max_rows:
                work = work.iloc[rng.choice(len(work), max_rows, replace=False)].copy()

            T = work[treatment_col].values
            Y = work[outcome_col].values
            X = work[covariate_cols].values

            # Try econml LinearDML first, fall back to manual IPW
            try:
                from econml.dml import LinearDML
                from sklearn.linear_model import LogisticRegression, Lasso

                est = LinearDML(
                    model_y=Lasso(alpha=0.01, random_state=ctx.run_seed),
                    model_t=LogisticRegression(max_iter=1000, random_state=ctx.run_seed),
                    discrete_treatment=True,
                    random_state=ctx.run_seed,
                )
                est.fit(Y, T, X=X)
                ate = float(est.ate(X))
                ci = est.ate_interval(X, alpha=0.05)
                ci_low = float(ci[0])
                ci_high = float(ci[1])
                ate_se = (ci_high - ci_low) / (2 * 1.96)
                method_used = "LinearDML (econml)"
            except ImportError:
                from sklearn.linear_model import LogisticRegression

                lr = LogisticRegression(max_iter=1000, random_state=ctx.run_seed, solver="lbfgs")
                lr.fit(X, T)
                ps = lr.predict_proba(X)[:, 1]
                ps = np.clip(ps, 0.01, 0.99)

                # Horvitz-Thompson IPW estimator
                w1 = T / ps
                w0 = (1 - T) / (1 - ps)
                ate = float(np.mean(w1 * Y) - np.mean(w0 * Y))

                # Bootstrap standard error
                n_boot = 200
                boot_ates = np.empty(n_boot)
                for b in range(n_boot):
                    idx = rng.choice(len(Y), len(Y), replace=True)
                    bT, bY, bps = T[idx], Y[idx], ps[idx]
                    bw1 = bT / bps
                    bw0 = (1 - bT) / (1 - bps)
                    boot_ates[b] = np.mean(bw1 * bY) - np.mean(bw0 * bY)
                ate_se = float(np.std(boot_ates))
                ci_low = float(np.percentile(boot_ates, 2.5))
                ci_high = float(np.percentile(boot_ates, 97.5))
                method_used = "Manual IPW (LogisticRegression)"

            # p-value from z-test
            z = ate / ate_se if ate_se > 0 else 0.0
            from scipy.stats import norm
            pvalue = float(2 * (1 - norm.cdf(abs(z))))

            findings = [{
                "kind": "causal",
                "measurement_type": "measured",
                "method": f"Inverse Propensity Weighting ({method_used})",
                "ate": round(ate, 6),
                "std_error": round(ate_se, 6),
                "p_value": round(pvalue, 6),
                "ci_95_lower": round(ci_low, 6),
                "ci_95_upper": round(ci_high, 6),
                "significant": pvalue < 0.05,
                "treatment_column": treatment_col,
                "outcome_column": outcome_col,
                "n_covariates": len(covariate_cols),
                "interpretation": (
                    f"IPW ATE: {ate:+.4f} (p={pvalue:.4f}, "
                    f"95% CI [{ci_low:.4f}, {ci_high:.4f}])"
                ),
            }]

            return PluginResult(
                status="ok",
                summary=f"IPW ATE: {ate:+.4f} (p={pvalue:.4f})",
                metrics={
                    "n_observations": len(work),
                    "ate": round(ate, 6),
                    "std_error": round(ate_se, 6),
                    "p_value": round(pvalue, 6),
                    "n_covariates": len(covariate_cols),
                    "method": method_used,
                },
                findings=findings,
                artifacts=[],
                error=None,
            )
        except Exception as e:
            logger.error(f"IPW analysis failed: {e}", exc_info=True)
            return PluginResult("error", f"IPW analysis failed: {e}", {}, [], [],
                PluginError(type=type(e).__name__, message=str(e), traceback=traceback.format_exc()))
