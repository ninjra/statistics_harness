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

            from sklearn.linear_model import LogisticRegression
            from sklearn.neighbors import NearestNeighbors

            T = work[treatment_col].values
            Y = work[outcome_col].values
            X = work[covariate_cols].values

            # Estimate propensity scores
            lr = LogisticRegression(max_iter=1000, random_state=ctx.run_seed, solver="lbfgs")
            lr.fit(X, T)
            ps = lr.predict_proba(X)[:, 1]

            # Clip propensity scores to avoid extreme weights
            ps = np.clip(ps, 0.01, 0.99)

            # Nearest-neighbor matching on propensity score (treated -> control)
            treated_idx = np.where(T == 1)[0]
            control_idx = np.where(T == 0)[0]

            if len(treated_idx) < 5 or len(control_idx) < 5:
                return PluginResult("na", "Too few treated or control units", {}, [], [], None)

            nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
            nn.fit(ps[control_idx].reshape(-1, 1))
            distances, indices = nn.kneighbors(ps[treated_idx].reshape(-1, 1))
            matched_control_idx = control_idx[indices.flatten()]

            # ATT: average treatment effect on the treated
            att = float(np.mean(Y[treated_idx] - Y[matched_control_idx]))
            att_se = float(np.std(Y[treated_idx] - Y[matched_control_idx]) / np.sqrt(len(treated_idx)))
            ci_low = att - 1.96 * att_se
            ci_high = att + 1.96 * att_se

            # Approximate p-value from z-test
            z = att / att_se if att_se > 0 else 0.0
            from scipy.stats import norm
            pvalue = float(2 * (1 - norm.cdf(abs(z))))

            mean_distance = float(np.mean(distances))

            findings = [{
                "kind": "causal",
                "measurement_type": "measured",
                "method": "Propensity Score Matching (Nearest Neighbor)",
                "att": round(att, 6),
                "std_error": round(att_se, 6),
                "p_value": round(pvalue, 6),
                "ci_95_lower": round(ci_low, 6),
                "ci_95_upper": round(ci_high, 6),
                "significant": pvalue < 0.05,
                "treatment_column": treatment_col,
                "outcome_column": outcome_col,
                "n_treated": int(len(treated_idx)),
                "n_control": int(len(control_idx)),
                "n_matched_pairs": int(len(treated_idx)),
                "mean_match_distance": round(mean_distance, 6),
                "interpretation": (
                    f"PSM ATT: {att:+.4f} (p={pvalue:.4f}, "
                    f"95% CI [{ci_low:.4f}, {ci_high:.4f}])"
                ),
            }]

            return PluginResult(
                status="ok",
                summary=f"Propensity Score Matching ATT: {att:+.4f} (p={pvalue:.4f})",
                metrics={
                    "n_observations": len(work),
                    "att": round(att, 6),
                    "std_error": round(att_se, 6),
                    "p_value": round(pvalue, 6),
                    "n_treated": int(len(treated_idx)),
                    "n_control": int(len(control_idx)),
                    "n_matched_pairs": int(len(treated_idx)),
                    "mean_match_distance": round(mean_distance, 6),
                    "n_covariates": len(covariate_cols),
                    "method": "Propensity Score Matching (Nearest Neighbor)",
                },
                findings=findings,
                artifacts=[],
                error=None,
            )
        except Exception as e:
            logger.error(f"Propensity score matching failed: {e}", exc_info=True)
            return PluginResult("error", f"Propensity score matching failed: {e}", {}, [], [],
                PluginError(type=type(e).__name__, message=str(e), traceback=traceback.format_exc()))
