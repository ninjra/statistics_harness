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

            settings = ctx.settings or {}
            outcome_col = settings.get("outcome_column")
            group_col = settings.get("group_column")

            # Auto-detect binary columns if not specified
            if not outcome_col or not group_col:
                binary_cols = []
                for c in df.columns:
                    nunique = df[c].dropna().nunique()
                    if nunique == 2:
                        binary_cols.append(c)
                if len(binary_cols) < 2:
                    return PluginResult(
                        "na",
                        "Need at least 2 binary columns (group + outcome)",
                        {}, [], [], None,
                    )
                if not group_col:
                    group_col = binary_cols[0]
                if not outcome_col:
                    outcome_col = binary_cols[1] if binary_cols[1] != group_col else binary_cols[0]
                    if outcome_col == group_col:
                        return PluginResult(
                            "na",
                            "Cannot use same column for group and outcome",
                            {}, [], [], None,
                        )

            work = df[[group_col, outcome_col]].dropna()
            if len(work) < 10:
                return PluginResult("na", f"Insufficient rows ({len(work)})", {}, [], [], None)

            # Encode group to 0/1
            groups = work[group_col].unique()
            if len(groups) != 2:
                return PluginResult("na", f"Group column must have exactly 2 levels, found {len(groups)}", {}, [], [], None)

            group_a_label, group_b_label = sorted(groups, key=str)
            mask_a = work[group_col] == group_a_label
            mask_b = work[group_col] == group_b_label

            # Encode outcome to 0/1
            outcome_vals = work[outcome_col].unique()
            if len(outcome_vals) != 2:
                return PluginResult("na", f"Outcome column must be binary, found {len(outcome_vals)} levels", {}, [], [], None)

            outcome_map = {sorted(outcome_vals, key=str)[0]: 0, sorted(outcome_vals, key=str)[1]: 1}
            outcomes = work[outcome_col].map(outcome_map)

            successes_a = int(outcomes[mask_a].sum())
            trials_a = int(mask_a.sum())
            successes_b = int(outcomes[mask_b].sum())
            trials_b = int(mask_b.sum())

            failures_a = trials_a - successes_a
            failures_b = trials_b - successes_b

            from scipy.stats import beta as beta_dist

            # Uniform prior: Beta(1, 1)
            alpha_prior, beta_prior = 1, 1

            alpha_a = alpha_prior + successes_a
            beta_a = beta_prior + failures_a
            alpha_b = alpha_prior + successes_b
            beta_b = beta_prior + failures_b

            # Monte Carlo estimation of P(B > A)
            rng = np.random.RandomState(ctx.run_seed)
            n_samples = 100_000
            samples_a = rng.beta(alpha_a, beta_a, size=n_samples)
            samples_b = rng.beta(alpha_b, beta_b, size=n_samples)
            prob_b_gt_a = float(np.mean(samples_b > samples_a))

            # Expected rates
            rate_a = alpha_a / (alpha_a + beta_a)
            rate_b = alpha_b / (alpha_b + beta_b)
            expected_lift = (rate_b - rate_a) / rate_a if rate_a > 0 else 0.0

            findings = [{
                "kind": "distribution",
                "measurement_type": "measured",
                "group_column": group_col,
                "outcome_column": outcome_col,
                "group_a": str(group_a_label),
                "group_b": str(group_b_label),
                "rate_a": round(rate_a, 6),
                "rate_b": round(rate_b, 6),
                "prob_b_greater_than_a": round(prob_b_gt_a, 6),
                "expected_lift": round(expected_lift, 6),
                "posterior_a_alpha": alpha_a,
                "posterior_a_beta": beta_a,
                "posterior_b_alpha": alpha_b,
                "posterior_b_beta": beta_b,
                "method": "Bayesian A/B test (Beta-Binomial)",
            }]

            return PluginResult(
                status="ok",
                summary=(
                    f"Bayesian A/B test: P(B>A)={prob_b_gt_a:.3f}, "
                    f"rate_A={rate_a:.4f}, rate_B={rate_b:.4f}, lift={expected_lift:.4f}"
                ),
                metrics={
                    "n_observations": len(work),
                    "trials_a": trials_a,
                    "trials_b": trials_b,
                    "successes_a": successes_a,
                    "successes_b": successes_b,
                    "prob_b_greater_than_a": round(prob_b_gt_a, 6),
                    "expected_lift": round(expected_lift, 6),
                    "method": "Bayesian A/B test (Beta-Binomial conjugate)",
                },
                findings=findings,
                artifacts=[],
                error=None,
            )
        except Exception as e:
            logger.error(f"Bayesian A/B test failed: {e}", exc_info=True)
            return PluginResult(
                status="error",
                summary=f"Bayesian A/B test failed: {e}",
                metrics={},
                findings=[],
                artifacts=[],
                error=PluginError(type=type(e).__name__, message=str(e), traceback=traceback.format_exc()),
            )
