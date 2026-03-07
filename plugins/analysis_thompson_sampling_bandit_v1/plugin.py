from __future__ import annotations
import logging
import traceback
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from statistic_harness.core.types import PluginResult, PluginArtifact, PluginError

logger = logging.getLogger(__name__)


class Plugin:
    def run(self, ctx) -> PluginResult:
        try:
            df = ctx.dataset_loader()
            if df.empty:
                return PluginResult("na", "Empty dataset", {}, [], [], None)

            settings = ctx.settings or {}
            group_col = settings.get("group_column")
            outcome_col = settings.get("outcome_column")
            n_mc = int(settings.get("n_mc_samples", 10_000))

            # Auto-detect: group = first categorical with >1 unique, outcome = first binary numeric
            if not group_col:
                for c in df.columns:
                    if df[c].dtype == object or str(df[c].dtype) == "category":
                        if df[c].dropna().nunique() >= 2:
                            group_col = c
                            break
            if not outcome_col:
                for c in df.columns:
                    if c == group_col:
                        continue
                    if pd.api.types.is_numeric_dtype(df[c]):
                        vals = df[c].dropna().unique()
                        if set(vals).issubset({0, 1, 0.0, 1.0}):
                            outcome_col = c
                            break

            if not group_col or not outcome_col:
                return PluginResult(
                    "na",
                    "Could not identify a group column and a binary outcome column",
                    {}, [], [], None,
                )

            work = df[[group_col, outcome_col]].dropna()
            work[outcome_col] = work[outcome_col].astype(float)
            arms = work[group_col].unique()
            if len(arms) < 2:
                return PluginResult("na", f"Need >=2 arms, found {len(arms)}", {}, [], [], None)

            # Compute posterior Beta(successes+1, failures+1) for each arm
            arm_stats = {}
            for arm in arms:
                mask = work[group_col] == arm
                successes = int(work.loc[mask, outcome_col].sum())
                trials = int(mask.sum())
                failures = trials - successes
                alpha_post = successes + 1
                beta_post = failures + 1
                arm_stats[arm] = {
                    "alpha": alpha_post,
                    "beta": beta_post,
                    "trials": trials,
                    "successes": successes,
                    "rate": alpha_post / (alpha_post + beta_post),
                }

            # Monte Carlo: probability each arm is best
            rng = np.random.RandomState(ctx.run_seed)
            samples = {
                arm: rng.beta(s["alpha"], s["beta"], size=n_mc)
                for arm, s in arm_stats.items()
            }
            arm_order = list(arm_stats.keys())
            sample_matrix = np.column_stack([samples[a] for a in arm_order])
            best_indices = sample_matrix.argmax(axis=1)
            prob_best = {}
            for i, arm in enumerate(arm_order):
                prob_best[arm] = float(np.mean(best_indices == i))

            best_arm = max(prob_best, key=prob_best.get)

            arm_details = []
            for arm in arm_order:
                arm_details.append({
                    "arm": str(arm),
                    "trials": arm_stats[arm]["trials"],
                    "successes": arm_stats[arm]["successes"],
                    "posterior_alpha": arm_stats[arm]["alpha"],
                    "posterior_beta": arm_stats[arm]["beta"],
                    "posterior_mean": round(arm_stats[arm]["rate"], 6),
                    "prob_best": round(prob_best[arm], 6),
                })

            findings = [{
                "kind": "distribution",
                "measurement_type": "measured",
                "group_column": group_col,
                "outcome_column": outcome_col,
                "best_arm": str(best_arm),
                "prob_best_arm": round(prob_best[best_arm], 6),
                "n_arms": len(arm_order),
                "arm_details": arm_details,
                "method": "Thompson Sampling (Beta-Binomial)",
            }]

            return PluginResult(
                status="ok",
                summary=(
                    f"Thompson sampling: best arm={best_arm} "
                    f"(P(best)={prob_best[best_arm]:.3f}), {len(arm_order)} arms"
                ),
                metrics={
                    "n_observations": len(work),
                    "n_arms": len(arm_order),
                    "best_arm": str(best_arm),
                    "prob_best_arm": round(prob_best[best_arm], 6),
                },
                findings=findings,
                artifacts=[],
                error=None,
            )
        except Exception as e:
            logger.error(f"Thompson sampling bandit failed: {e}", exc_info=True)
            return PluginResult(
                status="error",
                summary=f"Thompson sampling bandit failed: {e}",
                metrics={},
                findings=[],
                artifacts=[],
                error=PluginError(type=type(e).__name__, message=str(e), traceback=traceback.format_exc()),
            )
