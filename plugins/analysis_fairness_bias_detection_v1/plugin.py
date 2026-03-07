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
            sensitive_col = settings.get("sensitive_column")

            # Auto-detect binary columns if not specified
            if not outcome_col or not sensitive_col:
                binary_cols = []
                for c in df.columns:
                    nunique = df[c].dropna().nunique()
                    if nunique == 2:
                        binary_cols.append(c)
                if len(binary_cols) < 2:
                    return PluginResult(
                        "na",
                        "Need at least 2 binary columns (sensitive attribute + outcome)",
                        {}, [], [], None,
                    )
                if not sensitive_col:
                    sensitive_col = binary_cols[0]
                if not outcome_col:
                    outcome_col = binary_cols[1] if binary_cols[1] != sensitive_col else binary_cols[0]
                    if outcome_col == sensitive_col:
                        return PluginResult(
                            "na",
                            "Cannot use same column for sensitive attribute and outcome",
                            {}, [], [], None,
                        )

            work = df[[sensitive_col, outcome_col]].dropna()
            if len(work) < 10:
                return PluginResult("na", f"Insufficient rows ({len(work)})", {}, [], [], None)

            # Encode outcome to 0/1
            outcome_vals = sorted(work[outcome_col].unique(), key=str)
            if len(outcome_vals) != 2:
                return PluginResult(
                    "na",
                    f"Outcome column must be binary, found {len(outcome_vals)} levels",
                    {}, [], [], None,
                )
            outcome_map = {outcome_vals[0]: 0, outcome_vals[1]: 1}
            y_true = work[outcome_col].map(outcome_map).values
            sensitive = work[sensitive_col].values

            try:
                from fairlearn.metrics import (
                    MetricFrame,
                    demographic_parity_difference,
                    equalized_odds_difference,
                )
                from sklearn.metrics import accuracy_score
            except ImportError:
                return PluginResult(
                    "na",
                    "fairlearn or sklearn not installed",
                    {}, [], [], None,
                )

            # Use y_true as trivial "predictions" to measure base-rate disparity
            y_pred = y_true.copy()

            dp_diff = demographic_parity_difference(
                y_true, y_pred, sensitive_features=sensitive,
            )

            eo_diff = equalized_odds_difference(
                y_true, y_pred, sensitive_features=sensitive,
            )

            mf = MetricFrame(
                metrics=accuracy_score,
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive,
            )
            group_metrics = {str(k): float(v) for k, v in mf.by_group.items()}

            groups = sorted(work[sensitive_col].unique(), key=str)
            group_rates = {}
            for g in groups:
                mask = sensitive == g
                group_rates[str(g)] = float(np.mean(y_true[mask]))

            findings = [{
                "kind": "distribution",
                "measurement_type": "measured",
                "sensitive_column": sensitive_col,
                "outcome_column": outcome_col,
                "demographic_parity_difference": round(float(dp_diff), 6),
                "equalized_odds_difference": round(float(eo_diff), 6),
                "group_positive_rates": group_rates,
                "group_accuracy": group_metrics,
                "method": "Fairlearn MetricFrame",
            }]

            return PluginResult(
                status="ok",
                summary=(
                    f"Fairness bias detection: demographic_parity_diff={dp_diff:.4f}, "
                    f"equalized_odds_diff={eo_diff:.4f}"
                ),
                metrics={
                    "n_observations": len(work),
                    "n_groups": len(groups),
                    "demographic_parity_difference": round(float(dp_diff), 6),
                    "equalized_odds_difference": round(float(eo_diff), 6),
                },
                findings=findings,
                artifacts=[],
                error=None,
            )
        except Exception as e:
            logger.error(f"Fairness bias detection failed: {e}", exc_info=True)
            return PluginResult(
                status="error",
                summary=f"Fairness bias detection failed: {e}",
                metrics={},
                findings=[],
                artifacts=[],
                error=PluginError(type=type(e).__name__, message=str(e), traceback=traceback.format_exc()),
            )
