from __future__ import annotations

import numpy as np
from scipy.special import gammaln

from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json


def _bocpd_gaussian(series: np.ndarray, hazard_lambda: float = 200.0, prior_mu: float = 0.0,
                    prior_kappa: float = 1.0, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> np.ndarray:
    """Bayesian Online Changepoint Detection with Gaussian predictive (Adams & MacKay 2007).

    Returns an array of run-length posterior maxima (changepoint probability proxy).
    """
    n = len(series)
    # R[t, r] = P(run_length=r | x_{1:t})
    # We only track the current column of the run-length distribution
    R = np.zeros(n + 1, dtype=float)
    R[0] = 1.0  # P(r=0) = 1 at start

    # Sufficient statistics for Gaussian conjugate (Normal-Inverse-Gamma)
    mu = np.full(n + 1, prior_mu, dtype=float)
    kappa = np.full(n + 1, prior_kappa, dtype=float)
    alpha = np.full(n + 1, prior_alpha, dtype=float)
    beta = np.full(n + 1, prior_beta, dtype=float)

    changepoint_prob = np.zeros(n, dtype=float)
    hazard = 1.0 / hazard_lambda

    for t in range(n):
        x = series[t]

        # Predictive probability: Student-t distribution
        # p(x_t | r_t, x_{t-r:t-1}) for each run length
        df = 2.0 * alpha[:t + 1]
        pred_mean = mu[:t + 1]
        pred_var = beta[:t + 1] * (kappa[:t + 1] + 1.0) / (kappa[:t + 1] * alpha[:t + 1])
        pred_scale = np.sqrt(np.maximum(pred_var, 1e-12))

        # Student-t log pdf
        z = (x - pred_mean) / pred_scale
        log_pred = (
            gammaln((df + 1.0) / 2.0) - gammaln(df / 2.0)
            - 0.5 * np.log(df * np.pi) - np.log(pred_scale)
            - ((df + 1.0) / 2.0) * np.log1p(z * z / df)
        )
        pred = np.exp(log_pred - np.max(log_pred))  # numerical stability

        # Growth probabilities
        growth = R[:t + 1] * pred * (1.0 - hazard)

        # Changepoint probability
        cp = np.sum(R[:t + 1] * pred * hazard)

        # New run-length distribution
        new_R = np.zeros(t + 2, dtype=float)
        new_R[0] = cp
        new_R[1:t + 2] = growth

        # Normalize
        total = new_R.sum()
        if total > 0:
            new_R /= total

        changepoint_prob[t] = float(new_R[0])

        # Update sufficient statistics
        new_mu = np.zeros(t + 2, dtype=float)
        new_kappa = np.zeros(t + 2, dtype=float)
        new_alpha = np.zeros(t + 2, dtype=float)
        new_beta = np.zeros(t + 2, dtype=float)

        # Reset for new run (r=0)
        new_mu[0] = prior_mu
        new_kappa[0] = prior_kappa
        new_alpha[0] = prior_alpha
        new_beta[0] = prior_beta

        # Update for continuing runs
        old_kappa = kappa[:t + 1]
        new_kappa[1:t + 2] = old_kappa + 1.0
        new_mu[1:t + 2] = (old_kappa * mu[:t + 1] + x) / new_kappa[1:t + 2]
        new_alpha[1:t + 2] = alpha[:t + 1] + 0.5
        new_beta[1:t + 2] = (
            beta[:t + 1]
            + 0.5 * old_kappa * (x - mu[:t + 1]) ** 2 / new_kappa[1:t + 2]
        )

        R = np.zeros(t + 2, dtype=float)
        R[:t + 2] = new_R
        mu = new_mu
        kappa = new_kappa
        alpha = new_alpha
        beta = new_beta

    return changepoint_prob


class Plugin:
    def run(self, ctx) -> PluginResult:
        df = ctx.dataset_loader()
        value_col = ctx.settings.get("value_column")
        value_cols = ctx.settings.get("value_columns")
        columns: list[str]
        if value_col:
            columns = [value_col] if value_col in df.columns else []
        elif isinstance(value_cols, list):
            columns = [str(col) for col in value_cols if col in df.columns]
        else:
            numeric = df.select_dtypes(include="number")
            if numeric.empty:
                return PluginResult("na", "No numeric columns", {}, [], [], None)
            columns = list(numeric.columns)
        if not columns:
            return PluginResult("na", "No value columns", {}, [], [], None)

        peak_threshold = float(ctx.settings.get("peak_threshold", 0.3))
        hazard_lambda = float(ctx.settings.get("hazard_lambda", 200.0))
        changepoints = []
        for col in columns:
            series = df[col].dropna().to_numpy(dtype=float)
            n = len(series)
            if n < 10:
                continue
            if not np.isfinite(np.nanstd(series)) or np.nanstd(series) == 0:
                continue

            cp_prob = _bocpd_gaussian(series, hazard_lambda=hazard_lambda)

            # Find peaks above threshold
            for idx in range(1, n - 1):
                if (cp_prob[idx] >= peak_threshold
                        and cp_prob[idx] >= cp_prob[idx - 1]
                        and cp_prob[idx] >= cp_prob[idx + 1]):
                    changepoints.append({
                        "id": f"analysis_bocpd_gaussian:{col}:{idx}",
                        "severity": "warn" if cp_prob[idx] > 0.5 else "info",
                        "confidence": float(cp_prob[idx]),
                        "title": f"BOCPD changepoint at index {idx} in {col}",
                        "what": f"Bayesian Online Changepoint Detection identified a regime change at index {idx} in column {col} (posterior probability {cp_prob[idx]:.3f})",
                        "why": "A changepoint indicates a shift in the underlying data-generating process; investigate what changed at this time.",
                        "kind": "changepoint",
                        "column": col,
                        "index": int(idx),
                        "prob": float(cp_prob[idx]),
                    })

        artifacts_dir = ctx.artifacts_dir("analysis_bocpd_gaussian")
        out_path = artifacts_dir / "changepoints.json"
        write_json(out_path, changepoints)
        artifacts = [
            PluginArtifact(
                path=str(out_path.relative_to(ctx.run_dir)),
                type="json",
                description="BOCPD changepoints",
            )
        ]
        return PluginResult(
            "ok",
            f"BOCPD Gaussian detected {len(changepoints)} changepoint(s)",
            {
                "count": len(changepoints),
                "columns_scanned": len(columns),
                "columns_with_findings": len({c["column"] for c in changepoints}),
            },
            changepoints,
            artifacts,
            None,
        )
