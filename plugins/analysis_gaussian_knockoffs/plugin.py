from __future__ import annotations

import numpy as np

from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json


def _equicorrelated_knockoffs(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Construct equicorrelated Gaussian knockoff variables.

    Implements the equicorrelated construction from Barber & Candes (2015):
    X_tilde = X - X @ Sigma_inv @ S + U @ chol(2S - S @ Sigma_inv @ S)
    where S = s * I and s = min(2 * lambda_min(Sigma), 1).
    """
    n, p = X.shape
    # Center and scale
    mu = X.mean(axis=0)
    X_centered = X - mu

    # Sample covariance with shrinkage for stability
    Sigma = (X_centered.T @ X_centered) / max(1, n - 1)
    # Ledoit-Wolf-style shrinkage
    trace_S2 = float(np.sum(Sigma ** 2))
    trace_S = float(np.trace(Sigma))
    shrinkage = min(1.0, max(0.01, trace_S2 / (trace_S ** 2 * p) if trace_S > 0 else 0.5))
    Sigma = (1.0 - shrinkage) * Sigma + shrinkage * np.eye(p) * (trace_S / p)

    # Compute s for equicorrelated construction
    eigvals = np.linalg.eigvalsh(Sigma)
    lambda_min = max(float(eigvals[0]), 1e-6)
    s = min(2.0 * lambda_min, 1.0)
    S = s * np.eye(p)

    # Sigma inverse
    Sigma_inv = np.linalg.solve(Sigma, np.eye(p))

    # Knockoff construction
    # X_tilde = X - X @ Sigma_inv @ S + U @ C
    # where C = cholesky(2S - S @ Sigma_inv @ S)
    M = 2.0 * S - S @ Sigma_inv @ S
    # Ensure PSD
    eigvals_M, eigvecs_M = np.linalg.eigh(M)
    eigvals_M = np.maximum(eigvals_M, 1e-10)
    C = eigvecs_M @ np.diag(np.sqrt(eigvals_M)) @ eigvecs_M.T

    U = rng.standard_normal((n, p))
    X_tilde = X_centered - X_centered @ Sigma_inv @ S + U @ C
    X_tilde += mu  # re-center

    return X_tilde


class Plugin:
    def run(self, ctx) -> PluginResult:
        df = ctx.dataset_loader()
        numeric = df.select_dtypes(include="number")
        if numeric.shape[1] < 2:
            return PluginResult(
                "na", "Not enough numeric columns", {}, [], [], None
            )
        target_column = ctx.settings.get("target_column")
        target_columns = [target_column] if target_column else list(numeric.columns)

        seed = int(getattr(ctx, "run_seed", 0) or 0)
        rng = np.random.default_rng(seed)
        fdr_q = float(ctx.settings.get("fdr_q", 0.1))
        findings = []
        for target in target_columns:
            if target not in numeric.columns:
                continue
            frame = numeric.dropna(axis=0, how="any")
            if frame.shape[0] < 10:
                continue
            y = frame[target].to_numpy(dtype=float)
            X = frame.drop(columns=[target]).to_numpy(dtype=float)
            feature_names = [c for c in frame.columns if c != target]
            if X.shape[1] < 1:
                continue

            # Construct proper knockoff variables
            X_tilde = _equicorrelated_knockoffs(X, rng)

            # Compute knockoff statistics: W_j = |X_j^T y| - |X_tilde_j^T y|
            scores = {}
            for j, col in enumerate(feature_names):
                orig_stat = abs(float(X[:, j] @ y))
                knock_stat = abs(float(X_tilde[:, j] @ y))
                scores[col] = orig_stat - knock_stat

            if not scores:
                continue

            # Knockoff+ threshold for FDR control
            W_values = np.array(list(scores.values()))
            abs_W = np.abs(W_values)
            sorted_abs_W = np.sort(abs_W)[::-1]

            threshold = float("inf")
            for t in sorted_abs_W:
                if t <= 0:
                    continue
                n_above = int(np.sum(W_values >= t))
                n_below = int(np.sum(W_values <= -t))
                fdr_est = (1.0 + n_below) / max(1, n_above)
                if fdr_est <= fdr_q:
                    threshold = t
                    break

            for feature, score in sorted(scores.items(), key=lambda x: -abs(x[1])):
                selected = bool(score >= threshold) if np.isfinite(threshold) else False
                findings.append({
                    "id": f"analysis_gaussian_knockoffs:{target}:{feature}",
                    "severity": "info",
                    "confidence": min(1.0, abs(score) / max(abs(W_values).max(), 1e-9)),
                    "title": f"Knockoff feature selection: {feature} -> {target}",
                    "what": f"Gaussian knockoff statistic W={score:.4f} for {feature} predicting {target}",
                    "why": f"Feature {'selected' if selected else 'not selected'} at FDR q={fdr_q}. "
                           f"{'Investigate this feature as a significant predictor.' if selected else 'Not significant after FDR control.'}",
                    "kind": "feature_discovery",
                    "target": target,
                    "feature": feature,
                    "score": float(score),
                    "selected": selected,
                })

        artifacts_dir = ctx.artifacts_dir("analysis_gaussian_knockoffs")
        selection_path = artifacts_dir / "selection.json"
        write_json(selection_path, findings)
        artifacts = [
            PluginArtifact(
                path=str(selection_path.relative_to(ctx.run_dir)),
                type="json",
                description="Knockoff feature selection results",
            )
        ]
        selected_count = sum(1 for f in findings if f["selected"])
        return PluginResult(
            "ok",
            f"Gaussian knockoff FDR-controlled feature selection: {selected_count} selected",
            {"selected": selected_count, "total_features": len(findings)},
            findings,
            artifacts,
            None,
            debug={"targets_scanned": len(target_columns)},
        )
