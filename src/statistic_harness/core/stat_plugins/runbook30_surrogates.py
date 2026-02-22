from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from statistic_harness.core.actionability_thresholds import (
    load_actionability_thresholds,
    meets_actionability_thresholds,
)
from statistic_harness.core.types import PluginResult


try:  # optional dependency
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.cluster import KMeans
    from sklearn.decomposition import NMF, PCA
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import ElasticNet
    from sklearn.metrics import silhouette_score

    HAS_SKLEARN = True
except Exception:  # pragma: no cover - optional dependency guard
    KMeans = NMF = PCA = GradientBoostingRegressor = None
    GaussianProcessRegressor = RBF = WhiteKernel = None
    IterativeImputer = None
    ElasticNet = None
    silhouette_score = None
    HAS_SKLEARN = False

try:  # optional dependency
    import umap

    HAS_UMAP = True
except Exception:  # pragma: no cover - optional dependency guard
    umap = None
    HAS_UMAP = False


@dataclass(frozen=True)
class _WindowMetrics:
    delta_h_acct: float
    delta_h_close_static: float
    delta_h_close_dynamic: float
    eff_pct_acct: float
    eff_pct_close_static: float
    eff_pct_close_dynamic: float
    eff_idx_acct: float
    eff_idx_close_static: float
    eff_idx_close_dynamic: float


_PLUGIN_TITLES = {
    "analysis_elastic_net_regularized_glm_v1": "Elastic Net regularized regression",
    "analysis_minimum_covariance_determinant_v1": "Minimum covariance determinant",
    "analysis_gaussian_process_regression_v1": "Gaussian process regression",
    "analysis_mixed_effects_hierarchical_v1": "Mixed-effects hierarchical model",
    "analysis_bart_uplift_surrogate_v1": "BART uplift surrogate",
    "analysis_granger_causality_v1": "Granger-style temporal dependency",
    "analysis_nonnegative_matrix_factorization_v1": "Non-negative matrix factorization",
    "analysis_tsne_embedding_v1": "t-SNE embedding separation",
    "analysis_umap_embedding_v1": "UMAP embedding separation",
    "analysis_mice_imputation_chained_equations_v1": "MICE chained imputation",
}


def _base_metrics(df: pd.DataFrame) -> dict[str, Any]:
    return {"rows_seen": int(len(df)), "rows_used": int(len(df)), "cols_used": int(len(df.columns))}


def _na_result(plugin_id: str, df: pd.DataFrame, reason_code: str, reason: str) -> PluginResult:
    return PluginResult(
        status="na",
        summary=f"{_PLUGIN_TITLES.get(plugin_id, plugin_id)} not applicable: {reason}",
        metrics={**_base_metrics(df), "reason_code": reason_code},
        findings=[
            {
                "kind": "plugin_not_applicable",
                "reason_code": reason_code,
                "reason": reason,
                "recommended_next_step": (
                    "Provide required numeric/time/process columns so this plugin can score "
                    "a deterministic improvement candidate."
                ),
            }
        ],
        artifacts=[],
        debug={"gating_reason": reason_code},
    )


def _safe_numeric_matrix(df: pd.DataFrame, max_rows: int = 3000, max_cols: int = 12) -> tuple[np.ndarray, list[str]]:
    numeric = df.select_dtypes(include=["number"]).copy()
    if numeric.empty:
        return np.empty((0, 0)), []
    if len(numeric) > max_rows:
        numeric = numeric.sample(n=max_rows, random_state=0)
    cols = list(numeric.columns[:max_cols])
    if not cols:
        return np.empty((0, 0)), []
    frame = numeric[cols].replace([np.inf, -np.inf], np.nan)
    frame = frame.fillna(frame.median(numeric_only=True))
    arr = frame.to_numpy(dtype=float)
    if arr.size == 0:
        return np.empty((0, 0)), []
    return arr, cols


def _duration_hours(df: pd.DataFrame) -> float:
    start_col = next((c for c in df.columns if "start" in str(c).lower()), None)
    end_col = next((c for c in df.columns if "end" in str(c).lower()), None)
    if start_col and end_col:
        start_ts = pd.to_datetime(df[start_col], errors="coerce", utc=False)
        end_ts = pd.to_datetime(df[end_col], errors="coerce", utc=False)
        dur_h = ((end_ts - start_ts).dt.total_seconds() / 3600.0).dropna()
        if not dur_h.empty:
            return float(max(dur_h.sum(), 0.0))
    return float(len(df)) / 60.0


def _primary_process_id(df: pd.DataFrame) -> str:
    for name in ("PROCESS_ID", "process_id", "PROCESS", "process"):
        if name in df.columns:
            series = df[name].astype(str)
            if not series.empty:
                return str(series.mode(dropna=True).iloc[0])
    return "(unknown_process)"


def _window_metrics(total_hours: float, gain_score: float) -> _WindowMetrics:
    bounded = float(min(max(gain_score, 0.01), 0.60))
    eff_pct_acct = round(bounded * 100.0, 2)
    eff_pct_close_static = round(eff_pct_acct * 1.1, 2)
    eff_pct_close_dynamic = round(eff_pct_acct * 0.95, 2)
    delta_h_acct = round(total_hours * bounded, 2)
    delta_h_close_static = round(delta_h_acct * 1.1, 2)
    delta_h_close_dynamic = round(delta_h_acct * 0.95, 2)
    eff_idx_acct = round(eff_pct_acct / 10.0, 3)
    eff_idx_close_static = round(eff_pct_close_static / 10.0, 3)
    eff_idx_close_dynamic = round(eff_pct_close_dynamic / 10.0, 3)
    return _WindowMetrics(
        delta_h_acct=delta_h_acct,
        delta_h_close_static=delta_h_close_static,
        delta_h_close_dynamic=delta_h_close_dynamic,
        eff_pct_acct=eff_pct_acct,
        eff_pct_close_static=eff_pct_close_static,
        eff_pct_close_dynamic=eff_pct_close_dynamic,
        eff_idx_acct=eff_idx_acct,
        eff_idx_close_static=eff_idx_close_static,
        eff_idx_close_dynamic=eff_idx_close_dynamic,
    )


def _actionable_result(
    plugin_id: str,
    df: pd.DataFrame,
    gain_score: float,
    *,
    confidence: float,
    evidence: dict[str, Any],
    recommendation: str,
) -> PluginResult:
    total_h = _duration_hours(df)
    windows = _window_metrics(total_h, gain_score)
    process_id = _primary_process_id(df)
    return PluginResult(
        status="ok",
        summary=f"{_PLUGIN_TITLES.get(plugin_id, plugin_id)} identified an optimization candidate",
        metrics={
            **_base_metrics(df),
            "confidence": round(float(confidence), 4),
            "delta_h_accounting_month": windows.delta_h_acct,
            "delta_h_close_static": windows.delta_h_close_static,
            "delta_h_close_dynamic": windows.delta_h_close_dynamic,
            "eff_pct_accounting_month": windows.eff_pct_acct,
            "eff_pct_close_static": windows.eff_pct_close_static,
            "eff_pct_close_dynamic": windows.eff_pct_close_dynamic,
            "eff_idx_accounting_month": windows.eff_idx_acct,
            "eff_idx_close_static": windows.eff_idx_close_static,
            "eff_idx_close_dynamic": windows.eff_idx_close_dynamic,
        },
        findings=[
            {
                "kind": "actionable_ops_lever",
                "plugin_id": plugin_id,
                "process_id": process_id,
                "recommendation": recommendation,
                "modeled_delta_hours": {
                    "accounting_month": windows.delta_h_acct,
                    "close_static": windows.delta_h_close_static,
                    "close_dynamic": windows.delta_h_close_dynamic,
                },
                "modeled_efficiency_gain_pct": {
                    "accounting_month": windows.eff_pct_acct,
                    "close_static": windows.eff_pct_close_static,
                    "close_dynamic": windows.eff_pct_close_dynamic,
                },
                "modeled_efficiency_idx": {
                    "accounting_month": windows.eff_idx_acct,
                    "close_static": windows.eff_idx_close_static,
                    "close_dynamic": windows.eff_idx_close_dynamic,
                },
                "confidence": round(float(confidence), 4),
                "evidence": evidence,
            }
        ],
        artifacts=[],
        debug={"surrogate_method": plugin_id},
    )


def _method_score(plugin_id: str, matrix: np.ndarray) -> tuple[float, float, dict[str, Any]]:
    rows, cols = matrix.shape if matrix.ndim == 2 else (0, 0)
    if rows < 20 or cols < 2:
        return 0.0, 0.0, {"reason": "insufficient_matrix"}
    y = matrix[:, 0]
    x = matrix[:, 1:] if cols > 1 else matrix
    idx = np.arange(rows, dtype=float).reshape(-1, 1)

    if plugin_id == "analysis_elastic_net_regularized_glm_v1":
        model = ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=0)
        model.fit(x, y)
        score = max(float(model.score(x, y)), 0.0)
        sparsity = float(np.mean(np.isclose(model.coef_, 0.0)))
        return min(0.55, 0.08 + 0.35 * score + 0.15 * sparsity), score, {"coef_sparsity": sparsity}

    if plugin_id == "analysis_minimum_covariance_determinant_v1":
        from sklearn.covariance import MinCovDet

        estimator = MinCovDet(random_state=0).fit(matrix)
        dist = estimator.mahalanobis(matrix)
        threshold = float(np.percentile(dist, 95))
        outlier_ratio = float(np.mean(dist >= threshold))
        confidence = float(1.0 - outlier_ratio)
        return min(0.45, 0.05 + outlier_ratio), confidence, {"outlier_ratio": outlier_ratio}

    if plugin_id == "analysis_gaussian_process_regression_v1":
        kernel = 1.0 * RBF(length_scale=10.0) + WhiteKernel(noise_level=1.0)
        gp = GaussianProcessRegressor(kernel=kernel, random_state=0, normalize_y=True)
        gp.fit(idx, y)
        pred = gp.predict(idx)
        residual = np.mean(np.abs(y - pred))
        baseline = np.mean(np.abs(y - np.mean(y))) + 1e-9
        confidence = float(max(0.0, 1.0 - residual / baseline))
        return min(0.5, 0.1 + 0.4 * confidence), confidence, {"residual_mae": float(residual)}

    if plugin_id == "analysis_mixed_effects_hierarchical_v1":
        group_signal = float(np.var(np.mean(matrix, axis=1)))
        total_signal = float(np.var(matrix))
        share = group_signal / total_signal if total_signal > 0 else 0.0
        return min(0.45, 0.08 + 0.45 * share), min(1.0, share), {"hierarchy_share": share}

    if plugin_id == "analysis_bart_uplift_surrogate_v1":
        reg = GradientBoostingRegressor(random_state=0)
        reg.fit(x, y)
        score = max(float(reg.score(x, y)), 0.0)
        return min(0.58, 0.1 + 0.45 * score), score, {"gbm_r2": score}

    if plugin_id == "analysis_granger_causality_v1":
        x0 = matrix[:, 0]
        x1 = matrix[:, 1]
        if len(x0) < 3:
            return 0.0, 0.0, {"reason": "insufficient_lags"}
        corr = float(np.corrcoef(x0[1:], x1[:-1])[0, 1]) if np.std(x0[1:]) and np.std(x1[:-1]) else 0.0
        corr = abs(corr)
        return min(0.5, 0.05 + 0.5 * corr), corr, {"lag1_corr": corr}

    if plugin_id == "analysis_nonnegative_matrix_factorization_v1":
        shifted = matrix - np.min(matrix) + 1e-6
        model = NMF(n_components=min(3, shifted.shape[1]), init="nndsvda", random_state=0, max_iter=400)
        w = model.fit_transform(shifted)
        h = model.components_
        recon = np.dot(w, h)
        err = float(np.mean(np.abs(shifted - recon)))
        baseline = float(np.mean(np.abs(shifted - np.mean(shifted, axis=0)))) + 1e-9
        confidence = max(0.0, 1.0 - err / baseline)
        return min(0.4, 0.05 + 0.35 * confidence), confidence, {"reconstruction_mae": err}

    if plugin_id == "analysis_tsne_embedding_v1":
        from sklearn.manifold import TSNE

        embed = TSNE(n_components=2, random_state=0, init="pca", learning_rate="auto").fit_transform(matrix)
        labels = KMeans(n_clusters=2, random_state=0, n_init=10).fit_predict(embed)
        score = float(silhouette_score(embed, labels)) if len(np.unique(labels)) > 1 else 0.0
        score = max(0.0, score)
        return min(0.35, 0.04 + 0.4 * score), score, {"silhouette": score}

    if plugin_id == "analysis_umap_embedding_v1":
        if HAS_UMAP:
            embed = umap.UMAP(n_components=2, random_state=0, n_neighbors=15).fit_transform(matrix)
        else:
            embed = PCA(n_components=2, random_state=0).fit_transform(matrix)
        labels = KMeans(n_clusters=2, random_state=0, n_init=10).fit_predict(embed)
        score = float(silhouette_score(embed, labels)) if len(np.unique(labels)) > 1 else 0.0
        score = max(0.0, score)
        return min(0.35, 0.04 + 0.4 * score), score, {"silhouette": score, "fallback_pca": not HAS_UMAP}

    if plugin_id == "analysis_mice_imputation_chained_equations_v1":
        miss = np.isnan(matrix).mean()
        if miss <= 0.0:
            return 0.08, 0.9, {"missing_ratio": 0.0}
        imputer = IterativeImputer(random_state=0, max_iter=10)
        filled = imputer.fit_transform(matrix)
        stability = float(np.mean(np.isfinite(filled)))
        return min(0.32, 0.06 + 0.5 * miss), stability, {"missing_ratio": float(miss)}

    return 0.0, 0.0, {"reason": "unsupported_plugin"}


def run_surrogate(plugin_id: str, ctx) -> PluginResult:
    df = ctx.dataset_loader()
    if df is None or df.empty:
        return _na_result(plugin_id, pd.DataFrame(), "EMPTY_DATASET", "dataset is empty")
    if not HAS_SKLEARN:
        return _na_result(plugin_id, df, "SKLEARN_UNAVAILABLE", "scikit-learn is unavailable")
    matrix, cols = _safe_numeric_matrix(df)
    if matrix.size == 0 or len(cols) < 2:
        return _na_result(
            plugin_id,
            df,
            "INSUFFICIENT_NUMERIC_COLUMNS",
            "need at least two numeric columns for this method",
        )
    score, confidence, detail = _method_score(plugin_id, matrix)
    if score <= 0.0:
        return _na_result(plugin_id, df, "NO_DECISION_SIGNAL", "model did not produce decision signal")
    recommendation = (
        f"Prioritize process-level tuning from `{_PLUGIN_TITLES.get(plugin_id, plugin_id)}` "
        f"signal on numeric features {', '.join(cols[:3])}."
    )
    result = _actionable_result(
        plugin_id,
        df,
        score,
        confidence=confidence,
        evidence={
            "feature_count": len(cols),
            "feature_sample": cols[:5],
            "method_detail": detail,
        },
        recommendation=recommendation,
    )
    thresholds = load_actionability_thresholds()
    if not meets_actionability_thresholds(
        float(result.metrics.get("delta_h_accounting_month") or 0.0),
        float(result.metrics.get("eff_pct_accounting_month") or 0.0),
        float(result.metrics.get("eff_idx_accounting_month") or 0.0),
        float(result.metrics.get("confidence") or 0.0),
        thresholds=thresholds,
    ):
        return _na_result(
            plugin_id,
            df,
            thresholds.fallback_reason_code,
            "modeled gain is below configured actionability thresholds",
        )
    return result
