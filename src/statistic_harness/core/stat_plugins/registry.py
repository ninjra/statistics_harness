from __future__ import annotations

from importlib import import_module
from typing import Any, Callable, Iterable

import math
import statistics

import numpy as np
import pandas as pd

from statistic_harness.core.close_cycle import resolve_active_close_cycle_mask
from statistic_harness.core.stat_plugins import (
    BudgetTimer,
    bh_fdr,
    build_redactor,
    cramers_v,
    deterministic_sample,
    infer_columns,
    merge_config,
    robust_center_scale,
    robust_zscores,
    stable_id,
    standardized_median_diff,
)
from statistic_harness.core.stat_plugins.references import default_references_for_plugin
from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json
from statistic_harness.core.stat_plugins.topo_tda_addon import (
    HANDLERS as TOPO_TDA_ADDON_HANDLERS,
)
from statistic_harness.core.stat_plugins.ideaspace import (
    HANDLERS as IDEASPACE_HANDLERS,
)
from statistic_harness.core.stat_plugins.erp_next_wave import (
    HANDLERS as ERP_NEXT_WAVE_HANDLERS,
)
from statistic_harness.core.stat_plugins.next30_addon import (
    HANDLERS as NEXT30_HANDLERS,
)
from statistic_harness.core.stat_plugins.next30b_addon import (
    HANDLERS as NEXT30B_HANDLERS,
)

try:  # optional
    from scipy import stats as scipy_stats

    HAS_SCIPY = True
except Exception:  # pragma: no cover - optional
    scipy_stats = None
    HAS_SCIPY = False

try:  # optional
    from sklearn.covariance import LedoitWolf, GraphicalLasso, EmpiricalCovariance
    from sklearn.decomposition import PCA, LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import pairwise_distances
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM
    from sklearn.cluster import KMeans

    HAS_SKLEARN = True
except Exception:  # pragma: no cover - optional
    LedoitWolf = GraphicalLasso = EmpiricalCovariance = PCA = None
    LatentDirichletAllocation = None
    CountVectorizer = None
    pairwise_distances = None
    LocalOutlierFactor = OneClassSVM = None
    KMeans = None
    HAS_SKLEARN = False


ALIAS_MAP = {
    "analysis_isolation_forest": "analysis_isolation_forest_anomaly",
    "analysis_log_template_drain": "analysis_log_template_mining_drain",
    "analysis_robust_pca_pcp": "analysis_robust_pca_sparse_outliers",
}

_BUDGET_KEYS_TO_CONFIG = {
    # Only merge non-dimension-changing keys by default. Plugins can still read
    # ctx.budget directly for optional caps.
    "row_limit": "row_limit",
    "batch_size": "batch_size",
}


def _merge_budget_into_config(config: dict[str, Any], budget: dict[str, Any] | None) -> dict[str, Any]:
    """Allow harness-level caps (ctx.budget) to flow into stat-plugin configs.

    Stat plugins historically read caps from settings-derived `config`. The harness now
    injects large-dataset caps into `ctx.budget` (so dataset_loader can apply them even
    when plugins don't set settings). This helper bridges that gap without overriding
    explicit per-plugin settings.
    """

    if not budget:
        return config
    merged = dict(config)
    for bkey, ckey in _BUDGET_KEYS_TO_CONFIG.items():
        if ckey in merged and merged.get(ckey) is not None:
            continue
        val = budget.get(bkey)
        if val is None:
            continue
        merged[ckey] = val

    # Timer uses time_budget_ms; harness injects time_limit_ms in budgets.
    if merged.get("time_budget_ms") is None:
        tl = budget.get("time_limit_ms")
        if isinstance(tl, (int, float)) and tl:
            merged["time_budget_ms"] = int(tl)
    return merged


def run_plugin(plugin_id: str, ctx) -> PluginResult:
    if plugin_id in ALIAS_MAP:
        return _delegate_alias(ALIAS_MAP[plugin_id], ctx)

    # Merge harness-level caps into settings-derived config for stat plugin handlers.
    config = _merge_budget_into_config(merge_config(ctx.settings), getattr(ctx, "budget", None))
    # Determinism contract: default all plugin randomness to the per-run seed unless
    # the caller explicitly set `seed` in settings.
    if isinstance(getattr(ctx, "settings", None), dict) and "seed" not in ctx.settings:
        config["seed"] = int(getattr(ctx, "run_seed", config.get("seed", 1337)) or 0)
    timer = BudgetTimer(config.get("time_budget_ms"))

    df = ctx.dataset_loader()
    if df is None or df.empty:
        return PluginResult(
            status="skipped",
            summary="Empty dataset",
            metrics={"rows_seen": 0, "rows_used": 0, "cols_used": 0},
            findings=[],
            artifacts=[],
            references=default_references_for_plugin(plugin_id),
            debug={"warnings": ["empty_dataset"]},
        )

    active_row_filter_meta = {
        "enabled": bool(config.get("use_active_close_window_filter", True)),
        "applied": False,
        "source_plugin": None,
        "rows_before": int(len(df)),
        "rows_after": int(len(df)),
    }
    if active_row_filter_meta["enabled"]:
        time_col_hint = str(config.get("time_column", "auto") or "auto")
        if time_col_hint.lower() != "auto" and time_col_hint in df.columns:
            active_time_col = time_col_hint
        else:
            inferred_time = infer_columns(df, config).get("time_column")
            active_time_col = str(inferred_time) if inferred_time in df.columns else None

        if active_time_col:
            mask, used_dynamic, source_plugin, windows = resolve_active_close_cycle_mask(
                df[active_time_col], ctx.run_dir
            )
            if used_dynamic and mask is not None:
                mask_series = pd.Series(mask, index=df.index).fillna(False)
                narrowed = df.loc[mask_series]
                if not narrowed.empty:
                    df = narrowed
                    active_row_filter_meta.update(
                        {
                            "applied": True,
                            "source_plugin": source_plugin,
                            "time_column": active_time_col,
                            "rows_after": int(len(df)),
                            "windows_detected": int(len(windows)),
                        }
                    )
                else:
                    active_row_filter_meta.update(
                        {
                            "source_plugin": source_plugin,
                            "time_column": active_time_col,
                            "windows_detected": int(len(windows)),
                            "fallback_reason": "dynamic_mask_empty",
                        }
                    )
            elif source_plugin:
                active_row_filter_meta.update(
                    {
                        "source_plugin": source_plugin,
                        "time_column": active_time_col,
                        "fallback_reason": "dynamic_windows_unusable",
                    }
                )
        else:
            active_row_filter_meta["fallback_reason"] = "time_column_not_found"
    else:
        active_row_filter_meta["fallback_reason"] = "disabled_by_config"

    max_rows = config.get("max_rows")
    allow_sampling = bool(config.get("allow_row_sampling", False))
    if not allow_sampling:
        max_rows = None
    df, sample_meta = deterministic_sample(df, max_rows, seed=int(config.get("seed", 1337)))
    inferred = infer_columns(df, config)
    handler = HANDLERS.get(plugin_id)
    if handler is None:
        return PluginResult(
            status="error",
            summary=f"Plugin handler missing for {plugin_id}",
            metrics={"rows_seen": int(len(df)), "rows_used": int(len(df)), "cols_used": int(len(df.columns))},
            findings=[],
            artifacts=[],
            references=default_references_for_plugin(plugin_id),
            debug={"warnings": ["missing_handler"]},
        )

    try:
        result = handler(plugin_id, ctx, df, config, inferred, timer, sample_meta)
    except Exception as exc:  # pragma: no cover - runtime guard
        if "time_budget_exceeded" in str(exc):
            return PluginResult(
                status="na",
                summary="Not applicable: time_budget_exceeded",
                metrics={"rows_seen": int(len(df)), "rows_used": int(len(df)), "cols_used": int(len(df.columns))},
                findings=[
                    {
                        "kind": "plugin_not_applicable",
                        "reason_code": "TIME_BUDGET_EXCEEDED",
                        "reason": "time_budget_exceeded",
                        "recommended_next_step": (
                            "Increase plugin time budget or enable deterministic sampling/downsampling "
                            "for this plugin path."
                        ),
                    }
                ],
                artifacts=[],
                references=default_references_for_plugin(plugin_id),
                debug={
                    "warnings": ["exception"],
                    "exception": str(exc),
                    "gating_reason": "time_budget_exceeded",
                },
            )
        return PluginResult(
            status="error",
            summary=f"{plugin_id} failed",
            metrics={"rows_seen": int(len(df)), "rows_used": int(len(df)), "cols_used": int(len(df.columns))},
            findings=[],
            artifacts=[],
            references=default_references_for_plugin(plugin_id),
            debug={"warnings": ["exception"], "exception": str(exc)},
        )

    if result.references is None or len(result.references) == 0:
        result.references = default_references_for_plugin(plugin_id)
    result.debug = result.debug or {}
    result.debug.setdefault("column_inference", inferred)
    result.debug.setdefault("sample", sample_meta)
    result.debug.setdefault("active_row_filter", active_row_filter_meta)
    # Ensure report outputs can cite the effective harness budget/caps.
    try:
        budget = dict(getattr(ctx, "budget", None) or {})
    except Exception:
        budget = {}
    if budget:
        result.budget = {**budget, **(result.budget or {})}
    if result.status in {"skipped", "degraded"}:
        # Cross-cutting contract: when a plugin is not fully applicable, callers
        # should have a structured reason to display without parsing the summary.
        result.debug.setdefault("gating_reason", result.summary)
    return result


def _delegate_alias(base_id: str, ctx) -> PluginResult:
    module = import_module(f"plugins.{base_id}.plugin")
    plugin = module.Plugin()
    return plugin.run(ctx)


def _artifact(ctx, plugin_id: str, name: str, payload: Any, kind: str) -> PluginArtifact:
    artifact_dir = ctx.artifacts_dir(plugin_id)
    path = artifact_dir / name
    write_json(path, payload)
    return PluginArtifact(
        path=str(path.relative_to(ctx.run_dir)),
        type=kind,
        description=name,
    )


def _group_slices(
    df: pd.DataFrame, group_cols: list[str], max_groups: int
) -> list[tuple[str, pd.Index, dict[str, Any]]]:
    slices: list[tuple[str, pd.Index, dict[str, Any]]] = [("ALL", df.index, {})]
    for col in group_cols:
        if col not in df.columns:
            continue
        counts = df[col].value_counts(dropna=False)
        for value in counts.index[:max_groups]:
            label = f"{col}={value}"
            row_idx = df.index[df[col].eq(value)]
            if row_idx.empty:
                continue
            slices.append((label, row_idx, {"group": {col: value}}))
    return slices


def _sort_by_time(df: pd.DataFrame, time_col: str | None) -> pd.DataFrame:
    if not time_col or time_col not in df.columns:
        return df
    series = pd.to_datetime(df[time_col], errors="coerce")
    if series.notna().any():
        return df.assign(_time_sort=series).sort_values("_time_sort").drop(columns=["_time_sort"])
    return df


def _numeric_matrix(
    df: pd.DataFrame, numeric_cols: list[str], max_cols: int
) -> tuple[np.ndarray, list[str]]:
    cols = [col for col in numeric_cols if col in df.columns][: max_cols]
    if not cols:
        return np.empty((0, 0)), []
    frame = df[cols].astype(float)
    medians = frame.median()
    frame = frame.fillna(medians)
    matrix = frame.to_numpy(dtype=float)
    for idx in range(matrix.shape[1]):
        center, scale = robust_center_scale(matrix[:, idx])
        if scale <= 0:
            scale = 1.0
        matrix[:, idx] = (matrix[:, idx] - center) / scale
    return matrix, cols


def _top_numeric_columns(df: pd.DataFrame, numeric_cols: list[str], max_cols: int) -> list[str]:
    cols = [col for col in numeric_cols if col in df.columns]
    if not cols:
        return []
    variances = []
    for col in cols:
        series = df[col].dropna().to_numpy(dtype=float)
        if series.size == 0:
            variances.append(0.0)
        else:
            variances.append(float(np.nanvar(series)))
    order = np.argsort(variances)[::-1]
    return [cols[idx] for idx in order[:max_cols]]


def _safe_exemplars(
    series: pd.Series, privacy: dict[str, Any], max_exemplars: int
) -> list[str]:
    if series.empty:
        return []
    samples = series.dropna().astype(str).head(max_exemplars).tolist()
    redactor = build_redactor(privacy)
    return [redactor(value) for value in samples]


def _make_finding(
    plugin_id: str,
    key: str,
    title: str,
    what: str,
    why: str,
    evidence: dict[str, Any],
    where: dict[str, Any] | None = None,
    recommendation: str | None = None,
    severity: str = "info",
    confidence: float = 0.5,
    measurement_type: str = "measured",
    scope: dict[str, Any] | None = None,
    assumptions: list[str] | None = None,
) -> dict[str, Any]:
    finding = {
        "id": stable_id(f"{plugin_id}:{key}"),
        "severity": severity,
        "confidence": float(max(0.0, min(1.0, confidence))),
        "title": title,
        "what": what,
        "why": why,
        "evidence": evidence,
        "where": where or {},
        "recommendation": recommendation or "",
        "measurement_type": measurement_type,
    }
    if measurement_type == "modeled":
        finding["scope"] = scope or {"dataset": "full"}
        finding["assumptions"] = assumptions or ["modeled using simplified assumptions"]
    return finding


def _basic_metrics(df: pd.DataFrame, sample_meta: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        "rows_seen": int(sample_meta.get("rows_total", len(df))),
        "rows_used": int(sample_meta.get("rows_used", len(df))),
        "cols_used": int(len(df.columns)),
    }
    metrics.update(sample_meta)
    return metrics


def _summary_or_skip(summary: str, findings: list[dict[str, Any]]) -> str:
    if not findings:
        return summary or "No findings"
    return summary


def _effect_size(left: Iterable[float], right: Iterable[float]) -> float:
    return standardized_median_diff(left, right)


def _two_sample_numeric(
    left: np.ndarray,
    right: np.ndarray,
    test: str,
) -> tuple[float, float]:
    left = left[~np.isnan(left)]
    right = right[~np.isnan(right)]
    if left.size == 0 or right.size == 0:
        return 1.0, 0.0
    if test == "ks" and HAS_SCIPY:
        stat, p = scipy_stats.ks_2samp(left, right, alternative="two-sided", mode="auto")
        return float(p), float(stat)
    if test == "ad" and HAS_SCIPY:
        stat, crit, p = scipy_stats.anderson_ksamp([left, right])
        p = p / 100.0 if p > 1 else p
        return float(p), float(stat)
    if test == "mw" and HAS_SCIPY:
        stat, p = scipy_stats.mannwhitneyu(left, right, alternative="two-sided")
        return float(p), float(stat)
    # fallback: compare medians
    diff = float(np.median(left) - np.median(right))
    return 1.0, diff


def _wrap_two_sample(test_name: str) -> Callable[..., PluginResult]:
    def _handler(
        plugin_id: str,
        ctx,
        df: pd.DataFrame,
        config: dict[str, Any],
        inferred: dict[str, Any],
        timer: BudgetTimer,
        sample_meta: dict[str, Any],
    ) -> PluginResult:
        return _two_sample_numeric_plugin(
            plugin_id, test_name, ctx, df, config, inferred, timer, sample_meta
        )

    return _handler


def _local_outlier_factor(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    numeric_cols = inferred.get("numeric_columns") or []
    if not numeric_cols:
        return PluginResult("skipped", "No numeric columns detected", _basic_metrics(df, sample_meta), [], [], None)
    X, cols = _numeric_matrix(df, numeric_cols, max_cols=int(config.get("max_cols", 20)))
    if X.size == 0 or X.shape[0] < 20:
        return PluginResult("skipped", "Insufficient numeric data", _basic_metrics(df, sample_meta), [], [], None)
    max_findings = int(config.get("max_findings", 30))
    max_rows_for_lof = int(config.get("max_rows_for_lof", 50000))
    use_lof = (
        HAS_SKLEARN
        and LocalOutlierFactor is not None
        and int(X.shape[0]) <= max_rows_for_lof
    )
    debug: dict[str, Any] = {"rows": int(X.shape[0]), "cols": int(X.shape[1])}
    if use_lof:
        try:
            lof = LocalOutlierFactor(n_neighbors=min(20, X.shape[0] - 1), contamination="auto")
            labels = lof.fit_predict(X)
            scores = -lof.negative_outlier_factor_
            debug["model_path"] = "lof"
        except Exception:
            zscores = np.abs(robust_zscores(X[:, 0]))
            labels = np.where(zscores > 3.5, -1, 1)
            scores = zscores
            debug["model_path"] = "robust_z_fallback_exception"
    else:
        zscores = np.abs(robust_zscores(X[:, 0]))
        labels = np.where(zscores > 3.5, -1, 1)
        scores = zscores
        debug["model_path"] = "robust_z_fallback_large_n"
    outlier_idx = np.where(labels == -1)[0]
    top_idx = outlier_idx[np.argsort(scores[outlier_idx])[::-1][:max_findings]] if outlier_idx.size else np.array([], dtype=int)
    artifact_rows = [
        {"row_index": int(idx), "score": float(scores[idx])} for idx in top_idx
    ]
    findings = []
    if outlier_idx.size:
        evidence = {"metrics": {"outliers": int(outlier_idx.size), "top_score": float(scores[top_idx[0]]) if top_idx.size else None}}
        findings.append(
            _make_finding(
                plugin_id,
                "lof",
                "Local outlier factor anomalies",
                "Density-based outliers detected.",
                "LOF flagged points with low local density.",
                evidence,
                severity="warn",
                confidence=min(1.0, float(np.max(scores[outlier_idx])) / 10.0) if outlier_idx.size else 0.5,
            )
        )
    artifacts = []
    if artifact_rows:
        artifacts.append(_artifact(ctx, plugin_id, "lof_outliers.json", artifact_rows, "json"))
    summary = _summary_or_skip("LOF analysis complete", findings)
    return PluginResult("ok", summary, _basic_metrics(df, sample_meta), findings, artifacts, None, debug=debug)


def _one_class_svm_plugin(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    numeric_cols = inferred.get("numeric_columns") or []
    if not numeric_cols:
        return PluginResult("skipped", "No numeric columns detected", _basic_metrics(df, sample_meta), [], [], None)
    X, cols = _numeric_matrix(df, numeric_cols, max_cols=int(config.get("max_cols", 20)))
    if X.size == 0 or X.shape[0] < 20:
        return PluginResult("skipped", "Insufficient numeric data", _basic_metrics(df, sample_meta), [], [], None)
    # OCSVM can explode in CPU/RAM on large N due kernel complexity.
    # For large matrices, use deterministic robust-z fallback to keep the full gauntlet stable.
    max_rows_for_ocsvm = int(config.get("max_rows_for_ocsvm", 50000))
    use_ocsvm = (
        HAS_SKLEARN
        and OneClassSVM is not None
        and int(X.shape[0]) <= max_rows_for_ocsvm
    )
    debug: dict[str, Any] = {"rows": int(X.shape[0]), "cols": int(X.shape[1])}
    if use_ocsvm:
        try:
            svm = OneClassSVM(gamma="scale", nu=0.05)
            labels = svm.fit_predict(X)
            scores = -svm.score_samples(X)
            debug["model_path"] = "ocsvm"
        except Exception:
            zscores = np.abs(robust_zscores(X[:, 0]))
            labels = np.where(zscores > 3.5, -1, 1)
            scores = zscores
            debug["model_path"] = "robust_z_fallback_exception"
    else:
        zscores = np.abs(robust_zscores(X[:, 0]))
        labels = np.where(zscores > 3.5, -1, 1)
        scores = zscores
        debug["model_path"] = "robust_z_fallback_large_n"
    outlier_idx = np.where(labels == -1)[0]
    max_findings = int(config.get("max_findings", 30))
    top_idx = outlier_idx[np.argsort(scores[outlier_idx])[::-1][:max_findings]] if outlier_idx.size else np.array([], dtype=int)
    artifact_rows = [{"row_index": int(idx), "score": float(scores[idx])} for idx in top_idx]
    findings = []
    if outlier_idx.size:
        evidence = {"metrics": {"outliers": int(outlier_idx.size), "top_score": float(scores[top_idx[0]]) if top_idx.size else None}}
        findings.append(
            _make_finding(
                plugin_id,
                "ocsvm",
                "One-class SVM anomalies",
                "Model-based outliers detected.",
                "One-class SVM flagged points outside learned boundary.",
                evidence,
                severity="warn",
                confidence=min(1.0, float(np.max(scores[outlier_idx])) / 10.0),
            )
        )
    artifacts = []
    if artifact_rows:
        artifacts.append(_artifact(ctx, plugin_id, "ocsvm_outliers.json", artifact_rows, "json"))
    summary = _summary_or_skip("One-class SVM analysis complete", findings)
    return PluginResult("ok", summary, _basic_metrics(df, sample_meta), findings, artifacts, None, debug=debug)


def _robust_covariance_outliers(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    numeric_cols = inferred.get("numeric_columns") or []
    if not numeric_cols:
        return PluginResult("skipped", "No numeric columns detected", _basic_metrics(df, sample_meta), [], [], None)
    X, cols = _numeric_matrix(df, numeric_cols, max_cols=int(config.get("max_cols", 10)))
    if X.size == 0 or X.shape[0] < 20:
        return PluginResult("skipped", "Insufficient numeric data", _basic_metrics(df, sample_meta), [], [], None)
    if HAS_SKLEARN and EmpiricalCovariance is not None:
        cov = EmpiricalCovariance().fit(X)
        scores = cov.mahalanobis(X)
    else:
        scores = np.sum(X**2, axis=1)
    threshold = np.quantile(scores, 0.99) if scores.size else 0.0
    outlier_idx = np.where(scores >= threshold)[0]
    max_findings = int(config.get("max_findings", 30))
    top_idx = outlier_idx[np.argsort(scores[outlier_idx])[::-1][:max_findings]] if outlier_idx.size else np.array([], dtype=int)
    artifact_rows = [{"row_index": int(idx), "score": float(scores[idx])} for idx in top_idx]
    findings = []
    if outlier_idx.size:
        evidence = {"metrics": {"outliers": int(outlier_idx.size), "threshold": float(threshold)}}
        findings.append(
            _make_finding(
                plugin_id,
                "robust_cov",
                "Robust covariance outliers",
                "Mahalanobis-distance outliers detected.",
                "Points exceed robust covariance distance threshold.",
                evidence,
                severity="warn",
                confidence=min(1.0, float(np.max(scores[outlier_idx])) / max(threshold, 1e-6)),
            )
        )
    artifacts = []
    if artifact_rows:
        artifacts.append(_artifact(ctx, plugin_id, "robust_cov_outliers.json", artifact_rows, "json"))
    summary = _summary_or_skip("Robust covariance analysis complete", findings)
    return PluginResult("ok", summary, _basic_metrics(df, sample_meta), findings, artifacts, None)


def _evt_gumbel_tail(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    numeric_cols = _top_numeric_columns(df, inferred.get("numeric_columns") or [], max_cols=5)
    if not numeric_cols or not HAS_SCIPY:
        return PluginResult("skipped", "No numeric columns or scipy missing", _basic_metrics(df, sample_meta), [], [], None)
    findings: list[dict[str, Any]] = []
    for col in numeric_cols:
        series = df[col].dropna().to_numpy(dtype=float)
        if series.size < 50:
            continue
        block_size = max(5, int(series.size / 20))
        blocks = [series[i:i + block_size] for i in range(0, series.size, block_size)]
        maxima = np.array([np.max(block) for block in blocks if block.size > 0])
        if maxima.size < 5:
            continue
        loc, scale = float(np.mean(maxima)), float(np.std(maxima) or 1.0)
        threshold = float(np.quantile(series, 0.99))
        exceed_prob = float(np.mean(maxima > threshold))
        evidence = {"metrics": {"block_max_mean": loc, "block_max_std": scale, "threshold": threshold, "exceed_prob": exceed_prob}}
        findings.append(
            _make_finding(
                plugin_id,
                f"{col}:gumbel",
                f"Tail risk in {col}",
                "Block maxima show heavy tail risk.",
                "Gumbel-style tail estimate indicates exceedances.",
                evidence,
                where={"column": col},
                severity="warn" if exceed_prob > 0.2 else "info",
                confidence=min(1.0, exceed_prob * 2.0),
            )
        )
    artifacts = []
    if findings:
        artifacts.append(_artifact(ctx, plugin_id, "evt_gumbel_tail.json", findings, "json"))
    summary = _summary_or_skip("EVT Gumbel tail analysis complete", findings)
    return PluginResult("ok", summary, _basic_metrics(df, sample_meta), findings, artifacts, None)


def _evt_peaks_over_threshold(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    numeric_cols = _top_numeric_columns(df, inferred.get("numeric_columns") or [], max_cols=5)
    if not numeric_cols or not HAS_SCIPY:
        return PluginResult("skipped", "No numeric columns or scipy missing", _basic_metrics(df, sample_meta), [], [], None)
    findings: list[dict[str, Any]] = []
    for col in numeric_cols:
        series = df[col].dropna().to_numpy(dtype=float)
        if series.size < 50:
            continue
        threshold = float(np.quantile(series, 0.95))
        exceed = series[series > threshold] - threshold
        if exceed.size < 5:
            continue
        shape, loc, scale = scipy_stats.genpareto.fit(exceed, floc=0)
        tail_prob = float(exceed.size / max(series.size, 1))
        evidence = {"metrics": {"threshold": threshold, "shape": float(shape), "scale": float(scale), "exceed_rate": tail_prob}}
        findings.append(
            _make_finding(
                plugin_id,
                f"{col}:pot",
                f"Peaks-over-threshold tail in {col}",
                "Exceedances above high threshold are frequent.",
                "GPD fit indicates tail heaviness.",
                evidence,
                where={"column": col},
                severity="warn" if tail_prob > 0.1 else "info",
                confidence=min(1.0, tail_prob * 2.0),
            )
        )
    artifacts = []
    if findings:
        artifacts.append(_artifact(ctx, plugin_id, "evt_pot.json", findings, "json"))
    summary = _summary_or_skip("EVT POT analysis complete", findings)
    return PluginResult("ok", summary, _basic_metrics(df, sample_meta), findings, artifacts, None)


def _matrix_profile_discords(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    numeric_cols = _top_numeric_columns(df, inferred.get("numeric_columns") or [], max_cols=1)
    if not numeric_cols:
        return PluginResult("skipped", "No numeric columns detected", _basic_metrics(df, sample_meta), [], [], None)
    series = df[numeric_cols[0]].dropna().to_numpy(dtype=float)
    if series.size < 50:
        return PluginResult("skipped", "Insufficient series length", _basic_metrics(df, sample_meta), [], [], None)
    max_points = min(series.size, 2000)
    series = series[:max_points]
    m = max(10, int(len(series) * 0.05))
    if len(series) <= m * 2:
        return PluginResult("skipped", "Series too short for motifs", _basic_metrics(df, sample_meta), [], [], None)
    windows = np.lib.stride_tricks.sliding_window_view(series, m)
    window_norm = (windows - windows.mean(axis=1, keepdims=True)) / (windows.std(axis=1, keepdims=True) + 1e-6)
    # naive nearest-neighbor distance
    distances = []
    for idx in range(len(window_norm)):
        if timer.exceeded():
            break
        dists = np.linalg.norm(window_norm[idx] - window_norm, axis=1)
        dists[idx] = np.inf
        distances.append(float(np.min(dists)))
    if not distances:
        return PluginResult("skipped", "No motif distances computed", _basic_metrics(df, sample_meta), [], [], None)
    discord_idx = int(np.argmax(distances))
    evidence = {"metrics": {"discord_distance": float(np.max(distances)), "window": m, "index": discord_idx}}
    finding = _make_finding(
        plugin_id,
        f"{numeric_cols[0]}:{discord_idx}",
        f"Discord in {numeric_cols[0]}",
        "A subsequence is dissimilar to all others.",
        "Matrix-profile-style nearest-neighbor distance is maximal.",
        evidence,
        where={"column": numeric_cols[0]},
        severity="warn",
        confidence=min(1.0, float(np.max(distances)) / 10.0),
    )
    artifacts = [_artifact(ctx, plugin_id, "discords.json", evidence, "json")]
    summary = "Matrix profile scan complete"
    return PluginResult("ok", summary, _basic_metrics(df, sample_meta), [finding], artifacts, None)


def _burst_detection(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
    term_mode: bool = False,
) -> PluginResult:
    time_col = inferred.get("time_column")
    text_cols = inferred.get("text_columns") or []
    if not time_col or not text_cols:
        return PluginResult("skipped", "Need time + text columns", _basic_metrics(df, sample_meta), [], [], None)
    df = _sort_by_time(df, time_col)
    bucket = pd.to_datetime(df[time_col], errors="coerce").dt.floor("D")
    if bucket.isna().all():
        return PluginResult("skipped", "Time column unparseable", _basic_metrics(df, sample_meta), [], [], None)
    df = df.assign(_bucket=bucket)
    privacy = config.get("privacy", {})
    max_ex = int(privacy.get("max_exemplars", 3))
    findings: list[dict[str, Any]] = []
    for col in text_cols[:1]:
        counts = df.groupby("_bucket")[col].count()
        if counts.empty:
            continue
        mean = float(counts.mean())
        std = float(counts.std() or 1.0)
        burst_buckets = counts[counts > mean + 2.0 * std]
        if burst_buckets.empty:
            continue
        top_bucket = burst_buckets.index[0]
        evidence = {"metrics": {"mean": mean, "std": std, "burst_count": int(burst_buckets.iloc[0])}}
        exemplars = _safe_exemplars(df.loc[df["_bucket"] == top_bucket, col], privacy, max_ex)
        if exemplars:
            evidence["exemplars"] = exemplars
        findings.append(
            _make_finding(
                plugin_id,
                f"{col}:{top_bucket}",
                f"Burst detected in {col}",
                "Event volume spikes above baseline.",
                "Counts exceed mean + 2*std.",
                evidence,
                where={"column": col, "bucket": str(top_bucket)},
                severity="warn",
                confidence=min(1.0, (burst_buckets.iloc[0] - mean) / (std + 1e-6)),
            )
        )
        if term_mode:
            break
    artifacts = []
    if findings:
        artifacts.append(_artifact(ctx, plugin_id, "burst_results.json", findings, "json"))
    summary = _summary_or_skip("Burst detection complete", findings)
    return PluginResult("ok", summary, _basic_metrics(df, sample_meta), findings, artifacts, None)


def _event_count_bocpd_poisson(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    time_col = inferred.get("time_column")
    if not time_col:
        return PluginResult("skipped", "No time column detected", _basic_metrics(df, sample_meta), [], [], None)
    df = _sort_by_time(df, time_col)
    bucket = pd.to_datetime(df[time_col], errors="coerce").dt.floor("D")
    df = df.assign(_bucket=bucket)
    counts = df.groupby("_bucket").size()
    if counts.size < 10:
        return PluginResult("skipped", "Insufficient time buckets", _basic_metrics(df, sample_meta), [], [], None)
    rolling = counts.rolling(window=5, min_periods=3).mean()
    diff = rolling.diff().abs()
    idx = int(np.argmax(diff.fillna(0.0)))
    change_bucket = counts.index[idx]
    evidence = {"metrics": {"rolling_mean_delta": float(diff.iloc[idx]), "bucket": str(change_bucket)}}
    finding = _make_finding(
        plugin_id,
        f"bocpd:{change_bucket}",
        "Poisson count change detected",
        "Event counts shift in time.",
        "BOCPD-style rolling mean change.",
        evidence,
        where={"bucket": str(change_bucket)},
        severity="warn",
        confidence=min(1.0, float(diff.iloc[idx]) / max(counts.mean(), 1e-6)),
    )
    artifacts = [_artifact(ctx, plugin_id, "bocpd_poisson.json", evidence, "json")]
    return PluginResult("ok", "BOCPD Poisson scan complete", _basic_metrics(df, sample_meta), [finding], artifacts, None)


def _hawkes_self_exciting(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    time_col = inferred.get("time_column")
    if not time_col:
        return PluginResult("skipped", "No time column detected", _basic_metrics(df, sample_meta), [], [], None)
    times = pd.to_datetime(df[time_col], errors="coerce").dropna().sort_values()
    if times.size < 50:
        return PluginResult("skipped", "Insufficient events", _basic_metrics(df, sample_meta), [], [], None)
    deltas = times.diff().dropna().dt.total_seconds().to_numpy()
    if deltas.size == 0:
        return PluginResult("skipped", "No inter-arrival times", _basic_metrics(df, sample_meta), [], [], None)
    mean = float(np.mean(deltas))
    std = float(np.std(deltas))
    cv = float(std / (mean + 1e-6))
    evidence = {"metrics": {"mean_interarrival": mean, "std_interarrival": std, "cv": cv}}
    finding = _make_finding(
        plugin_id,
        "hawkes",
        "Self-exciting behavior",
        "Inter-arrival times show clustering.",
        "Coefficient of variation above 1 suggests self-excitation.",
        evidence,
        severity="warn" if cv > 1.2 else "info",
        confidence=min(1.0, max(cv - 1.0, 0.0)),
    )
    artifacts = [_artifact(ctx, plugin_id, "hawkes_metrics.json", evidence, "json")]
    return PluginResult("ok", "Hawkes-style clustering check complete", _basic_metrics(df, sample_meta), [finding], artifacts, None)


def _periodicity_spectral_scan(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    time_col = inferred.get("time_column")
    if not time_col:
        return PluginResult("skipped", "No time column detected", _basic_metrics(df, sample_meta), [], [], None)
    df = _sort_by_time(df, time_col)
    bucket = pd.to_datetime(df[time_col], errors="coerce").dt.floor("D")
    counts = bucket.value_counts().sort_index()
    if counts.size < 10:
        return PluginResult("skipped", "Insufficient time buckets", _basic_metrics(df, sample_meta), [], [], None)
    series = counts.to_numpy(dtype=float)
    spectrum = np.abs(np.fft.rfft(series - np.mean(series)))
    if spectrum.size <= 1:
        return PluginResult("skipped", "No spectral components", _basic_metrics(df, sample_meta), [], [], None)
    peak_idx = int(np.argmax(spectrum[1:]) + 1)
    peak = float(spectrum[peak_idx])
    median = float(np.median(spectrum[1:]))
    period = float(len(series) / peak_idx) if peak_idx > 0 else None
    evidence = {"metrics": {"peak_power": peak, "median_power": median, "period_estimate": period}}
    finding = _make_finding(
        plugin_id,
        "spectral",
        "Periodicity detected",
        "Spectral scan shows a dominant frequency.",
        "FFT peak significantly above median power.",
        evidence,
        severity="info" if peak < 3 * median else "warn",
        confidence=min(1.0, (peak / (median + 1e-6)) / 5.0),
    )
    artifacts = [_artifact(ctx, plugin_id, "spectral_scan.json", evidence, "json")]
    return PluginResult("ok", "Spectral periodicity scan complete", _basic_metrics(df, sample_meta), [finding], artifacts, None)


def _kalman_residuals(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    numeric_cols = _top_numeric_columns(df, inferred.get("numeric_columns") or [], max_cols=1)
    if not numeric_cols:
        return PluginResult("skipped", "No numeric columns detected", _basic_metrics(df, sample_meta), [], [], None)
    series = df[numeric_cols[0]].dropna().to_numpy(dtype=float)
    if series.size < 50:
        return PluginResult("skipped", "Insufficient series length", _basic_metrics(df, sample_meta), [], [], None)
    q = 1e-3
    r = np.var(series) * 0.1 if np.var(series) > 0 else 1.0
    x = series[0]
    p = 1.0
    residuals = []
    for value in series:
        # predict
        p = p + q
        # update
        k = p / (p + r)
        residual = value - x
        x = x + k * residual
        p = (1 - k) * p
        residuals.append(residual)
    residuals = np.array(residuals)
    zscores = robust_zscores(residuals)
    idx = int(np.argmax(np.abs(zscores)))
    evidence = {"metrics": {"max_residual_z": float(zscores[idx]), "index": idx}}
    finding = _make_finding(
        plugin_id,
        f"{numeric_cols[0]}:{idx}",
        f"Kalman residual spike in {numeric_cols[0]}",
        "State-space residuals show spike.",
        "Kalman filter residual z-score exceeds baseline.",
        evidence,
        where={"column": numeric_cols[0]},
        severity="warn" if abs(zscores[idx]) > 3.5 else "info",
        confidence=min(1.0, abs(float(zscores[idx])) / 5.0),
    )
    artifacts = [_artifact(ctx, plugin_id, "kalman_residuals.json", evidence, "json")]
    return PluginResult("ok", "Kalman residual scan complete", _basic_metrics(df, sample_meta), [finding], artifacts, None)


def _covariance_matrix(X: np.ndarray) -> np.ndarray:
    if X.size == 0:
        return np.eye(1)
    if HAS_SKLEARN and LedoitWolf is not None:
        return LedoitWolf().fit(X).covariance_
    return np.cov(X, rowvar=False) + np.eye(X.shape[1]) * 1e-6


def _multivariate_t2_control(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    numeric_cols = inferred.get("numeric_columns") or []
    X, cols = _numeric_matrix(df, numeric_cols, max_cols=int(config.get("max_cols", 20)))
    if X.size == 0 or X.shape[0] < 50:
        return PluginResult("skipped", "Insufficient numeric data", _basic_metrics(df, sample_meta), [], [], None)
    mu = np.mean(X, axis=0)
    cov = _covariance_matrix(X)
    inv = np.linalg.pinv(cov)
    diff = X - mu
    t2 = np.einsum("ij,jk,ik->i", diff, inv, diff)
    threshold = float(np.quantile(t2, 0.995))
    idx = int(np.argmax(t2))
    evidence = {"metrics": {"max_t2": float(t2[idx]), "threshold": threshold, "columns": cols[:5]}}
    finding = _make_finding(
        plugin_id,
        f"t2:{idx}",
        "Hotelling T² spike",
        "Multivariate deviation detected.",
        "Hotelling T² exceeds threshold.",
        evidence,
        severity="warn" if t2[idx] > threshold else "info",
        confidence=min(1.0, float(t2[idx]) / max(threshold, 1e-6)),
    )
    artifacts = [_artifact(ctx, plugin_id, "t2_control.json", evidence, "json")]
    return PluginResult("ok", "Multivariate T² control complete", _basic_metrics(df, sample_meta), [finding], artifacts, None)


def _multivariate_ewma_control(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    numeric_cols = inferred.get("numeric_columns") or []
    X, cols = _numeric_matrix(df, numeric_cols, max_cols=int(config.get("max_cols", 20)))
    if X.size == 0 or X.shape[0] < 50:
        return PluginResult("skipped", "Insufficient numeric data", _basic_metrics(df, sample_meta), [], [], None)
    lam = float(config.get("mv_control", {}).get("mewma_lambda", 0.2))
    mu = np.mean(X, axis=0)
    z = np.zeros_like(mu)
    cov = _covariance_matrix(X)
    inv = np.linalg.pinv(cov)
    stats = []
    for row in X:
        z = lam * row + (1 - lam) * z
        diff = z - mu
        stats.append(float(diff.T @ inv @ diff))
    stats_arr = np.asarray(stats)
    threshold = float(np.quantile(stats_arr, 0.995))
    idx = int(np.argmax(stats_arr))
    evidence = {"metrics": {"max_mewma": float(stats_arr[idx]), "threshold": threshold}}
    finding = _make_finding(
        plugin_id,
        f"mewma:{idx}",
        "MEWMA multivariate drift",
        "MEWMA statistic exceeds baseline.",
        "Multivariate EWMA indicates drift.",
        evidence,
        severity="warn" if stats_arr[idx] > threshold else "info",
        confidence=min(1.0, float(stats_arr[idx]) / max(threshold, 1e-6)),
    )
    artifacts = [_artifact(ctx, plugin_id, "mewma_control.json", evidence, "json")]
    return PluginResult("ok", "Multivariate MEWMA control complete", _basic_metrics(df, sample_meta), [finding], artifacts, None)


def _pca_control_chart(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    numeric_cols = inferred.get("numeric_columns") or []
    X, cols = _numeric_matrix(df, numeric_cols, max_cols=int(config.get("max_cols", 20)))
    if X.size == 0 or X.shape[0] < 50:
        return PluginResult("skipped", "Insufficient numeric data", _basic_metrics(df, sample_meta), [], [], None)
    n = X.shape[0]
    baseline_n = max(10, int(n * 0.3))
    base = X[:baseline_n]
    if HAS_SKLEARN and PCA is not None:
        pca = PCA(n_components=min(5, X.shape[1]))
        pca.fit(base)
        scores = pca.transform(X)
        recon = pca.inverse_transform(scores)
        resid = np.sum((X - recon) ** 2, axis=1)
    else:
        resid = np.sum((X - np.mean(base, axis=0)) ** 2, axis=1)
    threshold = float(np.quantile(resid, 0.995))
    idx = int(np.argmax(resid))
    evidence = {"metrics": {"max_residual": float(resid[idx]), "threshold": threshold}}
    finding = _make_finding(
        plugin_id,
        f"pca:{idx}",
        "PCA residual spike",
        "Residual variance exceeds baseline.",
        "PCA residual chart indicates drift.",
        evidence,
        severity="warn" if resid[idx] > threshold else "info",
        confidence=min(1.0, float(resid[idx]) / max(threshold, 1e-6)),
    )
    artifacts = [_artifact(ctx, plugin_id, "pca_control.json", evidence, "json")]
    return PluginResult("ok", "PCA control chart complete", _basic_metrics(df, sample_meta), [finding], artifacts, None)


def _binary_segmentation(
    series: np.ndarray, min_size: int, penalty: float
) -> list[int]:
    n = series.size
    if n < 2 * min_size:
        return []
    cumsum = np.cumsum(series)
    cumsum2 = np.cumsum(series**2)

    def _cost(start: int, end: int) -> float:
        length = end - start
        if length <= 0:
            return 0.0
        total = cumsum[end - 1] - (cumsum[start - 1] if start > 0 else 0.0)
        total2 = cumsum2[end - 1] - (cumsum2[start - 1] if start > 0 else 0.0)
        mean = total / length
        return total2 - 2 * mean * total + length * mean * mean

    base_cost = _cost(0, n)
    best_gain = 0.0
    best_idx = None
    for idx in range(min_size, n - min_size):
        cost = _cost(0, idx) + _cost(idx, n)
        gain = base_cost - cost
        if gain > best_gain:
            best_gain = gain
            best_idx = idx
    if best_idx is None or best_gain < penalty:
        return []
    left = _binary_segmentation(series[:best_idx], min_size, penalty)
    right = [best_idx + cp for cp in _binary_segmentation(series[best_idx:], min_size, penalty)]
    return left + [best_idx] + right


def _changepoint_pelt(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    numeric_cols = _top_numeric_columns(df, inferred.get("numeric_columns") or [], max_cols=1)
    if not numeric_cols:
        return PluginResult("skipped", "No numeric columns detected", _basic_metrics(df, sample_meta), [], [], None)
    series = df[numeric_cols[0]].dropna().to_numpy(dtype=float)
    if series.size < 100:
        return PluginResult("skipped", "Insufficient series length", _basic_metrics(df, sample_meta), [], [], None)
    max_points = int(config.get("pelt", {}).get("max_points", 20000))
    series = series[:max_points]
    min_size = int(config.get("pelt", {}).get("min_segment_size", 50))
    penalty = float(config.get("pelt", {}).get("penalty_beta", 2.0)) * math.log(len(series)) * 1.0
    changepoints = sorted(set(_binary_segmentation(series, min_size, penalty)))
    changepoints = changepoints[: int(config.get("pelt", {}).get("max_changepoints", 20))]
    findings = []
    for cp in changepoints:
        left = series[max(0, cp - min_size) : cp]
        right = series[cp : cp + min_size]
        effect = _effect_size(left, right)
        evidence = {"metrics": {"index": cp, "effect_size": effect}}
        findings.append(
            _make_finding(
                plugin_id,
                f"{numeric_cols[0]}:{cp}",
                f"Changepoint in {numeric_cols[0]}",
                "Segment boundary detected.",
                "Binary segmentation approximates PELT with SSE cost.",
                evidence,
                where={"column": numeric_cols[0], "index": int(cp)},
                severity="warn" if abs(effect) > 0.5 else "info",
                confidence=min(1.0, abs(effect)),
            )
        )
    artifacts = []
    if findings:
        artifacts.append(_artifact(ctx, plugin_id, "changepoints.json", findings, "json"))
    summary = _summary_or_skip("Changepoint detection complete", findings)
    return PluginResult("ok", summary, _basic_metrics(df, sample_meta), findings, artifacts, None)


def _changepoint_energy_edivisive(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    numeric_cols = _top_numeric_columns(df, inferred.get("numeric_columns") or [], max_cols=1)
    if not numeric_cols:
        return PluginResult("skipped", "No numeric columns detected", _basic_metrics(df, sample_meta), [], [], None)
    series = df[numeric_cols[0]].dropna().to_numpy(dtype=float)
    if series.size < 100:
        return PluginResult("skipped", "Insufficient series length", _basic_metrics(df, sample_meta), [], [], None)
    max_points = min(series.size, 2000)
    series = series[:max_points]
    best_stat = 0.0
    best_idx = None
    for idx in range(20, max_points - 20):
        left = series[:idx]
        right = series[idx:]
        if left.size == 0 or right.size == 0:
            continue
        dist_lr = np.mean(np.abs(left[:, None] - right[None, :]))
        dist_ll = np.mean(np.abs(left[:, None] - left[None, :]))
        dist_rr = np.mean(np.abs(right[:, None] - right[None, :]))
        stat = 2 * dist_lr - dist_ll - dist_rr
        if stat > best_stat:
            best_stat = stat
            best_idx = idx
    findings = []
    if best_idx is not None:
        evidence = {"metrics": {"index": best_idx, "energy_stat": best_stat}}
        findings.append(
            _make_finding(
                plugin_id,
                f"{numeric_cols[0]}:{best_idx}",
                f"Energy changepoint in {numeric_cols[0]}",
                "Energy statistic peaks at a boundary.",
                "E-divisive style energy statistic indicates change.",
                evidence,
                where={"column": numeric_cols[0], "index": int(best_idx)},
                severity="warn",
                confidence=min(1.0, best_stat / (abs(best_stat) + 1e-6)),
            )
        )
    artifacts = []
    if findings:
        artifacts.append(_artifact(ctx, plugin_id, "energy_changepoint.json", findings, "json"))
    summary = _summary_or_skip("Energy changepoint detection complete", findings)
    return PluginResult("ok", summary, _basic_metrics(df, sample_meta), findings, artifacts, None)


def _drift_adwin(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    numeric_cols = _top_numeric_columns(df, inferred.get("numeric_columns") or [], max_cols=1)
    if not numeric_cols:
        return PluginResult("skipped", "No numeric columns detected", _basic_metrics(df, sample_meta), [], [], None)
    series = df[numeric_cols[0]].dropna().to_numpy(dtype=float)
    if series.size < 100:
        return PluginResult("skipped", "Insufficient series length", _basic_metrics(df, sample_meta), [], [], None)
    window = min(200, series.size // 2)
    if window <= 10:
        return PluginResult("skipped", "Window too small", _basic_metrics(df, sample_meta), [], [], None)
    left = series[-2 * window : -window]
    right = series[-window:]
    mean_left = float(np.mean(left))
    mean_right = float(np.mean(right))
    diff = abs(mean_right - mean_left)
    eps = math.sqrt(math.log(4 * window) / (2 * window))
    evidence = {"metrics": {"mean_left": mean_left, "mean_right": mean_right, "diff": diff, "epsilon": eps}}
    findings = []
    if diff > eps:
        findings.append(
            _make_finding(
                plugin_id,
                f"{numeric_cols[0]}:adwin",
                f"ADWIN drift in {numeric_cols[0]}",
                "Recent window mean differs from previous window.",
                "ADWIN-style Hoeffding bound exceeded.",
                evidence,
                where={"column": numeric_cols[0]},
                severity="warn",
                confidence=min(1.0, diff / (eps + 1e-6)),
            )
        )
    artifacts = []
    if findings:
        artifacts.append(_artifact(ctx, plugin_id, "adwin_drift.json", findings, "json"))
    summary = _summary_or_skip("ADWIN drift check complete", findings)
    return PluginResult("ok", summary, _basic_metrics(df, sample_meta), findings, artifacts, None)


def _changepoint_method_survey_guided(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    pelt_result = _changepoint_pelt(plugin_id, ctx, df, config, inferred, timer, sample_meta)
    energy_result = _changepoint_energy_edivisive(plugin_id, ctx, df, config, inferred, timer, sample_meta)
    adwin_result = _drift_adwin(plugin_id, ctx, df, config, inferred, timer, sample_meta)
    combined = []
    for result in (pelt_result, energy_result, adwin_result):
        combined.extend(result.findings)
    seen = set()
    deduped = []
    for item in combined:
        if item["id"] in seen:
            continue
        deduped.append(item)
        seen.add(item["id"])
    artifacts = []
    if deduped:
        artifacts.append(_artifact(ctx, plugin_id, "changepoint_ensemble.json", deduped, "json"))
    summary = _summary_or_skip("Changepoint ensemble complete", deduped)
    return PluginResult("ok", summary, _basic_metrics(df, sample_meta), deduped, artifacts, None)


def _select_event_column(inferred: dict[str, Any]) -> str | None:
    candidates = inferred.get("categorical_columns") or []
    return candidates[0] if candidates else None


def _build_sequences(df: pd.DataFrame, event_col: str, time_col: str | None, case_col: str | None) -> list[list[str]]:
    if case_col and case_col in df.columns:
        work = df[[case_col, event_col] + ([time_col] if time_col and time_col in df.columns else [])].copy()
        if time_col and time_col in work.columns:
            # Sort once globally by case+time, then group; avoids expensive per-group re-sorting.
            parsed = pd.to_datetime(work[time_col], errors="coerce", utc=True)
            work = work.assign(__ts=parsed).sort_values([case_col, "__ts"]).drop(columns=["__ts"])
        seqs = (
            work.groupby(case_col, sort=False)[event_col]
            .apply(lambda s: s.astype(str).tolist())
            .tolist()
        )
        return [seq for seq in seqs if seq]
    if time_col and time_col in df.columns:
        df = _sort_by_time(df, time_col)
    return [df[event_col].astype(str).tolist()]


def _markov_transition_shift(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    event_col = _select_event_column(inferred)
    time_col = inferred.get("time_column")
    case_cols = inferred.get("id_like_columns") or []
    case_col = case_cols[0] if case_cols else None
    if not event_col:
        return PluginResult("skipped", "No event column detected", _basic_metrics(df, sample_meta), [], [], None)
    df = _sort_by_time(df, time_col)
    left_df, right_df = _split_pre_post(df, time_col, fraction=0.5)
    def _transition_matrix(frame: pd.DataFrame) -> dict[tuple[str, str], int]:
        seqs = _build_sequences(frame, event_col, time_col, case_col)
        counts: dict[tuple[str, str], int] = {}
        for seq in seqs:
            for a, b in zip(seq, seq[1:]):
                counts[(a, b)] = counts.get((a, b), 0) + 1
        return counts
    left = _transition_matrix(left_df)
    right = _transition_matrix(right_df)
    keys = set(left) | set(right)
    diffs = []
    for key in keys:
        diff = abs(left.get(key, 0) - right.get(key, 0))
        diffs.append((diff, key))
    diffs.sort(reverse=True)
    top = diffs[:10]
    findings = []
    if top:
        evidence = {"metrics": {"top_transitions": [{"from": k[0], "to": k[1], "delta": int(diff)} for diff, k in top]}}
        findings.append(
            _make_finding(
                plugin_id,
                "markov_shift",
                "Transition shift detected",
                "Transition frequencies changed between periods.",
                "Markov transition counts differ between early/late windows.",
                evidence,
                severity="warn" if top[0][0] > 5 else "info",
                confidence=min(1.0, top[0][0] / 20.0),
            )
        )
    artifacts = []
    if top:
        artifacts.append(_artifact(ctx, plugin_id, "transition_shift.json", evidence, "json"))
    summary = _summary_or_skip("Markov transition shift complete", findings)
    return PluginResult("ok", summary, _basic_metrics(df, sample_meta), findings, artifacts, None)


def _sequential_patterns_prefixspan(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    event_col = _select_event_column(inferred)
    time_col = inferred.get("time_column")
    case_cols = inferred.get("id_like_columns") or []
    case_col = case_cols[0] if case_cols else None
    if not event_col:
        return PluginResult("skipped", "No event column detected", _basic_metrics(df, sample_meta), [], [], None)
    seqs = _build_sequences(df, event_col, time_col, case_col)
    counts: dict[tuple[str, ...], int] = {}
    for seq in seqs:
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            counts[pair] = counts.get(pair, 0) + 1
    top = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:10]
    findings = []
    if top:
        evidence = {"metrics": {"top_patterns": [{"pattern": "->".join(pat), "count": int(cnt)} for pat, cnt in top]}}
        findings.append(
            _make_finding(
                plugin_id,
                "prefixspan",
                "Frequent sequence patterns",
                "Common subsequences detected.",
                "Prefix-span style bigram mining.",
                evidence,
                severity="info",
                confidence=0.5,
            )
        )
    artifacts = []
    if top:
        artifacts.append(_artifact(ctx, plugin_id, "frequent_patterns.json", evidence, "json"))
    summary = _summary_or_skip("Sequential pattern mining complete", findings)
    return PluginResult("ok", summary, _basic_metrics(df, sample_meta), findings, artifacts, None)


def _hmm_latent_state_sequences(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    numeric_cols = inferred.get("numeric_columns") or []
    X, cols = _numeric_matrix(df, numeric_cols, max_cols=int(config.get("max_cols", 5)))
    if X.size == 0 or X.shape[0] < 50:
        return PluginResult("skipped", "Insufficient numeric data", _basic_metrics(df, sample_meta), [], [], None)
    k = min(4, X.shape[0])
    if HAS_SKLEARN and KMeans is not None:
        model = KMeans(n_clusters=k, n_init=5, random_state=int(config.get("seed", 1337)))
        states = model.fit_predict(X)
    else:
        states = np.zeros(X.shape[0], dtype=int)
    transitions: dict[tuple[int, int], int] = {}
    for a, b in zip(states[:-1], states[1:]):
        transitions[(int(a), int(b))] = transitions.get((int(a), int(b)), 0) + 1
    top = sorted(transitions.items(), key=lambda item: item[1], reverse=True)[:10]
    evidence = {"metrics": {"top_transitions": [{"from": k[0], "to": k[1], "count": int(cnt)} for k, cnt in top]}}
    finding = _make_finding(
        plugin_id,
        "hmm_states",
        "Latent state sequence summary",
        "Latent clusters show dominant transitions.",
        "KMeans-based latent state approximation.",
        evidence,
        severity="info",
        confidence=0.5,
    )
    artifacts = [_artifact(ctx, plugin_id, "latent_states.json", evidence, "json")]
    return PluginResult("ok", "Latent state sequence complete", _basic_metrics(df, sample_meta), [finding], artifacts, None)


def _dependency_graph_change_detection(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    numeric_cols = inferred.get("numeric_columns") or []
    time_col = inferred.get("time_column")
    if len(numeric_cols) < 2:
        return PluginResult("skipped", "Need at least two numeric columns", _basic_metrics(df, sample_meta), [], [], None)
    left_df, right_df = _split_pre_post(df, time_col, fraction=0.5)
    def _corr(frame: pd.DataFrame) -> np.ndarray:
        return frame[numeric_cols].corr().fillna(0.0).to_numpy()
    left_corr = _corr(left_df)
    right_corr = _corr(right_df)
    diff = np.abs(left_corr - right_corr)
    np.fill_diagonal(diff, 0.0)
    idx = np.unravel_index(np.argmax(diff), diff.shape)
    evidence = {"metrics": {"col_a": numeric_cols[idx[0]], "col_b": numeric_cols[idx[1]], "delta_corr": float(diff[idx])}}
    finding = _make_finding(
        plugin_id,
        "dependency_shift",
        "Dependency graph shift",
        "Correlation structure changed between periods.",
        "Largest correlation delta detected.",
        evidence,
        severity="warn" if diff[idx] > 0.5 else "info",
        confidence=min(1.0, float(diff[idx])),
    )
    artifacts = [_artifact(ctx, plugin_id, "dependency_shift.json", evidence, "json")]
    return PluginResult("ok", "Dependency graph change detection complete", _basic_metrics(df, sample_meta), [finding], artifacts, None)


def _graphical_lasso_dependency_network(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    numeric_cols = inferred.get("numeric_columns") or []
    X, cols = _numeric_matrix(df, numeric_cols, max_cols=int(config.get("max_cols", 20)))
    if X.size == 0 or X.shape[0] < 50:
        return PluginResult("skipped", "Insufficient numeric data", _basic_metrics(df, sample_meta), [], [], None)
    if X.shape[0] > int(config.get("max_rows_for_covariance", 20000)):
        step = int(math.ceil(float(X.shape[0]) / float(config.get("max_rows_for_covariance", 20000))))
        X = X[:: max(1, step), :]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    finite_mask = np.all(np.isfinite(X), axis=1)
    if finite_mask.any():
        X = X[finite_mask]
    if X.shape[0] < 50 or X.shape[1] < 2:
        return PluginResult(
            "na",
            "Not applicable: insufficient_finite_numeric_data",
            _basic_metrics(df, sample_meta),
            [
                {
                    "kind": "plugin_not_applicable",
                    "reason_code": "INSUFFICIENT_FINITE_NUMERIC_DATA",
                    "reason": "insufficient_finite_numeric_data",
                    "recommended_next_step": "Provide at least 50 finite rows across 2+ numeric columns.",
                }
            ],
            [],
            None,
            debug={"gating_reason": "insufficient_finite_numeric_data"},
        )
    precision: np.ndarray | None = None
    fit_error: str | None = None
    if HAS_SKLEARN and GraphicalLasso is not None:
        try:
            model = GraphicalLasso(alpha=0.01, max_iter=100)
            model.fit(X)
            precision = model.precision_
        except Exception as exc:
            fit_error = str(exc)
    if precision is None:
        try:
            cov = np.cov(X, rowvar=False)
            cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
            precision = np.linalg.pinv(cov + np.eye(cov.shape[0]) * 1e-6)
        except Exception as exc:
            return PluginResult(
                "na",
                "Not applicable: ill_conditioned_dependency_covariance",
                _basic_metrics(df, sample_meta),
                [
                    {
                        "kind": "plugin_not_applicable",
                        "reason_code": "ILL_CONDITIONED_DEPENDENCY_COVARIANCE",
                        "reason": "ill_conditioned_dependency_covariance",
                        "recommended_next_step": "Normalize inputs and reduce collinearity in numeric features for dependency inference.",
                    }
                ],
                [],
                None,
                debug={
                    "gating_reason": "ill_conditioned_dependency_covariance",
                    "graphical_lasso_error": fit_error,
                    "fallback_error": str(exc),
                },
            )
    edges = []
    for i in range(precision.shape[0]):
        for j in range(i + 1, precision.shape[1]):
            if abs(precision[i, j]) > 0.1:
                edges.append({"a": cols[i], "b": cols[j], "weight": float(precision[i, j])})
    edges = sorted(edges, key=lambda e: abs(e["weight"]), reverse=True)[:20]
    evidence = {"metrics": {"edges": edges}}
    finding = _make_finding(
        plugin_id,
        "graphical_lasso",
        "Dependency network inferred",
        "Sparse dependency network estimated.",
        "Graphical lasso precision matrix provides edges.",
        evidence,
        severity="info",
        confidence=0.5,
    )
    artifacts = [_artifact(ctx, plugin_id, "dependency_network.json", evidence, "json")]
    return PluginResult("ok", "Graphical lasso network complete", _basic_metrics(df, sample_meta), [finding], artifacts, None)


def _mutual_information_screen(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    numeric_cols = inferred.get("numeric_columns") or []
    if len(numeric_cols) < 2:
        return PluginResult("skipped", "Need at least two numeric columns", _basic_metrics(df, sample_meta), [], [], None)
    max_cols = min(6, len(numeric_cols))
    cols = numeric_cols[:max_cols]
    scores = []
    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1 :]:
            if timer.exceeded():
                break
            a = pd.qcut(df[col_a].rank(method="first"), 5, duplicates="drop")
            b = pd.qcut(df[col_b].rank(method="first"), 5, duplicates="drop")
            joint = pd.crosstab(a, b)
            pxy = joint / joint.to_numpy().sum()
            px = pxy.sum(axis=1)
            py = pxy.sum(axis=0)
            mi = 0.0
            for r in pxy.index:
                for c in pxy.columns:
                    if pxy.loc[r, c] <= 0:
                        continue
                    mi += pxy.loc[r, c] * math.log(pxy.loc[r, c] / (px.loc[r] * py.loc[c] + 1e-9) + 1e-9)
            scores.append((mi, col_a, col_b))
    scores.sort(reverse=True)
    top = scores[:10]
    evidence = {"metrics": {"top_pairs": [{"a": a, "b": b, "mi": float(mi)} for mi, a, b in top]}}
    finding = _make_finding(
        plugin_id,
        "mutual_info",
        "Mutual information hotspots",
        "Strong dependencies between numeric columns.",
        "Discretized mutual information screening.",
        evidence,
        severity="info",
        confidence=0.5,
    )
    artifacts = [_artifact(ctx, plugin_id, "mutual_information.json", evidence, "json")]
    return PluginResult("ok", "Mutual information screen complete", _basic_metrics(df, sample_meta), [finding], artifacts, None)


def _transfer_entropy_directional(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    numeric_cols = inferred.get("numeric_columns") or []
    if len(numeric_cols) < 2:
        return PluginResult("skipped", "Need at least two numeric columns", _basic_metrics(df, sample_meta), [], [], None)
    cols = numeric_cols[:5]
    top: list[tuple[float, str, str]] = []
    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1 :]:
            if timer.exceeded():
                break
            a = df[col_a].to_numpy(dtype=float)
            b = df[col_b].to_numpy(dtype=float)
            if a.size < 5 or b.size < 5:
                continue
            a_lag = a[:-1]
            b_now = b[1:]
            corr = np.corrcoef(a_lag, b_now)[0, 1]
            if np.isnan(corr):
                continue
            top.append((abs(corr), col_a, col_b))
    top.sort(reverse=True)
    best = top[:10]
    evidence = {"metrics": {"top_pairs": [{"source": a, "target": b, "lag_corr": float(score)} for score, a, b in best]}}
    finding = _make_finding(
        plugin_id,
        "transfer_entropy",
        "Directional lag influence",
        "Lagged correlations suggest directional influence.",
        "Lag-1 correlation used as transfer-entropy proxy.",
        evidence,
        severity="info",
        confidence=0.5,
    )
    artifacts = [_artifact(ctx, plugin_id, "transfer_entropy.json", evidence, "json")]
    return PluginResult("ok", "Directional influence screen complete", _basic_metrics(df, sample_meta), [finding], artifacts, None)


def _lagged_predictability_test(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    numeric_cols = _top_numeric_columns(df, inferred.get("numeric_columns") or [], max_cols=1)
    if not numeric_cols:
        return PluginResult("skipped", "No numeric columns detected", _basic_metrics(df, sample_meta), [], [], None)
    series = df[numeric_cols[0]].dropna().to_numpy(dtype=float)
    if series.size < 10:
        return PluginResult("skipped", "Insufficient series length", _basic_metrics(df, sample_meta), [], [], None)
    x = series[:-1]
    y = series[1:]
    if x.size == 0:
        return PluginResult("skipped", "Insufficient series length", _basic_metrics(df, sample_meta), [], [], None)
    corr = float(np.corrcoef(x, y)[0, 1])
    evidence = {"metrics": {"lag1_corr": corr}}
    finding = _make_finding(
        plugin_id,
        "lagged_predictability",
        f"Lagged predictability in {numeric_cols[0]}",
        "Lag-1 correlation indicates predictability.",
        "Autocorrelation used as predictability proxy.",
        evidence,
        where={"column": numeric_cols[0]},
        severity="info" if abs(corr) < 0.5 else "warn",
        confidence=min(1.0, abs(corr)),
    )
    artifacts = [_artifact(ctx, plugin_id, "lagged_predictability.json", evidence, "json")]
    return PluginResult("ok", "Lagged predictability check complete", _basic_metrics(df, sample_meta), [finding], artifacts, None)


def _copula_dependence(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    numeric_cols = inferred.get("numeric_columns") or []
    if len(numeric_cols) < 2:
        return PluginResult("skipped", "Need at least two numeric columns", _basic_metrics(df, sample_meta), [], [], None)
    cols = numeric_cols[:5]
    top: list[tuple[float, str, str]] = []
    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1 :]:
            if timer.exceeded():
                break
            rho = df[[col_a, col_b]].corr(method="spearman").iloc[0, 1]
            if np.isnan(rho):
                continue
            top.append((abs(rho), col_a, col_b))
    top.sort(reverse=True)
    best = top[:10]
    evidence = {"metrics": {"top_pairs": [{"a": a, "b": b, "spearman_rho": float(score)} for score, a, b in best]}}
    finding = _make_finding(
        plugin_id,
        "copula_dependence",
        "Copula dependence signals",
        "Rank-based dependence detected.",
        "Spearman rho used as copula proxy.",
        evidence,
        severity="info",
        confidence=0.5,
    )
    artifacts = [_artifact(ctx, plugin_id, "copula_dependence.json", evidence, "json")]
    return PluginResult("ok", "Copula dependence screen complete", _basic_metrics(df, sample_meta), [finding], artifacts, None)


def _edit_distance(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = np.zeros((len(a) + 1, len(b) + 1), dtype=int)
    for i in range(len(a) + 1):
        dp[i, 0] = i
    for j in range(len(b) + 1):
        dp[0, j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + cost,
            )
    return int(dp[len(a), len(b)])


def _bounded_edit_distance(a: list[str], b: list[str], max_cells: int) -> int:
    if max_cells > 0 and (len(a) * len(b)) > max_cells:
        m = min(len(a), len(b))
        mismatches = sum(1 for i in range(m) if a[i] != b[i])
        return int(mismatches + abs(len(a) - len(b)))
    return _edit_distance(a, b)


def _truncate_seq(seq: list[str], max_len: int) -> list[str]:
    if max_len <= 0 or len(seq) <= max_len:
        return seq
    return seq[:max_len]


def _deterministic_seq_sample(seqs: list[list[str]], max_sequences: int) -> list[list[str]]:
    if max_sequences <= 0 or len(seqs) <= max_sequences:
        return seqs
    idx = np.linspace(0, len(seqs) - 1, num=max_sequences, dtype=int)
    return [seqs[int(i)] for i in idx]


def _conformance_alignments(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    event_col = _select_event_column(inferred)
    time_col = inferred.get("time_column")
    case_cols = inferred.get("id_like_columns") or []
    case_col = case_cols[0] if case_cols else None
    if not event_col or not case_col:
        return PluginResult("skipped", "Need event + case id columns", _basic_metrics(df, sample_meta), [], [], None)
    seqs = _build_sequences(df, event_col, time_col, case_col)
    if not seqs:
        return PluginResult("skipped", "No sequences detected", _basic_metrics(df, sample_meta), [], [], None)
    max_sequences = int(config.get("max_sequences", 3000))
    max_variant_length = int(config.get("max_variant_length", 80))
    max_edit_cells = int(config.get("max_edit_cells", 25000))

    variant_counts: dict[tuple[str, ...], int] = {}
    for seq in seqs:
        key = tuple(_truncate_seq(seq, max_variant_length))
        variant_counts[key] = variant_counts.get(key, 0) + 1
    top_variant = max(variant_counts.items(), key=lambda item: item[1])[0]
    top_seq = list(top_variant)
    seqs_for_distance = _deterministic_seq_sample(seqs, max_sequences=max_sequences)
    distances = [
        _bounded_edit_distance(
            _truncate_seq(seq, max_variant_length),
            top_seq,
            max_cells=max_edit_cells,
        )
        for seq in seqs_for_distance
    ]
    avg_dist = float(np.mean(distances)) if distances else 0.0
    evidence = {
        "metrics": {
            "top_variant": "->".join(top_seq),
            "avg_edit_distance": avg_dist,
            "variant_count": len(variant_counts),
            "distance_sequences_used": len(seqs_for_distance),
        }
    }
    finding = _make_finding(
        plugin_id,
        "conformance",
        "Conformance deviations",
        "Sequences deviate from dominant variant.",
        "Edit distance to top variant is elevated.",
        evidence,
        severity="warn" if avg_dist >= 2.0 else "info",
        confidence=min(1.0, avg_dist / 5.0),
    )
    artifacts = [_artifact(ctx, plugin_id, "conformance_alignments.json", evidence, "json")]
    return PluginResult("ok", "Conformance alignment check complete", _basic_metrics(df, sample_meta), [finding], artifacts, None)


def _process_drift_conformance_over_time(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    event_col = _select_event_column(inferred)
    time_col = inferred.get("time_column")
    case_cols = inferred.get("id_like_columns") or []
    case_col = case_cols[0] if case_cols else None
    if not event_col or not case_col:
        return PluginResult("skipped", "Need event + case id columns", _basic_metrics(df, sample_meta), [], [], None)
    left_df, right_df = _split_pre_post(df, time_col, fraction=0.5)
    max_variant_length = int(config.get("max_variant_length", 80))

    def _variant_counts(frame: pd.DataFrame) -> dict[tuple[str, ...], int]:
        seqs = _build_sequences(frame, event_col, time_col, case_col)
        counts: dict[tuple[str, ...], int] = {}
        for seq in seqs:
            key = tuple(_truncate_seq(seq, max_variant_length))
            counts[key] = counts.get(key, 0) + 1
        return counts
    left_counts = _variant_counts(left_df)
    right_counts = _variant_counts(right_df)
    all_keys = set(left_counts) | set(right_counts)
    l1 = sum(abs(left_counts.get(k, 0) - right_counts.get(k, 0)) for k in all_keys)
    evidence = {"metrics": {"variant_l1_distance": float(l1), "variants": len(all_keys)}}
    finding = _make_finding(
        plugin_id,
        "process_drift",
        "Process drift over time",
        "Variant mix differs between periods.",
        "Variant distribution L1 distance elevated.",
        evidence,
        severity="warn" if l1 > 5 else "info",
        confidence=min(1.0, l1 / 20.0),
    )
    artifacts = [_artifact(ctx, plugin_id, "process_drift.json", evidence, "json")]
    return PluginResult("ok", "Process drift analysis complete", _basic_metrics(df, sample_meta), [finding], artifacts, None)


def _variant_differential(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    event_col = _select_event_column(inferred)
    time_col = inferred.get("time_column")
    case_cols = inferred.get("id_like_columns") or []
    case_col = case_cols[0] if case_cols else None
    if not event_col or not case_col:
        return PluginResult("skipped", "Need event + case id columns", _basic_metrics(df, sample_meta), [], [], None)
    left_df, right_df = _split_pre_post(df, time_col, fraction=0.5)
    max_variant_length = int(config.get("max_variant_length", 80))

    def _variant_counts(frame: pd.DataFrame) -> dict[tuple[str, ...], int]:
        seqs = _build_sequences(frame, event_col, time_col, case_col)
        counts: dict[tuple[str, ...], int] = {}
        for seq in seqs:
            key = tuple(_truncate_seq(seq, max_variant_length))
            counts[key] = counts.get(key, 0) + 1
        return counts
    left_counts = _variant_counts(left_df)
    right_counts = _variant_counts(right_df)
    all_keys = set(left_counts) | set(right_counts)
    deltas = []
    for key in all_keys:
        delta = right_counts.get(key, 0) - left_counts.get(key, 0)
        deltas.append((abs(delta), key, delta))
    deltas.sort(reverse=True)
    top = deltas[:10]
    evidence = {
        "metrics": {
            "top_variants": [{"variant": "->".join(k), "delta": int(delta)} for _, k, delta in top]
        }
    }
    finding = _make_finding(
        plugin_id,
        "variant_diff",
        "Variant differential",
        "Variant frequencies differ between periods.",
        "Top variant deltas identified.",
        evidence,
        severity="warn" if top and top[0][0] > 5 else "info",
        confidence=min(1.0, top[0][0] / 20.0) if top else 0.3,
    )
    artifacts = [_artifact(ctx, plugin_id, "variant_differential.json", evidence, "json")]
    return PluginResult("ok", "Variant differential analysis complete", _basic_metrics(df, sample_meta), [finding], artifacts, None)


def _template_from_text(text: str) -> str:
    value = "".join(ch if not ch.isdigit() else "#" for ch in text)
    value = " ".join(value.split())
    return value[:200]


def _template_drift_two_sample(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    text_cols = inferred.get("text_columns") or []
    time_col = inferred.get("time_column")
    if not text_cols:
        return PluginResult("skipped", "No text columns detected", _basic_metrics(df, sample_meta), [], [], None)
    left_df, right_df = _split_pre_post(df, time_col, fraction=0.5)
    if left_df.empty or right_df.empty:
        return PluginResult("skipped", "Insufficient data for template drift", _basic_metrics(df, sample_meta), [], [], None)
    col = text_cols[0]
    privacy = config.get("privacy", {})
    def _counts(frame: pd.DataFrame) -> dict[str, int]:
        counts: dict[str, int] = {}
        for value in frame[col].dropna().astype(str).head(2000):
            templ = _template_from_text(value)
            templ = build_redactor(privacy)(templ)
            counts[templ] = counts.get(templ, 0) + 1
        return counts
    left_counts = _counts(left_df)
    right_counts = _counts(right_df)
    keys = set(left_counts) | set(right_counts)
    deltas = []
    for key in keys:
        delta = right_counts.get(key, 0) - left_counts.get(key, 0)
        deltas.append((abs(delta), key, delta))
    deltas.sort(reverse=True)
    top = deltas[:10]
    evidence = {"metrics": {"top_templates": [{"template": k, "delta": int(delta)} for _, k, delta in top]}}
    finding = _make_finding(
        plugin_id,
        "template_drift",
        "Template drift detected",
        "Template mix differs between periods.",
        "Template count deltas indicate drift.",
        evidence,
        severity="warn" if top and top[0][0] > 5 else "info",
        confidence=min(1.0, top[0][0] / 20.0) if top else 0.3,
    )
    artifacts = [_artifact(ctx, plugin_id, "template_drift.json", evidence, "json")]
    return PluginResult("ok", "Template drift analysis complete", _basic_metrics(df, sample_meta), [finding], artifacts, None)


def _message_entropy_drift(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    text_cols = inferred.get("text_columns") or []
    time_col = inferred.get("time_column")
    if not text_cols or not time_col:
        return PluginResult("skipped", "Need text + time columns", _basic_metrics(df, sample_meta), [], [], None)
    df = _sort_by_time(df, time_col)
    col = text_cols[0]
    privacy = config.get("privacy", {})
    def _entropy(values: Iterable[str]) -> float:
        counts: dict[str, int] = {}
        for value in values:
            tokens = str(value).split()
            for tok in tokens:
                tok = build_redactor(privacy)(tok)
                counts[tok] = counts.get(tok, 0) + 1
        total = sum(counts.values()) or 1
        ent = 0.0
        for count in counts.values():
            p = count / total
            ent -= p * math.log(p + 1e-9)
        return ent
    left_df, right_df = _split_pre_post(df, time_col, fraction=0.5)
    ent_left = _entropy(left_df[col].dropna().head(2000))
    ent_right = _entropy(right_df[col].dropna().head(2000))
    delta = ent_right - ent_left
    evidence = {"metrics": {"entropy_pre": ent_left, "entropy_post": ent_right, "delta": delta}}
    finding = _make_finding(
        plugin_id,
        "entropy_drift",
        "Message entropy drift",
        "Token entropy changes between periods.",
        "Entropy shift indicates content drift.",
        evidence,
        severity="warn" if abs(delta) > 0.2 else "info",
        confidence=min(1.0, abs(delta)),
    )
    artifacts = [_artifact(ctx, plugin_id, "message_entropy.json", evidence, "json")]
    return PluginResult("ok", "Message entropy drift complete", _basic_metrics(df, sample_meta), [finding], artifacts, None)


def _topic_model_lda(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    text_cols = inferred.get("text_columns") or []
    if not text_cols:
        return PluginResult("skipped", "No text columns detected", _basic_metrics(df, sample_meta), [], [], None)
    col = text_cols[0]
    docs = df[col].dropna().astype(str).head(2000).tolist()
    privacy = config.get("privacy", {})
    docs = [build_redactor(privacy)(doc) for doc in docs]
    if not docs:
        return PluginResult("skipped", "No text samples", _basic_metrics(df, sample_meta), [], [], None)
    topics = []
    if HAS_SKLEARN and CountVectorizer is not None and LatentDirichletAllocation is not None:
        vectorizer = CountVectorizer(max_features=500, stop_words="english")
        X = vectorizer.fit_transform(docs)
        if X.shape[1] == 0:
            return PluginResult("skipped", "No vocabulary for LDA", _basic_metrics(df, sample_meta), [], [], None)
        lda = LatentDirichletAllocation(n_components=3, random_state=int(config.get("seed", 1337)))
        lda.fit(X)
        terms = np.array(vectorizer.get_feature_names_out())
        for idx, topic in enumerate(lda.components_):
            top_terms = terms[np.argsort(topic)[::-1][:5]].tolist()
            topics.append({"topic": idx, "terms": top_terms})
    else:
        token_counts: dict[str, int] = {}
        for doc in docs:
            for tok in doc.split():
                token_counts[tok] = token_counts.get(tok, 0) + 1
        top_terms = [t for t, _ in sorted(token_counts.items(), key=lambda item: item[1], reverse=True)[:5]]
        topics.append({"topic": 0, "terms": top_terms})
    evidence = {"metrics": {"topics": topics}}
    finding = _make_finding(
        plugin_id,
        "lda",
        "Topic model summary",
        "Latent topics extracted from text.",
        "LDA or frequency-based topics.",
        evidence,
        severity="info",
        confidence=0.5,
    )
    artifacts = [_artifact(ctx, plugin_id, "topic_model.json", evidence, "json")]
    return PluginResult("ok", "Topic modeling complete", _basic_metrics(df, sample_meta), [finding], artifacts, None)


def _term_burst_kleinberg(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    text_cols = inferred.get("text_columns") or []
    time_col = inferred.get("time_column")
    if not text_cols or not time_col:
        return PluginResult("skipped", "Need text + time columns", _basic_metrics(df, sample_meta), [], [], None)
    df = _sort_by_time(df, time_col)
    col = text_cols[0]
    bucket = pd.to_datetime(df[time_col], errors="coerce").dt.floor("D")
    df = df.assign(_bucket=bucket)
    privacy = config.get("privacy", {})
    term_counts: dict[str, list[int]] = {}
    buckets = sorted(df["_bucket"].dropna().unique())
    bucket_index = {b: idx for idx, b in enumerate(buckets)}
    for _, row in df[[col, "_bucket"]].dropna().iterrows():
        tokens = str(row[col]).split()
        b_idx = bucket_index.get(row["_bucket"])
        if b_idx is None:
            continue
        for tok in tokens[:5]:
            tok = build_redactor(privacy)(tok)
            if tok not in term_counts:
                term_counts[tok] = [0] * len(buckets)
            term_counts[tok][b_idx] += 1
    best_term = None
    best_z = 0.0
    for term, counts in term_counts.items():
        mean = statistics.mean(counts) if counts else 0.0
        std = statistics.pstdev(counts) if counts else 0.0
        if std == 0:
            continue
        z = max((c - mean) / std for c in counts)
        if z > best_z:
            best_z = z
            best_term = term
    findings = []
    if best_term:
        evidence = {"metrics": {"term": best_term, "max_z": best_z}}
        findings.append(
            _make_finding(
                plugin_id,
                f"term:{best_term}",
                "Term burst detected",
                "A term spikes in frequency.",
                "Term count z-score exceeds baseline.",
                evidence,
                severity="warn" if best_z > 2.0 else "info",
                confidence=min(1.0, best_z / 5.0),
            )
        )
    artifacts = []
    if findings:
        artifacts.append(_artifact(ctx, plugin_id, "term_burst.json", findings, "json"))
    summary = _summary_or_skip("Term burst detection complete", findings)
    return PluginResult("ok", summary, _basic_metrics(df, sample_meta), findings, artifacts, None)


def _duration_columns(inferred: dict[str, Any], df: pd.DataFrame) -> list[str]:
    numeric_cols = inferred.get("numeric_columns") or []
    hints = ("duration", "wait", "latency", "elapsed", "cycle", "time_to", "lead")
    preferred = [col for col in numeric_cols if any(h in str(col).lower() for h in hints)]
    return preferred or numeric_cols


def _survival_kaplan_meier(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    duration_cols = _duration_columns(inferred, df)
    if not duration_cols:
        return PluginResult("skipped", "No duration columns detected", _basic_metrics(df, sample_meta), [], [], None)
    col = duration_cols[0]
    durations = df[col].dropna().to_numpy(dtype=float)
    if durations.size < 30:
        return PluginResult("skipped", "Insufficient duration data", _basic_metrics(df, sample_meta), [], [], None)
    median = float(np.median(durations))
    p90 = float(np.quantile(durations, 0.9))
    evidence = {"metrics": {"median": median, "p90": p90, "column": col}}
    finding = _make_finding(
        plugin_id,
        f"{col}:km",
        f"Survival summary for {col}",
        "Duration distribution summarized.",
        "Kaplan-Meier style summary (median, p90).",
        evidence,
        where={"column": col},
        severity="info",
        confidence=0.5,
    )
    artifacts = [_artifact(ctx, plugin_id, "survival_summary.json", evidence, "json")]
    return PluginResult("ok", "Survival analysis complete", _basic_metrics(df, sample_meta), [finding], artifacts, None)


def _proportional_hazards_duration(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    duration_cols = _duration_columns(inferred, df)
    group_cols = inferred.get("group_by") or []
    if not duration_cols or not group_cols:
        return PluginResult("skipped", "Need duration + group columns", _basic_metrics(df, sample_meta), [], [], None)
    col = duration_cols[0]
    group_col = group_cols[0]
    groups = df[group_col].dropna().astype(str)
    if groups.nunique() < 2:
        return PluginResult("skipped", "Insufficient group variety", _basic_metrics(df, sample_meta), [], [], None)
    top_group = groups.value_counts().index[0]
    left = df.loc[groups == top_group, col].dropna().to_numpy(dtype=float)
    right = df.loc[groups != top_group, col].dropna().to_numpy(dtype=float)
    if left.size == 0 or right.size == 0:
        return PluginResult("skipped", "Insufficient duration data", _basic_metrics(df, sample_meta), [], [], None)
    ratio = float(np.median(left) / (np.median(right) + 1e-6))
    evidence = {"metrics": {"hazard_ratio_proxy": ratio, "group": top_group, "column": col}}
    finding = _make_finding(
        plugin_id,
        f"{col}:{group_col}:{top_group}",
        "Proportional hazards proxy",
        "Median duration ratio differs by group.",
        "Median ratio used as hazard proxy.",
        evidence,
        where={"column": col, "group": {group_col: top_group}},
        severity="warn" if ratio > 1.5 else "info",
        confidence=min(1.0, abs(ratio - 1.0)),
    )
    artifacts = [_artifact(ctx, plugin_id, "hazards_proxy.json", evidence, "json")]
    return PluginResult("ok", "Proportional hazards proxy complete", _basic_metrics(df, sample_meta), [finding], artifacts, None)


def _quantile_regression_duration(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    duration_cols = _duration_columns(inferred, df)
    group_cols = inferred.get("group_by") or []
    if not duration_cols or not group_cols:
        return PluginResult("skipped", "Need duration + group columns", _basic_metrics(df, sample_meta), [], [], None)
    col = duration_cols[0]
    group_col = group_cols[0]
    groups = df[group_col].dropna().astype(str)
    if groups.nunique() < 2:
        return PluginResult("skipped", "Insufficient group variety", _basic_metrics(df, sample_meta), [], [], None)
    top_group = groups.value_counts().index[0]
    left = df.loc[groups == top_group, col].dropna().to_numpy(dtype=float)
    right = df.loc[groups != top_group, col].dropna().to_numpy(dtype=float)
    if left.size == 0 or right.size == 0:
        return PluginResult("skipped", "Insufficient duration data", _basic_metrics(df, sample_meta), [], [], None)
    q_left = float(np.quantile(left, 0.9))
    q_right = float(np.quantile(right, 0.9))
    delta = q_left - q_right
    evidence = {"metrics": {"p90_group": q_left, "p90_other": q_right, "delta": delta}}
    finding = _make_finding(
        plugin_id,
        f"{col}:{group_col}:q90",
        "Quantile duration difference",
        "High-percentile durations differ by group.",
        "Quantile comparison indicates tail risk differences.",
        evidence,
        where={"column": col, "group": {group_col: top_group}},
        severity="warn" if abs(delta) > 0 else "info",
        confidence=min(1.0, abs(delta) / (abs(q_right) + 1e-6)),
    )
    artifacts = [_artifact(ctx, plugin_id, "quantile_duration.json", evidence, "json")]
    return PluginResult("ok", "Quantile duration analysis complete", _basic_metrics(df, sample_meta), [finding], artifacts, None)


def _queue_model_fit(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    time_col = inferred.get("time_column")
    duration_cols = _duration_columns(inferred, df)
    if not time_col or not duration_cols:
        return PluginResult("skipped", "Need time + duration columns", _basic_metrics(df, sample_meta), [], [], None)
    times = pd.to_datetime(df[time_col], errors="coerce").dropna().sort_values()
    if times.size < 20:
        return PluginResult("skipped", "Insufficient timestamps", _basic_metrics(df, sample_meta), [], [], None)
    inter_arrivals = times.diff().dropna().dt.total_seconds().to_numpy()
    service = df[duration_cols[0]].dropna().to_numpy(dtype=float)
    if inter_arrivals.size == 0 or service.size == 0:
        return PluginResult("skipped", "Insufficient arrival/service data", _basic_metrics(df, sample_meta), [], [], None)
    lam = 1.0 / (np.mean(inter_arrivals) + 1e-6)
    mu = 1.0 / (np.mean(service) + 1e-6)
    rho = lam / mu if mu > 0 else 0.0
    evidence = {
        "metrics": {
            "arrival_rate": float(lam),
            "service_rate": float(mu),
            "utilization": float(rho),
        },
        "assumptions": ["Arrival/service times approximated as exponential."],
    }
    finding = _make_finding(
        plugin_id,
        "queue_fit",
        "Queue utilization estimate",
        "Arrival/service rates imply utilization.",
        "Simple M/M/1 utilization proxy.",
        evidence,
        severity="warn" if rho > 0.8 else "info",
        confidence=min(1.0, rho),
        measurement_type="measured",
    )
    artifacts = [_artifact(ctx, plugin_id, "queue_fit.json", evidence, "json")]
    return PluginResult("ok", "Queue model fit complete", _basic_metrics(df, sample_meta), [finding], artifacts, None)


def _littles_law_consistency(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    time_col = inferred.get("time_column")
    duration_cols = _duration_columns(inferred, df)
    if not time_col or not duration_cols:
        return PluginResult("skipped", "Need time + duration columns", _basic_metrics(df, sample_meta), [], [], None)
    times = pd.to_datetime(df[time_col], errors="coerce").dropna().sort_values()
    inter_arrivals = times.diff().dropna().dt.total_seconds().to_numpy()
    wait = df[duration_cols[0]].dropna().to_numpy(dtype=float)
    if inter_arrivals.size == 0 or wait.size == 0:
        return PluginResult("skipped", "Insufficient data", _basic_metrics(df, sample_meta), [], [], None)
    lam = 1.0 / (np.mean(inter_arrivals) + 1e-6)
    W = float(np.mean(wait))
    L_est = lam * W
    evidence = {
        "metrics": {"arrival_rate": float(lam), "mean_wait": W, "L_estimate": L_est},
        "assumptions": ["Steady-state assumptions applied."],
    }
    finding = _make_finding(
        plugin_id,
        "littles_law",
        "Little's Law consistency",
        "Estimated WIP from arrival rate and wait time.",
        "L = λW estimate computed.",
        evidence,
        severity="info",
        confidence=0.5,
        measurement_type="measured",
    )
    artifacts = [_artifact(ctx, plugin_id, "littles_law.json", evidence, "json")]
    return PluginResult("ok", "Little's Law check complete", _basic_metrics(df, sample_meta), [finding], artifacts, None)


def _kingman_vut_approx(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    time_col = inferred.get("time_column")
    duration_cols = _duration_columns(inferred, df)
    if not time_col or not duration_cols:
        return PluginResult("skipped", "Need time + duration columns", _basic_metrics(df, sample_meta), [], [], None)
    times = pd.to_datetime(df[time_col], errors="coerce").dropna().sort_values()
    inter_arrivals = times.diff().dropna().dt.total_seconds().to_numpy()
    service = df[duration_cols[0]].dropna().to_numpy(dtype=float)
    if inter_arrivals.size == 0 or service.size == 0:
        return PluginResult("skipped", "Insufficient data", _basic_metrics(df, sample_meta), [], [], None)
    ca2 = float(np.var(inter_arrivals) / (np.mean(inter_arrivals) ** 2 + 1e-6))
    cs2 = float(np.var(service) / (np.mean(service) ** 2 + 1e-6))
    lam = 1.0 / (np.mean(inter_arrivals) + 1e-6)
    mu = 1.0 / (np.mean(service) + 1e-6)
    rho = lam / mu if mu > 0 else 0.0
    Wq = (rho / max(1 - rho, 1e-6)) * (ca2 + cs2) / 2 * (1 / mu)
    evidence = {
        "metrics": {"ca2": ca2, "cs2": cs2, "rho": rho, "Wq_estimate": Wq},
        "assumptions": ["G/G/1 approximation", "Steady state"],
    }
    finding = _make_finding(
        plugin_id,
        "kingman_vut",
        "Kingman VUT waiting time",
        "Queueing approximation of waiting time.",
        "Kingman VUT formula applied.",
        evidence,
        severity="info",
        confidence=0.5,
        measurement_type="measured",
    )
    artifacts = [_artifact(ctx, plugin_id, "kingman_vut.json", evidence, "json")]
    return PluginResult("ok", "Kingman VUT approximation complete", _basic_metrics(df, sample_meta), [finding], artifacts, None)
def _control_chart_individuals(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    value_cols = inferred.get("value_columns") or []
    time_col = inferred.get("time_column")
    group_cols = inferred.get("group_by") or []
    if not value_cols:
        return PluginResult("skipped", "No numeric columns detected", _basic_metrics(df, sample_meta), [], [], None)
    df = _sort_by_time(df, time_col)
    max_groups = int(config.get("max_groups", 30))
    max_findings = int(config.get("max_findings", 30))
    z_thresh = float(config.get("control_chart", {}).get("z_thresh", 4.0))

    findings: list[dict[str, Any]] = []
    for col in value_cols:
        if timer.exceeded():
            break
        for label, row_idx, where in _group_slices(df, group_cols, max_groups):
            values = pd.to_numeric(df.loc[row_idx, col], errors="coerce").dropna().to_numpy(dtype=float)
            if values.size < 30:
                continue
            zscores = robust_zscores(values)
            mask = np.abs(zscores) > z_thresh
            if not mask.any():
                continue
            idx = int(np.max(np.where(mask)))
            direction = "high" if zscores[idx] > 0 else "low"
            key = f"{col}:{label}:{idx}"
            evidence = {"metrics": {"z_score": float(zscores[idx]), "threshold": z_thresh}}
            finding = _make_finding(
                plugin_id,
                key,
                f"Individuals chart alarm in {col} ({label})",
                f"Recent value is {direction} relative to robust baseline.",
                "Robust z-score exceeded threshold.",
                evidence,
                where=where,
                severity="warn" if abs(zscores[idx]) < z_thresh * 1.5 else "critical",
                confidence=min(1.0, abs(zscores[idx]) / max(z_thresh, 1e-6)),
            )
            findings.append(finding)
            if len(findings) >= max_findings:
                break
        if len(findings) >= max_findings:
            break

    artifacts = []
    if findings:
        artifact = _artifact(ctx, plugin_id, "individuals_alarms.json", findings, "json")
        artifacts.append(artifact)
    summary = _summary_or_skip("Individuals chart alarms detected", findings)
    return PluginResult("ok", summary, _basic_metrics(df, sample_meta), findings, artifacts, None)


def _control_chart_ewma(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    value_cols = inferred.get("value_columns") or []
    time_col = inferred.get("time_column")
    group_cols = inferred.get("group_by") or []
    if not value_cols:
        return PluginResult("skipped", "No numeric columns detected", _basic_metrics(df, sample_meta), [], [], None)
    df = _sort_by_time(df, time_col)
    max_groups = int(config.get("max_groups", 30))
    max_findings = int(config.get("max_findings", 30))
    settings = config.get("control_chart", {})
    lam = float(settings.get("ewma_lambda", 0.2))
    L = float(settings.get("ewma_L", 3.0))

    findings: list[dict[str, Any]] = []
    for col in value_cols:
        if timer.exceeded():
            break
        for label, row_idx, where in _group_slices(df, group_cols, max_groups):
            values = pd.to_numeric(df.loc[row_idx, col], errors="coerce").dropna().to_numpy(dtype=float)
            if values.size < 30:
                continue
            center, scale = robust_center_scale(values)
            sigma_ewma = scale * math.sqrt(lam / (2.0 - lam)) if scale > 0 else 0.0
            if sigma_ewma <= 0:
                continue
            ewma = np.zeros_like(values, dtype=float)
            ewma[0] = values[0]
            for idx in range(1, values.size):
                ewma[idx] = lam * values[idx] + (1.0 - lam) * ewma[idx - 1]
            mask = np.abs(ewma - center) > (L * sigma_ewma)
            if not mask.any():
                continue
            idx = int(np.max(np.where(mask)))
            direction = "high" if ewma[idx] > center else "low"
            key = f"{col}:{label}:{idx}"
            evidence = {"metrics": {"ewma": float(ewma[idx]), "threshold": float(L * sigma_ewma)}}
            finding = _make_finding(
                plugin_id,
                key,
                f"EWMA drift in {col} ({label})",
                f"EWMA is {direction} relative to robust baseline.",
                "EWMA exceeded control limit.",
                evidence,
                where=where,
                severity="warn",
                confidence=min(1.0, abs(ewma[idx] - center) / max(L * sigma_ewma, 1e-6)),
            )
            findings.append(finding)
            if len(findings) >= max_findings:
                break
        if len(findings) >= max_findings:
            break

    artifacts = []
    if findings:
        artifacts.append(_artifact(ctx, plugin_id, "ewma_alarms.json", findings, "json"))
    summary = _summary_or_skip("EWMA alarms detected", findings)
    return PluginResult("ok", summary, _basic_metrics(df, sample_meta), findings, artifacts, None)


def _control_chart_cusum(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    value_cols = inferred.get("value_columns") or []
    time_col = inferred.get("time_column")
    group_cols = inferred.get("group_by") or []
    if not value_cols:
        return PluginResult("skipped", "No numeric columns detected", _basic_metrics(df, sample_meta), [], [], None)
    df = _sort_by_time(df, time_col)
    max_groups = int(config.get("max_groups", 30))
    max_findings = int(config.get("max_findings", 30))
    settings = config.get("control_chart", {})
    k_sigma = float(settings.get("cusum_k", 0.5))
    h_sigma = float(settings.get("cusum_h", 5.0))

    findings: list[dict[str, Any]] = []
    for col in value_cols:
        if timer.exceeded():
            break
        for label, row_idx, where in _group_slices(df, group_cols, max_groups):
            values = pd.to_numeric(df.loc[row_idx, col], errors="coerce").dropna().to_numpy(dtype=float)
            if values.size < 30:
                continue
            center, scale = robust_center_scale(values)
            if scale <= 0:
                continue
            k = k_sigma * scale
            h = h_sigma * scale
            c_plus = np.zeros_like(values, dtype=float)
            c_minus = np.zeros_like(values, dtype=float)
            for idx, value in enumerate(values):
                if idx == 0:
                    c_plus[idx] = max(0.0, value - (center + k))
                    c_minus[idx] = max(0.0, (center - k) - value)
                else:
                    c_plus[idx] = max(0.0, c_plus[idx - 1] + value - (center + k))
                    c_minus[idx] = max(0.0, c_minus[idx - 1] + (center - k) - value)
            mask = (c_plus > h) | (c_minus > h)
            if not mask.any():
                continue
            idx = int(np.max(np.where(mask)))
            direction = "high" if c_plus[idx] > c_minus[idx] else "low"
            key = f"{col}:{label}:{idx}"
            evidence = {"metrics": {"cusum_plus": float(c_plus[idx]), "cusum_minus": float(c_minus[idx]), "threshold": h}}
            finding = _make_finding(
                plugin_id,
                key,
                f"CUSUM drift in {col} ({label})",
                f"CUSUM indicates a {direction} shift.",
                "CUSUM exceeded decision interval.",
                evidence,
                where=where,
                severity="warn",
                confidence=min(1.0, max(c_plus[idx], c_minus[idx]) / max(h, 1e-6)),
            )
            findings.append(finding)
            if len(findings) >= max_findings:
                break
        if len(findings) >= max_findings:
            break

    artifacts = []
    if findings:
        artifacts.append(_artifact(ctx, plugin_id, "cusum_alarms.json", findings, "json"))
    summary = _summary_or_skip("CUSUM alarms detected", findings)
    return PluginResult("ok", summary, _basic_metrics(df, sample_meta), findings, artifacts, None)


def _split_pre_post(df: pd.DataFrame, time_col: str | None, fraction: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    if time_col and time_col in df.columns:
        df = _sort_by_time(df, time_col)
    if df.empty:
        return df, df
    cut = max(1, int(len(df) * fraction))
    left = df.iloc[:cut]
    right = df.iloc[cut:]
    return left, right


def _two_sample_numeric_plugin(
    plugin_id: str,
    test_name: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    numeric_cols = inferred.get("numeric_columns") or []
    time_col = inferred.get("time_column")
    if not numeric_cols:
        return PluginResult("skipped", "No numeric columns detected", _basic_metrics(df, sample_meta), [], [], None)
    left_df, right_df = _split_pre_post(df, time_col, fraction=0.5)
    if left_df.empty or right_df.empty:
        return PluginResult("skipped", "Insufficient data for two-sample comparison", _basic_metrics(df, sample_meta), [], [], None)
    max_findings = int(config.get("max_findings", 30))
    findings: list[dict[str, Any]] = []
    scored: list[tuple[float, dict[str, Any]]] = []
    for col in numeric_cols:
        if timer.exceeded():
            break
        left = left_df[col].to_numpy(dtype=float)
        right = right_df[col].to_numpy(dtype=float)
        p_value, stat = _two_sample_numeric(left, right, test_name)
        effect = _effect_size(left, right)
        score = (1.0 - min(p_value, 1.0)) * abs(effect)
        evidence = {"metrics": {"p_value": p_value, "statistic": stat, "effect_size": effect, "test": test_name}}
        finding = _make_finding(
            plugin_id,
            f"{col}:{test_name}",
            f"Distribution shift in {col}",
            "Pre/post distributions differ.",
            "Two-sample test indicates a shift.",
            evidence,
            where={"column": col},
            severity="warn" if p_value < 0.1 else "info",
            confidence=1.0 - min(p_value, 1.0),
        )
        scored.append((score, finding))
    scored.sort(key=lambda item: item[0], reverse=True)
    findings = [item[1] for item in scored[:max_findings] if item[0] > 0]
    artifacts = []
    if findings:
        artifacts.append(_artifact(ctx, plugin_id, "two_sample_results.json", findings, "json"))
    summary = _summary_or_skip("Two-sample test completed", findings)
    return PluginResult("ok", summary, _basic_metrics(df, sample_meta), findings, artifacts, None)


def _two_sample_categorical_chi2(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    cat_cols = inferred.get("categorical_columns") or []
    time_col = inferred.get("time_column")
    if not cat_cols:
        return PluginResult("skipped", "No categorical columns detected", _basic_metrics(df, sample_meta), [], [], None)
    left_df, right_df = _split_pre_post(df, time_col, fraction=0.5)
    if left_df.empty or right_df.empty:
        return PluginResult("skipped", "Insufficient data for categorical comparison", _basic_metrics(df, sample_meta), [], [], None)
    max_findings = int(config.get("max_findings", 30))
    scored: list[tuple[float, dict[str, Any]]] = []
    for col in cat_cols:
        if timer.exceeded():
            break
        left_counts = left_df[col].astype(str).value_counts()
        right_counts = right_df[col].astype(str).value_counts()
        categories = list(set(left_counts.index).union(right_counts.index))
        table = np.vstack(
            [
                [left_counts.get(cat, 0) for cat in categories],
                [right_counts.get(cat, 0) for cat in categories],
            ]
        )
        if table.sum() == 0:
            continue
        if HAS_SCIPY:
            chi2, p, _, _ = scipy_stats.chi2_contingency(table)
            stat = float(chi2)
        else:
            p = 1.0
            stat = 0.0
        effect = cramers_v(table)
        score = (1.0 - min(p, 1.0)) * effect
        evidence = {"metrics": {"p_value": float(p), "statistic": stat, "effect_size": effect}}
        finding = _make_finding(
            plugin_id,
            f"{col}:chi2",
            f"Category mix shift in {col}",
            "Category distribution changed between periods.",
            "Chi-square test indicates a shift.",
            evidence,
            where={"column": col},
            severity="warn" if p < 0.1 else "info",
            confidence=1.0 - min(p, 1.0),
        )
        scored.append((score, finding))
    scored.sort(key=lambda item: item[0], reverse=True)
    findings = [item[1] for item in scored[:max_findings] if item[0] > 0]
    artifacts = []
    if findings:
        artifacts.append(_artifact(ctx, plugin_id, "chi2_results.json", findings, "json"))
    summary = _summary_or_skip("Categorical two-sample test completed", findings)
    return PluginResult("ok", summary, _basic_metrics(df, sample_meta), findings, artifacts, None)


def _kernel_two_sample_mmd(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    numeric_cols = inferred.get("numeric_columns") or []
    time_col = inferred.get("time_column")
    if not numeric_cols:
        return PluginResult("skipped", "No numeric columns detected", _basic_metrics(df, sample_meta), [], [], None)
    left_df, right_df = _split_pre_post(df, time_col, fraction=0.5)
    if left_df.empty or right_df.empty:
        return PluginResult("skipped", "Insufficient data for MMD", _basic_metrics(df, sample_meta), [], [], None)
    X = left_df[numeric_cols].to_numpy(dtype=float)
    Y = right_df[numeric_cols].to_numpy(dtype=float)
    max_points = min(1000, X.shape[0], Y.shape[0])
    X = X[:max_points]
    Y = Y[:max_points]
    if X.size == 0 or Y.size == 0:
        return PluginResult("skipped", "Insufficient numeric data", _basic_metrics(df, sample_meta), [], [], None)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
    clip_abs = float(config.get("mmd_clip_abs", 1e6))
    X = np.clip(X, -clip_abs, clip_abs)
    Y = np.clip(Y, -clip_abs, clip_abs)
    if not (np.isfinite(X).all() and np.isfinite(Y).all()):
        return PluginResult(
            "na",
            "Not applicable: invalid_numeric_matrix",
            _basic_metrics(df, sample_meta),
            [
                {
                    "kind": "plugin_not_applicable",
                    "reason_code": "INVALID_NUMERIC_MATRIX",
                    "reason": "invalid_numeric_matrix",
                    "recommended_next_step": "Ensure numeric columns contain finite values after normalization.",
                }
            ],
            [],
            None,
            debug={"gating_reason": "invalid_numeric_matrix"},
        )
    # RBF kernel with median heuristic
    combined = np.vstack([X, Y])
    try:
        if HAS_SKLEARN and pairwise_distances is not None:
            dists = pairwise_distances(combined, combined)
            median = np.median(dists)
        else:
            median = np.median(np.linalg.norm(combined[:, None, :] - combined[None, :, :], axis=-1))
    except Exception as exc:
        return PluginResult(
            "na",
            "Not applicable: mmd_distance_computation_failed",
            _basic_metrics(df, sample_meta),
            [
                {
                    "kind": "plugin_not_applicable",
                    "reason_code": "MMD_DISTANCE_COMPUTATION_FAILED",
                    "reason": "mmd_distance_computation_failed",
                    "recommended_next_step": "Reduce feature scale and verify finite numeric values for MMD.",
                }
            ],
            [],
            None,
            debug={"gating_reason": "mmd_distance_computation_failed", "exception": str(exc)},
        )
    gamma = 1.0 / max(median ** 2, 1e-6)

    def _rbf(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if HAS_SKLEARN and pairwise_distances is not None:
            d = pairwise_distances(a, b)
        else:
            d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
        return np.exp(-gamma * (d ** 2))

    k_xx = _rbf(X, X)
    k_yy = _rbf(Y, Y)
    k_xy = _rbf(X, Y)
    mmd2 = float(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())
    evidence = {"metrics": {"mmd2": mmd2, "gamma": gamma, "sample_size": int(max_points)}}
    finding = _make_finding(
        plugin_id,
        "mmd",
        "Kernel two-sample divergence",
        "Kernel MMD indicates distribution shift.",
        "RBF-kernel MMD positive between periods.",
        evidence,
        where={"columns": numeric_cols[:5]},
        severity="warn" if mmd2 > 0.1 else "info",
        confidence=min(1.0, max(mmd2, 0.0)),
    )
    artifacts = [_artifact(ctx, plugin_id, "mmd_result.json", evidence, "json")]
    summary = "Kernel MMD computed"
    return PluginResult("ok", summary, _basic_metrics(df, sample_meta), [finding], artifacts, None)


def _effect_size_report(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    numeric_cols = inferred.get("numeric_columns") or []
    group_cols = inferred.get("group_by") or []
    if not numeric_cols or not group_cols:
        return PluginResult("skipped", "Need numeric + group columns", _basic_metrics(df, sample_meta), [], [], None)
    max_findings = int(config.get("max_findings", 30))
    findings: list[dict[str, Any]] = []
    for group_col in group_cols:
        if timer.exceeded():
            break
        nonnull = df[group_col].dropna().astype(str)
        if nonnull.nunique() < 2:
            continue
        top_group = nonnull.value_counts().index[0]
        # IMPORTANT: build masks aligned to df.index; do not use dropna()-derived Series as a mask.
        mask_nonnull = df[group_col].notna()
        mask_top = mask_nonnull & (df[group_col].astype(str) == str(top_group))
        left_df = df.loc[mask_top]
        right_df = df.loc[mask_nonnull & (~mask_top)]
        for col in numeric_cols:
            if timer.exceeded():
                break
            left = left_df[col].to_numpy(dtype=float)
            right = right_df[col].to_numpy(dtype=float)
            effect = _effect_size(left, right)
            if abs(effect) < 0.5:
                continue
            evidence = {"metrics": {"effect_size": effect, "group": top_group}}
            finding = _make_finding(
                plugin_id,
                f"{group_col}:{col}:{top_group}",
                f"Effect size difference in {col}",
                f"{top_group} differs from other groups.",
                "Standardized median difference exceeds threshold.",
                evidence,
                where={"column": col, "group": {group_col: top_group}},
                severity="warn" if abs(effect) > 1.0 else "info",
                confidence=min(1.0, abs(effect)),
            )
            findings.append(finding)
            if len(findings) >= max_findings:
                break
        if len(findings) >= max_findings:
            break
    artifacts = []
    if findings:
        artifacts.append(_artifact(ctx, plugin_id, "effect_sizes.json", findings, "json"))
    summary = _summary_or_skip("Effect size report generated", findings)
    return PluginResult("ok", summary, _basic_metrics(df, sample_meta), findings, artifacts, None)


def _multiple_testing_fdr(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    numeric_cols = inferred.get("numeric_columns") or []
    time_col = inferred.get("time_column")
    if not numeric_cols:
        return PluginResult("skipped", "No numeric columns detected", _basic_metrics(df, sample_meta), [], [], None)
    left_df, right_df = _split_pre_post(df, time_col, fraction=0.5)
    if left_df.empty or right_df.empty:
        return PluginResult("skipped", "Insufficient data for FDR screening", _basic_metrics(df, sample_meta), [], [], None)
    pvals: list[float] = []
    stats: list[float] = []
    for col in numeric_cols:
        if timer.exceeded():
            break
        p, stat = _two_sample_numeric(
            left_df[col].to_numpy(dtype=float),
            right_df[col].to_numpy(dtype=float),
            "ks",
        )
        pvals.append(float(p))
        stats.append(float(stat))
    if not pvals:
        return PluginResult("skipped", "No p-values computed", _basic_metrics(df, sample_meta), [], [], None)
    qvals, _ = bh_fdr(pvals)
    findings = []
    for col, p, q, stat in zip(numeric_cols, pvals, qvals, stats):
        if q > 0.1:
            continue
        evidence = {"metrics": {"p_value": p, "q_value": float(q), "statistic": stat}}
        finding = _make_finding(
            plugin_id,
            f"{col}:fdr",
            f"FDR-significant shift in {col}",
            "Adjusted p-value passes FDR threshold.",
            "Benjamini-Hochberg procedure identifies a shift.",
            evidence,
            where={"column": col},
            severity="warn",
            confidence=1.0 - min(q, 1.0),
        )
        findings.append(finding)
    artifacts = []
    if findings:
        artifacts.append(_artifact(ctx, plugin_id, "fdr_results.json", findings, "json"))
    summary = _summary_or_skip("FDR screening completed", findings)
    return PluginResult("ok", summary, _basic_metrics(df, sample_meta), findings, artifacts, None)


def _change_impact_pre_post(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    numeric_cols = inferred.get("numeric_columns") or []
    time_col = inferred.get("time_column")
    if not numeric_cols:
        return PluginResult("skipped", "No numeric columns detected", _basic_metrics(df, sample_meta), [], [], None)
    left_df, right_df = _split_pre_post(df, time_col, fraction=0.5)
    if left_df.empty or right_df.empty:
        return PluginResult("skipped", "Insufficient data for pre/post impact", _basic_metrics(df, sample_meta), [], [], None)
    max_findings = int(config.get("max_findings", 30))
    scored: list[tuple[float, dict[str, Any]]] = []
    for col in numeric_cols:
        if timer.exceeded():
            break
        left = left_df[col].to_numpy(dtype=float)
        right = right_df[col].to_numpy(dtype=float)
        effect = _effect_size(left, right)
        delta = float(np.nanmedian(right) - np.nanmedian(left))
        score = abs(effect)
        evidence = {"metrics": {"pre_median": float(np.nanmedian(left)), "post_median": float(np.nanmedian(right)), "delta": delta, "effect_size": effect}}
        finding = _make_finding(
            plugin_id,
            f"{col}:prepost",
            f"Pre/post change in {col}",
            "Post period differs from pre period.",
            "Median shift observed between periods.",
            evidence,
            where={"column": col},
            severity="warn" if abs(effect) > 0.5 else "info",
            confidence=min(1.0, abs(effect)),
        )
        scored.append((score, finding))
    scored.sort(key=lambda item: item[0], reverse=True)
    findings = [item[1] for item in scored[:max_findings] if item[0] > 0]
    artifacts = []
    if findings:
        artifacts.append(_artifact(ctx, plugin_id, "pre_post_impact.json", findings, "json"))
    summary = _summary_or_skip("Pre/post impact computed", findings)
    return PluginResult("ok", summary, _basic_metrics(df, sample_meta), findings, artifacts, None)

HANDLERS: dict[str, Callable[..., PluginResult]] = {
    "analysis_control_chart_individuals": _control_chart_individuals,
    "analysis_control_chart_ewma": _control_chart_ewma,
    "analysis_control_chart_cusum": _control_chart_cusum,
    "analysis_two_sample_numeric_ks": _wrap_two_sample("ks"),
    "analysis_two_sample_numeric_ad": _wrap_two_sample("ad"),
    "analysis_two_sample_numeric_mann_whitney": _wrap_two_sample("mw"),
    "analysis_two_sample_categorical_chi2": _two_sample_categorical_chi2,
    "analysis_kernel_two_sample_mmd": _kernel_two_sample_mmd,
    "analysis_effect_size_report": _effect_size_report,
    "analysis_multiple_testing_fdr": _multiple_testing_fdr,
    "analysis_change_impact_pre_post": _change_impact_pre_post,
    "analysis_local_outlier_factor": _local_outlier_factor,
    "analysis_one_class_svm": _one_class_svm_plugin,
    "analysis_robust_covariance_outliers": _robust_covariance_outliers,
    "analysis_evt_gumbel_tail": _evt_gumbel_tail,
    "analysis_evt_peaks_over_threshold": _evt_peaks_over_threshold,
    "analysis_matrix_profile_motifs_discords": _matrix_profile_discords,
    "analysis_burst_detection_kleinberg": lambda *args: _burst_detection(*args, term_mode=False),
    "analysis_event_count_bocpd_poisson": _event_count_bocpd_poisson,
    "analysis_hawkes_self_exciting": _hawkes_self_exciting,
    "analysis_periodicity_spectral_scan": _periodicity_spectral_scan,
    "analysis_state_space_kalman_residuals": _kalman_residuals,
    "analysis_multivariate_t2_control": _multivariate_t2_control,
    "analysis_multivariate_ewma_control": _multivariate_ewma_control,
    "analysis_pca_control_chart": _pca_control_chart,
    "analysis_changepoint_pelt": _changepoint_pelt,
    "analysis_changepoint_energy_edivisive": _changepoint_energy_edivisive,
    "analysis_drift_adwin": _drift_adwin,
    "analysis_changepoint_method_survey_guided": _changepoint_method_survey_guided,
    "analysis_markov_transition_shift": _markov_transition_shift,
    "analysis_sequential_patterns_prefixspan": _sequential_patterns_prefixspan,
    "analysis_hmm_latent_state_sequences": _hmm_latent_state_sequences,
    "analysis_dependency_graph_change_detection": _dependency_graph_change_detection,
    "analysis_graphical_lasso_dependency_network": _graphical_lasso_dependency_network,
    "analysis_mutual_information_screen": _mutual_information_screen,
    "analysis_transfer_entropy_directional": _transfer_entropy_directional,
    "analysis_lagged_predictability_test": _lagged_predictability_test,
    "analysis_copula_dependence": _copula_dependence,
    "analysis_conformance_alignments": _conformance_alignments,
    "analysis_process_drift_conformance_over_time": _process_drift_conformance_over_time,
    "analysis_variant_differential": _variant_differential,
    "analysis_template_drift_two_sample": _template_drift_two_sample,
    "analysis_message_entropy_drift": _message_entropy_drift,
    "analysis_topic_model_lda": _topic_model_lda,
    "analysis_term_burst_kleinberg": _term_burst_kleinberg,
    "analysis_survival_kaplan_meier": _survival_kaplan_meier,
    "analysis_proportional_hazards_duration": _proportional_hazards_duration,
    "analysis_quantile_regression_duration": _quantile_regression_duration,
    "analysis_queue_model_fit": _queue_model_fit,
    "analysis_littles_law_consistency": _littles_law_consistency,
    "analysis_kingman_vut_approx": _kingman_vut_approx,
}

# Topo/TDA add-on pack handlers live in a separate module to keep this file tractable.
HANDLERS.update(TOPO_TDA_ADDON_HANDLERS)
HANDLERS.update(IDEASPACE_HANDLERS)
HANDLERS.update(ERP_NEXT_WAVE_HANDLERS)
HANDLERS.update(NEXT30_HANDLERS)
HANDLERS.update(NEXT30B_HANDLERS)
