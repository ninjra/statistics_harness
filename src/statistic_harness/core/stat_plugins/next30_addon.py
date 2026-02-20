from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from statistic_harness.core.stat_plugins import (
    BudgetTimer,
    bh_fdr,
    deterministic_sample,
    infer_columns,
    robust_center_scale,
    robust_zscores,
    stable_id,
)
from statistic_harness.core.types import PluginArtifact, PluginResult

try:  # optional
    from sklearn.cross_decomposition import CCA  # type: ignore
    from sklearn.decomposition import FastICA, SparsePCA, PCA  # type: ignore
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor  # type: ignore
    from sklearn.linear_model import Ridge  # type: ignore
    from sklearn.preprocessing import SplineTransformer  # type: ignore

    HAS_SKLEARN = True
except Exception:  # pragma: no cover - optional dependency
    CCA = FastICA = SparsePCA = PCA = None
    GradientBoostingRegressor = RandomForestRegressor = None
    Ridge = SplineTransformer = None
    HAS_SKLEARN = False


NEXT30_IDS: tuple[str, ...] = (
    "analysis_bsts_intervention_counterfactual_v1",
    "analysis_stl_seasonal_decompose_v1",
    "analysis_seasonal_holt_winters_forecast_residuals_v1",
    "analysis_lomb_scargle_periodogram_v1",
    "analysis_garch_volatility_shift_v1",
    "analysis_bayesian_online_changepoint_studentt_v1",
    "analysis_wild_binary_segmentation_v1",
    "analysis_fused_lasso_trend_filtering_v1",
    "analysis_cusum_on_model_residuals_v1",
    "analysis_change_score_consensus_v1",
    "analysis_benfords_law_anomaly_v1",
    "analysis_geometric_median_multivariate_location_v1",
    "analysis_random_matrix_marchenko_pastur_denoise_v1",
    "analysis_outlier_influence_cooks_distance_v1",
    "analysis_heavy_tail_index_hill_v1",
    "analysis_distance_correlation_screen_v1",
    "analysis_gam_spline_regression_v1",
    "analysis_quantile_loss_boosting_v1",
    "analysis_quantile_regression_forest_v1",
    "analysis_sparse_pca_interpretable_components_v1",
    "analysis_ica_source_separation_v1",
    "analysis_cca_crossblock_association_v1",
    "analysis_factor_rotation_varimax_v1",
    "analysis_subspace_tracking_oja_v1",
    "analysis_multicollinearity_vif_screen_v1",
    "analysis_zero_inflated_count_model_v1",
    "analysis_negative_binomial_overdispersion_v1",
    "analysis_dirichlet_multinomial_categorical_overdispersion_v1",
    "analysis_fisher_exact_enrichment_v1",
    "analysis_recurrence_quantification_rqa_v1",
)


class _QuadraticCapExceeded(RuntimeError):
    pass


def _safe_id(plugin_id: str, key: str) -> str:
    try:
        return stable_id(f"{plugin_id}:{key}")
    except Exception:
        return hashlib.sha256(f"{plugin_id}:{key}".encode("utf-8")).hexdigest()[:16]


def _artifact_json(
    ctx,
    plugin_id: str,
    filename: str,
    payload: Any,
    description: str,
) -> PluginArtifact:
    artifacts_dir = ctx.artifacts_dir(plugin_id)
    path = artifacts_dir / filename
    path.write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return PluginArtifact(
        path=str(path.relative_to(ctx.run_dir)),
        type="json",
        description=description,
    )


def _basic_metrics(df: pd.DataFrame, sample_meta: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        "rows_seen": int(sample_meta.get("rows_total", len(df))),
        "rows_used": int(sample_meta.get("rows_used", len(df))),
        "cols_used": int(len(df.columns)),
    }
    metrics.update(sample_meta or {})
    return metrics


def _make_finding(
    plugin_id: str,
    key: str,
    title: str,
    what: str,
    why: str,
    evidence: dict[str, Any],
    *,
    recommendation: str,
    severity: str = "info",
    confidence: float = 0.5,
    where: dict[str, Any] | None = None,
    measurement_type: str = "measured",
) -> dict[str, Any]:
    return {
        "id": _safe_id(plugin_id, key),
        "severity": severity,
        "confidence": float(max(0.0, min(1.0, confidence))),
        "title": title,
        "what": what,
        "why": why,
        "evidence": evidence,
        "where": where or {},
        "recommendation": recommendation,
        "measurement_type": measurement_type,
    }


def _ok_with_reason(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    sample_meta: dict[str, Any],
    reason: str,
    *,
    debug: dict[str, Any] | None = None,
) -> PluginResult:
    ctx.logger(f"SKIP reason={reason}")
    payload = dict(debug or {})
    payload.setdefault("gating_reason", reason)
    return PluginResult(
        "ok",
        f"No actionable result: {reason}",
        _basic_metrics(df, sample_meta),
        [],
        [],
        None,
        debug=payload,
    )


def _finalize(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    sample_meta: dict[str, Any],
    summary: str,
    findings: list[dict[str, Any]],
    artifacts: list[PluginArtifact],
    *,
    extra_metrics: dict[str, Any] | None = None,
    debug: dict[str, Any] | None = None,
) -> PluginResult:
    metrics = _basic_metrics(df, sample_meta)
    if extra_metrics:
        metrics.update(extra_metrics)
    ctx.logger(
        f"END runtime_ms={int(metrics.get('runtime_ms', 0))} findings={len(findings)}"
    )
    return PluginResult(
        "ok",
        summary,
        metrics,
        findings,
        artifacts,
        None,
        debug=debug or {},
    )


def _log_start(ctx, plugin_id: str, df: pd.DataFrame, config: dict[str, Any], inferred: dict[str, Any]) -> None:
    ctx.logger(
        f"START plugin_id={plugin_id} rows={len(df)} cols={len(df.columns)} seed={int(config.get('seed', 0))}"
    )
    ctx.logger(
        f"INFER time={inferred.get('time_column')} numeric={len(inferred.get('numeric_columns') or [])} cat={len(inferred.get('categorical_columns') or [])}"
    )


def _ensure_budget(timer: BudgetTimer) -> None:
    timer.ensure("time_budget_exceeded")


def _cap_for_quadratic(
    plugin_id: str,
    ctx,
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    max_points = int(config.get("plugin", {}).get("max_points_for_quadratic", 2000))
    if len(frame) <= max_points:
        return frame
    if bool(config.get("allow_row_sampling", False)):
        sampled, _ = deterministic_sample(frame, max_points, seed=int(config.get("seed", 1337)))
        ctx.logger(
            f"SKIP reason=quadratic_capped_sampling rows={len(frame)} cap={max_points} sampled={len(sampled)}"
        )
        return sampled
    raise _QuadraticCapExceeded(
        f"{plugin_id}: quadratic cap exceeded rows={len(frame)} cap={max_points}"
    )


def _numeric_columns(df: pd.DataFrame, inferred: dict[str, Any], max_cols: int | None = None) -> list[str]:
    cols = [str(c) for c in (inferred.get("numeric_columns") or []) if str(c) in df.columns]
    if not cols:
        cols = [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if max_cols is not None:
        cols = cols[: max(1, int(max_cols))]
    return cols


def _categorical_columns(df: pd.DataFrame, inferred: dict[str, Any]) -> list[str]:
    cols = [str(c) for c in (inferred.get("categorical_columns") or []) if str(c) in df.columns]
    if not cols:
        cols = [str(c) for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    return cols


def _series_from_time_value(
    df: pd.DataFrame,
    inferred: dict[str, Any],
    *,
    value_col: str | None = None,
) -> tuple[pd.Series, pd.Series, str, str] | None:
    def _parse_time(series: pd.Series) -> pd.Series:
        try:
            return pd.to_datetime(series, errors="coerce", utc=False)
        except Exception:
            return pd.to_datetime(series.astype(str), errors="coerce", utc=False)

    time_col = inferred.get("time_column")
    if not isinstance(time_col, str) or time_col not in df.columns:
        candidates = [c for c in df.columns if "time" in str(c).lower() or "date" in str(c).lower() or "dt" in str(c).lower()]
        if not candidates:
            return None
        time_col = str(candidates[0])
    numeric_cols = _numeric_columns(df, inferred, max_cols=80)
    if not numeric_cols:
        return None
    if value_col is None:
        best_var = -1.0
        best_col = None
        for c in numeric_cols:
            vals = pd.to_numeric(df[c], errors="coerce")
            var = float(np.nanvar(vals.to_numpy(dtype=float)))
            if var > best_var:
                best_var = var
                best_col = c
        if best_col is None:
            return None
        value_col = best_col
    ts = _parse_time(df[time_col])
    vals = pd.to_numeric(df[value_col], errors="coerce")
    ok = ts.notna() & vals.notna()
    if int(ok.sum()) < 20:
        return None
    return ts[ok], vals[ok], str(time_col), str(value_col)


def _bucket_series(ts: pd.Series, values: pd.Series) -> tuple[pd.Series, str]:
    frame = pd.DataFrame({"ts": ts, "value": values}).sort_values("ts")
    deltas = frame["ts"].diff().dt.total_seconds().dropna()
    freq = "D"
    if not deltas.empty and float(deltas.median()) <= 3600.0:
        # Pandas 2.x frequency aliases are lower-case.
        freq = "h"
    try:
        series = (
            frame.set_index("ts")["value"]
            .resample(freq)
            .mean()
            .dropna()
        )
    except Exception:
        freq = "D"
        series = (
            frame.set_index("ts")["value"]
            .resample(freq)
            .mean()
            .dropna()
        )
    return series, freq


def _severity_for_effect(value: float, warn: float = 0.5, critical: float = 1.0) -> str:
    if abs(value) >= critical:
        return "critical"
    if abs(value) >= warn:
        return "warn"
    return "info"


def _runtime_ms(timer: BudgetTimer) -> int:
    return int(max(0.0, timer.elapsed_ms()))


def _first_nonzero_digit(value: float) -> int | None:
    x = abs(float(value))
    if x == 0.0 or math.isnan(x) or math.isinf(x):
        return None
    while x < 1.0:
        x *= 10.0
    while x >= 10.0:
        x /= 10.0
    d = int(x)
    if 1 <= d <= 9:
        return d
    return None


def _soft_threshold(x: np.ndarray, lam: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


def _pick_integer_count_column(df: pd.DataFrame, inferred: dict[str, Any]) -> str | None:
    for c in _numeric_columns(df, inferred, max_cols=80):
        s = pd.to_numeric(df[c], errors="coerce")
        s = s[s.notna()]
        if s.empty:
            continue
        if float((s >= 0).mean()) < 0.9:
            continue
        int_like = float((np.isclose(s.to_numpy(dtype=float) % 1.0, 0.0)).mean())
        if int_like >= 0.9:
            return c
    return None


def _top_variance_pair(df: pd.DataFrame, inferred: dict[str, Any]) -> tuple[str, str] | None:
    cols = _numeric_columns(df, inferred, max_cols=20)
    if len(cols) < 2:
        return None
    scored = []
    for c in cols:
        vals = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        scored.append((float(np.nanvar(vals)), c))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return scored[0][1], scored[1][1]


def _best_latency_col(df: pd.DataFrame, inferred: dict[str, Any]) -> str | None:
    hint_tokens = ("duration", "wait", "latency", "elapsed", "queue", "time")
    candidates = _numeric_columns(df, inferred, max_cols=80)
    for c in candidates:
        lower = c.lower()
        if any(t in lower for t in hint_tokens):
            return c
    if candidates:
        return candidates[0]
    return None


def _cov_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return 0.0
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx <= 0.0 or sy <= 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _window_cp_score(y: np.ndarray, window: int) -> tuple[int, float]:
    n = len(y)
    if n < (2 * window + 1):
        return max(0, n // 2), 0.0
    best_i = window
    best_s = -1.0
    for i in range(window, n - window):
        pre = y[i - window : i]
        post = y[i : i + window]
        med_pre = float(np.median(pre))
        med_post = float(np.median(post))
        _, mad = robust_center_scale(pre)
        score = abs(med_post - med_pre) / max(1e-6, mad)
        if score > best_s:
            best_s = score
            best_i = i
    return best_i, float(max(best_s, 0.0))


def _to_array(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)


def _downsample_series(y: np.ndarray, max_points: int) -> tuple[np.ndarray, int]:
    cap = max(200, int(max_points))
    n = int(len(y))
    if n <= cap:
        return y, 1
    step = int(math.ceil(float(n) / float(cap)))
    return y[::step], step


def _handler_bsts_intervention_counterfactual_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    packed = _series_from_time_value(df, inferred)
    if packed is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "missing_time_or_numeric")
    ts, vals, time_col, value_col = packed
    series, freq = _bucket_series(ts, vals)
    y = series.to_numpy(dtype=float)
    n = len(y)
    if n < 30:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_points")
    window = max(7, n // 20)
    roll = pd.Series(y).rolling(window=window, min_periods=max(3, window // 2)).mean()
    deriv = roll.diff().abs().fillna(0.0).to_numpy(dtype=float)
    t0 = int(np.argmax(deriv))
    t0 = min(max(window, t0), n - max(5, window))
    pre = y[:t0]
    post = y[t0:]
    if len(pre) < 10 or len(post) < 5:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_pre_post_split")
    alpha = 0.2
    level = float(np.median(pre))
    for value in pre:
        level = level + alpha * (float(value) - level)
    post_pred = np.full(shape=len(post), fill_value=level, dtype=float)
    delta = float(np.mean(post) - np.mean(post_pred))
    _, scale = robust_center_scale(pre)
    effect = delta / max(scale, 1e-6)
    sev = _severity_for_effect(effect, 0.5, 1.0)
    finding = _make_finding(
        plugin_id,
        "counterfactual_effect",
        "Intervention counterfactual effect",
        "Detected a post-intervention level shift relative to a local-level counterfactual.",
        "A large standardized post-pre deviation suggests a structural shift.",
        {
            "metrics": {
                "t0_index": t0,
                "delta": delta,
                "effect_size": effect,
                "pre_mean": float(np.mean(pre)),
                "post_mean": float(np.mean(post)),
                "pred_post_mean": float(np.mean(post_pred)),
            }
        },
        recommendation="Review process/config changes near the intervention index and validate intended impact.",
        severity=sev,
        confidence=min(0.95, 0.55 + min(0.4, abs(effect) / 3.0)),
    )
    artifact_payload = {
        "bucket_freq": freq,
        "time_column": time_col,
        "value_column": value_col,
        "t0_index": t0,
        "pre_mean_actual": float(np.mean(pre)),
        "post_mean_actual": float(np.mean(post)),
        "post_mean_pred": float(np.mean(post_pred)),
        "delta": delta,
        "effect_size": effect,
    }
    artifacts = [
        _artifact_json(
            ctx, plugin_id, "bsts_counterfactual.json", artifact_payload, "BSTS-style counterfactual summary"
        )
    ]
    return _finalize(
        plugin_id,
        ctx,
        df,
        sample_meta,
        "Computed intervention counterfactual effect.",
        [finding],
        artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer)},
    )


def _handler_stl_seasonal_decompose_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    packed = _series_from_time_value(df, inferred)
    if packed is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "missing_time_or_numeric")
    ts, vals, time_col, value_col = packed
    series, freq = _bucket_series(ts, vals)
    y = series.to_numpy(dtype=float)
    n = len(y)
    if n < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_points")
    trend_window = int(config.get("plugin", {}).get("trend_window", 7))
    trend = pd.Series(y).rolling(window=max(3, trend_window), center=True, min_periods=1).median().to_numpy(dtype=float)
    detrended = y - trend
    period = int(config.get("plugin", {}).get("period", 24 if freq == "H" else 7))
    period = max(2, min(period, max(2, n // 2)))
    seasonal_template = np.zeros(period, dtype=float)
    for i in range(period):
        seasonal_template[i] = float(np.mean(detrended[i::period])) if len(detrended[i::period]) else 0.0
    seasonal = np.array([seasonal_template[i % period] for i in range(n)], dtype=float)
    residual = y - trend - seasonal
    rz = robust_zscores(residual)
    residual_z = float(config.get("plugin", {}).get("residual_z", 4.0))
    spike_idx = np.where(np.abs(rz) >= residual_z)[0]
    top_spikes = sorted(((int(i), float(rz[i]), float(residual[i])) for i in spike_idx), key=lambda t: (-abs(t[1]), t[0]))[: int(config.get("max_findings", 30))]
    findings: list[dict[str, Any]] = []
    for i, z, rv in top_spikes:
        findings.append(
            _make_finding(
                plugin_id,
                f"spike:{i}",
                "Seasonal decomposition residual spike",
                "Residual remains large after removing trend and seasonal components.",
                "Large residuals indicate anomalies not explained by baseline trend/seasonality.",
                {"metrics": {"index": i, "residual": rv, "robust_z": z, "period": period}},
                recommendation="Inspect raw events at this timestamp for one-off disruptions or data quality issues.",
                severity="warn" if abs(z) < 6 else "critical",
                confidence=min(0.95, 0.55 + min(0.4, abs(z) / 10.0)),
            )
        )
    artifact = {
        "time_column": time_col,
        "value_column": value_col,
        "bucket_freq": freq,
        "period": period,
        "residual_z_threshold": residual_z,
        "spike_count": int(len(top_spikes)),
        "top_spikes": top_spikes,
    }
    artifacts = [_artifact_json(ctx, plugin_id, "stl_components.json", artifact, "STL-style components and spikes")]
    return _finalize(
        plugin_id,
        ctx,
        df,
        sample_meta,
        "Computed STL-style seasonal decomposition.",
        findings,
        artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "spike_count": len(top_spikes)},
    )


def _handler_seasonal_holt_winters_forecast_residuals_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    packed = _series_from_time_value(df, inferred)
    if packed is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "missing_time_or_numeric")
    ts, vals, time_col, value_col = packed
    series, freq = _bucket_series(ts, vals)
    y = series.to_numpy(dtype=float)
    n = len(y)
    if n < 30:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_points")
    p = int(config.get("plugin", {}).get("period", 24 if freq == "H" else 7))
    p = max(2, min(p, max(2, n // 3)))
    alpha = float(config.get("plugin", {}).get("alpha", 0.2))
    beta = float(config.get("plugin", {}).get("beta", 0.05))
    gamma = float(config.get("plugin", {}).get("gamma", 0.1))
    level = float(y[0])
    trend = float((y[p] - y[0]) / p) if n > p else 0.0
    season = np.zeros(p, dtype=float)
    for i in range(p):
        season[i] = float(y[i] - np.mean(y[:p])) if i < n else 0.0
    one_step = np.zeros(n, dtype=float)
    for t in range(n):
        s = season[t % p]
        one_step[t] = level + trend + s
        obs = float(y[t])
        prev_level = level
        level = alpha * (obs - s) + (1.0 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1.0 - beta) * trend
        season[t % p] = gamma * (obs - level) + (1.0 - gamma) * s
        _ensure_budget(timer)
    residual = y - one_step
    rz = robust_zscores(residual)
    threshold = float(config.get("plugin", {}).get("residual_z", 4.0))
    idx = np.where(np.abs(rz) >= threshold)[0]
    findings = [
        _make_finding(
            plugin_id,
            f"residual_spike:{int(i)}",
            "Holt-Winters residual spike",
            "One-step-ahead forecast error exceeded robust threshold.",
            "Sustained residual spikes indicate model mismatch or process drift.",
            {"metrics": {"index": int(i), "residual": float(residual[i]), "robust_z": float(rz[i])}},
            recommendation="Check regime changes around this interval and adjust seasonal assumptions.",
            severity="warn" if abs(float(rz[i])) < 6 else "critical",
            confidence=min(0.95, 0.6 + min(0.3, abs(float(rz[i])) / 12.0)),
        )
        for i in idx[: int(config.get("max_findings", 30))]
    ]
    artifacts = [
        _artifact_json(
            ctx,
            plugin_id,
            "holt_winters.json",
            {
                "time_column": time_col,
                "value_column": value_col,
                "period": p,
                "bucket_freq": freq,
                "residual_threshold": threshold,
                "spike_count": len(findings),
            },
            "Holt-Winters residual diagnostics",
        )
    ]
    return _finalize(
        plugin_id,
        ctx,
        df,
        sample_meta,
        "Computed additive Holt-Winters residual diagnostics.",
        findings,
        artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "spike_count": len(findings)},
    )


def _handler_lomb_scargle_periodogram_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    packed = _series_from_time_value(df, inferred)
    if packed is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "missing_time_or_numeric")
    ts, vals, time_col, value_col = packed
    frame = pd.DataFrame({"ts": ts, "value": vals}).sort_values("ts")
    t = (frame["ts"] - frame["ts"].min()).dt.total_seconds().to_numpy(dtype=float)
    y = frame["value"].to_numpy(dtype=float)
    if len(t) < 30:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_points")
    deltas = np.diff(t)
    cv = float(np.std(deltas) / max(np.mean(deltas), 1e-6)) if len(deltas) else 0.0
    top_periods: list[float] = []
    peak_ratio = 0.0
    mode = "lomb_fallback"
    if cv < 0.1:
        mode = "fft_regular"
        # near-regular cadence path
        y0 = y - float(np.mean(y))
        spec = np.fft.rfft(y0)
        power = np.abs(spec) ** 2
        if len(power) > 1:
            power = power[1:]
            idx = np.argsort(power)[::-1][:3]
            for i in idx:
                freq = (i + 1) / max(1.0, (t[-1] - t[0]))
                if freq > 0:
                    top_periods.append(float(1.0 / freq))
            peak_ratio = float(np.max(power) / max(np.median(power), 1e-9))
    else:
        duration = max(1.0, float(t[-1] - t[0]))
        nfreq = int(config.get("plugin", {}).get("max_freqs", 256))
        freqs = np.linspace(1.0 / duration, nfreq / (2.0 * duration), nfreq)
        yc = y - float(np.mean(y))
        scores = []
        for f in freqs:
            ang = 2.0 * math.pi * f * t
            s = np.sin(ang)
            c = np.cos(ang)
            cs = _cov_corr(yc, s)
            cc = _cov_corr(yc, c)
            score = cs * cs + cc * cc
            scores.append(float(score))
        if scores:
            arr = np.array(scores, dtype=float)
            idx = np.argsort(arr)[::-1][:3]
            top_periods = [float(1.0 / freqs[i]) for i in idx if freqs[i] > 0]
            peak_ratio = float(np.max(arr) / max(np.median(arr), 1e-9))
    findings: list[dict[str, Any]] = []
    if peak_ratio > 3.0:
        findings.append(
            _make_finding(
                plugin_id,
                "periodicity_strength",
                "Strong periodic component detected",
                "Spectral peak is materially above median spectral density.",
                "Dominant periodicity often drives recurrent queue/latency bursts.",
                {"metrics": {"peak_to_median_ratio": peak_ratio, "top_periods_seconds": top_periods}},
                recommendation="Align staffing/scheduling buffers to the dominant cycle windows.",
                severity="warn" if peak_ratio < 6 else "critical",
                confidence=min(0.95, 0.55 + min(0.4, peak_ratio / 20.0)),
            )
        )
    artifacts = [
        _artifact_json(
            ctx,
            plugin_id,
            "lomb_scargle.json",
            {
                "mode": mode,
                "time_column": time_col,
                "value_column": value_col,
                "top_periods_seconds": top_periods,
                "peak_to_median_ratio": peak_ratio,
            },
            "Lomb-Scargle style periodicity summary",
        )
    ]
    return _finalize(
        plugin_id,
        ctx,
        df,
        sample_meta,
        "Computed periodicity scan.",
        findings,
        artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "peak_ratio": peak_ratio},
    )


def _handler_garch_volatility_shift_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    value_col = _best_latency_col(df, inferred)
    if value_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_column")
    y = _to_array(df[value_col])
    if len(y) < 80:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_points")
    ret = np.diff(y)
    window = int(config.get("plugin", {}).get("window", 50))
    window = min(max(20, window), max(20, len(ret) // 3))
    v_first = float(np.var(ret[:window]))
    v_last = float(np.var(ret[-window:]))
    ratio = v_last / max(v_first, 1e-9)
    sev = "critical" if ratio > 4.0 else "warn" if ratio > 2.0 else "info"
    findings = []
    if ratio > 2.0:
        findings.append(
            _make_finding(
                plugin_id,
                "volatility_ratio",
                "Volatility regime increased",
                "Recent return variance is materially higher than the initial baseline window.",
                "Higher volatility typically increases tail risk and queue instability.",
                {"metrics": {"variance_ratio": ratio, "first_window_var": v_first, "last_window_var": v_last}},
                recommendation="Investigate upstream instability and evaluate adaptive safety buffers.",
                severity=sev,
                confidence=min(0.95, 0.55 + min(0.35, ratio / 12.0)),
            )
        )
    artifacts = [
        _artifact_json(
            ctx,
            plugin_id,
            "volatility_shift.json",
            {"value_column": value_col, "variance_ratio": ratio, "window": window},
            "Volatility shift summary",
        )
    ]
    return _finalize(
        plugin_id,
        ctx,
        df,
        sample_meta,
        "Computed volatility shift ratio.",
        findings,
        artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "variance_ratio": ratio},
    )


def _handler_bayesian_online_changepoint_studentt_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    value_col = _best_latency_col(df, inferred)
    if value_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_column")
    y = _to_array(df[value_col])
    window = int(config.get("plugin", {}).get("window", 50))
    idx, score = _window_cp_score(y, max(10, window))
    sev = "critical" if score > 1.5 else "warn" if score > 0.8 else "info"
    findings = []
    if score > 0.8:
        findings.append(
            _make_finding(
                plugin_id,
                "studentt_cp",
                "Robust changepoint candidate",
                "Median shift detected between adjacent windows.",
                "Heavy-tail robust split indicates distributional shift.",
                {"metrics": {"changepoint_index": idx, "score": score, "window": window}},
                recommendation="Review release/process changes near the detected index.",
                severity=sev,
                confidence=min(0.95, 0.55 + min(0.35, score / 4.0)),
            )
        )
    artifacts = [
        _artifact_json(
            ctx,
            plugin_id,
            "bocpd_studentt.json",
            {"value_column": value_col, "changepoint_index": idx, "score": score, "window": window},
            "Student-t style changepoint summary",
        )
    ]
    return _finalize(
        plugin_id,
        ctx,
        df,
        sample_meta,
        "Computed robust online changepoint score.",
        findings,
        artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "cp_score": score},
    )


def _wbs_candidates(
    y: np.ndarray,
    seed: int,
    min_seg: int = 20,
    K: int = 128,
    timer: BudgetTimer | None = None,
) -> list[int]:
    n = len(y)
    rng = np.random.default_rng(seed)
    out: list[int] = []
    if n < (2 * min_seg + 2):
        return out
    y_eval, step = _downsample_series(y, max_points=4096)
    n_eval = len(y_eval)
    if n_eval < (2 * min_seg + 2):
        return out
    for _ in range(K):
        l = int(rng.integers(0, n_eval - 2 * min_seg))
        r = int(rng.integers(l + 2 * min_seg, n_eval))
        segment = y_eval[l:r]
        m = len(segment)
        if m < (2 * min_seg + 2):
            continue
        csum = np.concatenate(([0.0], np.cumsum(segment)))
        csum2 = np.concatenate(([0.0], np.cumsum(segment * segment)))
        total_sum = float(csum[m])
        total_sum2 = float(csum2[m])
        mean_seg = total_sum / float(m)
        var_seg = max(0.0, total_sum2 / float(m) - mean_seg * mean_seg)
        best_k = None
        best_gain = 0.0
        for k in range(min_seg, m - min_seg):
            left_n = float(k)
            right_n = float(m - k)
            left_sum = float(csum[k])
            left_sum2 = float(csum2[k])
            right_sum = total_sum - left_sum
            right_sum2 = total_sum2 - left_sum2
            left_mean = left_sum / left_n
            right_mean = right_sum / right_n
            left_var = max(0.0, left_sum2 / left_n - left_mean * left_mean)
            right_var = max(0.0, right_sum2 / right_n - right_mean * right_mean)
            gain = float(
                var_seg
                - ((left_var * left_n + right_var * right_n) / float(m))
            )
            if gain > best_gain:
                best_gain = gain
                best_k = (l + k) * step
        if best_k is not None:
            out.append(int(min(n - 1, max(0, int(best_k)))))
        if timer is not None:
            _ensure_budget(timer)
    return out


def _cluster_indices(indices: list[int], tolerance: int = 5, max_points: int = 10) -> list[dict[str, Any]]:
    if not indices:
        return []
    sorted_idx = sorted(indices)
    clusters: list[list[int]] = [[sorted_idx[0]]]
    for idx in sorted_idx[1:]:
        if idx - clusters[-1][-1] <= tolerance:
            clusters[-1].append(idx)
        else:
            clusters.append([idx])
    points = [
        {"index": int(round(float(np.mean(c)))), "votes": len(c), "members": c}
        for c in clusters
    ]
    points.sort(key=lambda x: (-int(x["votes"]), int(x["index"])))
    return points[:max_points]


def _handler_wild_binary_segmentation_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    value_col = _best_latency_col(df, inferred)
    if value_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_column")
    y_full = _to_array(df[value_col])
    y, downsample_step = _downsample_series(
        y_full, int(config.get("plugin", {}).get("max_points", 12000))
    )
    if len(y) < 120:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_points")
    seed = int(config.get("seed", 1337))
    K = int(config.get("plugin", {}).get("intervals", 128))
    try:
        candidates = _wbs_candidates(y, seed=seed, min_seg=20, K=max(16, K), timer=timer)
    except Exception as exc:
        if "time_budget_exceeded" in str(exc):
            return _ok_with_reason(
                plugin_id,
                ctx,
                df,
                sample_meta,
                "time_budget_exceeded",
                debug={"value_column": value_col, "downsample_step": int(downsample_step)},
            )
        raise
    if downsample_step > 1:
        candidates = [int(min(len(y_full) - 1, max(0, c * downsample_step))) for c in candidates]
    clustered = _cluster_indices(candidates, tolerance=5, max_points=int(config.get("plugin", {}).get("max_changepoints", 10)))
    findings = [
        _make_finding(
            plugin_id,
            f"cp:{p['index']}",
            "WBS changepoint cluster",
            "Repeated interval splits agree on a changepoint location.",
            "Consensus across random intervals reduces single-window bias.",
            {"metrics": {"index": int(p["index"]), "votes": int(p["votes"])}},
            recommendation="Prioritize high-vote changepoints for root-cause triage.",
            severity="warn" if int(p["votes"]) < 6 else "critical",
            confidence=min(0.95, 0.5 + min(0.4, int(p["votes"]) / 20.0)),
        )
        for p in clustered
    ]
    artifacts = [
        _artifact_json(
            ctx,
            plugin_id,
            "wbs_changepoints.json",
            {"value_column": value_col, "candidates": candidates[:1000], "clusters": clustered},
            "WBS changepoint candidates",
        )
    ]
    return _finalize(
        plugin_id,
        ctx,
        df,
        sample_meta,
        "Computed wild binary segmentation changepoints.",
        findings,
        artifacts,
        extra_metrics={
            "runtime_ms": _runtime_ms(timer),
            "clusters": len(clustered),
            "downsample_step": int(downsample_step),
        },
    )


def _handler_fused_lasso_trend_filtering_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    value_col = _best_latency_col(df, inferred)
    if value_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_column")
    y = _to_array(df[value_col])
    if len(y) < 40:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_points")
    level = y.copy()
    lam = float(config.get("plugin", {}).get("lam", max(0.1, float(np.std(y)) * 0.05)))
    iters = int(config.get("plugin", {}).get("iters", 50))
    for _ in range(max(5, iters)):
        d = np.diff(level)
        d2 = _soft_threshold(d, lam)
        level = np.concatenate([[level[0]], level[0] + np.cumsum(d2)])
        _ensure_budget(timer)
    break_idx = np.where(np.abs(np.diff(level)) > float(config.get("plugin", {}).get("eps", 1e-3)))[0]
    findings = [
        _make_finding(
            plugin_id,
            f"break:{int(i)}",
            "Trend filtering breakpoint",
            "Smoothed trend derivative indicates a local break.",
            "Piecewise trend changes often correspond to process phase shifts.",
            {"metrics": {"index": int(i), "delta": float(np.diff(level)[i])}},
            recommendation="Review process changes around breakpoint clusters.",
            severity="warn",
            confidence=0.55,
        )
        for i in break_idx[: int(config.get("max_findings", 30))]
    ]
    artifacts = [
        _artifact_json(
            ctx,
            plugin_id,
            "trend_filtering.json",
            {"value_column": value_col, "lambda": lam, "breakpoints": [int(i) for i in break_idx.tolist()]},
            "Fused-lasso-style trend filtering",
        )
    ]
    return _finalize(
        plugin_id,
        ctx,
        df,
        sample_meta,
        "Computed trend filtering breakpoints.",
        findings,
        artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "breakpoints": len(break_idx)},
    )


def _handler_cusum_on_model_residuals_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    value_col = _best_latency_col(df, inferred)
    if value_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_column")
    y = _to_array(df[value_col])
    if len(y) < 40:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_points")
    trend = pd.Series(y).rolling(window=25, min_periods=3, center=True).median().interpolate().bfill().ffill().to_numpy(dtype=float)
    r = y - trend
    center, scale = robust_center_scale(r)
    z = (r - center) / max(scale, 1e-6)
    k = float(config.get("plugin", {}).get("k", 0.5))
    h = float(config.get("plugin", {}).get("h", 5.0))
    pos = 0.0
    neg = 0.0
    alarm_idx = None
    direction = "none"
    magnitude = 0.0
    for i, zi in enumerate(z):
        pos = max(0.0, pos + zi - k)
        neg = min(0.0, neg + zi + k)
        if pos > h:
            alarm_idx = i
            direction = "positive"
            magnitude = pos
            break
        if abs(neg) > h:
            alarm_idx = i
            direction = "negative"
            magnitude = abs(neg)
            break
    findings = []
    if alarm_idx is not None:
        findings.append(
            _make_finding(
                plugin_id,
                f"cusum:{alarm_idx}",
                "CUSUM residual alarm",
                "CUSUM crossed control limit on model residuals.",
                "Persistent directional residual drift indicates process shift.",
                {"metrics": {"alarm_index": int(alarm_idx), "direction": direction, "magnitude": magnitude, "k": k, "h": h}},
                recommendation="Investigate sustained directional drift drivers near the alarm index.",
                severity="warn" if magnitude < h * 1.6 else "critical",
                confidence=0.7,
            )
        )
    artifacts = [
        _artifact_json(
            ctx,
            plugin_id,
            "cusum_residuals.json",
            {"value_column": value_col, "alarm_index": alarm_idx, "direction": direction, "magnitude": magnitude, "k": k, "h": h},
            "CUSUM residual diagnostics",
        )
    ]
    return _finalize(
        plugin_id,
        ctx,
        df,
        sample_meta,
        "Computed CUSUM on detrended residuals.",
        findings,
        artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "alarm": 1 if alarm_idx is not None else 0},
    )


def _handler_change_score_consensus_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    value_col = _best_latency_col(df, inferred)
    if value_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_column")
    y_full = _to_array(df[value_col])
    y, downsample_step = _downsample_series(
        y_full, int(config.get("plugin", {}).get("max_points", 8000))
    )
    if len(y) < 60:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_points")
    try:
        w_idx, _ = _window_cp_score(y, window=max(10, len(y) // 20))
        _ensure_budget(timer)
        wbs = _wbs_candidates(y, seed=int(config.get("seed", 1337)), min_seg=15, K=64, timer=timer)
    except Exception as exc:
        if "time_budget_exceeded" in str(exc):
            return _ok_with_reason(
                plugin_id,
                ctx,
                df,
                sample_meta,
                "time_budget_exceeded",
                debug={"value_column": value_col, "downsample_step": int(downsample_step)},
            )
        raise
    deriv = np.abs(
        np.diff(
            pd.Series(y)
            .rolling(window=max(5, len(y) // 30), min_periods=3)
            .mean()
            .bfill()
            .ffill()
            .to_numpy(dtype=float)
        )
    )
    d_idx = int(np.argmax(deriv)) if len(deriv) else len(y) // 2
    all_idx = [int(w_idx), int(d_idx)] + [int(i) for i in wbs[:100]]
    if downsample_step > 1:
        all_idx = [int(min(len(y_full) - 1, max(0, i * downsample_step))) for i in all_idx]
    consensus = _cluster_indices(all_idx, tolerance=10, max_points=10)
    findings = [
        _make_finding(
            plugin_id,
            f"consensus:{int(p['index'])}",
            "Consensus changepoint",
            "Multiple changepoint scorers agreed on this index neighborhood.",
            "Cross-method agreement usually indicates robust shift signal.",
            {"metrics": {"index": int(p["index"]), "votes": int(p["votes"])}},
            recommendation="Prioritize high-vote consensus changepoints for remediation planning.",
            severity="warn" if int(p["votes"]) < 4 else "critical",
            confidence=min(0.95, 0.6 + min(0.3, int(p["votes"]) / 12.0)),
        )
        for p in consensus
    ]
    artifacts = [
        _artifact_json(
            ctx,
            plugin_id,
            "changepoint_consensus.json",
            {"value_column": value_col, "consensus": consensus, "raw_count": len(all_idx)},
            "Changepoint consensus summary",
        )
    ]
    return _finalize(
        plugin_id,
        ctx,
        df,
        sample_meta,
        "Computed changepoint consensus.",
        findings,
        artifacts,
        extra_metrics={
            "runtime_ms": _runtime_ms(timer),
            "downsample_step": int(downsample_step),
        },
    )


def _handler_benfords_law_anomaly_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    cols = _numeric_columns(df, inferred, max_cols=int(config.get("max_cols", 80)))
    if not cols:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_columns")
    benford = np.array([math.log10(1.0 + 1.0 / d) for d in range(1, 10)], dtype=float)
    scored = []
    for col in cols:
        vals = _to_array(df[col])
        digits = [_first_nonzero_digit(v) for v in vals]
        digits = [d for d in digits if d is not None]
        if len(digits) < 30:
            continue
        counts = np.array([(np.array(digits) == d).sum() for d in range(1, 10)], dtype=float)
        n = float(np.sum(counts))
        phat = counts / max(n, 1.0)
        chi2 = float(np.sum(n * ((phat - benford) ** 2) / np.maximum(benford, 1e-9)))
        scored.append((chi2, col, int(n)))
    scored.sort(key=lambda t: (-t[0], t[1]))
    findings = [
        _make_finding(
            plugin_id,
            f"benford:{col}",
            "Benford deviation",
            "First-digit distribution diverges from Benford expectation.",
            "Strong Benford divergence can indicate process/systematic distortions.",
            {"metrics": {"column": col, "chi2": chi2, "n": n}},
            recommendation="Validate data generation rules and check for formatting/aggregation bias.",
            severity="warn" if chi2 < 25 else "critical",
            confidence=min(0.9, 0.5 + min(0.35, chi2 / 80.0)),
        )
        for chi2, col, n in scored[: int(config.get("max_findings", 30))]
    ]
    artifacts = [
        _artifact_json(
            ctx,
            plugin_id,
            "benford.json",
            {"top_columns": [{"column": c, "chi2": s, "n": n} for s, c, n in scored[:50]]},
            "Benford first-digit diagnostics",
        )
    ]
    return _finalize(plugin_id, ctx, df, sample_meta, "Computed Benford diagnostics.", findings, artifacts, extra_metrics={"runtime_ms": _runtime_ms(timer)})


def _handler_geometric_median_multivariate_location_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    cols = _numeric_columns(df, inferred, max_cols=min(int(config.get("max_cols", 80)), 10))
    if len(cols) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "need_at_least_two_numeric_columns")
    frame = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(frame) < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    try:
        frame = _cap_for_quadratic(plugin_id, ctx, frame, config)
    except _QuadraticCapExceeded as exc:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "quadratic_cap_exceeded", debug={"reason": str(exc)})
    X = frame.to_numpy(dtype=float)
    m = np.median(X, axis=0)
    eps = 1e-8
    for _ in range(50):
        dist = np.linalg.norm(X - m, axis=1)
        w = 1.0 / np.maximum(dist, eps)
        m_next = np.sum(X * w[:, None], axis=0) / max(np.sum(w), eps)
        if float(np.linalg.norm(m_next - m)) < 1e-6:
            m = m_next
            break
        m = m_next
        _ensure_budget(timer)
    d = np.linalg.norm(X - m, axis=1)
    top_idx = np.argsort(d)[::-1][: min(10, len(d))]
    findings = [
        _make_finding(
            plugin_id,
            f"outlier:{int(i)}",
            "Geometric-median distance outlier",
            "Point lies far from robust multivariate center.",
            "Large robust distance indicates atypical combined feature state.",
            {"metrics": {"row_position": int(i), "distance": float(d[i])}},
            recommendation="Inspect process context for high-distance outliers and validate feature quality.",
            severity="warn",
            confidence=0.6,
        )
        for i in top_idx
    ]
    artifacts = [
        _artifact_json(
            ctx,
            plugin_id,
            "geomed_outliers.json",
            {"columns": cols, "center": m.tolist(), "top_distances": [{"row_position": int(i), "distance": float(d[i])} for i in top_idx]},
            "Geometric median outlier summary",
        )
    ]
    return _finalize(plugin_id, ctx, df, sample_meta, "Computed geometric median location/outliers.", findings, artifacts, extra_metrics={"runtime_ms": _runtime_ms(timer)})


def _handler_random_matrix_marchenko_pastur_denoise_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    cols = _numeric_columns(df, inferred, max_cols=40)
    if len(cols) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "need_at_least_two_numeric_columns")
    frame = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(frame) < len(cols) + 5:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows_for_mp")
    X = frame.to_numpy(dtype=float)
    X = (X - np.nanmean(X, axis=0)) / np.maximum(np.nanstd(X, axis=0), 1e-9)
    n, p = X.shape
    corr = (X.T @ X) / float(n)
    lam, V = np.linalg.eigh(corr)
    q = float(p) / float(n)
    lam_max = float((1.0 + math.sqrt(max(q, 1e-9))) ** 2)
    keep = lam > lam_max
    eff_dim = int(np.sum(keep))
    den = V @ np.diag(np.where(keep, lam, 0.0)) @ V.T
    findings = []
    if eff_dim > 0:
        findings.append(
            _make_finding(
                plugin_id,
                "effective_dim",
                "Above-noise latent dimensions detected",
                "Eigen-spectrum exceeds Marchenko-Pastur upper edge.",
                "Dimensions above MP edge likely carry structured signal.",
                {"metrics": {"effective_dim": eff_dim, "lambda_max_mp": lam_max}},
                recommendation="Prioritize features loading on above-noise components.",
                severity="info",
                confidence=0.65,
            )
        )
    artifacts = [
        _artifact_json(
            ctx,
            plugin_id,
            "mp_denoise.json",
            {"columns": cols, "lambda": lam.tolist(), "lambda_max_mp": lam_max, "effective_dim": eff_dim, "denoised_corr": den.tolist()},
            "Marchenko-Pastur denoising summary",
        )
    ]
    return _finalize(plugin_id, ctx, df, sample_meta, "Computed MP denoising diagnostics.", findings, artifacts, extra_metrics={"runtime_ms": _runtime_ms(timer), "effective_dim": eff_dim})


def _handler_outlier_influence_cooks_distance_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    pair = _top_variance_pair(df, inferred)
    if pair is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "need_two_numeric_columns")
    x_col, y_col = pair
    frame = pd.DataFrame({"x": pd.to_numeric(df[x_col], errors="coerce"), "y": pd.to_numeric(df[y_col], errors="coerce")}).dropna()
    if len(frame) < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    x = frame["x"].to_numpy(dtype=float)
    y = frame["y"].to_numpy(dtype=float)
    xbar = float(np.mean(x))
    ybar = float(np.mean(y))
    sxx = float(np.sum((x - xbar) ** 2))
    if sxx <= 0:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "zero_variance_predictor")
    b = float(np.sum((x - xbar) * (y - ybar)) / sxx)
    a = ybar - b * xbar
    yhat = a + b * x
    e = y - yhat
    n = len(x)
    p = 2.0
    mse = float(np.sum(e ** 2) / max(1, n - 2))
    h = (1.0 / n) + (((x - xbar) ** 2) / max(sxx, 1e-9))
    D = (e ** 2) / max(p * mse, 1e-9) * (h / np.maximum((1.0 - h) ** 2, 1e-9))
    idx = np.argsort(D)[::-1][: min(10, n)]
    findings = [
        _make_finding(
            plugin_id,
            f"cook:{int(i)}",
            "High Cook's distance observation",
            "Observation has disproportionate influence on fitted linear relation.",
            "High-influence points can distort trend interpretation and policy actions.",
            {"metrics": {"row_position": int(i), "cooks_distance": float(D[i]), "x": float(x[i]), "y": float(y[i])}},
            recommendation="Audit high-influence rows before acting on linear model conclusions.",
            severity="warn" if float(D[i]) < 1.0 else "critical",
            confidence=min(0.9, 0.5 + min(0.35, float(D[i]) / 5.0)),
        )
        for i in idx
    ]
    artifacts = [
        _artifact_json(
            ctx,
            plugin_id,
            "cooks_distance.json",
            {"x_column": x_col, "y_column": y_col, "top": [{"row_position": int(i), "cooks_distance": float(D[i])} for i in idx]},
            "Cook's distance summary",
        )
    ]
    return _finalize(plugin_id, ctx, df, sample_meta, "Computed Cook's distance influence diagnostics.", findings, artifacts, extra_metrics={"runtime_ms": _runtime_ms(timer)})


def _handler_heavy_tail_index_hill_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    col = _best_latency_col(df, inferred)
    if col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_column")
    x = _to_array(df[col])
    x = x[x > 0]
    n = len(x)
    if n < 50:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_positive_samples")
    x_sorted = np.sort(x)[::-1]
    k = max(20, min(200, n // 10))
    xk = max(float(x_sorted[k - 1]), 1e-9)
    hill = float(np.mean(np.log(np.maximum(x_sorted[:k], 1e-9) / xk)))
    alpha = float(1.0 / max(hill, 1e-9))
    sev = "critical" if alpha < 1.5 else "warn" if alpha < 2.5 else "info"
    findings = []
    if alpha < 2.5:
        findings.append(
            _make_finding(
                plugin_id,
                "hill_alpha",
                "Heavy-tail behavior detected",
                "Estimated Hill tail index indicates strong tail heaviness.",
                "Heavy tails increase extreme-delay probability and tail-risk exposure.",
                {"metrics": {"column": col, "hill": hill, "alpha": alpha, "k": k}},
                recommendation="Use tail-aware SLOs and mitigation strategies for extreme outliers.",
                severity=sev,
                confidence=0.75,
            )
        )
    artifacts = [
        _artifact_json(
            ctx,
            plugin_id,
            "hill_tail_index.json",
            {"column": col, "hill": hill, "tail_index_alpha": alpha, "k": k},
            "Hill tail index summary",
        )
    ]
    return _finalize(plugin_id, ctx, df, sample_meta, "Computed Hill tail index.", findings, artifacts, extra_metrics={"runtime_ms": _runtime_ms(timer), "tail_index_alpha": alpha})


def _distance_corr(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    if n <= 2:
        return 0.0
    A = np.abs(x[:, None] - x[None, :])
    B = np.abs(y[:, None] - y[None, :])
    A = A - A.mean(axis=0)[None, :] - A.mean(axis=1)[:, None] + A.mean()
    B = B - B.mean(axis=0)[None, :] - B.mean(axis=1)[:, None] + B.mean()
    dcov2 = float(np.mean(A * B))
    dvarx = float(np.mean(A * A))
    dvary = float(np.mean(B * B))
    if dvarx <= 0 or dvary <= 0:
        return 0.0
    return float(max(0.0, dcov2) / math.sqrt(max(1e-12, dvarx * dvary)))


def _handler_distance_correlation_screen_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    cols = _numeric_columns(df, inferred, max_cols=10)
    if len(cols) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "need_two_numeric_columns")
    frame = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(frame) < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    try:
        frame = _cap_for_quadratic(plugin_id, ctx, frame, config)
    except _QuadraticCapExceeded as exc:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "quadratic_cap_exceeded", debug={"reason": str(exc)})
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            x = frame[cols[i]].to_numpy(dtype=float)
            y = frame[cols[j]].to_numpy(dtype=float)
            dcor = _distance_corr(x, y)
            pairs.append((dcor, cols[i], cols[j]))
            _ensure_budget(timer)
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))
    findings = [
        _make_finding(
            plugin_id,
            f"dcor:{a}:{b}",
            "High distance correlation pair",
            "Features show strong nonlinear dependence.",
            "Distance correlation captures nonlinear relationships beyond Pearson correlation.",
            {"metrics": {"feature_a": a, "feature_b": b, "distance_correlation": s}},
            recommendation="Model this pair jointly in root-cause and forecasting analyses.",
            severity="warn" if s < 0.6 else "critical",
            confidence=min(0.95, 0.55 + min(0.35, s / 1.2)),
        )
        for s, a, b in pairs[: int(config.get("max_findings", 30))]
        if s >= 0.3
    ]
    artifacts = [
        _artifact_json(
            ctx,
            plugin_id,
            "distance_correlation.json",
            {"top_pairs": [{"distance_correlation": s, "a": a, "b": b} for s, a, b in pairs[:50]]},
            "Distance correlation screen",
        )
    ]
    return _finalize(plugin_id, ctx, df, sample_meta, "Computed distance correlation screen.", findings, artifacts, extra_metrics={"runtime_ms": _runtime_ms(timer)})


def _select_target_predictors(df: pd.DataFrame, inferred: dict[str, Any], max_features: int = 6) -> tuple[str, list[str]] | None:
    cols = _numeric_columns(df, inferred, max_cols=20)
    if len(cols) < 2:
        return None
    ranked = sorted(
        ((float(np.nanvar(pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float))), c) for c in cols),
        key=lambda t: (-t[0], t[1]),
    )
    target = ranked[0][1]
    predictors = [c for _, c in ranked[1 : 1 + max_features]]
    if not predictors:
        return None
    return target, predictors


def _fit_linear_r2(X: np.ndarray, y: np.ndarray, lam: float = 1e-6) -> tuple[np.ndarray, float]:
    XtX = X.T @ X + lam * np.eye(X.shape[1])
    beta = np.linalg.pinv(XtX) @ X.T @ y
    pred = X @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 0.0 if ss_tot <= 0 else max(0.0, 1.0 - ss_res / ss_tot)
    return beta, r2


def _handler_gam_spline_regression_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    selected = _select_target_predictors(df, inferred, max_features=6)
    if selected is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_numeric_features")
    target, predictors = selected
    frame = df[[target] + predictors].apply(pd.to_numeric, errors="coerce").dropna()
    if len(frame) < 30:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    y = frame[target].to_numpy(dtype=float)
    X = frame[predictors].to_numpy(dtype=float)
    if HAS_SKLEARN and SplineTransformer is not None and Ridge is not None:
        spl = SplineTransformer(degree=3, n_knots=5, include_bias=False)
        Xs = spl.fit_transform(X)
        reg = Ridge(alpha=1.0, random_state=int(config.get("seed", 1337)))  # type: ignore[arg-type]
        reg.fit(Xs, y)
        pred = reg.predict(Xs)
        coeff = np.asarray(reg.coef_, dtype=float)
        # approximate per-feature importance by grouped L2 norm across spline basis chunks
        per = max(1, coeff.size // len(predictors))
        top = []
        for i, name in enumerate(predictors):
            seg = coeff[i * per : (i + 1) * per]
            top.append((float(np.linalg.norm(seg)), name))
        top.sort(key=lambda t: (-t[0], t[1]))
    else:
        # polynomial fallback
        Xpoly = np.hstack([X, X**2])
        _, r2 = _fit_linear_r2(Xpoly, y, lam=1e-4)
        pred = np.mean(y) + np.zeros_like(y)
        top = [(float(np.nanvar(frame[c].to_numpy(dtype=float))), c) for c in predictors]
        top.sort(key=lambda t: (-t[0], t[1]))
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 0.0 if ss_tot <= 0 else max(0.0, 1.0 - ss_res / ss_tot)
    findings = [
        _make_finding(
            plugin_id,
            "gam_fit",
            "Spline regression fit summary",
            "Fitted spline-style regression against selected predictors.",
            "Captures nonlinear predictor effects on target variability.",
            {"metrics": {"target": target, "predictors": predictors, "r2": r2, "top_features": top[:5]}},
            recommendation="Prioritize top nonlinear drivers for intervention testing.",
            severity="info",
            confidence=min(0.9, 0.5 + min(0.35, r2)),
        )
    ]
    artifacts = [
        _artifact_json(
            ctx,
            plugin_id,
            "gam_spline.json",
            {"target": target, "predictors": predictors, "r2": r2, "top_features": top[:10]},
            "GAM spline regression summary",
        )
    ]
    return _finalize(plugin_id, ctx, df, sample_meta, "Computed spline regression diagnostics.", findings, artifacts, extra_metrics={"runtime_ms": _runtime_ms(timer), "r2": r2})


def _handler_quantile_loss_boosting_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    selected = _select_target_predictors(df, inferred, max_features=6)
    if selected is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_numeric_features")
    target, predictors = selected
    frame = df[[target] + predictors].apply(pd.to_numeric, errors="coerce").dropna()
    if len(frame) < 40:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    X = frame[predictors].to_numpy(dtype=float)
    y = frame[target].to_numpy(dtype=float)
    q50 = np.quantile(y, 0.5) + np.zeros_like(y)
    q90 = np.quantile(y, 0.9) + np.zeros_like(y)
    importances = {c: float(np.nanvar(frame[c].to_numpy(dtype=float))) for c in predictors}
    if HAS_SKLEARN and GradientBoostingRegressor is not None:
        model50 = GradientBoostingRegressor(loss="quantile", alpha=0.5, random_state=int(config.get("seed", 1337)))
        model90 = GradientBoostingRegressor(loss="quantile", alpha=0.9, random_state=int(config.get("seed", 1337)))
        model50.fit(X, y)
        model90.fit(X, y)
        q50 = model50.predict(X)
        q90 = model90.predict(X)
        importances = {c: float(v) for c, v in zip(predictors, np.asarray(model90.feature_importances_, dtype=float))}
    scale = max(1e-6, np.median(np.abs(y - np.median(y))))
    drift = float(np.mean(q90[len(q90) // 2 :]) - np.mean(q90[: len(q90) // 2])) / scale
    findings = []
    if abs(drift) > 0.5:
        findings.append(
            _make_finding(
                plugin_id,
                "q90_drift",
                "Quantile drift detected",
                "Upper-quantile forecast shifted between early and late segments.",
                "Tail drift often signals worsening burst risk under changing conditions.",
                {"metrics": {"target": target, "q90_drift_scaled": drift, "top_feature_importances": sorted(importances.items(), key=lambda t: (-t[1], t[0]))[:5]}},
                recommendation="Investigate late-segment drivers with highest importance weights.",
                severity="warn" if abs(drift) < 1.0 else "critical",
                confidence=0.72,
            )
        )
    artifacts = [
        _artifact_json(
            ctx,
            plugin_id,
            "quantile_boosting.json",
            {"target": target, "predictors": predictors, "q90_drift_scaled": drift, "feature_importances": importances},
            "Quantile boosting summary",
        )
    ]
    return _finalize(plugin_id, ctx, df, sample_meta, "Computed quantile boosting diagnostics.", findings, artifacts, extra_metrics={"runtime_ms": _runtime_ms(timer), "q90_drift_scaled": drift})


def _handler_quantile_regression_forest_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    selected = _select_target_predictors(df, inferred, max_features=6)
    if selected is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_numeric_features")
    target, predictors = selected
    frame = df[[target] + predictors].apply(pd.to_numeric, errors="coerce").dropna()
    if len(frame) < 50:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    X = frame[predictors].to_numpy(dtype=float)
    y = frame[target].to_numpy(dtype=float)
    q90_pred = np.quantile(y, 0.9) + np.zeros_like(y)
    importances = {c: float(np.nanvar(frame[c].to_numpy(dtype=float))) for c in predictors}
    if HAS_SKLEARN and RandomForestRegressor is not None:
        rf = RandomForestRegressor(
            n_estimators=80,
            random_state=int(config.get("seed", 1337)),
            max_depth=8,
            n_jobs=1,
        )
        rf.fit(X, y)
        tree_preds = np.vstack([t.predict(X) for t in rf.estimators_])
        q90_pred = np.quantile(tree_preds, 0.9, axis=0)
        importances = {c: float(v) for c, v in zip(predictors, np.asarray(rf.feature_importances_, dtype=float))}
    resid = y - q90_pred
    idx = np.argsort(resid)[::-1][: min(10, len(resid))]
    findings = [
        _make_finding(
            plugin_id,
            f"qrf_tail:{int(i)}",
            "High q90 residual sample",
            "Observed value exceeded estimated q90 prediction.",
            "Large positive q90 residuals indicate under-modeled tail conditions.",
            {"metrics": {"row_position": int(i), "actual": float(y[i]), "q90_pred": float(q90_pred[i]), "residual": float(resid[i])}},
            recommendation="Inspect contextual features around high q90 residual rows.",
            severity="warn",
            confidence=0.65,
        )
        for i in idx
    ]
    artifacts = [
        _artifact_json(
            ctx,
            plugin_id,
            "qrf.json",
            {"target": target, "predictors": predictors, "feature_importances": importances, "top_q90_residual_rows": [int(i) for i in idx]},
            "Quantile regression forest summary",
        )
    ]
    return _finalize(plugin_id, ctx, df, sample_meta, "Computed quantile regression forest diagnostics.", findings, artifacts, extra_metrics={"runtime_ms": _runtime_ms(timer)})


def _handler_sparse_pca_interpretable_components_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    cols = _numeric_columns(df, inferred, max_cols=20)
    if len(cols) < 3:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_numeric_columns")
    frame = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(frame) < 30:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    X = frame.to_numpy(dtype=float)
    X = (X - np.mean(X, axis=0)) / np.maximum(np.std(X, axis=0), 1e-9)
    n_comp = min(5, len(cols))
    if HAS_SKLEARN and SparsePCA is not None:
        model = SparsePCA(n_components=n_comp, random_state=int(config.get("seed", 1337)), alpha=1.0)
        model.fit(X)
        loadings = np.asarray(model.components_, dtype=float)
    else:
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        loadings = Vt[:n_comp].copy()
        thresh = float(np.quantile(np.abs(loadings), 0.75))
        loadings[np.abs(loadings) < thresh] = 0.0
    components = []
    for i, row in enumerate(loadings):
        top_idx = np.argsort(np.abs(row))[::-1][:8]
        components.append({"component": i, "top_loadings": [{"column": cols[int(j)], "loading": float(row[int(j)])} for j in top_idx]})
    findings = [
        _make_finding(
            plugin_id,
            "sparse_components",
            "Sparse interpretable components extracted",
            "Dominant sparse loadings identified across principal components.",
            "Sparse component structure helps isolate high-impact feature clusters.",
            {"metrics": {"components": components[:3]}},
            recommendation="Use top loadings per component as candidate driver groups.",
            severity="info",
            confidence=0.65,
        )
    ]
    artifacts = [
        _artifact_json(ctx, plugin_id, "sparse_pca.json", {"columns": cols, "components": components}, "Sparse PCA summary")
    ]
    return _finalize(plugin_id, ctx, df, sample_meta, "Computed sparse PCA components.", findings, artifacts, extra_metrics={"runtime_ms": _runtime_ms(timer)})


def _handler_ica_source_separation_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    cols = _numeric_columns(df, inferred, max_cols=12)
    if len(cols) < 3:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_numeric_columns")
    frame = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(frame) < 30:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    X = frame.to_numpy(dtype=float)
    X = (X - np.mean(X, axis=0)) / np.maximum(np.std(X, axis=0), 1e-9)
    n_comp = min(5, X.shape[1])
    mode = "pca_fallback"
    if HAS_SKLEARN and FastICA is not None:
        model = FastICA(n_components=n_comp, random_state=int(config.get("seed", 1337)), max_iter=400)
        S = model.fit_transform(X)
        M = np.asarray(model.mixing_, dtype=float)
        mode = "fastica"
    else:
        U, Svals, Vt = np.linalg.svd(X, full_matrices=False)
        S = U[:, :n_comp]
        M = Vt[:n_comp].T
    kurt = np.mean(((S - np.mean(S, axis=0)) / np.maximum(np.std(S, axis=0), 1e-9)) ** 4, axis=0) - 3.0
    top = []
    for j in range(M.shape[1]):
        idx = np.argsort(np.abs(M[:, j]))[::-1][:5]
        top.append({"component": int(j), "kurtosis": float(kurt[j]), "top_columns": [{"column": cols[int(i)], "loading": float(M[int(i), j])} for i in idx]})
    findings = [
        _make_finding(
            plugin_id,
            "ica_components",
            "Independent source components extracted",
            "Latent source components and loading structure were identified.",
            "Independent component structure highlights mixed-process latent factors.",
            {"metrics": {"mode": mode, "components": top[:3]}},
            recommendation="Inspect components with highest kurtosis for burst-like latent drivers.",
            severity="info",
            confidence=0.62,
        )
    ]
    artifacts = [_artifact_json(ctx, plugin_id, "ica_sources.json", {"mode": mode, "components": top}, "ICA source separation summary")]
    return _finalize(plugin_id, ctx, df, sample_meta, "Computed ICA/PCA source separation.", findings, artifacts, extra_metrics={"runtime_ms": _runtime_ms(timer)})


def _handler_cca_crossblock_association_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    cols = _numeric_columns(df, inferred, max_cols=12)
    if len(cols) < 4:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "need_four_numeric_columns")
    hints_a = ("queue", "wait", "eligible")
    hints_b = ("duration", "service", "runtime")
    block_a = [c for c in cols if any(h in c.lower() for h in hints_a)]
    block_b = [c for c in cols if any(h in c.lower() for h in hints_b) and c not in block_a]
    if len(block_a) < 2 or len(block_b) < 2:
        half = len(cols) // 2
        block_a = cols[:half]
        block_b = cols[half:]
    if len(block_a) < 1 or len(block_b) < 1:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "unable_to_build_blocks")
    frame = df[list(dict.fromkeys(block_a + block_b))].apply(pd.to_numeric, errors="coerce").dropna()
    if len(frame) < 30:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    XA = frame[block_a].to_numpy(dtype=float)
    XB = frame[block_b].to_numpy(dtype=float)
    XA = (XA - np.mean(XA, axis=0)) / np.maximum(np.std(XA, axis=0), 1e-9)
    XB = (XB - np.mean(XB, axis=0)) / np.maximum(np.std(XB, axis=0), 1e-9)
    mode = "cca"
    corrs: list[float] = []
    if HAS_SKLEARN and CCA is not None:
        n_comp = min(2, XA.shape[1], XB.shape[1])
        cca = CCA(n_components=n_comp, max_iter=500)
        UA, UB = cca.fit_transform(XA, XB)
        for i in range(n_comp):
            corrs.append(abs(_cov_corr(UA[:, i], UB[:, i])))
    else:
        mode = "pc_corr_fallback"
        u1 = np.linalg.svd(XA, full_matrices=False)[0][:, 0]
        u2 = np.linalg.svd(XB, full_matrices=False)[0][:, 0]
        corrs = [abs(_cov_corr(u1, u2))]
    top_corr = max(corrs) if corrs else 0.0
    findings = []
    if top_corr > 0.4:
        findings.append(
            _make_finding(
                plugin_id,
                "cca_association",
                "Cross-block association detected",
                "Canonical latent factors show meaningful cross-block correlation.",
                "Shared latent structure across operational blocks suggests coupled dynamics.",
                {"metrics": {"mode": mode, "canonical_correlations": corrs, "block_a": block_a, "block_b": block_b}},
                recommendation="Coordinate optimization across both blocks instead of single-block tuning.",
                severity="warn" if top_corr < 0.7 else "critical",
                confidence=min(0.92, 0.55 + min(0.35, top_corr / 1.2)),
            )
        )
    artifacts = [_artifact_json(ctx, plugin_id, "cca.json", {"mode": mode, "block_a": block_a, "block_b": block_b, "canonical_correlations": corrs}, "CCA summary")]
    return _finalize(plugin_id, ctx, df, sample_meta, "Computed cross-block CCA diagnostics.", findings, artifacts, extra_metrics={"runtime_ms": _runtime_ms(timer), "top_corr": top_corr})


def _varimax(Phi: np.ndarray, gamma: float = 1.0, q: int = 50, tol: float = 1e-6) -> np.ndarray:
    p, k = Phi.shape
    R = np.eye(k)
    d_old = 0.0
    for _ in range(q):
        Lambda = Phi @ R
        u, s, vh = np.linalg.svd(
            Phi.T @ (Lambda**3 - (gamma / p) * Lambda @ np.diag(np.diag(Lambda.T @ Lambda))),
            full_matrices=False,
        )
        R = u @ vh
        d = float(np.sum(s))
        if d_old != 0 and d / d_old < 1 + tol:
            break
        d_old = d
    return Phi @ R


def _handler_factor_rotation_varimax_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    cols = _numeric_columns(df, inferred, max_cols=12)
    if len(cols) < 3:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_numeric_columns")
    frame = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(frame) < 30:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    X = frame.to_numpy(dtype=float)
    X = (X - np.mean(X, axis=0)) / np.maximum(np.std(X, axis=0), 1e-9)
    k = min(3, X.shape[1])
    if HAS_SKLEARN and PCA is not None:
        pca = PCA(n_components=k, random_state=int(config.get("seed", 1337)))
        pca.fit(X)
        loadings = pca.components_.T
    else:
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        loadings = Vt[:k].T
    rotated = _varimax(loadings)
    sparsity = float(np.mean(np.abs(rotated) < 0.1))
    comps = []
    for j in range(rotated.shape[1]):
        idx = np.argsort(np.abs(rotated[:, j]))[::-1][:6]
        comps.append({"component": int(j), "top_columns": [{"column": cols[int(i)], "loading": float(rotated[int(i), j])} for i in idx]})
    findings = [
        _make_finding(
            plugin_id,
            "varimax_rotation",
            "Varimax-rotated factors computed",
            "Rotated factors produced a sparser interpretable loading structure.",
            "Varimax sparsity improves explainability of latent factor assignments.",
            {"metrics": {"sparsity": sparsity, "components": comps[:3]}},
            recommendation="Use rotated factors to map clearer ownership between features and latent drivers.",
            severity="info",
            confidence=0.65,
        )
    ]
    artifacts = [_artifact_json(ctx, plugin_id, "varimax.json", {"columns": cols, "sparsity": sparsity, "components": comps}, "Varimax factor rotation summary")]
    return _finalize(plugin_id, ctx, df, sample_meta, "Computed varimax rotation diagnostics.", findings, artifacts, extra_metrics={"runtime_ms": _runtime_ms(timer), "sparsity": sparsity})


def _oja_basis(X: np.ndarray, k: int, eta: float = 0.01, steps: int | None = None, seed: int = 1337) -> np.ndarray:
    n, p = X.shape
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(p, k))
    # orthonormalize
    Q, _ = np.linalg.qr(W)
    W = Q[:, :k]
    max_steps = n if steps is None else min(n, steps)
    for i in range(max_steps):
        x = X[i : i + 1].T
        W = W + eta * x @ (x.T @ W)
        Q, _ = np.linalg.qr(W)
        W = Q[:, :k]
    return W


def _handler_subspace_tracking_oja_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    cols = _numeric_columns(df, inferred, max_cols=12)
    if len(cols) < 3:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_numeric_columns")
    frame = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(frame) < 80:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    X = frame.to_numpy(dtype=float)
    X = (X - np.mean(X, axis=0)) / np.maximum(np.std(X, axis=0), 1e-9)
    mid = len(X) // 2
    k = min(3, X.shape[1])
    seed = int(config.get("seed", 1337))
    W1 = _oja_basis(X[:mid], k=k, eta=0.01, seed=seed)
    W2 = _oja_basis(X[mid:], k=k, eta=0.01, seed=seed + 1)
    s = np.linalg.svd(W1.T @ W2, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    angles = np.degrees(np.arccos(s))
    max_angle = float(np.max(angles)) if len(angles) else 0.0
    sev = "critical" if max_angle > 35 else "warn" if max_angle > 20 else "info"
    findings = []
    if max_angle > 20:
        findings.append(
            _make_finding(
                plugin_id,
                "oja_angle_shift",
                "Subspace drift detected",
                "Principal subspace orientation shifted between early and late segments.",
                "Subspace angle drift indicates latent structure change over time.",
                {"metrics": {"angles_deg": angles.tolist(), "max_angle_deg": max_angle}},
                recommendation="Investigate regime changes that altered multivariate dependency structure.",
                severity=sev,
                confidence=min(0.95, 0.6 + min(0.3, max_angle / 90.0)),
            )
        )
    artifacts = [_artifact_json(ctx, plugin_id, "oja_subspace.json", {"columns": cols, "angles_deg": angles.tolist(), "max_angle_deg": max_angle}, "Oja subspace tracking summary")]
    return _finalize(plugin_id, ctx, df, sample_meta, "Computed Oja subspace tracking diagnostics.", findings, artifacts, extra_metrics={"runtime_ms": _runtime_ms(timer), "max_angle_deg": max_angle})


def _handler_multicollinearity_vif_screen_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    cols = _numeric_columns(df, inferred, max_cols=8)
    if len(cols) < 3:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "need_three_numeric_columns")
    frame = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(frame) < 30:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    X = frame.to_numpy(dtype=float)
    X = (X - np.mean(X, axis=0)) / np.maximum(np.std(X, axis=0), 1e-9)
    vif_rows = []
    for j, col in enumerate(cols):
        y = X[:, j]
        Xo = np.delete(X, j, axis=1)
        _, r2 = _fit_linear_r2(Xo, y, lam=1e-4)
        vif = float(1.0 / max(1e-6, 1.0 - r2))
        vif_rows.append((vif, col))
    vif_rows.sort(key=lambda t: (-t[0], t[1]))
    findings = [
        _make_finding(
            plugin_id,
            f"vif:{col}",
            "High multicollinearity risk",
            "Feature can be strongly explained by other predictors (high VIF).",
            "High VIF can destabilize coefficient-based interpretation and policy ranking.",
            {"metrics": {"column": col, "vif": vif}},
            recommendation="Remove/recombine correlated predictors or use regularization-aware interpretation.",
            severity="critical" if vif > 10 else "warn",
            confidence=min(0.95, 0.55 + min(0.35, vif / 20.0)),
        )
        for vif, col in vif_rows
        if vif > 5
    ]
    artifacts = [_artifact_json(ctx, plugin_id, "vif.json", {"vif": [{"column": c, "vif": v} for v, c in vif_rows]}, "VIF multicollinearity screen")]
    return _finalize(plugin_id, ctx, df, sample_meta, "Computed VIF multicollinearity diagnostics.", findings, artifacts, extra_metrics={"runtime_ms": _runtime_ms(timer)})


def _handler_zero_inflated_count_model_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    col = _pick_integer_count_column(df, inferred)
    if col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_integer_like_count_column")
    x = _to_array(df[col])
    x = np.clip(np.round(x), 0, None)
    if len(x) < 40:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    pi = float(np.clip(np.mean(x == 0), 0.0, 0.95))
    lam = float(np.mean(x) / max(1e-6, 1.0 - pi))
    lam = max(1e-6, lam)
    for _ in range(50):
        p0_pois = math.exp(-lam)
        z = np.zeros_like(x, dtype=float)
        mask0 = x == 0
        denom = pi + (1.0 - pi) * p0_pois
        z[mask0] = pi / max(1e-9, denom)
        pi = float(np.clip(np.mean(z), 1e-6, 0.99))
        lam = float(np.sum((1.0 - z) * x) / max(1e-6, np.sum(1.0 - z)))
        lam = max(1e-6, lam)
        _ensure_budget(timer)
    findings = [
        _make_finding(
            plugin_id,
            "zip_params",
            "Zero-inflation estimated",
            "EM fit estimated non-trivial structural-zero mixture component.",
            "High structural-zero mass changes expected queue/count behavior.",
            {"metrics": {"column": col, "pi": pi, "lambda": lam}},
            recommendation="Separate structural-zero conditions from stochastic counts in operational policies.",
            severity="warn" if pi < 0.4 else "critical",
            confidence=min(0.9, 0.55 + min(0.3, pi)),
        )
    ]
    artifacts = [_artifact_json(ctx, plugin_id, "zip_em.json", {"column": col, "pi": pi, "lambda": lam}, "ZIP EM parameters")]
    return _finalize(plugin_id, ctx, df, sample_meta, "Computed ZIP EM parameters.", findings, artifacts, extra_metrics={"runtime_ms": _runtime_ms(timer)})


def _handler_negative_binomial_overdispersion_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    col = _pick_integer_count_column(df, inferred)
    if col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_integer_like_count_column")
    x = _to_array(df[col])
    x = np.clip(np.round(x), 0, None)
    if len(x) < 40:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    m = float(np.mean(x))
    v = float(np.var(x))
    ratio = v / max(m, 1e-9)
    r = (m * m) / max(v - m, 1e-9)
    findings = []
    if ratio > 2.0:
        findings.append(
            _make_finding(
                plugin_id,
                "nb_overdispersion",
                "Count overdispersion detected",
                "Variance materially exceeds Poisson-equivalent mean.",
                "Overdispersion indicates latent heterogeneity and burstiness in count process.",
                {"metrics": {"column": col, "mean": m, "variance": v, "dispersion_ratio": ratio, "nb_r": r}},
                recommendation="Use NB/mixture models for planning instead of Poisson assumptions.",
                severity="warn" if ratio <= 5 else "critical",
                confidence=0.75,
            )
        )
    artifacts = [_artifact_json(ctx, plugin_id, "nb_overdispersion.json", {"column": col, "mean": m, "variance": v, "dispersion_ratio": ratio, "nb_r": r}, "Negative-binomial overdispersion summary")]
    return _finalize(plugin_id, ctx, df, sample_meta, "Computed negative-binomial overdispersion metrics.", findings, artifacts, extra_metrics={"runtime_ms": _runtime_ms(timer), "dispersion_ratio": ratio})


def _handler_dirichlet_multinomial_categorical_overdispersion_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    cats = _categorical_columns(df, inferred)
    if not cats:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_categorical_column")
    cat_col = None
    for c in cats:
        card = int(df[c].astype(str).nunique(dropna=True))
        if 2 <= card <= 50:
            cat_col = c
            break
    if cat_col is None:
        cat_col = cats[0]
    values = df[cat_col].astype(str)
    time_col = inferred.get("time_column")
    parts = []
    if isinstance(time_col, str) and time_col in df.columns:
        ts = pd.to_datetime(df[time_col], errors="coerce")
        ok = ts.notna()
        if int(ok.sum()) >= 20:
            frame = pd.DataFrame({"ts": ts[ok], "cat": values[ok]}).sort_values("ts")
            frame["bucket"] = np.where(frame["ts"] <= frame["ts"].median(), "early", "late")
            parts = [g["cat"] for _, g in frame.groupby("bucket")]
    if not parts:
        parts = [values.iloc[: len(values) // 2], values.iloc[len(values) // 2 :]]
    uniq = sorted(values.unique().tolist())[:50]
    P = []
    for part in parts:
        counts = np.array([(part == u).sum() for u in uniq], dtype=float)
        p = counts / max(1.0, float(np.sum(counts)))
        P.append(p)
    Pm = np.vstack(P)
    dispersion = float(np.mean(np.var(Pm, axis=0)))
    threshold = float(config.get("plugin", {}).get("dispersion_threshold", 0.01))
    findings = []
    if dispersion > threshold:
        findings.append(
            _make_finding(
                plugin_id,
                "cat_dispersion",
                "Categorical overdispersion signal",
                "Category composition varies materially across partitions.",
                "Cross-partition instability suggests latent regime or process-mix shifts.",
                {"metrics": {"column": cat_col, "dispersion_score": dispersion, "threshold": threshold}},
                recommendation="Segment analysis by partition drivers before pooled policy conclusions.",
                severity="warn" if dispersion < 0.03 else "critical",
                confidence=0.68,
            )
        )
    artifacts = [_artifact_json(ctx, plugin_id, "dirichlet_multinomial.json", {"column": cat_col, "dispersion_score": dispersion, "threshold": threshold, "categories": uniq[:20]}, "Dirichlet-multinomial overdispersion proxy")]
    return _finalize(plugin_id, ctx, df, sample_meta, "Computed categorical overdispersion proxy.", findings, artifacts, extra_metrics={"runtime_ms": _runtime_ms(timer), "dispersion_score": dispersion})


def _log_factorials_upto(n: int) -> np.ndarray:
    arr = np.zeros(n + 1, dtype=float)
    for i in range(2, n + 1):
        arr[i] = arr[i - 1] + math.log(i)
    return arr


def _log_choose(logf: np.ndarray, n: int, k: int) -> float:
    if k < 0 or k > n:
        return -1e18
    return float(logf[n] - logf[k] - logf[n - k])


def _fisher_right_tail(a: int, b: int, c: int, d: int) -> float:
    n1 = a + b
    n2 = c + d
    m1 = a + c
    N = n1 + n2
    lo = max(0, m1 - n2)
    hi = min(m1, n1)
    logf = _log_factorials_upto(N)
    terms = []
    for x in range(a, hi + 1):
        lp = _log_choose(logf, m1, x) + _log_choose(logf, N - m1, n1 - x) - _log_choose(logf, N, n1)
        terms.append(lp)
    if not terms:
        return 1.0
    m = max(terms)
    s = sum(math.exp(t - m) for t in terms)
    return float(min(1.0, math.exp(m) * s))


def _handler_fisher_exact_enrichment_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    cats = _categorical_columns(df, inferred)
    if not cats:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_categorical_column")
    cat_col = cats[0]
    values = df[cat_col].astype(str)
    case = np.zeros(len(df), dtype=bool)
    time_col = inferred.get("time_column")
    if isinstance(time_col, str) and time_col in df.columns:
        ts = pd.to_datetime(df[time_col], errors="coerce")
        ok = ts.notna()
        if int(ok.sum()) >= 20:
            med = ts[ok].median()
            case = (ts >= med).fillna(False).to_numpy(dtype=bool)
    if not np.any(case):
        # fallback split by index
        case[len(df) // 2 :] = True
    uniq = values.value_counts().index.tolist()[:100]
    rows = []
    for u in uniq:
        v = values == u
        a = int(np.sum(case & v.to_numpy(dtype=bool)))
        b = int(np.sum(case & (~v.to_numpy(dtype=bool))))
        c = int(np.sum((~case) & v.to_numpy(dtype=bool)))
        d = int(np.sum((~case) & (~v.to_numpy(dtype=bool))))
        p = _fisher_right_tail(a, b, c, d)
        rows.append({"value": u, "a": a, "b": b, "c": c, "d": d, "p_value": p})
    pvals = [float(r["p_value"]) for r in rows]
    qvals, _ = bh_fdr(pvals)
    for i, q in enumerate(qvals):
        rows[i]["q_value"] = float(q)
    rows.sort(key=lambda r: (float(r["q_value"]), float(r["p_value"]), str(r["value"])))
    findings = [
        _make_finding(
            plugin_id,
            f"fisher:{r['value']}",
            "Categorical enrichment in case partition",
            "Category is enriched in case vs control split under Fisher exact test.",
            "Significant enrichment indicates categorical composition shift.",
            {"metrics": {"value": r["value"], "p_value": r["p_value"], "q_value": r["q_value"], "table": [r["a"], r["b"], r["c"], r["d"]]}},
            recommendation="Inspect enriched categories and map to workflow/policy changes.",
            severity="warn" if float(r["q_value"]) > 0.01 else "critical",
            confidence=0.7,
        )
        for r in rows
        if float(r["q_value"]) <= 0.1
    ][: int(config.get("max_findings", 30))]
    artifacts = [_artifact_json(ctx, plugin_id, "fisher_enrichment.json", {"column": cat_col, "rows": rows[:100]}, "Fisher exact enrichment summary")]
    return _finalize(plugin_id, ctx, df, sample_meta, "Computed Fisher enrichment scan.", findings, artifacts, extra_metrics={"runtime_ms": _runtime_ms(timer), "significant_count": len(findings)})


def _line_fraction(mask: np.ndarray, axis: int) -> float:
    arr = mask.astype(np.uint8)
    if axis == 0:
        arr = arr.T
    total = int(np.sum(arr))
    if total <= 0:
        return 0.0
    count = 0
    for row in arr:
        run = 0
        for v in row:
            if v:
                run += 1
            else:
                if run >= 2:
                    count += run
                run = 0
        if run >= 2:
            count += run
    return float(count) / float(total)


def _handler_recurrence_quantification_rqa_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    col = _best_latency_col(df, inferred)
    if col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_column")
    x = _to_array(df[col])
    if len(x) < 60:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_points")
    frame = pd.DataFrame({"x": x})
    try:
        frame = _cap_for_quadratic(plugin_id, ctx, frame, config)
    except _QuadraticCapExceeded as exc:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "quadratic_cap_exceeded", debug={"reason": str(exc)})
    x = frame["x"].to_numpy(dtype=float)
    if len(x) < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_points_after_cap")
    # m=2, delay=1
    V = np.stack([x[:-1], x[1:]], axis=1)
    D = np.linalg.norm(V[:, None, :] - V[None, :, :], axis=2)
    tri = D[np.triu_indices_from(D, k=1)]
    eps = float(0.1 * np.median(tri)) if len(tri) else 0.0
    eps = max(eps, 1e-9)
    R = D <= eps
    rr = float(np.mean(R))
    det = _line_fraction(R, axis=1)
    lam = _line_fraction(R, axis=0)
    mid = len(V) // 2
    early = V[:mid]
    late = V[mid:]
    if len(early) > 2 and len(late) > 2:
        De = np.linalg.norm(early[:, None, :] - early[None, :, :], axis=2)
        Dl = np.linalg.norm(late[:, None, :] - late[None, :, :], axis=2)
        Re = De <= eps
        Rl = Dl <= eps
        rr_e = float(np.mean(Re))
        rr_l = float(np.mean(Rl))
    else:
        rr_e = rr
        rr_l = rr
    delta_rr = rr_l - rr_e
    findings = []
    if abs(delta_rr) > 0.1:
        findings.append(
            _make_finding(
                plugin_id,
                "rqa_shift",
                "Recurrence regime shift detected",
                "Recurrence metrics differ materially between early and late segments.",
                "Recurrence metric drift indicates temporal dynamics changes.",
                {"metrics": {"recurrence_rate": rr, "determinism": det, "laminarity": lam, "early_rr": rr_e, "late_rr": rr_l, "delta_rr": delta_rr}},
                recommendation="Investigate dynamics changes across the early/late boundary.",
                severity="warn" if abs(delta_rr) <= 0.2 else "critical",
                confidence=0.72,
            )
        )
    artifacts = [_artifact_json(ctx, plugin_id, "rqa.json", {"column": col, "eps": eps, "recurrence_rate": rr, "determinism": det, "laminarity": lam, "early_rr": rr_e, "late_rr": rr_l, "delta_rr": delta_rr}, "RQA summary")]
    return _finalize(plugin_id, ctx, df, sample_meta, "Computed recurrence quantification metrics.", findings, artifacts, extra_metrics={"runtime_ms": _runtime_ms(timer)})


def _handler_alias_sparse_pca_interpretable_components_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    return _handler_sparse_pca_interpretable_components_v1(plugin_id, ctx, df, config, inferred, timer, sample_meta)


HANDLERS: dict[str, Callable[..., PluginResult]] = {
    "analysis_bsts_intervention_counterfactual_v1": _handler_bsts_intervention_counterfactual_v1,
    "analysis_stl_seasonal_decompose_v1": _handler_stl_seasonal_decompose_v1,
    "analysis_seasonal_holt_winters_forecast_residuals_v1": _handler_seasonal_holt_winters_forecast_residuals_v1,
    "analysis_lomb_scargle_periodogram_v1": _handler_lomb_scargle_periodogram_v1,
    "analysis_garch_volatility_shift_v1": _handler_garch_volatility_shift_v1,
    "analysis_bayesian_online_changepoint_studentt_v1": _handler_bayesian_online_changepoint_studentt_v1,
    "analysis_wild_binary_segmentation_v1": _handler_wild_binary_segmentation_v1,
    "analysis_fused_lasso_trend_filtering_v1": _handler_fused_lasso_trend_filtering_v1,
    "analysis_cusum_on_model_residuals_v1": _handler_cusum_on_model_residuals_v1,
    "analysis_change_score_consensus_v1": _handler_change_score_consensus_v1,
    "analysis_benfords_law_anomaly_v1": _handler_benfords_law_anomaly_v1,
    "analysis_geometric_median_multivariate_location_v1": _handler_geometric_median_multivariate_location_v1,
    "analysis_random_matrix_marchenko_pastur_denoise_v1": _handler_random_matrix_marchenko_pastur_denoise_v1,
    "analysis_outlier_influence_cooks_distance_v1": _handler_outlier_influence_cooks_distance_v1,
    "analysis_heavy_tail_index_hill_v1": _handler_heavy_tail_index_hill_v1,
    "analysis_distance_correlation_screen_v1": _handler_distance_correlation_screen_v1,
    "analysis_gam_spline_regression_v1": _handler_gam_spline_regression_v1,
    "analysis_quantile_loss_boosting_v1": _handler_quantile_loss_boosting_v1,
    "analysis_quantile_regression_forest_v1": _handler_quantile_regression_forest_v1,
    "analysis_sparse_pca_interpretable_components_v1": _handler_alias_sparse_pca_interpretable_components_v1,
    "analysis_ica_source_separation_v1": _handler_ica_source_separation_v1,
    "analysis_cca_crossblock_association_v1": _handler_cca_crossblock_association_v1,
    "analysis_factor_rotation_varimax_v1": _handler_factor_rotation_varimax_v1,
    "analysis_subspace_tracking_oja_v1": _handler_subspace_tracking_oja_v1,
    "analysis_multicollinearity_vif_screen_v1": _handler_multicollinearity_vif_screen_v1,
    "analysis_zero_inflated_count_model_v1": _handler_zero_inflated_count_model_v1,
    "analysis_negative_binomial_overdispersion_v1": _handler_negative_binomial_overdispersion_v1,
    "analysis_dirichlet_multinomial_categorical_overdispersion_v1": _handler_dirichlet_multinomial_categorical_overdispersion_v1,
    "analysis_fisher_exact_enrichment_v1": _handler_fisher_exact_enrichment_v1,
    "analysis_recurrence_quantification_rqa_v1": _handler_recurrence_quantification_rqa_v1,
}
