from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable

import numpy as np
import pandas as pd

from statistic_harness.core.stat_plugins import (
    BudgetTimer,
    deterministic_sample,
    robust_center_scale,
    stable_id,
    standardized_median_diff,
)
from statistic_harness.core.types import PluginArtifact, PluginResult

try:  # optional
    from sklearn.covariance import LedoitWolf  # type: ignore
    from sklearn.linear_model import HuberRegressor, PoissonRegressor, RANSACRegressor  # type: ignore

    HAS_SKLEARN = True
except Exception:  # pragma: no cover - optional dependency
    LedoitWolf = None
    HuberRegressor = None
    PoissonRegressor = None
    RANSACRegressor = None
    HAS_SKLEARN = False

try:  # optional
    import networkx as nx  # type: ignore

    HAS_NETWORKX = True
except Exception:  # pragma: no cover - optional dependency
    nx = None
    HAS_NETWORKX = False


NEXT30B_IDS: tuple[str, ...] = (
    "analysis_beta_binomial_overdispersion_v1",
    "analysis_circular_time_of_day_drift_v1",
    "analysis_mann_kendall_trend_test_v1",
    "analysis_quantile_mapping_drift_qq_v1",
    "analysis_constraints_violation_detector_v1",
    "analysis_negative_binomial_overdispersion_v1",
    "analysis_partial_correlation_network_shift_v1",
    "analysis_piecewise_linear_trend_changepoints_v1",
    "analysis_poisson_regression_rate_drivers_v1",
    "analysis_quantile_sketch_p2_streaming_v1",
    "analysis_robust_regression_huber_ransac_v1",
    "analysis_state_space_smoother_level_shift_v1",
    "analysis_aft_survival_lognormal_v1",
    "analysis_competing_risks_cif_v1",
    "analysis_haar_wavelet_transient_detector_v1",
    "analysis_hurst_exponent_long_memory_v1",
    "analysis_permutation_entropy_drift_v1",
    "analysis_capacity_frontier_envelope_v1",
    "analysis_graph_assortativity_shift_v1",
    "analysis_graph_pagerank_hotspots_v1",
    "analysis_higuchi_fractal_dimension_v1",
    "analysis_marked_point_process_intensity_v1",
    "analysis_spectral_radius_stability_v1",
    "analysis_bootstrap_ci_effect_sizes_v1",
    "analysis_energy_distance_two_sample_v1",
    "analysis_randomization_test_median_shift_v1",
    "analysis_distance_covariance_dependence_v1",
    "analysis_graph_motif_triads_shift_v1",
    "analysis_multiscale_entropy_mse_v1",
    "analysis_sample_entropy_irregularity_v1",
)


@dataclass(frozen=True)
class _Split:
    pre: pd.DataFrame
    post: pd.DataFrame
    mode: str
    time_column: str | None


def _safe_id(plugin_id: str, key: str) -> str:
    try:
        return stable_id((plugin_id, key))
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
    ctx.logger(f"END runtime_ms={int(metrics.get('runtime_ms', 0))} findings={len(findings)}")
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


def _runtime_ms(timer: BudgetTimer) -> int:
    return int(max(0.0, timer.elapsed_ms()))


def _numeric_columns(df: pd.DataFrame, inferred: dict[str, Any], max_cols: int | None = None) -> list[str]:
    cols = [str(c) for c in (inferred.get("numeric_columns") or []) if str(c) in df.columns]
    if not cols:
        cols = [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if max_cols is not None:
        cols = cols[: max(1, int(max_cols))]
    return cols


def _categorical_columns(df: pd.DataFrame, inferred: dict[str, Any], max_cols: int | None = None) -> list[str]:
    cols = [str(c) for c in (inferred.get("categorical_columns") or []) if str(c) in df.columns]
    if not cols:
        cols = [str(c) for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if max_cols is not None:
        cols = cols[: max(1, int(max_cols))]
    return cols


def _time_series(df: pd.DataFrame, inferred: dict[str, Any]) -> tuple[str | None, pd.Series | None]:
    time_col = inferred.get("time_column")
    if isinstance(time_col, str) and time_col in df.columns:
        parsed = pd.to_datetime(df[time_col], errors="coerce", utc=False)
        if parsed.notna().sum() >= 10:
            return time_col, parsed
    for col in df.columns:
        lname = str(col).lower()
        if "time" not in lname and "date" not in lname and not lname.endswith("_dt"):
            continue
        parsed = pd.to_datetime(df[col], errors="coerce", utc=False)
        if parsed.notna().sum() >= 10:
            return str(col), parsed
    return None, None


def _split_pre_post(df: pd.DataFrame, inferred: dict[str, Any]) -> _Split:
    time_col, ts = _time_series(df, inferred)
    if ts is not None:
        frame = df.copy()
        frame["_ts"] = ts
        frame = frame.sort_values("_ts", kind="mergesort")
        frame = frame.drop(columns=["_ts"])
        cut = max(1, len(frame) // 2)
        pre = frame.iloc[:cut]
        post = frame.iloc[cut:]
        if len(pre) > 0 and len(post) > 0:
            return _Split(pre=pre, post=post, mode="time_half", time_column=time_col)
    cut = max(1, len(df) // 2)
    return _Split(pre=df.iloc[:cut], post=df.iloc[cut:], mode="index_half", time_column=None)


def _cap_quadratic(df: pd.DataFrame, config: dict[str, Any], seed: int, ctx) -> tuple[pd.DataFrame, dict[str, Any]]:
    cap = int(config.get("plugin", {}).get("max_points_for_quadratic", 2000))
    if len(df) <= cap:
        return df, {"quadratic_capped": False, "quadratic_cap": cap}
    sampled, _ = deterministic_sample(df, cap, seed=seed)
    ctx.logger(f"SKIP reason=quadratic_cap_applied rows={len(df)} cap={cap} sampled={len(sampled)}")
    return sampled, {"quadratic_capped": True, "quadratic_cap": cap, "rows_before": int(len(df)), "rows_after": int(len(sampled))}


def _count_column(df: pd.DataFrame, inferred: dict[str, Any]) -> str | None:
    hints = ("count", "qty", "volume", "num", "events")
    candidates = _numeric_columns(df, inferred, max_cols=100)
    for col in candidates:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if vals.empty:
            continue
        if not any(h in col.lower() for h in hints):
            continue
        if float((vals >= 0).mean()) < 0.9:
            continue
        if float(np.mean(np.isclose(vals.to_numpy(dtype=float) % 1.0, 0.0))) >= 0.9:
            return col
    for col in candidates:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if vals.empty:
            continue
        if float((vals >= 0).mean()) < 0.9:
            continue
        if float(np.mean(np.isclose(vals.to_numpy(dtype=float) % 1.0, 0.0))) >= 0.9:
            return col
    return None


def _binary_columns(df: pd.DataFrame, inferred: dict[str, Any]) -> list[str]:
    cols: list[str] = []
    for col in _numeric_columns(df, inferred, max_cols=100):
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if vals.empty:
            continue
        uniq = set(np.unique(vals.to_numpy(dtype=float)).tolist())
        if uniq.issubset({0.0, 1.0}):
            cols.append(col)
    for col in df.columns:
        if pd.api.types.is_bool_dtype(df[col]):
            c = str(col)
            if c not in cols:
                cols.append(c)
    return cols


def _duration_column(df: pd.DataFrame, inferred: dict[str, Any]) -> str | None:
    hints = ("duration", "latency", "wait", "elapsed", "runtime", "service", "time")
    for col in _numeric_columns(df, inferred, max_cols=100):
        if any(h in col.lower() for h in hints):
            return col
    cols = _numeric_columns(df, inferred, max_cols=100)
    return cols[0] if cols else None


def _variance_sorted_numeric(df: pd.DataFrame, inferred: dict[str, Any], limit: int = 8) -> list[str]:
    cols = _numeric_columns(df, inferred, max_cols=80)
    scored: list[tuple[float, str]] = []
    for col in cols:
        vals = pd.to_numeric(df[col], errors="coerce")
        scored.append((float(np.nanvar(vals.to_numpy(dtype=float))), col))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [c for _, c in scored[:limit]]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        x = float(value)
        if not math.isfinite(x):
            return default
        return x
    except Exception:
        return default


def _hash_node(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]


def _fit_linear(x: np.ndarray, y: np.ndarray, lam: float = 1e-6) -> tuple[np.ndarray, float]:
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    X = np.concatenate([np.ones((len(x), 1), dtype=float), x], axis=1)
    XtX = X.T @ X
    XtX += lam * np.eye(X.shape[1])
    beta = np.linalg.pinv(XtX) @ (X.T @ y)
    pred = X @ beta
    resid = y - pred
    sse = float(np.sum(resid * resid))
    return beta, sse


def _fit_linear_r2(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    beta, sse = _fit_linear(x, y, lam=1e-6)
    yvar = float(np.var(y))
    if yvar <= 0:
        return beta, 0.0
    r2 = 1.0 - sse / max(1e-9, yvar * len(y))
    return beta, float(max(-1.0, min(1.0, r2)))


def _mad_scale(values: np.ndarray) -> float:
    _, mad = robust_center_scale(values)
    return float(max(1e-9, mad))


def _distance_covariance(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    a = np.sqrt(np.maximum(0.0, (x - x.T) ** 2))
    b = np.sqrt(np.maximum(0.0, (y - y.T) ** 2))
    A = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
    B = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()
    dcov2 = float(np.mean(A * B))
    dvarx = float(np.mean(A * A))
    dvary = float(np.mean(B * B))
    dcov = math.sqrt(max(0.0, dcov2))
    dcor = dcov / max(1e-9, math.sqrt(max(0.0, dvarx * dvary)))
    return dcov, float(max(0.0, min(1.0, dcor)))


def _sample_entropy(x: np.ndarray, m: int = 2, r: float | None = None) -> float:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < (m + 3):
        return 0.0
    if r is None:
        r = 0.2 * max(np.std(x), 1e-9)
    r = max(float(r), 1e-9)
    count_m = 0
    count_m1 = 0
    for i in range(n - m):
        template_m = x[i : i + m]
        template_m1 = x[i : i + m + 1]
        for j in range(i + 1, n - m):
            if np.max(np.abs(template_m - x[j : j + m])) <= r:
                count_m += 1
                if j < n - m - 1 and np.max(np.abs(template_m1 - x[j : j + m + 1])) <= r:
                    count_m1 += 1
    if count_m <= 0 or count_m1 <= 0:
        return 0.0
    return float(-math.log(count_m1 / count_m))


def _permutation_entropy(x: np.ndarray, m: int = 3, tau: int = 1) -> float:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < m * tau + 1:
        return 0.0
    counts: dict[tuple[int, ...], int] = defaultdict(int)
    total = 0
    for i in range(n - (m - 1) * tau):
        window = x[i : i + m * tau : tau]
        ranks = tuple(np.argsort(window, kind="mergesort").tolist())
        counts[ranks] += 1
        total += 1
    if total <= 0:
        return 0.0
    probs = np.array([v / total for v in counts.values()], dtype=float)
    h = -float(np.sum(probs * np.log(np.maximum(probs, 1e-12))))
    return float(h / math.log(math.factorial(m)))


def _coarse_grain(x: np.ndarray, scale: int) -> np.ndarray:
    n = len(x) // scale
    if n <= 0:
        return np.array([], dtype=float)
    return x[: n * scale].reshape(n, scale).mean(axis=1)


def _hurst_rs(x: np.ndarray) -> tuple[float, float]:
    n = len(x)
    if n < 60:
        return 0.5, 0.0
    windows = [8, 12, 16, 24, 32, 48, 64, 96]
    xs = []
    ys = []
    for w in windows:
        if w >= n // 2:
            continue
        rs_vals = []
        for i in range(0, n - w + 1, w):
            segment = x[i : i + w]
            if len(segment) < w:
                continue
            seg = segment - np.mean(segment)
            z = np.cumsum(seg)
            R = float(np.max(z) - np.min(z))
            S = float(np.std(segment))
            if S > 0:
                rs_vals.append(R / S)
        if rs_vals:
            xs.append(math.log(w))
            ys.append(math.log(float(np.mean(rs_vals))))
    if len(xs) < 2:
        return 0.5, 0.0
    beta, r2 = _fit_linear(np.array(xs, dtype=float), np.array(ys, dtype=float))
    H = float(beta[1]) if len(beta) > 1 else 0.5
    return H, r2


def _higuchi_fd(x: np.ndarray, k_max: int = 10) -> float:
    n = len(x)
    if n < 30:
        return 1.0
    Lk = []
    k_vals = list(range(1, max(2, k_max + 1)))
    for k in k_vals:
        Lm = []
        for m in range(k):
            idx = np.arange(m, n, k)
            if len(idx) < 2:
                continue
            diffs = np.abs(np.diff(x[idx]))
            norm = (n - 1) / (len(idx) * k)
            L = float(np.sum(diffs) * norm)
            if L > 0:
                Lm.append(L)
        if Lm:
            Lk.append(float(np.mean(Lm)))
        else:
            Lk.append(1e-9)
    X = np.log(np.array([1.0 / k for k in k_vals], dtype=float))
    Y = np.log(np.maximum(np.array(Lk, dtype=float), 1e-12))
    beta, _ = _fit_linear(X, Y)
    return float(beta[1]) if len(beta) > 1 else 1.0


def _find_graph_columns(df: pd.DataFrame, inferred: dict[str, Any]) -> tuple[str | None, str | None]:
    cols = list(df.columns)
    low = {c: str(c).lower() for c in cols}
    src_hints = ("src", "from", "parent", "source", "caller")
    dst_hints = ("dst", "to", "child", "target", "callee")
    src = None
    dst = None
    for col in cols:
        if any(h in low[col] for h in src_hints):
            src = col
            break
    for col in cols:
        if col == src:
            continue
        if any(h in low[col] for h in dst_hints):
            dst = col
            break
    if src and dst:
        return str(src), str(dst)
    cats = _categorical_columns(df, inferred, max_cols=6)
    if len(cats) >= 2:
        return cats[0], cats[1]
    return None, None


def _build_edges(df: pd.DataFrame, inferred: dict[str, Any], max_edges: int = 20000) -> tuple[pd.DataFrame, str, str]:
    src_col, dst_col = _find_graph_columns(df, inferred)
    if src_col is None or dst_col is None:
        return pd.DataFrame(columns=["src", "dst"]), "", ""
    edges = pd.DataFrame(
        {
            "src": df[src_col].astype(str),
            "dst": df[dst_col].astype(str),
        }
    )
    edges = edges[(edges["src"] != "") & (edges["dst"] != "") & (edges["src"] != "nan") & (edges["dst"] != "nan")]
    edges = edges[edges["src"] != edges["dst"]]
    if len(edges) > max_edges:
        edges = edges.iloc[:max_edges]
    return edges, src_col, dst_col


def _degree_assortativity(edges: pd.DataFrame) -> float:
    if edges.empty:
        return 0.0
    src_deg = Counter(edges["src"].tolist())
    dst_deg = Counter(edges["dst"].tolist())
    x = []
    y = []
    for row in edges.itertuples(index=False):
        x.append(float(src_deg[row.src]))
        y.append(float(dst_deg[row.dst]))
    if len(x) < 5:
        return 0.0
    if np.std(x) <= 0 or np.std(y) <= 0:
        return 0.0
    return float(np.corrcoef(np.array(x), np.array(y))[0, 1])


def _pagerank_power_iteration(edges: pd.DataFrame, alpha: float = 0.85, iters: int = 60) -> dict[str, float]:
    nodes = sorted(set(edges["src"].tolist()) | set(edges["dst"].tolist()))
    if not nodes:
        return {}
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    out = [[] for _ in range(n)]
    for row in edges.itertuples(index=False):
        out[idx[row.src]].append(idx[row.dst])
    pr = np.full(n, 1.0 / n, dtype=float)
    for _ in range(iters):
        new = np.full(n, (1.0 - alpha) / n, dtype=float)
        for i in range(n):
            if out[i]:
                share = alpha * pr[i] / len(out[i])
                for j in out[i]:
                    new[j] += share
            else:
                new += alpha * pr[i] / n
        pr = new
    return {nodes[i]: float(pr[i]) for i in range(n)}


def _event_type_column(df: pd.DataFrame, inferred: dict[str, Any]) -> str | None:
    hints = ("status", "type", "outcome", "event", "result")
    for col in _categorical_columns(df, inferred, max_cols=20):
        if any(h in col.lower() for h in hints):
            return col
    cols = _categorical_columns(df, inferred, max_cols=20)
    return cols[0] if cols else None


def _poisson_regression(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, str]:
    if HAS_SKLEARN and PoissonRegressor is not None:
        model = PoissonRegressor(alpha=1e-6, fit_intercept=True, max_iter=400)
        model.fit(X, y)
        coef = np.concatenate([[float(model.intercept_)], model.coef_.astype(float)])
        return coef, "sklearn_poisson"
    ylog = np.log1p(np.maximum(0.0, y))
    beta, _ = _fit_linear(X, ylog, lam=1e-3)
    return beta, "log_ols_fallback"


def _bootstrap_effect_ci(
    left: np.ndarray, right: np.ndarray, B: int, rng: np.random.Generator, timer: BudgetTimer
) -> tuple[float, float, float]:
    obs = standardized_median_diff(left, right)
    vals = []
    if len(left) == 0 or len(right) == 0:
        return obs, 0.0, 0.0
    for _ in range(B):
        li = rng.integers(0, len(left), size=len(left))
        ri = rng.integers(0, len(right), size=len(right))
        vals.append(standardized_median_diff(left[li], right[ri]))
        _ensure_budget(timer)
    arr = np.array(vals, dtype=float)
    lo, hi = np.quantile(arr, [0.025, 0.975]).tolist()
    return float(obs), float(lo), float(hi)


def _perm_pvalue(
    a: np.ndarray,
    b: np.ndarray,
    score_fn: Callable[[np.ndarray, np.ndarray], float],
    B: int,
    rng: np.random.Generator,
    timer: BudgetTimer,
) -> tuple[float, float]:
    obs = float(score_fn(a, b))
    pool = np.concatenate([a, b])
    n = len(a)
    ge = 1
    for _ in range(B):
        perm = rng.permutation(pool)
        stat = float(score_fn(perm[:n], perm[n:]))
        if abs(stat) >= abs(obs):
            ge += 1
        _ensure_budget(timer)
    p = ge / (B + 1)
    return obs, float(p)


def _energy_stat(a: np.ndarray, b: np.ndarray) -> float:
    A = np.abs(a[:, None] - b[None, :]).mean()
    AA = np.abs(a[:, None] - a[None, :]).mean()
    BB = np.abs(b[:, None] - b[None, :]).mean()
    return float(2.0 * A - AA - BB)


def _partial_corr(frame: pd.DataFrame) -> np.ndarray:
    X = frame.to_numpy(dtype=float)
    X = (X - np.mean(X, axis=0)) / np.maximum(np.std(X, axis=0), 1e-9)
    if HAS_SKLEARN and LedoitWolf is not None:
        cov = LedoitWolf().fit(X).covariance_
    else:
        cov = np.cov(X, rowvar=False)
        cov = cov + 1e-3 * np.eye(cov.shape[0])
    P = np.linalg.pinv(cov)
    d = np.sqrt(np.maximum(1e-9, np.diag(P)))
    pc = -P / np.outer(d, d)
    np.fill_diagonal(pc, 1.0)
    return pc


def _handler_beta_binomial_overdispersion_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    binary_cols = _binary_columns(df, inferred)
    if not binary_cols:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_binary_columns")
    split = _split_pre_post(df, inferred)
    rows = []
    for col in binary_cols[: int(config.get("max_cols", 10))]:
        pre = pd.to_numeric(split.pre[col], errors="coerce").dropna().to_numpy(dtype=float)
        post = pd.to_numeric(split.post[col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(pre) < 10 or len(post) < 10:
            continue
        rates = np.array([np.mean(pre), np.mean(post)], dtype=float)
        ns = np.array([len(pre), len(post)], dtype=float)
        pbar = float(np.mean(rates))
        bin_proxy = float(np.mean(np.maximum(1e-9, pbar * (1.0 - pbar) / np.maximum(ns, 1.0))))
        ratio = float(np.var(rates) / max(1e-9, bin_proxy))
        rows.append({"column": col, "rates": rates.tolist(), "sizes": ns.tolist(), "dispersion_ratio": ratio})
        _ensure_budget(timer)
    if not rows:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_partition_data")
    rows.sort(key=lambda r: (-float(r["dispersion_ratio"]), str(r["column"])))
    top = rows[0]
    findings = []
    if float(top["dispersion_ratio"]) > 1.2:
        findings.append(
            _make_finding(
                plugin_id,
                f"dispersion:{top['column']}",
                "Beta-binomial overdispersion signal",
                "Partition-level binary rates vary more than binomial expectation.",
                "Excess rate variability indicates heterogeneity beyond iid binomial assumptions.",
                {"metrics": top},
                recommendation="Segment drivers of binary outcome by process/team and calibrate controls per segment.",
                severity="warn" if float(top["dispersion_ratio"]) < 2.0 else "critical",
                confidence=min(0.92, 0.55 + min(0.35, float(top["dispersion_ratio"]) / 4.0)),
            )
        )
    artifacts = [
        _artifact_json(
            ctx,
            plugin_id,
            "beta_binomial_overdispersion.json",
            {"split_mode": split.mode, "rows": rows[:30]},
            "Beta-binomial overdispersion summary",
        )
    ]
    return _finalize(
        plugin_id,
        ctx,
        df,
        sample_meta,
        "Computed beta-binomial overdispersion diagnostics.",
        findings,
        artifacts,
        extra_metrics={
            "runtime_ms": _runtime_ms(timer),
            "binary_columns_scanned": int(len(rows)),
            "partitions": 2,
            "dispersion_ratio_max": float(top["dispersion_ratio"]),
        },
    )


def _handler_circular_time_of_day_drift_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    split = _split_pre_post(df, inferred)
    if split.time_column is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "time_column_not_found")

    def _angles(frame: pd.DataFrame) -> np.ndarray:
        ts = pd.to_datetime(frame[split.time_column], errors="coerce")
        h = ts.dt.hour.fillna(0).astype(float).to_numpy(dtype=float)
        m = ts.dt.minute.fillna(0).astype(float).to_numpy(dtype=float)
        return 2.0 * np.pi * ((h + (m / 60.0)) / 24.0)

    a_pre = _angles(split.pre)
    a_post = _angles(split.post)
    if len(a_pre) < 20 or len(a_post) < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_time_samples")

    def _cstats(angles: np.ndarray) -> tuple[float, float, float]:
        C = float(np.mean(np.cos(angles)))
        S = float(np.mean(np.sin(angles)))
        R = float(math.sqrt(C * C + S * S))
        mu = float(math.atan2(S, C))
        return C, R, mu

    _, pre_R, pre_mu = _cstats(a_pre)
    _, post_R, post_mu = _cstats(a_post)
    delta_mu = float(abs(math.atan2(math.sin(post_mu - pre_mu), math.cos(post_mu - pre_mu))))
    delta_R = float(abs(post_R - pre_R))
    findings = []
    if delta_mu > 1.0 or delta_R > 0.2:
        findings.append(
            _make_finding(
                plugin_id,
                "circular_drift",
                "Time-of-day circular drift detected",
                "Circular distribution of event times shifted between early and late periods.",
                "Shifted operational timing can change queue pressure and staffing alignment.",
                {"metrics": {"delta_mu_rad": delta_mu, "delta_R": delta_R, "pre_R": pre_R, "post_R": post_R}},
                recommendation="Re-align cutoffs/staffing to new peak-time pattern and validate SLA impact.",
                severity="warn" if delta_mu <= 1.5 else "critical",
                confidence=min(0.95, 0.6 + min(0.3, delta_mu / 3.0)),
            )
        )
    artifacts = [
        _artifact_json(
            ctx,
            plugin_id,
            "circular_time_drift.json",
            {"time_column": split.time_column, "delta_mu_rad": delta_mu, "delta_R": delta_R, "pre_R": pre_R, "post_R": post_R},
            "Circular time-of-day drift summary",
        )
    ]
    return _finalize(
        plugin_id,
        ctx,
        df,
        sample_meta,
        "Computed circular time-of-day drift metrics.",
        findings,
        artifacts,
        extra_metrics={
            "runtime_ms": _runtime_ms(timer),
            "delta_mu_rad": delta_mu,
            "delta_R": delta_R,
            "pre_R": pre_R,
            "post_R": post_R,
        },
    )


def _handler_mann_kendall_trend_test_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    time_col, ts = _time_series(df, inferred)
    if time_col is None or ts is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "time_column_not_found")
    cols = _variance_sorted_numeric(df, inferred, limit=6)
    if not cols:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_columns")
    value_col = cols[0]
    frame = pd.DataFrame({"ts": ts, "y": pd.to_numeric(df[value_col], errors="coerce")}).dropna().sort_values("ts")
    if len(frame) < 25:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_points")
    frame, cap_meta = _cap_quadratic(frame, config, int(config.get("seed", 1337)), ctx)
    y = frame["y"].to_numpy(dtype=float)
    n = len(y)
    S = 0
    slopes = []
    for i in range(n - 1):
        diff = y[i + 1 :] - y[i]
        S += int(np.sum(np.sign(diff)))
        if i % max(1, n // 50) == 0:
            denom = np.arange(i + 1, n) - i
            slopes.extend((diff / np.maximum(1, denom)).tolist())
        _ensure_budget(timer)
    var_s = n * (n - 1) * (2 * n + 5) / 18.0
    if var_s <= 0:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "degenerate_variance")
    if S > 0:
        z = (S - 1) / math.sqrt(var_s)
    elif S < 0:
        z = (S + 1) / math.sqrt(var_s)
    else:
        z = 0.0
    p = float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0)))))
    sen = float(np.median(np.array(slopes, dtype=float))) if slopes else 0.0
    findings = []
    if p < 0.05 and abs(sen) > 0:
        findings.append(
            _make_finding(
                plugin_id,
                "mann_kendall_trend",
                "Monotonic trend detected",
                "Mann-Kendall test indicates a statistically significant monotonic trend.",
                "Persistent monotonic drift suggests structural process change rather than noise.",
                {"metrics": {"S": int(S), "z": z, "p_value": p, "sen_slope": sen, "column": value_col}},
                recommendation="Investigate policy/process changes over the trend horizon and recalibrate baselines.",
                severity="warn" if abs(z) < 3.0 else "critical",
                confidence=min(0.95, 0.6 + min(0.35, abs(z) / 6.0)),
            )
        )
    artifacts = [
        _artifact_json(
            ctx,
            plugin_id,
            "mann_kendall.json",
            {"column": value_col, "S": int(S), "z": z, "p_value": p, "sen_slope": sen, **cap_meta},
            "Mann-Kendall trend diagnostics",
        )
    ]
    return _finalize(
        plugin_id,
        ctx,
        df,
        sample_meta,
        "Computed Mann-Kendall trend statistics.",
        findings,
        artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "S": int(S), "z": z, "p_value": p, "sen_slope": sen},
    )


def _handler_quantile_mapping_drift_qq_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    split = _split_pre_post(df, inferred)
    num_cols = _variance_sorted_numeric(df, inferred, limit=int(config.get("max_cols", 20)))
    if not num_cols:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_columns")
    qs = np.arange(0.05, 1.0, 0.05)
    rows = []
    for col in num_cols:
        pre = pd.to_numeric(split.pre[col], errors="coerce").dropna().to_numpy(dtype=float)
        post = pd.to_numeric(split.post[col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(pre) < 20 or len(post) < 20:
            continue
        q_pre = np.quantile(pre, qs)
        q_post = np.quantile(post, qs)
        scale = _mad_scale(pre)
        qq_l1 = float(np.mean(np.abs(q_pre - q_post) / scale))
        rows.append(
            {
                "column": col,
                "qq_l1": qq_l1,
                "q_pre": q_pre.tolist(),
                "q_post": q_post.tolist(),
            }
        )
        _ensure_budget(timer)
    if not rows:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_pre_post_numeric")
    rows.sort(key=lambda r: (-float(r["qq_l1"]), str(r["column"])))
    top = rows[0]
    findings = []
    if float(top["qq_l1"]) > 0.5:
        findings.append(
            _make_finding(
                plugin_id,
                f"qq:{top['column']}",
                "Q-Q distribution drift detected",
                "Pre/post quantile mapping shows a material distribution shift.",
                "Distribution-shape drift often degrades static thresholds and old SLAs.",
                {"metrics": {"column": top["column"], "qq_l1": top["qq_l1"]}},
                recommendation="Rebaseline thresholds/alerts for shifted distributions and validate tail risk controls.",
                severity="warn" if float(top["qq_l1"]) < 1.0 else "critical",
                confidence=min(0.95, 0.58 + min(0.35, float(top["qq_l1"]) / 3.0)),
            )
        )
    artifacts = [
        _artifact_json(
            ctx,
            plugin_id,
            "qq_drift.json",
            {"split_mode": split.mode, "quantiles": qs.tolist(), "rows": rows[:30]},
            "Q-Q drift diagnostics",
        )
    ]
    return _finalize(
        plugin_id,
        ctx,
        df,
        sample_meta,
        "Computed Q-Q quantile mapping drift.",
        findings,
        artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "cols_scanned": len(rows), "top_qq_l1": float(top["qq_l1"])},
    )


def _handler_constraints_violation_detector_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    split = _split_pre_post(df, inferred)
    rows = []
    for col in _numeric_columns(df, inferred, max_cols=int(config.get("max_cols", 40))):
        all_vals = pd.to_numeric(df[col], errors="coerce")
        if all_vals.notna().sum() < 20:
            continue
        hints_nonneg = any(k in col.lower() for k in ("count", "qty", "num", "volume", "duration", "time"))
        inferred_nonneg = float((all_vals.dropna() >= 0).mean()) >= 0.99
        require_nonneg = hints_nonneg or inferred_nonneg
        require_int = any(k in col.lower() for k in ("count", "qty", "num")) and float(
            np.mean(np.isclose(all_vals.dropna().to_numpy(dtype=float) % 1.0, 0.0))
        ) >= 0.9
        pre_vals = pd.to_numeric(split.pre[col], errors="coerce").dropna().to_numpy(dtype=float)
        post_vals = pd.to_numeric(split.post[col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(pre_vals) < 10 or len(post_vals) < 10:
            continue
        pre_viol = 0.0
        post_viol = 0.0
        if require_nonneg:
            pre_viol += float(np.mean(pre_vals < 0.0))
            post_viol += float(np.mean(post_vals < 0.0))
        if require_int:
            pre_viol += float(np.mean(~np.isclose(pre_vals % 1.0, 0.0)))
            post_viol += float(np.mean(~np.isclose(post_vals % 1.0, 0.0)))
        rows.append(
            {
                "column": col,
                "require_nonneg": require_nonneg,
                "require_int_like": require_int,
                "pre_violation_rate": pre_viol,
                "post_violation_rate": post_viol,
                "delta_violation_rate": post_viol - pre_viol,
            }
        )
        _ensure_budget(timer)
    if not rows:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_constraints_detected")
    rows.sort(key=lambda r: (-float(max(r["post_violation_rate"], r["delta_violation_rate"])), str(r["column"])))
    top = rows[0]
    findings = []
    if float(top["post_violation_rate"]) > 0.0 or float(top["delta_violation_rate"]) > 0.0:
        findings.append(
            _make_finding(
                plugin_id,
                f"constraints:{top['column']}",
                "Constraint violations detected",
                "Column violates inferred data constraints or violation rate increased over time.",
                "Violation growth indicates quality/process regressions that can invalidate downstream analytics.",
                {"metrics": top},
                recommendation="Add pre-ingest guards for this field and trace upstream producer changes.",
                severity="warn" if float(top["post_violation_rate"]) < 0.05 else "critical",
                confidence=0.72,
            )
        )
    artifacts = [_artifact_json(ctx, plugin_id, "constraint_violations.json", {"rows": rows[:50]}, "Constraint violation summary")]
    return _finalize(
        plugin_id,
        ctx,
        df,
        sample_meta,
        "Computed invariant constraint violations.",
        findings,
        artifacts,
        extra_metrics={
            "runtime_ms": _runtime_ms(timer),
            "constraints_checked": len(rows),
            "violations_total": int(sum(1 for r in rows if float(r["post_violation_rate"]) > 0.0)),
            "max_violation_rate": float(max(float(r["post_violation_rate"]) for r in rows)),
        },
    )


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
    col = _count_column(df, inferred)
    if col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_integer_like_count_column")
    x = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
    x = np.clip(np.round(x), 0, None)
    if len(x) < 40:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    m = float(np.mean(x))
    v = float(np.var(x))
    ratio = float(v / max(m, 1e-9))
    k = float((m * m) / max(v - m, 1e-9))
    findings = []
    if ratio > 2.0:
        findings.append(
            _make_finding(
                plugin_id,
                f"nb:{col}",
                "Negative-binomial overdispersion detected",
                "Count variance materially exceeds Poisson-equivalent mean.",
                "Overdispersion indicates latent heterogeneity/burstiness in count-generating process.",
                {"metrics": {"column": col, "mean": m, "variance": v, "overdispersion_ratio": ratio, "nb_size_k": k}},
                recommendation="Use NB/mixture assumptions for capacity planning and anomaly thresholds.",
                severity="warn" if ratio <= 5 else "critical",
                confidence=min(0.94, 0.58 + min(0.3, ratio / 10.0)),
            )
        )
    artifacts = [
        _artifact_json(
            ctx,
            plugin_id,
            "nb_overdispersion.json",
            {"column": col, "mean": m, "variance": v, "overdispersion_ratio": ratio, "nb_size_k": k},
            "Negative-binomial overdispersion summary",
        )
    ]
    return _finalize(
        plugin_id,
        ctx,
        df,
        sample_meta,
        "Computed negative-binomial overdispersion metrics.",
        findings,
        artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "count_cols_scanned": 1, "overdispersion_ratio_max": ratio},
    )


def _handler_partial_correlation_network_shift_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    split = _split_pre_post(df, inferred)
    cols = _variance_sorted_numeric(df, inferred, limit=min(25, int(config.get("max_cols", 25))))
    if len(cols) < 3:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "need_three_numeric_columns")
    pre = split.pre[cols].apply(pd.to_numeric, errors="coerce").dropna()
    post = split.post[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(pre) < max(30, len(cols) + 5) or len(post) < max(30, len(cols) + 5):
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows_for_precision")
    pc_pre = _partial_corr(pre)
    pc_post = _partial_corr(post)
    delta = np.abs(pc_post - pc_pre)
    np.fill_diagonal(delta, 0.0)
    i, j = np.unravel_index(int(np.argmax(delta)), delta.shape)
    max_delta = float(delta[i, j])
    findings = []
    if max_delta > 0.2:
        findings.append(
            _make_finding(
                plugin_id,
                f"pcorr:{cols[i]}:{cols[j]}",
                "Partial-correlation network shift detected",
                "Conditional dependency between feature pair changed between pre/post windows.",
                "Shift in conditional structure implies changed driver interactions, not just marginal drift.",
                {"metrics": {"edge": [cols[i], cols[j]], "delta_partial_corr": max_delta, "pre": float(pc_pre[i, j]), "post": float(pc_post[i, j])}},
                recommendation="Investigate coupled control changes across this feature pair and downstream effects.",
                severity="warn" if max_delta < 0.35 else "critical",
                confidence=min(0.93, 0.58 + min(0.32, max_delta)),
            )
        )
    top_edges = []
    for a, b in combinations(range(len(cols)), 2):
        top_edges.append(
            {
                "left": cols[a],
                "right": cols[b],
                "delta_partial_corr": float(delta[a, b]),
                "pre_partial_corr": float(pc_pre[a, b]),
                "post_partial_corr": float(pc_post[a, b]),
            }
        )
    top_edges.sort(key=lambda r: (-abs(float(r["delta_partial_corr"])), str(r["left"]), str(r["right"])))
    artifacts = [
        _artifact_json(
            ctx,
            plugin_id,
            "partial_corr_shift.json",
            {"p_cols": len(cols), "top_edges": top_edges[:50]},
            "Partial-correlation network shift summary",
        )
    ]
    return _finalize(
        plugin_id,
        ctx,
        df,
        sample_meta,
        "Computed partial-correlation network shift.",
        findings,
        artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "p_cols": len(cols), "max_delta_partial_corr": max_delta},
    )


def _piecewise_single_fit_sse(x: np.ndarray, y: np.ndarray) -> float:
    _, sse = _fit_linear(x.reshape(-1, 1), y, lam=1e-6)
    return sse


def _piecewise_best_split(x: np.ndarray, y: np.ndarray, lo: int, hi: int, min_size: int) -> tuple[int | None, float]:
    if (hi - lo) < (2 * min_size):
        return None, 0.0
    base = _piecewise_single_fit_sse(x[lo:hi], y[lo:hi])
    best_i = None
    best_gain = 0.0
    for i in range(lo + min_size, hi - min_size + 1):
        left = _piecewise_single_fit_sse(x[lo:i], y[lo:i])
        right = _piecewise_single_fit_sse(x[i:hi], y[i:hi])
        gain = base - (left + right)
        if gain > best_gain:
            best_gain = gain
            best_i = i
    return best_i, float(best_gain)


def _handler_piecewise_linear_trend_changepoints_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    time_col, ts = _time_series(df, inferred)
    if time_col is None or ts is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "time_column_not_found")
    cols = _variance_sorted_numeric(df, inferred, limit=3)
    if not cols:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_columns")
    value_col = cols[0]
    frame = pd.DataFrame({"ts": ts, "y": pd.to_numeric(df[value_col], errors="coerce")}).dropna().sort_values("ts")
    if len(frame) < 40:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_points")
    frame, cap_meta = _cap_quadratic(frame, config, int(config.get("seed", 1337)), ctx)
    y = frame["y"].to_numpy(dtype=float)
    x = np.arange(len(y), dtype=float)
    k_max = int(config.get("plugin", {}).get("max_segments", 5))
    min_size = int(config.get("plugin", {}).get("min_segment_size", 12))
    penalty = float(config.get("plugin", {}).get("split_penalty", 2.0))
    segments = [(0, len(y))]
    changepoints = []
    while len(segments) < k_max:
        best_gain = 0.0
        best_seg_idx = -1
        best_split = None
        for sidx, (lo, hi) in enumerate(segments):
            split_idx, gain = _piecewise_best_split(x, y, lo, hi, min_size=min_size)
            if split_idx is not None and gain > best_gain:
                best_gain = gain
                best_seg_idx = sidx
                best_split = split_idx
        if best_split is None or best_gain <= penalty:
            break
        lo, hi = segments.pop(best_seg_idx)
        segments.append((lo, best_split))
        segments.append((best_split, hi))
        segments.sort(key=lambda p: p[0])
        changepoints.append(int(best_split))
        _ensure_budget(timer)
    slopes = []
    sse_total = 0.0
    for lo, hi in segments:
        beta, sse = _fit_linear(x[lo:hi], y[lo:hi], lam=1e-6)
        slope = float(beta[1]) if len(beta) > 1 else 0.0
        slopes.append({"start": int(lo), "end": int(hi), "slope": slope})
        sse_total += float(sse)
    findings = []
    if changepoints:
        biggest = float(max(abs(s["slope"]) for s in slopes))
        findings.append(
            _make_finding(
                plugin_id,
                "piecewise_changepoints",
                "Piecewise linear changepoints identified",
                "Segmented trend fit found structural breakpoints in the series.",
                "Breakpoint-aligned slope shifts are consistent with regime changes.",
                {"metrics": {"segments": len(segments), "changepoints": changepoints, "max_abs_slope": biggest}},
                recommendation="Tie detected breakpoints to release/config/process-change timeline.",
                severity="warn" if len(changepoints) <= 2 else "critical",
                confidence=min(0.93, 0.57 + min(0.33, len(changepoints) / 6.0)),
            )
        )
    artifacts = [
        _artifact_json(
            ctx,
            plugin_id,
            "piecewise_trend.json",
            {"time_column": time_col, "value_column": value_col, "segments": slopes, "changepoints": changepoints, "best_sse": sse_total, **cap_meta},
            "Piecewise linear trend summary",
        )
    ]
    return _finalize(
        plugin_id,
        ctx,
        df,
        sample_meta,
        "Computed piecewise linear trend changepoints.",
        findings,
        artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "segments": len(segments), "changepoints": len(changepoints), "best_sse": sse_total},
    )


def _build_poisson_features(df: pd.DataFrame, inferred: dict[str, Any], target_col: str, max_cols: int) -> tuple[pd.DataFrame, list[str]]:
    ncols = [c for c in _variance_sorted_numeric(df, inferred, limit=max_cols) if c != target_col]
    frame = pd.DataFrame(index=df.index)
    names = []
    for col in ncols:
        vals = pd.to_numeric(df[col], errors="coerce")
        frame[col] = vals
        names.append(col)
    for col in _categorical_columns(df, inferred, max_cols=6):
        if col == target_col:
            continue
        cats = df[col].astype(str)
        top = cats.value_counts().index.tolist()[:3]
        for cat in top:
            name = f"{col}=={cat}"
            frame[name] = (cats == cat).astype(float)
            names.append(name)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    return frame, names


def _handler_poisson_regression_rate_drivers_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    target = _count_column(df, inferred)
    if target is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "count_target_not_found")
    y_all = pd.to_numeric(df[target], errors="coerce")
    X_frame, names = _build_poisson_features(df, inferred, target, max_cols=min(8, int(config.get("max_cols", 8))))
    if X_frame.empty:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_predictors")
    frame = X_frame.copy()
    frame["target"] = y_all.reindex(frame.index)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 30:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    yv = np.clip(frame["target"].to_numpy(dtype=float), 0.0, None)
    Xv = frame.drop(columns=["target"]).to_numpy(dtype=float)
    coef, mode = _poisson_regression(Xv, yv, seed=int(config.get("seed", 1337)))
    pred = np.exp(np.concatenate([np.ones((len(Xv), 1), dtype=float), Xv], axis=1) @ coef)
    dev = float(np.mean((yv - pred) ** 2))
    coeffs = []
    for i, name in enumerate(frame.drop(columns=["target"]).columns.tolist(), start=1):
        coeffs.append({"feature": name, "coef": float(coef[i])})
    coeffs.sort(key=lambda r: (-abs(float(r["coef"])), str(r["feature"])))
    findings = []
    if coeffs:
        findings.append(
            _make_finding(
                plugin_id,
                "poisson_rate_drivers",
                "Poisson rate drivers identified",
                "Poisson-style model attributes count-rate changes to top predictors.",
                "High-magnitude coefficients indicate high leverage on expected event rate.",
                {"metrics": {"target": target, "deviance": dev, "top_coefficients": coeffs[:5], "mode": mode}},
                recommendation="Prioritize controls on top positive-rate drivers and monitor counterfactual impact.",
                severity="info",
                confidence=0.66,
            )
        )
    artifacts = [_artifact_json(ctx, plugin_id, "poisson_regression.json", {"target": target, "mode": mode, "deviance": dev, "coefficients": coeffs[:50]}, "Poisson regression summary")]
    return _finalize(
        plugin_id,
        ctx,
        df,
        sample_meta,
        "Computed Poisson-style rate driver model.",
        findings,
        artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "target": target, "n_features": len(coeffs), "deviance": dev},
    )


class _P2Quantile:
    def __init__(self, q: float):
        self.q = float(q)
        self.initial: list[float] = []
        self.n = np.zeros(5, dtype=float)
        self.np = np.zeros(5, dtype=float)
        self.dn = np.array([0.0, self.q / 2.0, self.q, (1.0 + self.q) / 2.0, 1.0], dtype=float)
        self.h = np.zeros(5, dtype=float)
        self.count = 0

    def update(self, value: float) -> None:
        x = float(value)
        self.count += 1
        if len(self.initial) < 5:
            self.initial.append(x)
            if len(self.initial) == 5:
                self.initial.sort()
                self.h = np.array(self.initial, dtype=float)
                self.n = np.array([1, 2, 3, 4, 5], dtype=float)
                self.np = np.array([1, 1 + 2 * self.q, 1 + 4 * self.q, 3 + 2 * self.q, 5], dtype=float)
            return

        if x < self.h[0]:
            self.h[0] = x
            k = 0
        elif x >= self.h[4]:
            self.h[4] = x
            k = 3
        else:
            k = int(np.searchsorted(self.h, x) - 1)
            k = max(0, min(3, k))
        for i in range(k + 1, 5):
            self.n[i] += 1
        self.np += self.dn
        for i in (1, 2, 3):
            d = self.np[i] - self.n[i]
            if (d >= 1 and self.n[i + 1] - self.n[i] > 1) or (d <= -1 and self.n[i - 1] - self.n[i] < -1):
                dsign = float(np.sign(d))
                hp = self.h[i] + dsign / (self.n[i + 1] - self.n[i - 1]) * (
                    (self.n[i] - self.n[i - 1] + dsign) * (self.h[i + 1] - self.h[i]) / (self.n[i + 1] - self.n[i])
                    + (self.n[i + 1] - self.n[i] - dsign) * (self.h[i] - self.h[i - 1]) / (self.n[i] - self.n[i - 1])
                )
                if self.h[i - 1] < hp < self.h[i + 1]:
                    self.h[i] = hp
                else:
                    self.h[i] = self.h[i] + dsign * (self.h[i + int(dsign)] - self.h[i]) / (self.n[i + int(dsign)] - self.n[i])
                self.n[i] += dsign

    def value(self) -> float:
        if len(self.initial) < 5:
            if not self.initial:
                return 0.0
            return float(np.quantile(np.array(self.initial, dtype=float), self.q))
        return float(self.h[2])


def _handler_quantile_sketch_p2_streaming_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    num_cols = _variance_sorted_numeric(df, inferred, limit=min(12, int(config.get("max_cols", 12))))
    if not num_cols:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_columns")
    quantiles = [0.5, 0.9, 0.99]
    rows = []
    for col in num_cols:
        estimators = {q: _P2Quantile(q) for q in quantiles}
        vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals) < 10:
            continue
        for i, v in enumerate(vals):
            for est in estimators.values():
                est.update(float(v))
            if i % 200 == 0:
                _ensure_budget(timer)
        rows.append({"column": col, "q50": estimators[0.5].value(), "q90": estimators[0.9].value(), "q99": estimators[0.99].value()})
    if not rows:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_numeric_values")
    artifacts = [_artifact_json(ctx, plugin_id, "p2_quantiles.json", {"rows": rows}, "P2 streaming quantile summary")]
    return _finalize(
        plugin_id,
        ctx,
        df,
        sample_meta,
        "Computed P2 streaming quantile sketches.",
        [],
        artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "cols_processed": len(rows), "quantiles": quantiles},
    )


def _handler_robust_regression_huber_ransac_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    cols = _variance_sorted_numeric(df, inferred, limit=4)
    if len(cols) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "need_two_numeric_columns")
    ycol = cols[0]
    xcol = cols[1]
    frame = df[[xcol, ycol]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(frame) < 40:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    x = frame[xcol].to_numpy(dtype=float).reshape(-1, 1)
    y = frame[ycol].to_numpy(dtype=float)
    if HAS_SKLEARN and HuberRegressor is not None:
        hub = HuberRegressor(alpha=1e-4)
        hub.fit(x, y)
        slope_h = float(hub.coef_[0])
        pred_h = hub.predict(x)
    else:
        beta, _ = _fit_linear(x.ravel(), y, lam=1e-2)
        slope_h = float(beta[1]) if len(beta) > 1 else 0.0
        pred_h = beta[0] + slope_h * x.ravel()
    if HAS_SKLEARN and RANSACRegressor is not None:
        rs = RANSACRegressor(random_state=int(config.get("seed", 1337)))
        rs.fit(x, y)
        slope_r = float(rs.estimator_.coef_[0]) if hasattr(rs, "estimator_") else slope_h
        pred_r = rs.predict(x)
        inliers = rs.inlier_mask_
        outliers = int((~inliers).sum()) if inliers is not None else 0
    else:
        slope_r = slope_h
        pred_r = pred_h
        resid = np.abs(y - pred_h)
        outliers = int(np.sum(resid > (3.0 * _mad_scale(resid))))
    findings = []
    if outliers > 0:
        findings.append(
            _make_finding(
                plugin_id,
                "robust_outliers",
                "Robust regression outliers detected",
                "Huber/RANSAC comparison indicates influential outliers in linear relation.",
                "Outlier-sensitive slope instability can mislead driver attribution.",
                {"metrics": {"x_column": xcol, "y_column": ycol, "slope_huber": slope_h, "slope_ransac": slope_r, "outliers": outliers}},
                recommendation="Review outlier segments and prefer robust models for driver ranking.",
                severity="warn" if outliers < 10 else "critical",
                confidence=0.72,
            )
        )
    artifacts = [_artifact_json(ctx, plugin_id, "robust_regression.json", {"x_column": xcol, "y_column": ycol, "slope_huber": slope_h, "slope_ransac": slope_r, "outliers": outliers}, "Robust regression summary")]
    return _finalize(
        plugin_id,
        ctx,
        df,
        sample_meta,
        "Computed robust regression diagnostics.",
        findings,
        artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "slope_huber": slope_h, "slope_ransac": slope_r, "outliers": outliers},
    )


def _local_level_smoother(y: np.ndarray, q: float = 1e-2, r: float = 1.0) -> np.ndarray:
    n = len(y)
    x_f = np.zeros(n, dtype=float)
    P_f = np.zeros(n, dtype=float)
    x = float(y[0])
    P = 1.0
    for t in range(n):
        x_pred = x
        P_pred = P + q
        K = P_pred / (P_pred + r)
        x = x_pred + K * (y[t] - x_pred)
        P = (1.0 - K) * P_pred
        x_f[t] = x
        P_f[t] = P
    x_s = np.copy(x_f)
    P_s = np.copy(P_f)
    for t in range(n - 2, -1, -1):
        C = P_f[t] / max(1e-9, P_f[t] + q)
        x_s[t] = x_f[t] + C * (x_s[t + 1] - x_f[t])
        P_s[t] = P_f[t] + C * (P_s[t + 1] - (P_f[t] + q)) * C
    return x_s


def _handler_state_space_smoother_level_shift_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    time_col, ts = _time_series(df, inferred)
    if time_col is None or ts is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "time_column_not_found")
    value_col = _duration_column(df, inferred)
    if value_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_column")
    frame = pd.DataFrame({"ts": ts, "y": pd.to_numeric(df[value_col], errors="coerce")}).dropna().sort_values("ts")
    if len(frame) < 30:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_points")
    y = frame["y"].to_numpy(dtype=float)
    y, cap_meta = _cap_quadratic(pd.DataFrame({"y": y}), config, int(config.get("seed", 1337)), ctx)
    yv = y["y"].to_numpy(dtype=float)
    smooth = _local_level_smoother(yv)
    delta = np.diff(smooth)
    idx = int(np.argmax(np.abs(delta))) if len(delta) else 0
    max_shift = float(abs(delta[idx])) if len(delta) else 0.0
    findings = []
    if max_shift > (2.5 * _mad_scale(delta if len(delta) else np.array([0.0]))):
        findings.append(
            _make_finding(
                plugin_id,
                "state_space_shift",
                "Smoothed level shift detected",
                "RTS-style local-level smoothing indicates a major level change point.",
                "Smoothed level shifts are robust indicators of structural baseline changes.",
                {"metrics": {"index": idx, "max_level_shift": max_shift, "value_column": value_col}},
                recommendation="Investigate operational/control changes around the inferred level-shift index.",
                severity="warn" if max_shift < 3.0 else "critical",
                confidence=0.73,
            )
        )
    artifacts = [_artifact_json(ctx, plugin_id, "state_space_level_shift.json", {"value_column": value_col, "index": idx, "max_level_shift": max_shift, **cap_meta}, "State-space smoother level-shift summary")]
    return _finalize(
        plugin_id,
        ctx,
        df,
        sample_meta,
        "Computed state-space smoother level shift.",
        findings,
        artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "max_level_shift": max_shift, "index": idx},
    )


def _handler_aft_survival_lognormal_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    dcol = _duration_column(df, inferred)
    if dcol is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "duration_column_not_found")
    y = np.log1p(np.clip(pd.to_numeric(df[dcol], errors="coerce").to_numpy(dtype=float), 0.0, None))
    feature_cols = [c for c in _variance_sorted_numeric(df, inferred, limit=6) if c != dcol]
    if not feature_cols:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_covariates")
    frame = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    frame["y"] = y
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 30:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    X = frame[feature_cols].to_numpy(dtype=float)
    yy = frame["y"].to_numpy(dtype=float)
    beta, r2 = _fit_linear_r2(X, yy)
    coeffs = []
    for i, col in enumerate(feature_cols, start=1):
        coeffs.append({"feature": col, "coef": float(beta[i])})
    coeffs.sort(key=lambda r: (-abs(float(r["coef"])), str(r["feature"])))
    findings = []
    if coeffs:
        findings.append(
            _make_finding(
                plugin_id,
                "aft_coeffs",
                "AFT lognormal proxy drivers identified",
                "Log-duration regression surfaced top covariates associated with duration shifts.",
                "AFT-style coefficients approximate multiplicative effects on completion time.",
                {"metrics": {"duration_col": dcol, "r2": r2, "top_coefficients": coeffs[:5]}},
                recommendation="Prioritize improvement levers on largest-magnitude duration drivers.",
                severity="info",
                confidence=0.64,
            )
        )
    artifacts = [_artifact_json(ctx, plugin_id, "aft_lognormal.json", {"duration_col": dcol, "r2": r2, "coefficients": coeffs[:50]}, "AFT lognormal proxy summary")]
    return _finalize(
        plugin_id,
        ctx,
        df,
        sample_meta,
        "Computed AFT lognormal proxy model.",
        findings,
        artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "duration_col": dcol, "n_covariates": len(feature_cols), "r2": r2},
    )


def _handler_competing_risks_cif_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    et_col = _event_type_column(df, inferred)
    dcol = _duration_column(df, inferred)
    if et_col is None or dcol is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "event_type_or_duration_missing")
    frame = pd.DataFrame({"event": df[et_col].astype(str), "duration": pd.to_numeric(df[dcol], errors="coerce")}).dropna()
    if len(frame) < 40:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    frame = frame.sort_values("duration")
    t50 = float(np.quantile(frame["duration"], 0.5))
    t90 = float(np.quantile(frame["duration"], 0.9))
    events = frame["event"].value_counts().index.tolist()[:10]
    rows = []
    for e in events:
        d = frame["duration"].to_numpy(dtype=float)
        is_e = (frame["event"] == e).to_numpy(dtype=float)
        cif50 = float(np.mean((d <= t50) * is_e))
        cif90 = float(np.mean((d <= t90) * is_e))
        rows.append({"event_type": e, "cif_t50": cif50, "cif_t90": cif90})
        _ensure_budget(timer)
    rows.sort(key=lambda r: (-float(r["cif_t90"]), str(r["event_type"])))
    findings = []
    if rows:
        findings.append(
            _make_finding(
                plugin_id,
                f"cif:{rows[0]['event_type']}",
                "Competing risk incidence profile computed",
                "Cumulative incidence differs by event type over duration horizon.",
                "Event-type incidence concentration identifies dominant failure/exit modes.",
                {"metrics": {"duration_col": dcol, "top_event": rows[0]}},
                recommendation="Target dominant early-incidence event type for first mitigation wave.",
                severity="info",
                confidence=0.62,
            )
        )
    artifacts = [_artifact_json(ctx, plugin_id, "competing_risks.json", {"duration_col": dcol, "event_type_col": et_col, "t50": t50, "t90": t90, "rows": rows}, "Competing-risks CIF summary")]
    return _finalize(
        plugin_id,
        ctx,
        df,
        sample_meta,
        "Computed competing-risks cumulative incidence profile.",
        findings,
        artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "event_types": len(rows), "cif_at_t": {"t50": t50, "t90": t90}},
    )


def _handler_haar_wavelet_transient_detector_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    col = _duration_column(df, inferred)
    if col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_column")
    y = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
    if len(y) < 40:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_points")
    max_scale_pow = int(math.floor(math.log2(max(2, len(y) // 8))))
    scales = [2**k for k in range(0, max(1, max_scale_pow + 1))]
    rows = []
    best = {"scale": 1, "index": 0, "z": 0.0}
    for s in scales:
        if len(y) < 2 * s:
            continue
        coeffs = []
        idxs = []
        for i in range(0, len(y) - 2 * s + 1):
            c = float(np.mean(y[i + s : i + 2 * s]) - np.mean(y[i : i + s]))
            coeffs.append(c)
            idxs.append(i + s)
        arr = np.array(coeffs, dtype=float)
        scale = _mad_scale(arr)
        z = np.abs(arr) / scale
        zmax_idx = int(np.argmax(z))
        zmax = float(z[zmax_idx])
        rows.append({"scale": int(s), "index": int(idxs[zmax_idx]), "max_coeff_z": zmax})
        if zmax > float(best["z"]):
            best = {"scale": int(s), "index": int(idxs[zmax_idx]), "z": zmax}
        _ensure_budget(timer)
    if not rows:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_valid_scales")
    findings = []
    if float(best["z"]) > 4.0:
        findings.append(
            _make_finding(
                plugin_id,
                "haar_transient",
                "Wavelet transient detected",
                "Haar multiscale coefficients indicate abrupt local change.",
                "High normalized Haar coefficient marks likely burst/step transient.",
                {"metrics": {"column": col, "scale": best["scale"], "index": best["index"], "max_coeff_z": best["z"]}},
                recommendation="Inspect process events around transient index and evaluate step-change root causes.",
                severity="warn" if float(best["z"]) < 6.0 else "critical",
                confidence=min(0.93, 0.58 + min(0.3, float(best["z"]) / 10.0)),
            )
        )
    artifacts = [_artifact_json(ctx, plugin_id, "haar_transients.json", {"column": col, "rows": rows}, "Haar transient detection summary")]
    return _finalize(
        plugin_id,
        ctx,
        df,
        sample_meta,
        "Computed Haar wavelet transient metrics.",
        findings,
        artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "max_coeff_z": float(best["z"]), "scale": int(best["scale"])},
    )


def _handler_hurst_exponent_long_memory_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    col = _duration_column(df, inferred)
    if col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_column")
    y = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
    if len(y) < 80:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_points")
    y, cap_meta = _cap_quadratic(pd.DataFrame({"y": y}), config, int(config.get("seed", 1337)), ctx)
    H, r2 = _hurst_rs(y["y"].to_numpy(dtype=float))
    findings = []
    if H > 0.7 or H < 0.3:
        findings.append(
            _make_finding(
                plugin_id,
                "hurst_memory",
                "Long-memory persistence anomaly",
                "Estimated Hurst exponent deviates from near-random benchmark.",
                "Extreme H suggests persistent/anti-persistent dynamics affecting predictability.",
                {"metrics": {"column": col, "hurst_H": H, "fit_r2": r2}},
                recommendation="Adjust forecast assumptions for long-memory dynamics and retune control horizons.",
                severity="warn" if 0.2 < H < 0.8 else "critical",
                confidence=min(0.9, 0.55 + min(0.25, abs(H - 0.5))),
            )
        )
    artifacts = [_artifact_json(ctx, plugin_id, "hurst.json", {"column": col, "hurst_H": H, "fit_r2": r2, **cap_meta}, "Hurst exponent summary")]
    return _finalize(
        plugin_id,
        ctx,
        df,
        sample_meta,
        "Computed Hurst exponent long-memory diagnostics.",
        findings,
        artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "hurst_H": H, "fit_r2": r2},
    )


def _handler_permutation_entropy_drift_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    split = _split_pre_post(df, inferred)
    col = _duration_column(df, inferred)
    if col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_column")
    pre = pd.to_numeric(split.pre[col], errors="coerce").dropna().to_numpy(dtype=float)
    post = pd.to_numeric(split.post[col], errors="coerce").dropna().to_numpy(dtype=float)
    if len(pre) < 30 or len(post) < 30:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_points")
    h_pre = _permutation_entropy(pre, m=3, tau=1)
    h_post = _permutation_entropy(post, m=3, tau=1)
    delta = float(h_post - h_pre)
    findings = []
    if abs(delta) > 0.1:
        findings.append(
            _make_finding(
                plugin_id,
                "perm_entropy_drift",
                "Permutation entropy drift detected",
                "Ordinal-pattern complexity changed between pre and post periods.",
                "Complexity drift can indicate regime changes in temporal process generation.",
                {"metrics": {"column": col, "H_pre": h_pre, "H_post": h_post, "delta": delta}},
                recommendation="Investigate process control changes around entropy drift onset.",
                severity="warn" if abs(delta) < 0.2 else "critical",
                confidence=0.7,
            )
        )
    artifacts = [_artifact_json(ctx, plugin_id, "permutation_entropy.json", {"column": col, "H_pre": h_pre, "H_post": h_post, "delta": delta}, "Permutation entropy drift summary")]
    return _finalize(
        plugin_id,
        ctx,
        df,
        sample_meta,
        "Computed permutation entropy drift.",
        findings,
        artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "H_pre": h_pre, "H_post": h_post, "delta": delta},
    )


def _handler_capacity_frontier_envelope_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    num_cols = _numeric_columns(df, inferred, max_cols=30)
    if len(num_cols) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "need_two_numeric_columns")
    out_hints = ("completed", "processed", "throughput", "count", "success")
    in_hints = ("cpu", "duration", "time", "server", "resource", "cost")
    out_col = next((c for c in num_cols if any(h in c.lower() for h in out_hints)), num_cols[0])
    in_col = next((c for c in num_cols if c != out_col and any(h in c.lower() for h in in_hints)), num_cols[1] if len(num_cols) > 1 else num_cols[0])
    frame = pd.DataFrame({"x": pd.to_numeric(df[in_col], errors="coerce"), "y": pd.to_numeric(df[out_col], errors="coerce")}).dropna()
    if len(frame) < 30:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    frame = frame.sort_values("x")
    x = frame["x"].to_numpy(dtype=float)
    y = frame["y"].to_numpy(dtype=float)
    best_y = np.maximum.accumulate(y)
    eff = y / np.maximum(best_y, 1e-9)
    min_eff = float(np.min(eff))
    frontier = [{"x": float(xi), "y_frontier": float(yi)} for xi, yi in zip(x.tolist(), best_y.tolist())]
    findings = []
    if min_eff < 0.8:
        findings.append(
            _make_finding(
                plugin_id,
                "capacity_frontier_gap",
                "Capacity frontier inefficiency detected",
                "Observed points fall materially below the empirical output frontier.",
                "Low frontier efficiency indicates untapped capacity at given resource/time input.",
                {"metrics": {"input_col": in_col, "output_col": out_col, "min_efficiency": min_eff}},
                recommendation="Target low-efficiency segments for process redesign or load-balancing improvements.",
                severity="warn" if min_eff >= 0.6 else "critical",
                confidence=0.71,
            )
        )
    artifacts = [_artifact_json(ctx, plugin_id, "capacity_frontier.json", {"input_col": in_col, "output_col": out_col, "min_efficiency": min_eff, "frontier_points": frontier[:200]}, "Capacity frontier envelope summary")]
    return _finalize(
        plugin_id,
        ctx,
        df,
        sample_meta,
        "Computed capacity frontier envelope.",
        findings,
        artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "frontier_points": len(frontier), "min_efficiency": min_eff},
    )


def _handler_graph_assortativity_shift_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    split = _split_pre_post(df, inferred)
    pre_edges, src_col, dst_col = _build_edges(split.pre, inferred)
    post_edges, _, _ = _build_edges(split.post, inferred)
    if pre_edges.empty or post_edges.empty:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "graph_edges_not_found")
    assort_pre = _degree_assortativity(pre_edges)
    assort_post = _degree_assortativity(post_edges)
    delta = float(assort_post - assort_pre)
    findings = []
    if abs(delta) > 0.15:
        findings.append(
            _make_finding(
                plugin_id,
                "assortativity_shift",
                "Graph assortativity shift detected",
                "Degree-assortativity changed across pre/post graph windows.",
                "Mixing-pattern shifts can indicate topology changes in dependency routing.",
                {"metrics": {"assort_pre": assort_pre, "assort_post": assort_post, "delta": delta}},
                recommendation="Review routing/dependency changes that altered graph mixing behavior.",
                severity="warn" if abs(delta) < 0.25 else "critical",
                confidence=0.69,
            )
        )
    artifacts = [_artifact_json(ctx, plugin_id, "graph_assortativity.json", {"src_col": src_col, "dst_col": dst_col, "assort_pre": assort_pre, "assort_post": assort_post, "delta": delta}, "Graph assortativity shift summary")]
    return _finalize(
        plugin_id,
        ctx,
        df,
        sample_meta,
        "Computed graph assortativity shift.",
        findings,
        artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "assort_pre": assort_pre, "assort_post": assort_post, "delta": delta},
    )


def _handler_graph_pagerank_hotspots_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    edges, src_col, dst_col = _build_edges(df, inferred)
    if edges.empty:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "graph_edges_not_found")
    pr = _pagerank_power_iteration(edges)
    if not pr:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "pagerank_empty")
    ranked = sorted(pr.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))
    top = ranked[:10]
    top_share = float(sum(v for _, v in top))
    findings = []
    if top_share > 0.4:
        findings.append(
            _make_finding(
                plugin_id,
                "pagerank_concentration",
                "PageRank hotspot concentration",
                "Top nodes hold a large share of centrality mass.",
                "Centrality concentration indicates potential bottlenecks/single points of failure.",
                {"metrics": {"top10_share": top_share, "top_nodes": [{"node_hash": _hash_node(n), "pagerank": float(v)} for n, v in top]}},
                recommendation="Add redundancy and load distribution around top-centrality nodes.",
                severity="warn" if top_share < 0.6 else "critical",
                confidence=min(0.9, 0.58 + min(0.25, top_share)),
            )
        )
    artifacts = [_artifact_json(ctx, plugin_id, "pagerank_hotspots.json", {"src_col": src_col, "dst_col": dst_col, "nodes": len(pr), "edges": len(edges), "top10_share": top_share, "top_nodes": [{"node_hash": _hash_node(n), "pagerank": float(v)} for n, v in top]}, "PageRank hotspot summary")]
    return _finalize(
        plugin_id,
        ctx,
        df,
        sample_meta,
        "Computed graph PageRank hotspots.",
        findings,
        artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "nodes": len(pr), "edges": len(edges), "top10_share": top_share},
    )


def _handler_higuchi_fractal_dimension_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    col = _duration_column(df, inferred)
    if col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_column")
    y = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
    if len(y) < 60:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_points")
    k_max = int(config.get("plugin", {}).get("k_max", 10))
    fd = _higuchi_fd(y, k_max=k_max)
    findings = []
    if fd > 1.4 or fd < 1.05:
        findings.append(
            _make_finding(
                plugin_id,
                "higuchi_fd",
                "Fractal complexity deviation",
                "Higuchi fractal dimension indicates complexity outside expected band.",
                "Complexity shift can reflect regime changes in process irregularity.",
                {"metrics": {"column": col, "fd": fd, "k_max": k_max}},
                recommendation="Track complexity trend over time and correlate with release/process events.",
                severity="warn" if 1.0 <= fd <= 1.6 else "critical",
                confidence=0.66,
            )
        )
    artifacts = [_artifact_json(ctx, plugin_id, "higuchi_fd.json", {"column": col, "fd": fd, "k_max": k_max}, "Higuchi fractal dimension summary")]
    return _finalize(
        plugin_id,
        ctx,
        df,
        sample_meta,
        "Computed Higuchi fractal dimension.",
        findings,
        artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "fd": fd, "k_max": k_max},
    )


def _handler_marked_point_process_intensity_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    time_col, ts = _time_series(df, inferred)
    if time_col is None or ts is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "time_column_not_found")
    mark_col = _duration_column(df, inferred)
    if mark_col is None:
        cats = _categorical_columns(df, inferred, max_cols=5)
        mark_col = cats[0] if cats else None
    if mark_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "mark_column_not_found")
    frame = pd.DataFrame({"ts": ts, "mark": df[mark_col]}).dropna().sort_values("ts")
    if len(frame) < 30:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    freq = "h"
    deltas = frame["ts"].diff().dt.total_seconds().dropna()
    if not deltas.empty and float(deltas.median()) > 3600:
        freq = "d"
    bucket = frame.set_index("ts").groupby(pd.Grouper(freq=freq))
    counts = bucket.size().astype(float)
    if counts.empty:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "empty_buckets")
    lam = counts.to_numpy(dtype=float)
    lam_mean = float(np.mean(lam))
    lam_p95 = float(np.quantile(lam, 0.95))
    shift = float(np.max(np.abs(np.diff(pd.Series(lam).rolling(window=max(2, len(lam) // 5), min_periods=1).mean().to_numpy(dtype=float))))) if len(lam) > 2 else 0.0
    if pd.api.types.is_numeric_dtype(frame["mark"]):
        mark_summary = {
            str(k): float(v)
            for k, v in bucket["mark"].mean().fillna(0.0).to_dict().items()
        }
    else:
        mark_summary = {}
        for key, grp in bucket:
            probs = grp["mark"].astype(str).value_counts(normalize=True)
            ent = -float(np.sum(probs.to_numpy(dtype=float) * np.log(np.maximum(1e-12, probs.to_numpy(dtype=float))))) if not probs.empty else 0.0
            mark_summary[str(key)] = ent
    findings = []
    if shift > max(1.0, 0.5 * lam_mean):
        findings.append(
            _make_finding(
                plugin_id,
                "intensity_shift",
                "Marked point-process intensity shift",
                "Bucketed event intensity exhibits a strong shift.",
                "Intensity shifts indicate changed arrival process or backlog clearing dynamics.",
                {"metrics": {"lambda_mean": lam_mean, "lambda_p95": lam_p95, "max_shift": shift, "freq": freq}},
                recommendation="Investigate workload routing and release scheduling around shift window.",
                severity="warn" if shift < (1.5 * lam_mean) else "critical",
                confidence=0.7,
            )
        )
    artifacts = [_artifact_json(ctx, plugin_id, "marked_intensity.json", {"time_col": time_col, "mark_col": mark_col, "freq": freq, "lambda_mean": lam_mean, "lambda_p95": lam_p95, "max_shift": shift, "mark_summary": mark_summary}, "Marked point-process intensity summary")]
    return _finalize(
        plugin_id,
        ctx,
        df,
        sample_meta,
        "Computed marked point-process intensity metrics.",
        findings,
        artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "lambda_mean": lam_mean, "lambda_p95": lam_p95, "max_shift": shift},
    )


def _handler_spectral_radius_stability_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    split = _split_pre_post(df, inferred)
    cols = _variance_sorted_numeric(df, inferred, limit=min(12, int(config.get("max_cols", 12))))
    if len(cols) >= 2:
        pre = split.pre[cols].apply(pd.to_numeric, errors="coerce").dropna()
        post = split.post[cols].apply(pd.to_numeric, errors="coerce").dropna()
        if len(pre) >= 20 and len(post) >= 20:
            corr_pre = np.nan_to_num(np.corrcoef(pre.to_numpy(dtype=float), rowvar=False))
            corr_post = np.nan_to_num(np.corrcoef(post.to_numpy(dtype=float), rowvar=False))
            lam_pre = float(np.max(np.abs(np.linalg.eigvals(corr_pre))))
            lam_post = float(np.max(np.abs(np.linalg.eigvals(corr_post))))
            mode = "numeric_corr"
        else:
            lam_pre = lam_post = 0.0
            mode = "numeric_insufficient"
    else:
        lam_pre = lam_post = 0.0
        mode = "numeric_missing"
    if mode != "numeric_corr":
        edges, _, _ = _build_edges(df, inferred)
        if edges.empty:
            return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_or_graph_for_spectral_radius")
        nodes = sorted(set(edges["src"].tolist()) | set(edges["dst"].tolist()))
        idx = {n: i for i, n in enumerate(nodes)}
        A = np.zeros((len(nodes), len(nodes)), dtype=float)
        for row in edges.itertuples(index=False):
            A[idx[row.src], idx[row.dst]] += 1.0
        lam_pre = float(np.max(np.abs(np.linalg.eigvals(A))))
        lam_post = lam_pre
        mode = "graph_adj"
    delta = float(lam_post - lam_pre)
    findings = []
    if abs(delta) > 0.2:
        findings.append(
            _make_finding(
                plugin_id,
                "spectral_radius_shift",
                "Spectral radius stability shift",
                "Leading eigenvalue shifted between pre and post structures.",
                "Spectral radius shift indicates changed systemic coupling/stability margin.",
                {"metrics": {"mode": mode, "lambda_pre": lam_pre, "lambda_post": lam_post, "delta": delta}},
                recommendation="Investigate coupling pathways and enforce dampening controls on unstable segments.",
                severity="warn" if abs(delta) < 0.5 else "critical",
                confidence=0.68,
            )
        )
    artifacts = [_artifact_json(ctx, plugin_id, "spectral_radius.json", {"mode": mode, "lambda_pre": lam_pre, "lambda_post": lam_post, "delta": delta}, "Spectral radius stability summary")]
    return _finalize(
        plugin_id,
        ctx,
        df,
        sample_meta,
        "Computed spectral radius stability metrics.",
        findings,
        artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "lambda_pre": lam_pre, "lambda_post": lam_post, "delta": delta},
    )


def _handler_bootstrap_ci_effect_sizes_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    split = _split_pre_post(df, inferred)
    cols = _variance_sorted_numeric(df, inferred, limit=min(8, int(config.get("max_cols", 8))))
    if not cols:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_columns")
    B = int(config.get("plugin", {}).get("max_resamples", 200))
    B = max(20, min(B, 500))
    rng = np.random.default_rng(int(config.get("seed", 1337)))
    rows = []
    for col in cols:
        left = pd.to_numeric(split.pre[col], errors="coerce").dropna().to_numpy(dtype=float)
        right = pd.to_numeric(split.post[col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(left) < 20 or len(right) < 20:
            continue
        obs, lo, hi = _bootstrap_effect_ci(left, right, B, rng, timer)
        rows.append({"column": col, "effect": obs, "ci_low": lo, "ci_high": hi})
    if not rows:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_pre_post_numeric")
    rows.sort(key=lambda r: (-abs(float(r["effect"])), str(r["column"])))
    findings = []
    for r in rows[: int(config.get("max_findings", 30))]:
        if float(r["ci_low"]) * float(r["ci_high"]) > 0:
            findings.append(
                _make_finding(
                    plugin_id,
                    f"bootstrap:{r['column']}",
                    "Bootstrap effect CI excludes zero",
                    "Bootstrap confidence interval indicates non-zero effect size.",
                    "Non-zero CI improves confidence that observed shift is not sampling noise.",
                    {"metrics": r},
                    recommendation="Prioritize interventions on columns with largest non-zero effect CI magnitude.",
                    severity="warn" if abs(float(r["effect"])) < 1.0 else "critical",
                    confidence=0.75,
                )
            )
    artifacts = [_artifact_json(ctx, plugin_id, "bootstrap_effect_ci.json", {"B": B, "rows": rows[:50]}, "Bootstrap CI effect-size summary")]
    return _finalize(
        plugin_id,
        ctx,
        df,
        sample_meta,
        "Computed bootstrap confidence intervals for effect sizes.",
        findings,
        artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "B": B, "effects_with_ci": len(rows)},
    )


def _handler_energy_distance_two_sample_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    split = _split_pre_post(df, inferred)
    col = _duration_column(df, inferred)
    if col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_column")
    left = pd.to_numeric(split.pre[col], errors="coerce").dropna().to_numpy(dtype=float)
    right = pd.to_numeric(split.post[col], errors="coerce").dropna().to_numpy(dtype=float)
    if len(left) < 20 or len(right) < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_pre_post")
    merged = pd.DataFrame({"x": np.concatenate([left, right])})
    merged, cap_meta = _cap_quadratic(merged, config, int(config.get("seed", 1337)), ctx)
    half = len(merged) // 2
    a = merged["x"].to_numpy(dtype=float)[:half]
    b = merged["x"].to_numpy(dtype=float)[half:]
    if len(a) < 10 or len(b) < 10:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_points_after_cap")
    B = int(config.get("plugin", {}).get("max_resamples", 200))
    rng = np.random.default_rng(int(config.get("seed", 1337)))
    stat, p = _perm_pvalue(a, b, _energy_stat, B, rng, timer)
    findings = []
    if p <= 0.05:
        findings.append(
            _make_finding(
                plugin_id,
                "energy_distance_shift",
                "Energy-distance two-sample shift",
                "Energy distance indicates significant distribution difference across windows.",
                "Energy distance captures broad shape and tail changes, not only mean shifts.",
                {"metrics": {"column": col, "energy_stat": stat, "p_value": p, "B": B}},
                recommendation="Rebaseline distribution-aware thresholds for this metric.",
                severity="warn" if p > 0.01 else "critical",
                confidence=0.8,
            )
        )
    artifacts = [_artifact_json(ctx, plugin_id, "energy_distance.json", {"column": col, "energy_stat": stat, "p_value": p, "B": B, **cap_meta}, "Energy-distance two-sample summary")]
    return _finalize(
        plugin_id,
        ctx,
        df,
        sample_meta,
        "Computed energy-distance two-sample test.",
        findings,
        artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "energy_stat": stat, "p_value": p, "B": B},
    )


def _handler_randomization_test_median_shift_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    split = _split_pre_post(df, inferred)
    col = _duration_column(df, inferred)
    if col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_column")
    left = pd.to_numeric(split.pre[col], errors="coerce").dropna().to_numpy(dtype=float)
    right = pd.to_numeric(split.post[col], errors="coerce").dropna().to_numpy(dtype=float)
    if len(left) < 20 or len(right) < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_pre_post")
    B = int(config.get("plugin", {}).get("max_resamples", 200))
    rng = np.random.default_rng(int(config.get("seed", 1337)))

    def _score(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.median(b) - np.median(a))

    delta, p = _perm_pvalue(left, right, _score, B, rng, timer)
    findings = []
    if p <= 0.05:
        findings.append(
            _make_finding(
                plugin_id,
                "median_shift",
                "Randomization median shift is significant",
                "Permutation test indicates a significant median shift.",
                "Median-shift tests remain robust under heavy-tailed noise/outliers.",
                {"metrics": {"column": col, "delta_median": delta, "p_value": p, "B": B}},
                recommendation="Quantify operational impact of median shift and update runbook thresholds.",
                severity="warn" if p > 0.01 else "critical",
                confidence=0.79,
            )
        )
    artifacts = [_artifact_json(ctx, plugin_id, "median_randomization.json", {"column": col, "delta_median": delta, "p_value": p, "B": B}, "Randomization median-shift summary")]
    return _finalize(
        plugin_id,
        ctx,
        df,
        sample_meta,
        "Computed randomization test for median shift.",
        findings,
        artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "delta_median": delta, "p_value": p, "B": B},
    )


def _handler_distance_covariance_dependence_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    cols = _variance_sorted_numeric(df, inferred, limit=4)
    if len(cols) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "need_two_numeric_columns")
    x = pd.to_numeric(df[cols[0]], errors="coerce")
    y = pd.to_numeric(df[cols[1]], errors="coerce")
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 30:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    frame, cap_meta = _cap_quadratic(frame, config, int(config.get("seed", 1337)), ctx)
    xv = frame["x"].to_numpy(dtype=float)
    yv = frame["y"].to_numpy(dtype=float)
    dcov, dcor = _distance_covariance(xv, yv)
    B = int(config.get("plugin", {}).get("max_resamples", 200))
    rng = np.random.default_rng(int(config.get("seed", 1337)))
    ge = 1
    for _ in range(B):
        yp = rng.permutation(yv)
        _, dcor_p = _distance_covariance(xv, yp)
        if dcor_p >= dcor:
            ge += 1
        _ensure_budget(timer)
    p = ge / (B + 1)
    findings = []
    if p <= 0.05 and dcor > 0.1:
        findings.append(
            _make_finding(
                plugin_id,
                "distance_cov_dependence",
                "Distance-covariance dependence detected",
                "Nonlinear dependence between numeric pair is statistically significant.",
                "Distance correlation captures dependence missed by linear correlation.",
                {"metrics": {"x_col": cols[0], "y_col": cols[1], "dcov": dcov, "dcor": dcor, "p_value": p, "B": B}},
                recommendation="Model these variables jointly when building controls/forecasts.",
                severity="warn" if dcor < 0.25 else "critical",
                confidence=0.78,
            )
        )
    artifacts = [_artifact_json(ctx, plugin_id, "distance_covariance.json", {"x_col": cols[0], "y_col": cols[1], "dcov": dcov, "dcor": dcor, "p_value": p, "B": B, **cap_meta}, "Distance covariance dependence summary")]
    return _finalize(
        plugin_id,
        ctx,
        df,
        sample_meta,
        "Computed distance covariance dependence test.",
        findings,
        artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "dcov": dcov, "dcor": dcor, "p_value": p},
    )


def _triad_profile(edges: pd.DataFrame, max_nodes: int = 50) -> dict[str, int]:
    nodes = sorted(set(edges["src"].tolist()) | set(edges["dst"].tolist()))
    if len(nodes) > max_nodes:
        nodes = nodes[:max_nodes]
    idx = {n: i for i, n in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)), dtype=np.uint8)
    for row in edges.itertuples(index=False):
        if row.src in idx and row.dst in idx:
            A[idx[row.src], idx[row.dst]] = 1
    out = Counter()
    for i, j, k in combinations(range(len(nodes)), 3):
        e = int(A[i, j] + A[j, i] + A[i, k] + A[k, i] + A[j, k] + A[k, j])
        if e == 0:
            out["empty"] += 1
        elif e == 1:
            out["single"] += 1
        elif e == 2:
            out["double"] += 1
        elif e >= 5:
            out["dense"] += 1
        else:
            cyc = int((A[i, j] and A[j, k] and A[k, i]) or (A[j, i] and A[k, j] and A[i, k]))
            if cyc:
                out["cycle3"] += 1
            else:
                out["other"] += 1
    return dict(out)


def _handler_graph_motif_triads_shift_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    split = _split_pre_post(df, inferred)
    pre_edges, src_col, dst_col = _build_edges(split.pre, inferred)
    post_edges, _, _ = _build_edges(split.post, inferred)
    if pre_edges.empty or post_edges.empty:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "graph_edges_not_found")
    pre_prof = _triad_profile(pre_edges, max_nodes=50)
    post_prof = _triad_profile(post_edges, max_nodes=50)
    keys = sorted(set(pre_prof.keys()) | set(post_prof.keys()))
    pre_vec = np.array([float(pre_prof.get(k, 0)) for k in keys], dtype=float)
    post_vec = np.array([float(post_prof.get(k, 0)) for k in keys], dtype=float)
    pre_p = pre_vec / max(1.0, float(np.sum(pre_vec)))
    post_p = post_vec / max(1.0, float(np.sum(post_vec)))
    l1 = float(np.sum(np.abs(pre_p - post_p)))
    findings = []
    if l1 > 0.2:
        findings.append(
            _make_finding(
                plugin_id,
                "triad_shift",
                "Triad motif distribution drift",
                "Directed triad motif profile changed between pre and post graphs.",
                "Motif distribution shifts indicate altered local interaction structures.",
                {"metrics": {"l1_drift": l1, "keys": keys}},
                recommendation="Inspect topology changes around triad classes with largest share deltas.",
                severity="warn" if l1 < 0.35 else "critical",
                confidence=0.7,
            )
        )
    artifacts = [_artifact_json(ctx, plugin_id, "triad_motifs.json", {"src_col": src_col, "dst_col": dst_col, "keys": keys, "pre_profile": pre_prof, "post_profile": post_prof, "l1_drift": l1}, "Graph triad motif shift summary")]
    return _finalize(
        plugin_id,
        ctx,
        df,
        sample_meta,
        "Computed graph triad motif drift.",
        findings,
        artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "triads_total": int(np.sum(pre_vec) + np.sum(post_vec)), "l1_drift": l1},
    )


def _handler_multiscale_entropy_mse_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    split = _split_pre_post(df, inferred)
    col = _duration_column(df, inferred)
    if col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_column")
    pre = pd.to_numeric(split.pre[col], errors="coerce").dropna().to_numpy(dtype=float)
    post = pd.to_numeric(split.post[col], errors="coerce").dropna().to_numpy(dtype=float)
    if len(pre) < 80 or len(post) < 80:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_points")
    scales = int(config.get("plugin", {}).get("scales", 8))
    scales = max(2, min(scales, 10))
    rows = []
    mse_pre = []
    mse_post = []
    for s in range(1, scales + 1):
        cg_pre = _coarse_grain(pre, s)
        cg_post = _coarse_grain(post, s)
        if len(cg_pre) < 20 or len(cg_post) < 20:
            continue
        se_pre = _sample_entropy(cg_pre, m=2)
        se_post = _sample_entropy(cg_post, m=2)
        rows.append({"scale": s, "mse_pre": se_pre, "mse_post": se_post, "delta": se_post - se_pre})
        mse_pre.append(se_pre)
        mse_post.append(se_post)
        _ensure_budget(timer)
    if not rows:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_points_after_coarse_grain")
    avg_pre = float(np.mean(np.array(mse_pre, dtype=float)))
    avg_post = float(np.mean(np.array(mse_post, dtype=float)))
    delta = float(avg_post - avg_pre)
    findings = []
    if abs(delta) > 0.1:
        findings.append(
            _make_finding(
                plugin_id,
                "mse_delta",
                "Multiscale entropy drift detected",
                "Average multiscale entropy changed across pre/post periods.",
                "Complexity changes across scales indicate altered system dynamics.",
                {"metrics": {"column": col, "mse_pre": avg_pre, "mse_post": avg_post, "delta": delta, "scales": scales}},
                recommendation="Investigate controls affecting variability across multiple timescales.",
                severity="warn" if abs(delta) < 0.2 else "critical",
                confidence=0.69,
            )
        )
    artifacts = [_artifact_json(ctx, plugin_id, "mse.json", {"column": col, "scales": scales, "rows": rows, "mse_pre": avg_pre, "mse_post": avg_post, "delta": delta}, "Multiscale entropy summary")]
    return _finalize(
        plugin_id,
        ctx,
        df,
        sample_meta,
        "Computed multiscale entropy profile.",
        findings,
        artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "scales": scales, "mse_pre": avg_pre, "mse_post": avg_post, "delta": delta},
    )


def _handler_sample_entropy_irregularity_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    split = _split_pre_post(df, inferred)
    col = _duration_column(df, inferred)
    if col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_column")
    pre = pd.to_numeric(split.pre[col], errors="coerce").dropna().to_numpy(dtype=float)
    post = pd.to_numeric(split.post[col], errors="coerce").dropna().to_numpy(dtype=float)
    if len(pre) < 60 or len(post) < 60:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_points")
    cap_df = pd.DataFrame({"v": np.concatenate([pre, post])})
    cap_df, cap_meta = _cap_quadratic(cap_df, config, int(config.get("seed", 1337)), ctx)
    half = len(cap_df) // 2
    pre2 = cap_df["v"].to_numpy(dtype=float)[:half]
    post2 = cap_df["v"].to_numpy(dtype=float)[half:]
    if len(pre2) < 20 or len(post2) < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_points_after_cap")
    s_pre = _sample_entropy(pre2, m=2)
    s_post = _sample_entropy(post2, m=2)
    delta = float(s_post - s_pre)
    findings = []
    if abs(delta) > 0.1:
        findings.append(
            _make_finding(
                plugin_id,
                "sampen_delta",
                "Sample-entropy irregularity shift",
                "Sample entropy changed materially between pre/post windows.",
                "Irregularity drift can signal stability loss or process mode transition.",
                {"metrics": {"column": col, "sampen_pre": s_pre, "sampen_post": s_post, "delta": delta}},
                recommendation="Investigate noise/instability sources and tighten process controls.",
                severity="warn" if abs(delta) < 0.25 else "critical",
                confidence=0.7,
            )
        )
    artifacts = [_artifact_json(ctx, plugin_id, "sample_entropy.json", {"column": col, "sampen_pre": s_pre, "sampen_post": s_post, "delta": delta, **cap_meta}, "Sample entropy irregularity summary")]
    return _finalize(
        plugin_id,
        ctx,
        df,
        sample_meta,
        "Computed sample entropy irregularity metrics.",
        findings,
        artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "sampen_pre": s_pre, "sampen_post": s_post, "delta": delta},
    )


HANDLERS: dict[str, Callable[..., PluginResult]] = {
    "analysis_beta_binomial_overdispersion_v1": _handler_beta_binomial_overdispersion_v1,
    "analysis_circular_time_of_day_drift_v1": _handler_circular_time_of_day_drift_v1,
    "analysis_mann_kendall_trend_test_v1": _handler_mann_kendall_trend_test_v1,
    "analysis_quantile_mapping_drift_qq_v1": _handler_quantile_mapping_drift_qq_v1,
    "analysis_constraints_violation_detector_v1": _handler_constraints_violation_detector_v1,
    "analysis_negative_binomial_overdispersion_v1": _handler_negative_binomial_overdispersion_v1,
    "analysis_partial_correlation_network_shift_v1": _handler_partial_correlation_network_shift_v1,
    "analysis_piecewise_linear_trend_changepoints_v1": _handler_piecewise_linear_trend_changepoints_v1,
    "analysis_poisson_regression_rate_drivers_v1": _handler_poisson_regression_rate_drivers_v1,
    "analysis_quantile_sketch_p2_streaming_v1": _handler_quantile_sketch_p2_streaming_v1,
    "analysis_robust_regression_huber_ransac_v1": _handler_robust_regression_huber_ransac_v1,
    "analysis_state_space_smoother_level_shift_v1": _handler_state_space_smoother_level_shift_v1,
    "analysis_aft_survival_lognormal_v1": _handler_aft_survival_lognormal_v1,
    "analysis_competing_risks_cif_v1": _handler_competing_risks_cif_v1,
    "analysis_haar_wavelet_transient_detector_v1": _handler_haar_wavelet_transient_detector_v1,
    "analysis_hurst_exponent_long_memory_v1": _handler_hurst_exponent_long_memory_v1,
    "analysis_permutation_entropy_drift_v1": _handler_permutation_entropy_drift_v1,
    "analysis_capacity_frontier_envelope_v1": _handler_capacity_frontier_envelope_v1,
    "analysis_graph_assortativity_shift_v1": _handler_graph_assortativity_shift_v1,
    "analysis_graph_pagerank_hotspots_v1": _handler_graph_pagerank_hotspots_v1,
    "analysis_higuchi_fractal_dimension_v1": _handler_higuchi_fractal_dimension_v1,
    "analysis_marked_point_process_intensity_v1": _handler_marked_point_process_intensity_v1,
    "analysis_spectral_radius_stability_v1": _handler_spectral_radius_stability_v1,
    "analysis_bootstrap_ci_effect_sizes_v1": _handler_bootstrap_ci_effect_sizes_v1,
    "analysis_energy_distance_two_sample_v1": _handler_energy_distance_two_sample_v1,
    "analysis_randomization_test_median_shift_v1": _handler_randomization_test_median_shift_v1,
    "analysis_distance_covariance_dependence_v1": _handler_distance_covariance_dependence_v1,
    "analysis_graph_motif_triads_shift_v1": _handler_graph_motif_triads_shift_v1,
    "analysis_multiscale_entropy_mse_v1": _handler_multiscale_entropy_mse_v1,
    "analysis_sample_entropy_irregularity_v1": _handler_sample_entropy_irregularity_v1,
}
