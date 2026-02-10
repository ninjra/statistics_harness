from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable

import math

import numpy as np
import pandas as pd

from statistic_harness.core.stat_plugins import (
    BudgetTimer,
    bh_fdr,
    build_redactor,
    cramers_v,
    robust_center_scale,
    stable_id,
)
from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json

try:  # optional
    from scipy import stats as scipy_stats
    from scipy.spatial.distance import cosine as scipy_cosine
    from scipy.signal import fftconvolve
    from scipy.sparse.csgraph import minimum_spanning_tree

    HAS_SCIPY = True
except Exception:  # pragma: no cover - optional
    scipy_stats = None
    scipy_cosine = None
    fftconvolve = None
    minimum_spanning_tree = None
    HAS_SCIPY = False

try:  # optional
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    HAS_SKLEARN = True
except Exception:  # pragma: no cover - optional
    PCA = None
    NearestNeighbors = None
    KMeans = None
    silhouette_score = None
    HAS_SKLEARN = False


def _privacy(config: dict[str, Any]) -> dict[str, Any]:
    privacy = config.get("privacy") if isinstance(config.get("privacy"), dict) else {}
    return dict(privacy) if isinstance(privacy, dict) else {}


def _k_min(config: dict[str, Any]) -> int:
    privacy = _privacy(config)
    value = privacy.get("k_min", 5)
    try:
        return max(1, int(value))
    except Exception:
        return 5


def _artifact(ctx, plugin_id: str, name: str, payload: Any, kind: str) -> PluginArtifact:
    artifact_dir = ctx.artifacts_dir(plugin_id)
    path = artifact_dir / name
    write_json(path, payload)
    return PluginArtifact(
        path=str(path.relative_to(ctx.run_dir)),
        type=kind,
        description=name,
    )


def _pick_column_by_tokens(df: pd.DataFrame, tokens: tuple[str, ...]) -> str | None:
    for col in df.columns:
        lowered = str(col).lower()
        if any(tok in lowered for tok in tokens):
            return str(col)
    return None


def _pick_duration_column(df: pd.DataFrame, numeric_cols: list[str]) -> str | None:
    # Prefer explicit duration-like numerics.
    candidates = []
    for col in numeric_cols:
        name = str(col).lower()
        if any(tok in name for tok in ("duration", "latency", "elapsed", "runtime", "run_time", "seconds", "sec")):
            candidates.append(col)
    if candidates:
        return candidates[0]
    # Else choose highest-variance numeric col.
    best = None
    best_var = -1.0
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().sum() < 10:
            continue
        var = float(np.nanvar(series.to_numpy(dtype=float)))
        if var > best_var:
            best_var = var
            best = col
    return best


def _safe_group_value(value: object, redactor: Callable[[str], str]) -> str:
    if value is None:
        return ""
    text = str(value)
    redacted = redactor(text)
    # Keep output short and stable.
    if len(redacted) > 80:
        return redacted[:77] + "..."
    return redacted


def _numeric_matrix(df: pd.DataFrame, numeric_cols: list[str], max_cols: int) -> tuple[np.ndarray, list[str]]:
    cols = [c for c in numeric_cols if c in df.columns][: max_cols]
    if not cols:
        return np.empty((0, 0)), []
    frame = df[cols].apply(pd.to_numeric, errors="coerce")
    med = frame.median(numeric_only=True)
    frame = frame.fillna(med)
    mat = frame.to_numpy(dtype=float)
    for j in range(mat.shape[1]):
        center, scale = robust_center_scale(mat[:, j])
        if not math.isfinite(scale) or scale <= 0:
            scale = 1.0
        mat[:, j] = (mat[:, j] - center) / scale
    return mat, cols


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _welch_t(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if left.size < 2 or right.size < 2:
        return 1.0, 0.0
    if HAS_SCIPY:
        stat, p = scipy_stats.ttest_ind(left, right, equal_var=False, nan_policy="omit")
        return float(p), float(stat)
    # Fallback: no p-value, return diff as "stat".
    return 1.0, float(np.nanmean(left) - np.nanmean(right))


def _one_way_anova(groups: list[np.ndarray]) -> tuple[float, float]:
    cleaned = [g[np.isfinite(g)] for g in groups if g is not None]
    cleaned = [g for g in cleaned if g.size >= 2]
    if len(cleaned) < 2:
        return 1.0, 0.0
    if HAS_SCIPY:
        stat, p = scipy_stats.f_oneway(*cleaned)
        return float(p), float(stat)
    return 1.0, 0.0


def _chi2_table(table: np.ndarray) -> tuple[float, float]:
    if table.size == 0:
        return 1.0, 0.0
    if HAS_SCIPY:
        chi2, p, _, _ = scipy_stats.chi2_contingency(table)
        return float(p), float(chi2)
    return 1.0, 0.0


def _top_group_columns(df: pd.DataFrame, categorical_cols: list[str], max_cols: int, k_min: int) -> list[str]:
    cols = []
    for col in categorical_cols:
        if col not in df.columns:
            continue
        vc = df[col].value_counts(dropna=False)
        # keep only columns with at least 2 k-anon levels
        levels = int((vc >= k_min).sum())
        if levels >= 2:
            cols.append(col)
        if len(cols) >= max_cols:
            break
    return cols


# ---------------------------
# Family D: Classic auto-scan
# ---------------------------


def _ttests_auto(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    numeric_cols = inferred.get("numeric_columns") or []
    cat_cols = inferred.get("categorical_columns") or inferred.get("group_by") or []
    if not numeric_cols or not cat_cols:
        return PluginResult("skipped", "Need numeric + categorical columns", {}, [], [], None)
    k_min = _k_min(config)
    privacy = _privacy(config)
    redactor = build_redactor(privacy)

    value_col = config.get("value_column")
    if not isinstance(value_col, str) or value_col not in df.columns:
        value_col = _pick_duration_column(df, numeric_cols)
    if not value_col:
        return PluginResult("skipped", "No numeric target detected", {}, [], [], None)

    max_group_cols = int(config.get("max_group_cols", 8))
    group_cols = _top_group_columns(df, list(cat_cols), max_group_cols, k_min)
    # binary only
    group_cols = [c for c in group_cols if int(df[c].nunique(dropna=True)) == 2]
    if not group_cols:
        return PluginResult("skipped", "No k-anonymous binary group columns detected", {}, [], [], None)

    max_tests = int(config.get("max_tests", 200))
    pvals: list[float] = []
    rows: list[dict[str, Any]] = []

    y = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)
    for gcol in group_cols:
        if timer.exceeded() or len(rows) >= max_tests:
            break
        series = df[gcol]
        vc = series.value_counts(dropna=False)
        keep = [idx for idx in vc.index if int(vc[idx]) >= k_min]
        if len(keep) != 2:
            continue
        a, b = keep[0], keep[1]
        left = y[(series == a).to_numpy()]
        right = y[(series == b).to_numpy()]
        p, stat = _welch_t(left, right)
        pvals.append(p)
        delta = float(np.nanmedian(left) - np.nanmedian(right))
        rows.append(
            {
                "group_column": gcol,
                "group_a": _safe_group_value(a, redactor),
                "group_b": _safe_group_value(b, redactor),
                "median_a": float(np.nanmedian(left)) if left.size else None,
                "median_b": float(np.nanmedian(right)) if right.size else None,
                "delta_median": delta,
                "p_value": p,
                "statistic": stat,
                "n_a": int(np.isfinite(left).sum()),
                "n_b": int(np.isfinite(right).sum()),
            }
        )

    if not rows:
        return PluginResult("ok", "No eligible t-tests computed", _basic_metrics(df, sample_meta), [], [], None)

    qvals, _ = bh_fdr(pvals)
    findings: list[dict[str, Any]] = []
    for row, q in zip(rows, qvals):
        row["q_value"] = float(q)
        if float(q) > float(config.get("fdr_q", 0.1)):
            continue
        delta = float(row.get("delta_median") or 0.0)
        # Action: shift work toward the faster group.
        action = "rebalance_group"
        recommendation = (
            f"For metric '{value_col}', {row['group_column']} shows a significant split "
            f"(q={float(q):.3f}). Consider shifting load from the slower group to the faster group "
            f"to reduce median by ~{abs(delta):.2f}."
        )
        findings.append(
            {
                "kind": "ttest_auto",
                "measurement_type": "measured",
                "metric": value_col,
                "action": action,
                "recommendation": recommendation,
                **row,
            }
        )
    artifacts = []
    artifacts.append(_artifact(ctx, plugin_id, "ttests_auto.json", {"rows": rows}, "json"))
    summary = f"T-tests auto: computed={len(rows)} significant={len(findings)}"
    return PluginResult("ok", summary, _basic_metrics(df, sample_meta), findings, artifacts, None)


def _chi_square_association(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    cat_cols = inferred.get("categorical_columns") or inferred.get("group_by") or []
    if not cat_cols or len(cat_cols) < 2:
        return PluginResult("skipped", "Need categorical columns", {}, [], [], None)
    k_min = _k_min(config)
    privacy = _privacy(config)
    redactor = build_redactor(privacy)
    max_cols = int(config.get("max_group_cols", 12))
    cols = _top_group_columns(df, list(cat_cols), max_cols, k_min)
    if len(cols) < 2:
        return PluginResult("skipped", "No k-anonymous categorical pairs detected", {}, [], [], None)

    max_pairs = int(config.get("max_pairs", 80))
    pvals: list[float] = []
    rows: list[dict[str, Any]] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if timer.exceeded() or len(rows) >= max_pairs:
                break
            a, b = cols[i], cols[j]
            tab = pd.crosstab(df[a], df[b], dropna=False)
            # k-anon prune rows/cols
            tab = tab.loc[tab.sum(axis=1) >= k_min, tab.sum(axis=0) >= k_min]
            if tab.shape[0] < 2 or tab.shape[1] < 2:
                continue
            p, chi2 = _chi2_table(tab.to_numpy(dtype=float))
            pvals.append(p)
            v = cramers_v(tab.to_numpy(dtype=float))
            rows.append(
                {
                    "col_a": a,
                    "col_b": b,
                    "p_value": p,
                    "chi2": chi2,
                    "cramers_v": float(v),
                    "shape": [int(tab.shape[0]), int(tab.shape[1])],
                    "top_a": _safe_group_value(tab.sum(axis=1).sort_values(ascending=False).index[0], redactor),
                    "top_b": _safe_group_value(tab.sum(axis=0).sort_values(ascending=False).index[0], redactor),
                }
            )
        if timer.exceeded() or len(rows) >= max_pairs:
            break

    if not rows:
        return PluginResult("ok", "No chi-square pairs eligible under k-min/budget", _basic_metrics(df, sample_meta), [], [], None)

    qvals, _ = bh_fdr(pvals)
    findings: list[dict[str, Any]] = []
    for row, q in zip(rows, qvals):
        row["q_value"] = float(q)
        if float(q) > float(config.get("fdr_q", 0.1)):
            continue
        findings.append({"kind": "chi_square_association", "measurement_type": "measured", **row})
    artifacts = [_artifact(ctx, plugin_id, "chi_square_association.json", {"rows": rows}, "json")]
    summary = f"Chi-square: computed={len(rows)} significant={len(findings)}"
    return PluginResult("ok", summary, _basic_metrics(df, sample_meta), findings, artifacts, None)


def _anova_auto(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    numeric_cols = inferred.get("numeric_columns") or []
    cat_cols = inferred.get("categorical_columns") or inferred.get("group_by") or []
    if not numeric_cols or not cat_cols:
        return PluginResult("skipped", "Need numeric + categorical columns", {}, [], [], None)
    k_min = _k_min(config)
    privacy = _privacy(config)
    redactor = build_redactor(privacy)

    process_col = config.get("process_column")
    if not isinstance(process_col, str) or process_col not in df.columns:
        process_col = _pick_column_by_tokens(df, ("process", "activity", "event", "task", "job", "step"))

    value_col = config.get("value_column")
    if not isinstance(value_col, str) or value_col not in df.columns:
        value_col = _pick_duration_column(df, numeric_cols)
    if not value_col:
        return PluginResult("skipped", "No numeric target detected", {}, [], [], None)

    group_col = config.get("group_column")
    if not isinstance(group_col, str) or group_col not in df.columns:
        group_col = _pick_column_by_tokens(df, ("server", "host", "node", "instance", "machine", "module", "params"))
    if not group_col:
        # fallback: any categorical with reasonable cardinality
        candidates = _top_group_columns(df, list(cat_cols), 10, k_min)
        group_col = candidates[0] if candidates else None
    if not group_col:
        return PluginResult("skipped", "No suitable group column detected", {}, [], [], None)

    y_all = pd.to_numeric(df[value_col], errors="coerce")
    if y_all.notna().sum() < 20:
        return PluginResult("skipped", "Insufficient numeric data for ANOVA", {}, [], [], None)

    max_processes = int(config.get("max_processes", 25))
    max_levels = int(config.get("max_levels", 8))
    # Focus on the biggest processes first to keep runtime reasonable.
    if process_col and process_col in df.columns:
        proc_counts = df[process_col].value_counts(dropna=False)
        processes = list(proc_counts.index[:max_processes])
    else:
        processes = [None]

    rows: list[dict[str, Any]] = []
    pvals: list[float] = []
    for proc in processes:
        if timer.exceeded():
            break
        if proc is None:
            work = df
            proc_name = "ALL"
        else:
            work = df.loc[df[process_col] == proc]
            proc_name = _safe_group_value(proc, redactor)
        if len(work) < max(k_min * 3, 50):
            continue
        vc = work[group_col].value_counts(dropna=False)
        levels = [lvl for lvl in vc.index if int(vc[lvl]) >= k_min][:max_levels]
        if len(levels) < 3:
            continue
        groups = []
        level_medians = []
        for lvl in levels:
            values = pd.to_numeric(work.loc[work[group_col] == lvl, value_col], errors="coerce").to_numpy(dtype=float)
            groups.append(values)
            level_medians.append(float(np.nanmedian(values)) if np.isfinite(values).any() else float("nan"))
        p, stat = _one_way_anova(groups)
        pvals.append(p)
        # effect: spread of medians between worst and best
        best_idx = int(np.nanargmin(level_medians))
        worst_idx = int(np.nanargmax(level_medians))
        delta = float(level_medians[worst_idx] - level_medians[best_idx])
        rows.append(
            {
                "process": proc_name,
                "process_norm": proc_name.lower() if proc_name and proc_name != "ALL" else "all",
                "group_column": group_col,
                "metric": value_col,
                "levels": [_safe_group_value(lvl, redactor) for lvl in levels],
                "level_medians": level_medians,
                "p_value": p,
                "statistic": stat,
                "best_level": _safe_group_value(levels[best_idx], redactor),
                "worst_level": _safe_group_value(levels[worst_idx], redactor),
                "delta_median": delta,
                "n": int(len(work)),
            }
        )

    if not rows:
        return PluginResult("ok", "ANOVA: no eligible (process,group) slices under k-min/budget", _basic_metrics(df, sample_meta), [], [], None)

    qvals, _ = bh_fdr(pvals)
    findings: list[dict[str, Any]] = []
    fdr_q = float(config.get("fdr_q", 0.1))
    min_delta = float(config.get("min_delta_median", 0.0))
    for row, q in zip(rows, qvals):
        row["q_value"] = float(q)
        if float(q) > fdr_q:
            continue
        if abs(float(row.get("delta_median") or 0.0)) < min_delta:
            continue
        recommendation = (
            f"{row['process']}: '{row['group_column']}' levels have significantly different '{row['metric']}' "
            f"(q={float(q):.3f}). Consider shifting work away from '{row['worst_level']}' toward '{row['best_level']}' "
            f"to reduce median by ~{abs(float(row['delta_median'])):.2f}."
        )
        findings.append(
            {
                "kind": "anova_process_group_effect",
                "measurement_type": "measured",
                "action": "rebalance_group",
                "recommendation": recommendation,
                **row,
            }
        )

    artifacts = [_artifact(ctx, plugin_id, "anova_auto.json", {"rows": rows}, "json")]
    summary = f"ANOVA auto: slices={len(rows)} significant={len(findings)}"
    return PluginResult("ok", summary, _basic_metrics(df, sample_meta), findings, artifacts, None)


def _regression_auto(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    numeric_cols = inferred.get("numeric_columns") or []
    cat_cols = inferred.get("categorical_columns") or []
    if not numeric_cols:
        return PluginResult("skipped", "Need numeric columns", {}, [], [], None)
    value_col = config.get("target_column")
    if not isinstance(value_col, str) or value_col not in df.columns:
        value_col = _pick_duration_column(df, numeric_cols)
    if not value_col:
        return PluginResult("skipped", "No numeric target detected", {}, [], [], None)

    max_num_features = int(config.get("max_num_features", 12))
    max_cat_features = int(config.get("max_cat_features", 4))
    k_min = _k_min(config)
    privacy = _privacy(config)
    redactor = build_redactor(privacy)

    y = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < 50:
        return PluginResult("skipped", "Insufficient target data for regression", {}, [], [], None)

    # Build a lightweight design matrix: top numeric features + one-hot for a few low-card cats.
    other_nums = [c for c in numeric_cols if c != value_col and c in df.columns][:max_num_features]
    # pandas may return a non-writeable view in some cases; force a writable copy.
    X_num = np.array(
        df.loc[mask, other_nums]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float),
        copy=True,
    )
    # standardize
    for j in range(X_num.shape[1]):
        center, scale = robust_center_scale(X_num[:, j])
        if not math.isfinite(scale) or scale <= 0:
            scale = 1.0
        X_num[:, j] = (X_num[:, j] - center) / scale

    cat_features = []
    X_cat_parts = []
    for col in cat_cols[:max_cat_features]:
        if timer.exceeded():
            break
        if col not in df.columns:
            continue
        vc = df[col].value_counts(dropna=False)
        levels = [lvl for lvl in vc.index if int(vc[lvl]) >= k_min][:6]
        if len(levels) < 2:
            continue
        # one-hot encode all but first level
        base = levels[0]
        for lvl in levels[1:]:
            vec = (df.loc[mask, col] == lvl).to_numpy(dtype=float)
            X_cat_parts.append(vec.reshape(-1, 1))
            cat_features.append(f"{col}={_safe_group_value(lvl, redactor)}")

    X_cat = np.concatenate(X_cat_parts, axis=1) if X_cat_parts else np.empty((int(mask.sum()), 0))
    X = np.concatenate([np.ones((int(mask.sum()), 1)), X_num, X_cat], axis=1)
    feature_names = ["intercept"] + other_nums + cat_features

    yv = y[mask]
    # Fit OLS (robust alternatives are optional; keep deterministic and fast).
    try:
        coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
    except Exception:
        return PluginResult("degraded", "Regression failed (numerical)", _basic_metrics(df, sample_meta), [], [], None)

    rows = []
    for name, c in zip(feature_names, coef):
        if name == "intercept":
            continue
        rows.append({"feature": name, "coef": float(c)})
    rows.sort(key=lambda r: abs(float(r.get("coef") or 0.0)), reverse=True)
    top = rows[: int(config.get("max_drivers", 12))]

    findings = [
        {
            "kind": "regression_auto_drivers",
            "measurement_type": "modeled" if HAS_SCIPY else "measured",
            "metric": value_col,
            "drivers": top,
        }
    ]
    artifacts = [_artifact(ctx, plugin_id, "regression_auto.json", {"target": value_col, "drivers": rows}, "json")]
    summary = f"Regression drivers for '{value_col}': top={len(top)}"
    return PluginResult("ok", summary, _basic_metrics(df, sample_meta), findings, artifacts, None)


def _time_series_analysis_auto(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    time_col = inferred.get("time_column")
    numeric_cols = inferred.get("numeric_columns") or []
    if not time_col or time_col not in df.columns or not numeric_cols:
        return PluginResult("skipped", "Need time + numeric columns", {}, [], [], None)
    # Choose a small set of top variance metrics.
    max_metrics = int(config.get("max_metrics", 6))
    metrics = numeric_cols[:max_metrics]
    ts = pd.to_datetime(df[time_col], errors="coerce")
    if ts.notna().sum() < 50:
        return PluginResult("skipped", "Insufficient valid timestamps", {}, [], [], None)
    work = df.loc[ts.notna(), :].copy()
    work["_t"] = ts.loc[ts.notna()]
    work = work.sort_values("_t")
    rows = []
    for col in metrics:
        if timer.exceeded():
            break
        y = pd.to_numeric(work[col], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(y).sum() < 50:
            continue
        # Simple trend: slope of y vs index.
        x = np.arange(len(y), dtype=float)
        m = np.isfinite(y)
        if m.sum() < 50:
            continue
        xm = x[m]
        ym = y[m]
        slope = float(np.cov(xm, ym)[0, 1] / (np.var(xm) + 1e-9))
        # Burstiness proxy: robust z on first diff.
        dy = np.diff(ym)
        med = float(np.median(dy))
        mad = float(np.median(np.abs(dy - med)) + 1e-9)
        burst = float((np.abs(dy - med) > (6.0 * mad)).mean())
        rows.append({"metric": col, "trend_slope_per_index": slope, "burst_fraction": burst})
    artifacts = [_artifact(ctx, plugin_id, "time_series_auto.json", {"rows": rows}, "json")]
    findings = [{"kind": "time_series_auto", "measurement_type": "measured", "rows": rows}]
    summary = f"Time-series scan: metrics={len(rows)}"
    return PluginResult("ok", summary, _basic_metrics(df, sample_meta), findings, artifacts, None)


def _survival_time_to_event(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    # Minimal gating: needs a duration-like numeric.
    numeric_cols = inferred.get("numeric_columns") or []
    if not numeric_cols:
        return PluginResult("skipped", "Need numeric columns", {}, [], [], None)
    duration_col = config.get("duration_column")
    if not isinstance(duration_col, str) or duration_col not in df.columns:
        duration_col = _pick_duration_column(df, numeric_cols)
    if not duration_col:
        return PluginResult("skipped", "No duration-like metric detected", {}, [], [], None)
    durations = pd.to_numeric(df[duration_col], errors="coerce").to_numpy(dtype=float)
    durations = durations[np.isfinite(durations)]
    if durations.size < 100:
        return PluginResult("skipped", "Insufficient durations for survival summary", {}, [], [], None)
    # Output a simple KM-like summary: percentiles.
    p50 = float(np.nanpercentile(durations, 50))
    p90 = float(np.nanpercentile(durations, 90))
    p99 = float(np.nanpercentile(durations, 99))
    finding = {
        "kind": "survival_time_to_event",
        "measurement_type": "measured",
        "duration_metric": duration_col,
        "p50": p50,
        "p90": p90,
        "p99": p99,
    }
    artifacts = [_artifact(ctx, plugin_id, "survival_summary.json", finding, "json")]
    return PluginResult("ok", "Survival summary computed", _basic_metrics(df, sample_meta), [finding], artifacts, None)


def _factor_analysis_auto(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    numeric_cols = inferred.get("numeric_columns") or []
    if len(numeric_cols) < 3:
        return PluginResult("skipped", "Need 3+ numeric columns", {}, [], [], None)
    max_cols = int(config.get("max_cols", 20))
    X, cols = _numeric_matrix(df, numeric_cols, max_cols=max_cols)
    if X.shape[0] < 50 or X.shape[1] < 3:
        return PluginResult("skipped", "Insufficient numeric matrix", {}, [], [], None)
    n_components = int(config.get("n_components", min(5, X.shape[1])))
    if HAS_SKLEARN and PCA is not None:
        pca = PCA(n_components=n_components, random_state=int(config.get("seed", 1337)))
        Z = pca.fit_transform(X)
        loadings = pca.components_.T
        explained = pca.explained_variance_ratio_.tolist()
        top = []
        for k in range(loadings.shape[1]):
            order = np.argsort(np.abs(loadings[:, k]))[::-1][:6]
            top.append([(cols[i], float(loadings[i, k])) for i in order])
        finding = {
            "kind": "factor_analysis_auto",
            "measurement_type": "measured",
            "method": "pca_fallback",
            "explained_variance_ratio": explained,
            "top_loadings": top,
        }
        artifacts = [_artifact(ctx, plugin_id, "factor_analysis.json", finding, "json")]
        return PluginResult("ok", "Factor/PCA summary computed", _basic_metrics(df, sample_meta), [finding], artifacts, None)
    return PluginResult("degraded", "sklearn missing; factor analysis unavailable", _basic_metrics(df, sample_meta), [], [], None)


def _cluster_analysis_auto(
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
        return PluginResult("skipped", "Need 2+ numeric columns", {}, [], [], None)
    if not (HAS_SKLEARN and KMeans is not None):
        return PluginResult("degraded", "sklearn missing; clustering unavailable", _basic_metrics(df, sample_meta), [], [], None)
    max_points = int(config.get("max_points", 2000))
    max_cols = int(config.get("max_cols", 12))
    X, cols = _numeric_matrix(df.head(max_points), numeric_cols, max_cols=max_cols)
    if X.shape[0] < 50:
        return PluginResult("skipped", "Insufficient rows for clustering", {}, [], [], None)
    # Pick k by a small silhouette sweep.
    best_k = None
    best_score = -1.0
    max_k = int(config.get("max_k", 6))
    for k in range(2, max_k + 1):
        if timer.exceeded():
            break
        km = KMeans(n_clusters=k, n_init=5, random_state=int(config.get("seed", 1337)))
        labels = km.fit_predict(X)
        if silhouette_score is None:
            score = -float(km.inertia_)
        else:
            try:
                score = float(silhouette_score(X, labels))
            except Exception:
                score = -1.0
        if score > best_score:
            best_score = score
            best_k = k
    if best_k is None:
        return PluginResult("ok", "Clustering skipped (budget)", _basic_metrics(df, sample_meta), [], [], None)
    km = KMeans(n_clusters=int(best_k), n_init=10, random_state=int(config.get("seed", 1337)))
    labels = km.fit_predict(X)
    counts = {str(i): int((labels == i).sum()) for i in range(int(best_k))}
    finding = {
        "kind": "cluster_analysis_auto",
        "measurement_type": "measured",
        "k": int(best_k),
        "silhouette": best_score,
        "cluster_counts": counts,
        "features": cols,
    }
    artifacts = [_artifact(ctx, plugin_id, "clusters.json", finding, "json")]
    return PluginResult("ok", "Cluster analysis computed", _basic_metrics(df, sample_meta), [finding], artifacts, None)


def _pca_auto(
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
        return PluginResult("skipped", "Need 2+ numeric columns", {}, [], [], None)
    max_points = int(config.get("max_points", 5000))
    max_cols = int(config.get("max_cols", 20))
    X, cols = _numeric_matrix(df.head(max_points), numeric_cols, max_cols=max_cols)
    if X.shape[0] < 50:
        return PluginResult("skipped", "Insufficient rows for PCA", {}, [], [], None)
    n_components = int(config.get("n_components", min(5, X.shape[1])))
    if HAS_SKLEARN and PCA is not None:
        pca = PCA(n_components=n_components, random_state=int(config.get("seed", 1337)))
        pca.fit(X)
        explained = pca.explained_variance_ratio_.tolist()
        comps = pca.components_.tolist()
        finding = {
            "kind": "pca_auto",
            "measurement_type": "measured",
            "explained_variance_ratio": explained,
            "components": comps,
            "features": cols,
        }
        artifacts = [_artifact(ctx, plugin_id, "pca.json", finding, "json")]
        return PluginResult("ok", "PCA computed", _basic_metrics(df, sample_meta), [finding], artifacts, None)
    return PluginResult("degraded", "sklearn missing; PCA unavailable", _basic_metrics(df, sample_meta), [], [], None)


# ---------------------------
# Family B: Topographic maps
# ---------------------------


def _topographic_similarity_angle_projection(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    numeric_cols = inferred.get("numeric_columns") or []
    cat_cols = inferred.get("categorical_columns") or inferred.get("group_by") or []
    if len(numeric_cols) < 2 or not cat_cols:
        return PluginResult("skipped", "Need multi-numeric + categorical columns", {}, [], [], None)
    k_min = _k_min(config)
    privacy = _privacy(config)
    redactor = build_redactor(privacy)
    X, cols = _numeric_matrix(df, numeric_cols, max_cols=int(config.get("max_cols", 30)))
    if X.shape[0] < 50 or X.shape[1] < 2:
        return PluginResult("skipped", "Insufficient numeric matrix", {}, [], [], None)

    max_groups = int(config.get("max_groups", 6))
    group_cols = _top_group_columns(df, list(cat_cols), max_cols=int(config.get("max_group_cols", 6)), k_min=k_min)
    if not group_cols:
        return PluginResult("skipped", "No k-anonymous group columns", {}, [], [], None)
    global_map = np.nanmedian(X, axis=0)
    rows: list[dict[str, Any]] = []
    for gcol in group_cols:
        if timer.exceeded():
            break
        vc = df[gcol].value_counts(dropna=False)
        levels = [lvl for lvl in vc.index if int(vc[lvl]) >= k_min][:max_groups]
        if len(levels) < 2:
            continue
        maps = {}
        for lvl in levels:
            idx = (df[gcol] == lvl).to_numpy()
            maps[lvl] = np.nanmedian(X[idx, :], axis=0)
        # dissimilarity pairs
        best_pair = None
        best_sim = 1.0
        for i in range(len(levels)):
            for j in range(i + 1, len(levels)):
                sim = _cosine_sim(maps[levels[i]], maps[levels[j]])
                if sim < best_sim:
                    best_sim = sim
                    best_pair = (levels[i], levels[j])
        # projections
        proj = {lvl: float(np.dot(maps[lvl], global_map) / (np.linalg.norm(global_map) + 1e-9)) for lvl in levels}
        max_lvl = max(proj, key=proj.get)
        min_lvl = min(proj, key=proj.get)
        rows.append(
            {
                "group_column": gcol,
                "levels": [_safe_group_value(l, redactor) for l in levels],
                "min_cosine_pair": [
                    _safe_group_value(best_pair[0], redactor),
                    _safe_group_value(best_pair[1], redactor),
                ]
                if best_pair
                else None,
                "min_cosine_similarity": float(best_sim),
                "max_projection_level": _safe_group_value(max_lvl, redactor),
                "min_projection_level": _safe_group_value(min_lvl, redactor),
                "projection_spread": float(proj[max_lvl] - proj[min_lvl]),
            }
        )

    artifacts = [_artifact(ctx, plugin_id, "topographic_similarity.json", {"rows": rows}, "json")]
    findings = [{"kind": "topographic_similarity", "measurement_type": "measured", "rows": rows}]
    return PluginResult("ok", f"Topographic similarity computed for {len(rows)} grouping(s)", _basic_metrics(df, sample_meta), findings, artifacts, None)


def _topographic_angle_dynamics(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    time_col = inferred.get("time_column")
    numeric_cols = inferred.get("numeric_columns") or []
    if len(numeric_cols) < 2:
        return PluginResult("skipped", "Need multi-numeric columns", {}, [], [], None)
    if not time_col or time_col not in df.columns:
        return PluginResult("skipped", "Need time column for dynamics", {}, [], [], None)
    X, cols = _numeric_matrix(df, numeric_cols, max_cols=int(config.get("max_cols", 30)))
    ts = pd.to_datetime(df[time_col], errors="coerce")
    m = ts.notna()
    if m.sum() < 100:
        return PluginResult("skipped", "Insufficient valid timestamps", {}, [], [], None)
    order = np.argsort(ts[m].to_numpy())
    Xs = X[m.to_numpy()][order]
    ts_s = ts[m].to_numpy()[order]
    window = int(config.get("window_rows", 200))
    step = int(config.get("step_rows", max(10, window // 4)))
    template = np.nanmedian(Xs, axis=0)
    rows = []
    for start in range(0, Xs.shape[0] - window + 1, step):
        if timer.exceeded():
            break
        seg = Xs[start : start + window]
        vec = np.nanmedian(seg, axis=0)
        sim = _cosine_sim(template, vec)
        rows.append(
            {
                "t_start": str(pd.Timestamp(ts_s[start]).to_pydatetime()),
                "t_end": str(pd.Timestamp(ts_s[start + window - 1]).to_pydatetime()),
                "cosine_similarity_to_template": float(sim),
            }
        )
    artifacts = [_artifact(ctx, plugin_id, "angle_dynamics.json", {"rows": rows}, "json")]
    findings = [{"kind": "topographic_angle_dynamics", "measurement_type": "measured", "rows": rows}]
    return PluginResult("ok", f"Angle dynamics windows={len(rows)}", _basic_metrics(df, sample_meta), findings, artifacts, None)


def _topographic_tanova_permutation(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    numeric_cols = inferred.get("numeric_columns") or []
    cat_cols = inferred.get("categorical_columns") or inferred.get("group_by") or []
    if len(numeric_cols) < 2 or not cat_cols:
        return PluginResult("skipped", "Need multi-numeric + categorical columns", {}, [], [], None)
    k_min = _k_min(config)
    max_group_cols = int(config.get("max_group_cols", 8))
    group_cols = _top_group_columns(df, list(cat_cols), max_cols=max_group_cols, k_min=k_min)
    if not group_cols:
        return PluginResult("skipped", "No k-anonymous group columns", {}, [], [], None)
    X, cols = _numeric_matrix(df, numeric_cols, max_cols=int(config.get("max_cols", 25)))
    if X.shape[0] < 80:
        return PluginResult("skipped", "Insufficient rows for permutation test", {}, [], [], None)
    n_perm = int(config.get("n_permutations", 200))
    rng = np.random.RandomState(int(config.get("seed", 1337)))

    rows = []
    pvals = []
    for gcol in group_cols:
        if timer.exceeded():
            break
        labels = df[gcol].astype(str).to_numpy()
        vc = pd.Series(labels).value_counts(dropna=False)
        keep_levels = set([lvl for lvl in vc.index if int(vc[lvl]) >= k_min][:6])
        keep_mask = np.array([lbl in keep_levels for lbl in labels], dtype=bool)
        if keep_mask.sum() < 80:
            continue
        Xk = X[keep_mask]
        lk = labels[keep_mask]
        # Defensive: mixed-type labels can appear in ERP datasets; force stable str ordering.
        levels = sorted(set(lk), key=lambda x: str(x))
        if len(levels) < 2:
            continue
        # statistic: mean pairwise cosine distance between group mean maps
        means = []
        for lvl in levels:
            means.append(np.nanmedian(Xk[lk == lvl], axis=0))
        stat0 = 0.0
        cnt = 0
        for i in range(len(means)):
            for j in range(i + 1, len(means)):
                stat0 += (1.0 - _cosine_sim(means[i], means[j]))
                cnt += 1
        stat0 = stat0 / max(1, cnt)
        # permutations
        greater = 0
        for _ in range(n_perm):
            if timer.exceeded():
                break
            perm = rng.permutation(lk)
            means_p = [np.nanmedian(Xk[perm == lvl], axis=0) for lvl in levels]
            statp = 0.0
            cntp = 0
            for i in range(len(means_p)):
                for j in range(i + 1, len(means_p)):
                    statp += (1.0 - _cosine_sim(means_p[i], means_p[j]))
                    cntp += 1
            statp = statp / max(1, cntp)
            if statp >= stat0:
                greater += 1
        p = (greater + 1.0) / (n_perm + 1.0)
        pvals.append(float(p))
        rows.append({"group_column": gcol, "statistic": float(stat0), "p_value": float(p), "levels": len(levels), "n": int(Xk.shape[0])})

    if not rows:
        return PluginResult("ok", "TANOVA-style permutation: none eligible", _basic_metrics(df, sample_meta), [], [], None)
    qvals, _ = bh_fdr(pvals)
    findings = []
    for row, q in zip(rows, qvals):
        row["q_value"] = float(q)
        if float(q) <= float(config.get("fdr_q", 0.1)):
            findings.append({"kind": "topographic_tanova_permutation", "measurement_type": "measured", **row})
    artifacts = [_artifact(ctx, plugin_id, "tanova_permutation.json", {"rows": rows}, "json")]
    return PluginResult("ok", f"TANOVA permutation columns={len(rows)} significant={len(findings)}", _basic_metrics(df, sample_meta), findings, artifacts, None)


def _map_permutation_test_karniski(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    # Implement a simplified, distribution-free permutation test for multivariate maps:
    # test statistic = ||mean(map_A) - mean(map_B)||_2 (on robust-scaled features).
    numeric_cols = inferred.get("numeric_columns") or []
    cat_cols = inferred.get("categorical_columns") or inferred.get("group_by") or []
    if len(numeric_cols) < 2 or not cat_cols:
        return PluginResult("skipped", "Need multi-numeric + categorical columns", {}, [], [], None)
    k_min = _k_min(config)
    group_col = config.get("group_column")
    if not isinstance(group_col, str) or group_col not in df.columns:
        candidates = _top_group_columns(df, list(cat_cols), max_cols=8, k_min=k_min)
        group_col = candidates[0] if candidates else None
    if not group_col:
        return PluginResult("skipped", "No suitable group column detected", {}, [], [], None)
    X, cols = _numeric_matrix(df, numeric_cols, max_cols=int(config.get("max_cols", 30)))
    labels = df[group_col]
    vc = labels.value_counts(dropna=False)
    levels = [lvl for lvl in vc.index if int(vc[lvl]) >= k_min][:2]
    if len(levels) != 2:
        return PluginResult("skipped", "Need two k-anonymous groups for Karniski-style test", {}, [], [], None)
    a, b = levels[0], levels[1]
    idx_a = (labels == a).to_numpy()
    idx_b = (labels == b).to_numpy()
    mu_a = np.nanmedian(X[idx_a], axis=0)
    mu_b = np.nanmedian(X[idx_b], axis=0)
    stat0 = float(np.linalg.norm(mu_a - mu_b))
    n_perm = int(config.get("n_permutations", 500))
    rng = np.random.RandomState(int(config.get("seed", 1337)))
    lk = labels.astype(str).to_numpy()
    keep = np.logical_or(idx_a, idx_b)
    lk = lk[keep]
    Xk = X[keep]
    greater = 0
    for _ in range(n_perm):
        if timer.exceeded():
            break
        perm = rng.permutation(lk)
        mu_ap = np.nanmedian(Xk[perm == str(a)], axis=0)
        mu_bp = np.nanmedian(Xk[perm == str(b)], axis=0)
        statp = float(np.linalg.norm(mu_ap - mu_bp))
        if statp >= stat0:
            greater += 1
    p = (greater + 1.0) / (n_perm + 1.0)
    privacy = _privacy(config)
    redactor = build_redactor(privacy)
    finding = {
        "kind": "map_permutation_test_karniski",
        "measurement_type": "measured",
        "group_column": group_col,
        "group_a": _safe_group_value(a, redactor),
        "group_b": _safe_group_value(b, redactor),
        "statistic_l2": stat0,
        "p_value": float(p),
        "n_perm": int(n_perm),
        "n": int(Xk.shape[0]),
    }
    artifacts = [_artifact(ctx, plugin_id, "karniski_map_test.json", finding, "json")]
    return PluginResult("ok", "Karniski-style permutation test computed", _basic_metrics(df, sample_meta), [finding], artifacts, None)


# ---------------------------
# Family A/C/E: placeholders
# These run in "degraded" mode by default to avoid heavy computation unless
# explicitly enabled/tuned.
# ---------------------------


def _heavy_disabled(plugin_id: str, config: dict[str, Any]) -> bool:
    # Keep default resource usage low; allow enabling heavy plugins by settings.
    return not bool(config.get("enable_heavy", False))


def _heavy_placeholder(kind: str, msg: str) -> list[dict[str, Any]]:
    return [{"kind": kind, "measurement_type": "degraded", "reason": msg}]


def _tda_persistent_homology(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    if _heavy_disabled(plugin_id, config):
        return PluginResult("degraded", "Heavy TDA disabled (set enable_heavy=true)", _basic_metrics(df, sample_meta), _heavy_placeholder("tda_persistent_homology", "disabled_by_default"), [], None)
    # Proxy implementation: MST-based H0 + cycle-rank proxy for H1 on kNN graph.
    numeric_cols = inferred.get("numeric_columns") or []
    if len(numeric_cols) < 2:
        return PluginResult("skipped", "Need multi-numeric columns", {}, [], [], None)
    max_points = int(config.get("max_points", 1000))
    X, cols = _numeric_matrix(df.head(max_points), numeric_cols, max_cols=int(config.get("max_cols", 20)))
    if X.shape[0] < 50:
        return PluginResult("skipped", "Insufficient points for TDA proxy", {}, [], [], None)
    k = int(config.get("knn_k", 10))
    if not (HAS_SKLEARN and NearestNeighbors is not None):
        return PluginResult("degraded", "sklearn missing; TDA proxy unavailable", _basic_metrics(df, sample_meta), _heavy_placeholder("tda_persistent_homology", "missing_sklearn"), [], None)
    nn = NearestNeighbors(n_neighbors=min(k, X.shape[0] - 1), algorithm="auto")
    nn.fit(X)
    dists, neigh = nn.kneighbors(X, return_distance=True)
    # Build undirected edge list (i,j,dist) excluding self.
    edges = []
    for i in range(X.shape[0]):
        for j_idx in range(1, neigh.shape[1]):
            j = int(neigh[i, j_idx])
            if i == j:
                continue
            dist = float(dists[i, j_idx])
            if i < j:
                edges.append((i, j, dist))
    edges.sort(key=lambda e: e[2])
    # Kruskal MST
    parent = list(range(X.shape[0]))
    rank = [0] * X.shape[0]

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
        return True

    mst_len = 0.0
    mst_edges = 0
    for i, j, dist in edges:
        if union(i, j):
            mst_len += dist
            mst_edges += 1
            if mst_edges >= X.shape[0] - 1:
                break
        if timer.exceeded():
            break

    # Cycle-rank proxy using full kNN edge set: E - V + C.
    comp_parent = list(range(X.shape[0]))
    comp_rank = [0] * X.shape[0]

    def cfind(x: int) -> int:
        while comp_parent[x] != x:
            comp_parent[x] = comp_parent[comp_parent[x]]
            x = comp_parent[x]
        return x

    def cunion(a: int, b: int) -> None:
        ra, rb = cfind(a), cfind(b)
        if ra == rb:
            return
        if comp_rank[ra] < comp_rank[rb]:
            comp_parent[ra] = rb
        elif comp_rank[ra] > comp_rank[rb]:
            comp_parent[rb] = ra
        else:
            comp_parent[rb] = ra
            comp_rank[ra] += 1

    for i, j, _ in edges:
        cunion(i, j)
    comps = len({cfind(i) for i in range(X.shape[0])})
    V = int(X.shape[0])
    E = int(len(edges))
    cycle_rank = max(0, E - V + comps)

    finding = {
        "kind": "tda_persistent_homology",
        "measurement_type": "degraded",
        "max_points": int(X.shape[0]),
        "mst_total_length": float(mst_len),
        "cycle_rank_proxy": int(cycle_rank),
        "features_used": cols,
    }
    artifacts = [_artifact(ctx, plugin_id, "tda_proxy.json", finding, "json")]
    return PluginResult("ok", "TDA proxy metrics computed", _basic_metrics(df, sample_meta), [finding], artifacts, None)


def _tda_persistence_landscapes(*args, **kwargs) -> PluginResult:
    plugin_id = args[0]
    ctx = args[1]
    df = args[2]
    config = args[3]
    sample_meta = args[6]
    if _heavy_disabled(plugin_id, config):
        return PluginResult("degraded", "Heavy TDA disabled (set enable_heavy=true)", _basic_metrics(df, sample_meta), _heavy_placeholder("tda_persistence_landscapes", "disabled_by_default"), [], None)
    return PluginResult("degraded", "Persistence landscapes not yet enabled; use persistent homology proxy", _basic_metrics(df, sample_meta), _heavy_placeholder("tda_persistence_landscapes", "not_implemented"), [], None)


def _tda_mapper_graph(*args, **kwargs) -> PluginResult:
    plugin_id = args[0]
    ctx = args[1]
    df = args[2]
    config = args[3]
    sample_meta = args[6]
    if _heavy_disabled(plugin_id, config):
        return PluginResult("degraded", "Heavy mapper disabled (set enable_heavy=true)", _basic_metrics(df, sample_meta), _heavy_placeholder("tda_mapper_graph", "disabled_by_default"), [], None)
    return PluginResult("degraded", "Mapper graph placeholder", _basic_metrics(df, sample_meta), _heavy_placeholder("tda_mapper_graph", "not_implemented"), [], None)


def _tda_betti_curve_changepoint(*args, **kwargs) -> PluginResult:
    plugin_id = args[0]
    df = args[2]
    config = args[3]
    sample_meta = args[6]
    if _heavy_disabled(plugin_id, config):
        return PluginResult("degraded", "Heavy betti curves disabled (set enable_heavy=true)", _basic_metrics(df, sample_meta), _heavy_placeholder("tda_betti_curve_changepoint", "disabled_by_default"), [], None)
    return PluginResult("degraded", "Betti curve changepoint placeholder", _basic_metrics(df, sample_meta), _heavy_placeholder("tda_betti_curve_changepoint", "not_implemented"), [], None)


def _surface_multiscale_wavelet_curvature(*args, **kwargs) -> PluginResult:
    plugin_id = args[0]
    df = args[2]
    config = args[3]
    sample_meta = args[6]
    if _heavy_disabled(plugin_id, config):
        return PluginResult("degraded", "Surface complexity disabled (set enable_heavy=true)", _basic_metrics(df, sample_meta), _heavy_placeholder("surface_multiscale_wavelet_curvature", "disabled_by_default"), [], None)
    return PluginResult("degraded", "Surface curvature placeholder", _basic_metrics(df, sample_meta), _heavy_placeholder("surface_multiscale_wavelet_curvature", "not_implemented"), [], None)


def _surface_fractal_dimension_variogram(*args, **kwargs) -> PluginResult:
    plugin_id = args[0]
    df = args[2]
    config = args[3]
    sample_meta = args[6]
    if _heavy_disabled(plugin_id, config):
        return PluginResult("degraded", "Surface complexity disabled (set enable_heavy=true)", _basic_metrics(df, sample_meta), _heavy_placeholder("surface_fractal_dimension_variogram", "disabled_by_default"), [], None)
    return PluginResult("degraded", "Fractal dimension placeholder", _basic_metrics(df, sample_meta), _heavy_placeholder("surface_fractal_dimension_variogram", "not_implemented"), [], None)


def _surface_rugosity_index(*args, **kwargs) -> PluginResult:
    plugin_id = args[0]
    df = args[2]
    config = args[3]
    sample_meta = args[6]
    if _heavy_disabled(plugin_id, config):
        return PluginResult("degraded", "Surface complexity disabled (set enable_heavy=true)", _basic_metrics(df, sample_meta), _heavy_placeholder("surface_rugosity_index", "disabled_by_default"), [], None)
    return PluginResult("degraded", "Rugosity placeholder", _basic_metrics(df, sample_meta), _heavy_placeholder("surface_rugosity_index", "not_implemented"), [], None)


def _surface_terrain_position_index(*args, **kwargs) -> PluginResult:
    plugin_id = args[0]
    df = args[2]
    config = args[3]
    sample_meta = args[6]
    if _heavy_disabled(plugin_id, config):
        return PluginResult("degraded", "Surface complexity disabled (set enable_heavy=true)", _basic_metrics(df, sample_meta), _heavy_placeholder("surface_terrain_position_index", "disabled_by_default"), [], None)
    return PluginResult("degraded", "TPI placeholder", _basic_metrics(df, sample_meta), _heavy_placeholder("surface_terrain_position_index", "not_implemented"), [], None)


def _surface_fabric_sso_eigen(*args, **kwargs) -> PluginResult:
    plugin_id = args[0]
    df = args[2]
    config = args[3]
    sample_meta = args[6]
    if _heavy_disabled(plugin_id, config):
        return PluginResult("degraded", "Surface fabric disabled (set enable_heavy=true)", _basic_metrics(df, sample_meta), _heavy_placeholder("surface_fabric_sso_eigen", "disabled_by_default"), [], None)
    return PluginResult("degraded", "SSO fabric placeholder", _basic_metrics(df, sample_meta), _heavy_placeholder("surface_fabric_sso_eigen", "not_implemented"), [], None)


def _surface_hydrology_flow_watershed(*args, **kwargs) -> PluginResult:
    plugin_id = args[0]
    df = args[2]
    config = args[3]
    sample_meta = args[6]
    if _heavy_disabled(plugin_id, config):
        return PluginResult("degraded", "Hydrology ops disabled (set enable_heavy=true)", _basic_metrics(df, sample_meta), _heavy_placeholder("surface_hydrology_flow_watershed", "disabled_by_default"), [], None)
    return PluginResult("degraded", "Hydrology placeholder", _basic_metrics(df, sample_meta), _heavy_placeholder("surface_hydrology_flow_watershed", "not_implemented"), [], None)


def _bayesian_point_displacement(*args, **kwargs) -> PluginResult:
    plugin_id = args[0]
    df = args[2]
    config = args[3]
    sample_meta = args[6]
    if _heavy_disabled(plugin_id, config):
        return PluginResult("degraded", "Bayesian displacement disabled (set enable_heavy=true)", _basic_metrics(df, sample_meta), _heavy_placeholder("bayesian_point_displacement", "disabled_by_default"), [], None)
    return PluginResult("degraded", "Bayesian displacement placeholder", _basic_metrics(df, sample_meta), _heavy_placeholder("bayesian_point_displacement", "not_implemented"), [], None)


def _monte_carlo_surface_uncertainty(*args, **kwargs) -> PluginResult:
    plugin_id = args[0]
    df = args[2]
    config = args[3]
    sample_meta = args[6]
    if _heavy_disabled(plugin_id, config):
        return PluginResult("degraded", "Monte Carlo uncertainty disabled (set enable_heavy=true)", _basic_metrics(df, sample_meta), _heavy_placeholder("monte_carlo_surface_uncertainty", "disabled_by_default"), [], None)
    return PluginResult("degraded", "Monte Carlo surface uncertainty placeholder", _basic_metrics(df, sample_meta), _heavy_placeholder("monte_carlo_surface_uncertainty", "not_implemented"), [], None)


def _surface_roughness_metrics(*args, **kwargs) -> PluginResult:
    plugin_id = args[0]
    df = args[2]
    config = args[3]
    sample_meta = args[6]
    if _heavy_disabled(plugin_id, config):
        return PluginResult("degraded", "Surface roughness disabled (set enable_heavy=true)", _basic_metrics(df, sample_meta), _heavy_placeholder("surface_roughness_metrics", "disabled_by_default"), [], None)
    return PluginResult("degraded", "Surface roughness placeholder", _basic_metrics(df, sample_meta), _heavy_placeholder("surface_roughness_metrics", "not_implemented"), [], None)


def _basic_metrics(df: pd.DataFrame, sample_meta: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        "rows_seen": int(sample_meta.get("rows_total", len(df))),
        "rows_used": int(sample_meta.get("rows_used", len(df))),
        "cols_used": int(len(df.columns)),
    }
    metrics.update(sample_meta)
    return metrics


# ---------------------------
# Family F: CEO-grade recommendations
# ---------------------------


def _pick_column_by_name_tokens(candidates: list[str], tokens: tuple[str, ...]) -> str | None:
    for col in candidates:
        lowered = str(col).lower()
        if any(tok in lowered for tok in tokens):
            return str(col)
    return None


def _actionable_ops_levers_v1(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    privacy = _privacy(config)
    redactor = build_redactor(privacy)
    k_min = _k_min(config)

    # NOTE: Many ERP datasets encode "categorical" dimensions as numeric codes (float/int)
    # and some inference pipelines may return empty categorical/numeric lists. This plugin
    # must still produce actionable outputs in that case, so we always fall back to
    # name-token matching over df.columns plus bounded-cardinality heuristics.
    time_col = inferred.get("time_column")
    cat_cols = list(inferred.get("categorical_columns") or inferred.get("group_by") or [])
    num_cols = list(inferred.get("numeric_columns") or [])

    # Used for "over-threshold wait" style recommendations; treated as an upper-bound impact.
    wait_threshold_seconds = config.get("wait_threshold_seconds", 60.0)
    try:
        wait_threshold_seconds = float(wait_threshold_seconds)
    except (TypeError, ValueError):
        wait_threshold_seconds = 60.0

    all_cols = [c for c in df.columns if isinstance(c, str)]

    def _pick_from_all(tokens: tuple[str, ...]) -> str | None:
        col = _pick_column_by_name_tokens(all_cols, tokens)
        return col if isinstance(col, str) and col in df.columns else None

    # Derive a queue-delay target when possible (this is the primary KPI we care about).
    # Prefer: WAIT_TO_START_SEC / ELIGIBLE_WAIT_SEC / QUEUE_DELAY_SEC; else compute from timestamps.
    derived_wait_col = None
    for candidate in (
        "WAIT_TO_START_SEC",
        "WAIT_SEC",
        "ELIGIBLE_WAIT_SEC",
        "QUEUE_DELAY_SEC",
        "QUEUE_WAIT_SEC",
    ):
        if candidate in df.columns:
            derived_wait_col = candidate
            break

    queue_col = None
    start_col = None
    end_col = None
    for name in ("QUEUE_DT", "ELIGIBLE_DT", "ENQUEUE_DT", "ENQUEUED_AT", "QUEUED_AT"):
        if name in df.columns:
            queue_col = name
            break
    if not queue_col:
        queue_col = _pick_from_all(("queue", "eligible", "enq", "enqueue"))

    for name in ("START_DT", "STARTED_AT", "RUN_START_DT", "BEGIN_DT", "BEGIN_TS"):
        if name in df.columns:
            start_col = name
            break
    if not start_col:
        start_col = _pick_from_all(("start", "begin", "run_start"))

    for name in ("END_DT", "ENDED_AT", "FINISH_DT", "COMPLETE_DT", "DONE_DT"):
        if name in df.columns:
            end_col = name
            break
    if not end_col:
        end_col = _pick_from_all(("end", "finish", "complete", "done"))

    if derived_wait_col is None and queue_col and start_col:
        q = pd.to_datetime(df[queue_col], errors="coerce")
        s = pd.to_datetime(df[start_col], errors="coerce")
        valid = q.notna() & s.notna()
        # Don't require a high parse rate: many ERP logs have missing timestamps for subsets.
        # If there is enough evidence to compute wait on a non-trivial subset, do it.
        if int(valid.sum()) >= max(50, int(k_min * 2)):
            derived_wait_col = "__WAIT_TO_START_SEC"
            df = df.copy()
            df[derived_wait_col] = (s - q).dt.total_seconds()
            # If we didn't have a usable time column, the queue timestamp is the most
            # meaningful for burst/scheduling analyses.
            if not isinstance(time_col, str) or time_col not in df.columns:
                time_col = queue_col

    # Derive a run duration if needed (fallback target).
    derived_run_col = None
    for candidate in ("RUN_SEC", "DURATION_SEC", "PROC_SEC", "ELAPSED_SEC"):
        if candidate in df.columns:
            derived_run_col = candidate
            break
    if derived_run_col is None and start_col and end_col:
        s = pd.to_datetime(df[start_col], errors="coerce")
        e = pd.to_datetime(df[end_col], errors="coerce")
        valid = s.notna() & e.notna()
        if int(valid.sum()) >= max(50, int(k_min * 2)):
            derived_run_col = "__RUN_SEC"
            df = df.copy()
            df[derived_run_col] = (e - s).dt.total_seconds()

    # Identify a duration-like target.
    duration_col = config.get("duration_column")
    if not isinstance(duration_col, str) or duration_col not in df.columns:
        # Prefer queue-delay style targets if we derived them; else fall back to generic numeric pick.
        if isinstance(derived_wait_col, str) and derived_wait_col in df.columns:
            duration_col = derived_wait_col
        elif isinstance(derived_run_col, str) and derived_run_col in df.columns:
            duration_col = derived_run_col
        else:
            duration_col = _pick_duration_column(df, num_cols)
    if not duration_col:
        return PluginResult("skipped", "No numeric duration/target column detected", {}, [], [], None)

    process_col = config.get("process_column")
    if not isinstance(process_col, str) or process_col not in df.columns:
        # Use name tokens over all columns (not just inferred categoricals) because many process IDs
        # are numeric-coded and can be misclassified.
        for candidate in ("PROCESS_ID", "PROCESS", "ACTIVITY", "TASK", "JOB"):
            if candidate in df.columns:
                process_col = candidate
                break
        if not process_col:
            process_col = _pick_from_all(("process", "activity", "task", "action", "job", "step", "workflow"))
    if not process_col:
        # Without a process key we cannot make specific recommendations.
        return PluginResult("skipped", "No process/activity column detected", {}, [], [], None)

    server_col = config.get("server_column")
    if not isinstance(server_col, str) or server_col not in df.columns:
        for candidate in ("ASSIGNED_MACHINE_ID", "LOCAL_MACHINE_ID", "HOST", "SERVER"):
            if candidate in df.columns:
                server_col = candidate
                break
        if not server_col:
            server_col = _pick_from_all(("host", "server", "node", "instance", "machine"))

    # Parameter-like columns: bounded-cardinality categoricals excluding process/server.
    param_cols: list[str] = []
    # Start from inferred categoricals, but also allow numeric-coded dimensions.
    param_candidates = list(dict.fromkeys([*cat_cols, *all_cols]))
    for col in param_candidates:
        if col == process_col or col == server_col or col not in df.columns:
            continue
        if col == duration_col or col == time_col:
            continue
        vc = df[col].value_counts(dropna=False)
        if 2 <= int(vc.shape[0]) <= int(config.get("max_param_cardinality", 20)):
            param_cols.append(col)
        if len(param_cols) >= int(config.get("max_param_cols", 3)):
            break

    y = pd.to_numeric(df[duration_col], errors="coerce")
    proc = df[process_col].astype(str)
    over = (y - float(wait_threshold_seconds)).clip(lower=0.0)

    # Used to give batching/throttling levers a measured, citeable "potential" magnitude.
    # This is an upper bound: not all over-threshold time is necessarily recoverable.
    try:
        _over_sum_by_proc = over.groupby(proc, dropna=False).sum()
    except Exception:
        _over_sum_by_proc = pd.Series(dtype=float)
    metrics: dict[str, Any] = _basic_metrics(df, sample_meta)
    metrics.update(
        {
            "process_column": process_col,
            "server_column": server_col,
            "duration_column": duration_col,
            "time_column": time_col,
            "param_columns": param_cols,
            "wait_threshold_seconds": float(wait_threshold_seconds),
        }
    )

    # Observation window (helps turn "total seconds" into interpretable rates like "hours/day").
    obs_days: float | None = None
    if time_col and time_col in df.columns:
        parsed_all = pd.to_datetime(df[time_col], errors="coerce")
        if parsed_all.notna().mean() >= 0.7:
            try:
                t0 = parsed_all.min()
                t1 = parsed_all.max()
                if pd.notna(t0) and pd.notna(t1):
                    span_s = float((t1 - t0).total_seconds())
                    if math.isfinite(span_s) and span_s > 0:
                        obs_days = span_s / 86400.0
                        metrics["observation_start"] = str(t0)
                        metrics["observation_end"] = str(t1)
                        metrics["observation_days"] = float(obs_days)
            except Exception:
                obs_days = None

    findings: list[dict[str, Any]] = []
    artifacts: list[PluginArtifact] = []

    # Helper to emit a finding with consistent required fields.
    def emit(
        *,
        title: str,
        action_type: str,
        process_value: object,
        expected_delta_seconds: float | None,
        expected_delta_percent: float | None,
        confidence: float,
        assumptions: list[str],
        evidence: dict[str, Any],
        measurement_type: str = "measured",
        extra: dict[str, Any] | None = None,
    ) -> None:
        label = redactor(str(process_value))
        proc_norm = str(process_value).strip().lower()
        proc_id = stable_id([proc_norm], prefix="proc")

        impact_hours: float | None = None
        impact_hours_per_day: float | None = None
        if isinstance(expected_delta_seconds, (int, float)) and math.isfinite(float(expected_delta_seconds)):
            impact_hours = float(expected_delta_seconds) / 3600.0
            if obs_days and obs_days > 0:
                impact_hours_per_day = impact_hours / float(obs_days)

        def _fmt_hours(hours: float | None) -> str:
            if hours is None or not isinstance(hours, (int, float)) or not math.isfinite(float(hours)):
                return "unknown"
            return f"{float(hours):,.1f}h"

        # Plain-English recommendation text (preferred by report synthesis).
        recommendation: str | None = None
        thr = float(wait_threshold_seconds)
        if action_type == "route_process":
            frm = str((extra or {}).get("from") or "").strip()
            to = str((extra or {}).get("to") or "").strip()
            recommendation = (
                f"Shift some runs of {label} from {frm} to {to}. "
                f"Observed median gap implies up to {_fmt_hours(impact_hours)} total wait reduction over the observation window "
                f"(based on medians; not guaranteed). "
                "Validate by re-running and confirming the former worst host’s median queue-delay shrinks."
            )
        elif action_type == "unblock_dependency_chain":
            parent_raw = str((extra or {}).get("parent_process") or evidence.get("parent_process") or "").strip()
            child_raw = str((extra or {}).get("child_process") or evidence.get("child_process") or "").strip()
            parent = redactor(parent_raw) if parent_raw else "(unknown parent)"
            child = redactor(child_raw) if child_raw else label
            per_day = _fmt_hours(impact_hours_per_day) if impact_hours_per_day is not None else None
            per_day_txt = f" (~{per_day}/day)" if per_day and per_day != "unknown" else ""
            recommendation = (
                f"Investigate the specific dependency {parent} -> {child}. "
                f"Upper bound: {_fmt_hours(impact_hours)}{per_day_txt} of wait time above {thr:.0f}s attributable to this linkage in the observed data. "
                "Next: confirm the linkage subset in the dependency artifact, then decide whether to relax the dependency rule, allow overlap/parallelism, or reduce the upstream step duration."
            )
        elif action_type == "reschedule":
            avoid = (extra or {}).get("avoid_window") or {}
            target = (extra or {}).get("target_window") or {}
            recommendation = (
                f"Move {label} away from the high-delay window (hour={avoid.get('hour')}) toward the low-delay window (hour={target.get('hour')}). "
                f"Estimated upper bound: {_fmt_hours(impact_hours)} of wait reduction above {thr:.0f}s over the observation window. "
                "Validate by comparing median wait by hour after the schedule change."
            )
        elif action_type == "batch_or_cache":
            key = str((extra or {}).get("key") or evidence.get("param_column") or "").strip()
            per_day = _fmt_hours(impact_hours_per_day) if impact_hours_per_day is not None else None
            per_day_txt = f" (~{per_day}/day)" if per_day and per_day != "unknown" else ""
            recommendation = (
                f"For {label}, many runs repeat the same {key}. "
                f"Consider caching or batching by {key} to reduce queued work. "
                f"Upper bound: {_fmt_hours(impact_hours)}{per_day_txt} of wait time above {thr:.0f}s associated with this process over the window. "
                "Validate by measuring queue-delay distribution before/after batching and ensuring any SLAs still hold."
            )
        elif action_type == "throttle_or_dedupe":
            per_day = _fmt_hours(impact_hours_per_day) if impact_hours_per_day is not None else None
            per_day_txt = f" (~{per_day}/day)" if per_day and per_day != "unknown" else ""
            recommendation = (
                f"For {label}, bursty arrivals correlate with slower median wait/duration. "
                "Apply throttling or deduplication at the arrival point to smooth bursts. "
                f"Upper bound: {_fmt_hours(impact_hours)}{per_day_txt} of wait time above {thr:.0f}s. "
                "Validate by re-running and checking the burst-correlation and queue-delay distribution."
            )
        row = {
            "kind": "actionable_ops_lever",
            "measurement_type": measurement_type,
            "title": title,
            "recommendation": recommendation,
            "process": label,
            "process_norm": proc_norm,
            "process_id": proc_id,
            "action_type": action_type,
            "expected_delta_seconds": expected_delta_seconds,
            "expected_delta_percent": expected_delta_percent,
            "confidence": float(max(0.0, min(1.0, confidence))),
            "assumptions": assumptions,
            "evidence": evidence,
        }
        if extra:
            row.update(extra)
        findings.append(row)

    # ---- Routing (server imbalance/outlier) ----
    if server_col and server_col in df.columns:
        rows: list[dict[str, Any]] = []
        for pval, sub in df.assign(_y=y).groupby(proc, dropna=False):
            if timer.exceeded():
                break
            if len(sub) < (k_min * 2):
                continue
            by = sub.groupby(server_col)["_y"]
            med = by.median().sort_values(ascending=False)
            cnt = by.count().sort_values(ascending=False)
            eligible = [s for s in med.index if int(cnt.get(s, 0)) >= k_min]
            if len(eligible) < 2:
                continue
            worst = eligible[0]
            best = eligible[-1]
            worst_med = float(med[worst])
            best_med = float(med[best])
            delta = worst_med - best_med
            if not math.isfinite(delta) or delta <= 0:
                continue
            pct = (delta / worst_med) * 100.0 if worst_med else None
            rows.append(
                {
                    "process": str(pval),
                    "server_worst": redactor(str(worst)),
                    "server_best": redactor(str(best)),
                    "median_worst": worst_med,
                    "median_best": best_med,
                    "delta_seconds": float(delta),
                    "delta_percent": float(pct) if pct is not None else None,
                    "n_worst": int(cnt.get(worst, 0)),
                    "n_best": int(cnt.get(best, 0)),
                }
            )

        if rows:
            rows = sorted(rows, key=lambda r: float(r.get("delta_seconds") or 0.0), reverse=True)
            artifacts.append(_artifact(ctx, plugin_id, "routing_candidates.json", rows[:50], "json"))
            for row in rows[:5]:
                emit(
                    title=f"Route {row['process']} from {row['server_worst']} to {row['server_best']}",
                    action_type="route_process",
                    process_value=row["process"],
                    expected_delta_seconds=float(row["delta_seconds"]),
                    expected_delta_percent=float(row["delta_percent"]) if row.get("delta_percent") is not None else None,
                    confidence=min(1.0, 0.3 + 0.05 * min(row["n_worst"], row["n_best"])),
                    assumptions=[
                        "Server labels map to actual execution nodes (not client-origin).",
                        "The observed median gap persists after routing changes.",
                    ],
                    evidence={
                        "process_column": process_col,
                        "server_column": server_col,
                        "duration_column": duration_col,
                        "n_worst": row["n_worst"],
                        "n_best": row["n_best"],
                        "routing_candidates_artifact": str(artifacts[-1].path),
                    },
                    extra={
                        "from": row["server_worst"],
                        "to": row["server_best"],
                    },
                )

    # ---- Dependency chain hotspot (parent/dep relationships) ----
    # If a delayed process is frequently blocked behind a specific upstream process, surface that as a
    # sequence-of-processes recommendation.
    pqid_col = None
    for candidate in ("PROCESS_QUEUE_ID", "PROC_QUEUE_ID", "QUEUE_ID"):
        if candidate in df.columns:
            pqid_col = candidate
            break
    dep_cols = [c for c in ("DEP_PROCESS_QUEUE_ID", "PARENT_PROCESS_QUEUE_ID", "MASTER_PROCESS_QUEUE_ID") if c in df.columns]
    dep_col = None
    if dep_cols:
        # Pick the dependency column with the most evidence.
        dep_col = max(dep_cols, key=lambda c: int(pd.to_numeric(df[c], errors="coerce").notna().sum()))
    if pqid_col and dep_col and pqid_col in df.columns and dep_col in df.columns:
        if not timer.exceeded():
            pq = pd.to_numeric(df[pqid_col], errors="coerce")
            depq = pd.to_numeric(df[dep_col], errors="coerce")
            # Build a best-effort mapping from queue_id -> process_id for parent resolution.
            map_df = (
                pd.DataFrame({pqid_col: pq, "_proc": proc})
                .dropna(subset=[pqid_col, "_proc"])
                .drop_duplicates(subset=[pqid_col])
            )
            pairs = pd.DataFrame({"_child": proc, "_over": over, dep_col: depq}).dropna(subset=[dep_col])
            pairs = pairs.merge(map_df.rename(columns={pqid_col: dep_col, "_proc": "_parent"}), on=dep_col, how="left")
            pairs = pairs.dropna(subset=["_parent"])
            # Focus on top child processes by total over-threshold contribution to keep output tight.
            top_children = (
                pairs.groupby("_child")["_over"].sum().sort_values(ascending=False).head(int(config.get("max_dependency_children", 6))).index.tolist()
            )
            if top_children:
                pairs = pairs[pairs["_child"].isin(top_children)]
            if not pairs.empty:
                grp = pairs.groupby(["_child", "_parent"])["_over"].agg(["sum", "count", "median"]).reset_index()
                # Only consider meaningful evidence.
                k_min_dep = int(config.get("k_min_dependency", k_min))
                grp = grp[grp["count"] >= k_min_dep]
                if not grp.empty:
                    artifacts.append(
                        _artifact(
                            ctx,
                            plugin_id,
                            "dependency_hotspots.json",
                            grp.sort_values("sum", ascending=False).head(200).to_dict(orient="records"),
                            "json",
                        )
                    )
                    for child in grp["_child"].unique().tolist()[: int(config.get("max_dependency_actions", 5))]:
                        sub = grp[grp["_child"] == child].sort_values("sum", ascending=False).head(1)
                        if sub.empty:
                            continue
                        row = sub.iloc[0].to_dict()
                        parent = str(row.get("_parent") or "")
                        sum_over = float(row.get("sum") or 0.0)
                        n = int(row.get("count") or 0)
                        if not parent or sum_over <= 0.0:
                            continue
                        emit(
                            title=f"Unblock {child} when preceded by {parent}",
                            action_type="unblock_dependency_chain",
                            process_value=child,
                            # Upper-bound: if this parent->child blockage were eliminated entirely.
                            expected_delta_seconds=sum_over,
                            expected_delta_percent=None,
                            confidence=min(1.0, 0.25 + 0.02 * float(n)),
                            assumptions=[
                                f"{dep_col} links to an upstream {process_col} via {pqid_col}.",
                                "Impact is an upper-bound based on observed over-threshold wait in this linkage subset.",
                                "Fix may require adjusting dependency ordering, concurrency, or upstream completion time.",
                            ],
                            evidence={
                                "process_column": process_col,
                                "dependency_column": dep_col,
                                "queue_id_column": pqid_col,
                                "duration_column": duration_col,
                                "wait_threshold_seconds": float(wait_threshold_seconds),
                                "rows_with_dependency": int(len(pairs)),
                                "parent_process": parent,
                                "child_process": child,
                                "subset_runs": n,
                                "subset_over_threshold_wait_sec_total": sum_over,
                                "dependency_hotspots_artifact": str(artifacts[-1].path) if artifacts else None,
                            },
                            extra={
                                "parent_process": redactor(parent),
                                "child_process": redactor(str(child)),
                            },
                        )

    # ---- Scheduling (time bucket effects) ----
    if time_col and time_col in df.columns:
        parsed = pd.to_datetime(df[time_col], errors="coerce")
        if parsed.notna().mean() >= 0.7:
            tmp = df.assign(_t=parsed, _y=y).dropna(subset=["_t"])
            tmp = tmp.assign(_hour=tmp["_t"].dt.hour.astype(int))
            rows: list[dict[str, Any]] = []
            for pval, sub in tmp.groupby(process_col, dropna=False):
                if timer.exceeded():
                    break
                by = sub.groupby("_hour")["_y"]
                cnt = by.count()
                med = by.median()
                eligible_hours = [h for h in med.index if int(cnt.get(h, 0)) >= k_min]
                if len(eligible_hours) < 3:
                    continue
                worst_h = int(med.loc[eligible_hours].idxmax())
                best_h = int(med.loc[eligible_hours].idxmin())
                worst = float(med[worst_h])
                best = float(med[best_h])
                delta = worst - best
                if not math.isfinite(delta) or delta <= 0:
                    continue
                pct = (delta / worst) * 100.0 if worst else None
                rows.append(
                    {
                        "process": str(pval),
                        "worst_hour": worst_h,
                        "best_hour": best_h,
                        "median_worst": worst,
                        "median_best": best,
                        "delta_seconds": float(delta),
                        "delta_percent": float(pct) if pct is not None else None,
                        "n_worst": int(cnt.get(worst_h, 0)),
                        "n_best": int(cnt.get(best_h, 0)),
                    }
                )
            if rows:
                rows = sorted(rows, key=lambda r: float(r.get("delta_seconds") or 0.0), reverse=True)
                artifacts.append(_artifact(ctx, plugin_id, "schedule_candidates.json", rows[:50], "json"))
                for row in rows[:5]:
                    total_over = None
                    try:
                        total_over = float(_over_sum_by_proc.get(str(row["process"]), 0.0))
                    except Exception:
                        total_over = None
                    # Prefer a volume-aware estimate so cross-process ranking is meaningful.
                    # Use observed median delta times the smaller sample size as a conservative
                    # "swappable" count; cap by over-threshold total as an upper bound.
                    est_total = float(row["delta_seconds"]) * float(min(row["n_worst"], row["n_best"]))
                    if isinstance(total_over, (int, float)) and math.isfinite(float(total_over)):
                        est_total = min(est_total, float(total_over))
                    emit(
                        title=f"Reschedule {row['process']} away from hour {row['worst_hour']} to {row['best_hour']}",
                        action_type="reschedule",
                        process_value=row["process"],
                        expected_delta_seconds=float(est_total),
                        expected_delta_percent=float(row["delta_percent"]) if row.get("delta_percent") is not None else None,
                        confidence=min(1.0, 0.25 + 0.05 * min(row["n_worst"], row["n_best"])),
                        assumptions=[
                            "The time column reflects when work executes (not only when enqueued).",
                            "The hour-of-day pattern is stable across periods.",
                        ],
                        evidence={
                            "process_column": process_col,
                            "time_column": time_col,
                            "duration_column": duration_col,
                            "schedule_candidates_artifact": str(artifacts[-1].path),
                            "median_delta_seconds": float(row["delta_seconds"]),
                            "n_worst": int(row["n_worst"]),
                            "n_best": int(row["n_best"]),
                            "over_threshold_seconds_total_upper_bound": float(total_over) if isinstance(total_over, (int, float)) else None,
                        },
                        extra={
                            "avoid_window": {"hour": row["worst_hour"]},
                            "target_window": {"hour": row["best_hour"]},
                        },
                    )

    # ---- Batching/caching (low parameter diversity + high volume) ----
    if param_cols:
        rows: list[dict[str, Any]] = []
        for pval, sub in df.assign(_y=y).groupby(process_col, dropna=False):
            if timer.exceeded():
                break
            n = int(len(sub))
            if n < int(config.get("min_volume_for_batching", k_min * 5)):
                continue
            best_col = None
            best_score = 0.0
            best_vals: list[str] = []
            for col in param_cols:
                series = sub[col].dropna()
                if series.empty:
                    continue
                nunique = int(series.nunique())
                if nunique < 1:
                    continue
                diversity = float(nunique) / float(max(len(series), 1))
                score = (1.0 - diversity) * math.log1p(n)
                if score > best_score:
                    best_score = score
                    best_col = col
                    top_vals = series.value_counts().index[:3].tolist()
                    best_vals = [redactor(str(v)) for v in top_vals]
            if not best_col:
                continue
            if best_score < 2.0:
                continue
            rows.append(
                {
                    "process": str(pval),
                    "param_column": best_col,
                    "top_values": best_vals,
                    "volume": n,
                    "score": float(best_score),
                }
            )
        if rows:
            rows = sorted(rows, key=lambda r: float(r.get("score") or 0.0), reverse=True)
            artifacts.append(_artifact(ctx, plugin_id, "batching_candidates.json", rows[:50], "json"))
            for row in rows[:5]:
                total_over = None
                try:
                    total_over = float(_over_sum_by_proc.get(str(row["process"]), 0.0))
                except Exception:
                    total_over = None
                emit(
                    title=f"Batch/cache {row['process']} by {row['param_column']} (upper bound {float(total_over or 0.0)/3600.0:.1f}h)",
                    action_type="batch_or_cache",
                    process_value=row["process"],
                    expected_delta_seconds=float(total_over) if isinstance(total_over, (int, float)) else None,
                    expected_delta_percent=None,
                    confidence=min(1.0, 0.2 + 0.02 * float(row["volume"])),
                    assumptions=[
                        "Repeated parameter patterns imply cacheability or batchability.",
                        "Batching does not violate external SLAs.",
                    ],
                    evidence={
                        "process_column": process_col,
                        "param_column": row["param_column"],
                        "volume": int(row["volume"]),
                        "top_param_values_redacted": row["top_values"],
                        "batching_candidates_artifact": str(artifacts[-1].path),
                        "over_threshold_seconds_total_upper_bound": float(total_over) if isinstance(total_over, (int, float)) else None,
                    },
                    extra={"key": row["param_column"]},
                )

    # ---- Throttle/dedupe (bursts correlate with slowdown) ----
    if time_col and time_col in df.columns:
        parsed = pd.to_datetime(df[time_col], errors="coerce")
        if parsed.notna().mean() >= 0.7:
            tmp = df.assign(_t=parsed, _y=y).dropna(subset=["_t"])
            tmp = tmp.assign(_bin=tmp["_t"].dt.floor(str(config.get("burst_bin", "15min"))))
            rows: list[dict[str, Any]] = []
            for pval, sub in tmp.groupby(process_col, dropna=False):
                if timer.exceeded():
                    break
                grouped = sub.groupby("_bin")["_y"]
                counts = grouped.count()
                meds = grouped.median()
                if len(counts) < 6:
                    continue
                xs = counts.to_numpy(dtype=float)
                ys = meds.to_numpy(dtype=float)
                if np.isfinite(xs).sum() < 6 or np.isfinite(ys).sum() < 6:
                    continue
                # Correlation (deterministic).
                x0 = xs - float(np.nanmean(xs))
                y0 = ys - float(np.nanmean(ys))
                denom = float(np.linalg.norm(x0) * np.linalg.norm(y0))
                corr = float(np.dot(x0, y0) / denom) if denom > 0 else 0.0
                if corr < float(config.get("min_burst_corr", 0.5)):
                    continue
                rows.append(
                    {
                        "process": str(pval),
                        "corr": float(corr),
                        "bins": int(len(counts)),
                        "burst_bin": str(config.get("burst_bin", "15min")),
                    }
                )
            if rows:
                rows = sorted(rows, key=lambda r: float(r.get("corr") or 0.0), reverse=True)
                artifacts.append(_artifact(ctx, plugin_id, "burst_candidates.json", rows[:50], "json"))
                for row in rows[:5]:
                    total_over = None
                    try:
                        total_over = float(_over_sum_by_proc.get(str(row["process"]), 0.0))
                    except Exception:
                        total_over = None
                    emit(
                        title=f"Throttle/dedupe bursty arrivals for {row['process']}",
                        action_type="throttle_or_dedupe",
                        process_value=row["process"],
                        expected_delta_seconds=float(total_over) if isinstance(total_over, (int, float)) else None,
                        expected_delta_percent=None,
                        confidence=min(1.0, 0.2 + 0.8 * float(row["corr"])),
                        assumptions=[
                            "Time bins approximate arrival bursts meaningfully.",
                            "Burst mitigation is possible via scheduling/throttling controls.",
                        ],
                        evidence={
                            "process_column": process_col,
                            "time_column": time_col,
                            "duration_column": duration_col,
                            "corr_count_vs_median_duration": float(row["corr"]),
                            "burst_bin": row["burst_bin"],
                            "burst_candidates_artifact": str(artifacts[-1].path),
                            "over_threshold_seconds_total_upper_bound": float(total_over) if isinstance(total_over, (int, float)) else None,
                        },
                    )

    # Cap and rank findings to avoid overwhelming output.
    max_actions = int(config.get("max_actions", 20))
    if findings:
        findings = sorted(
            findings,
            key=lambda f: float(f.get("expected_delta_seconds") or 0.0),
            reverse=True,
        )[:max_actions]

    if not findings:
        return PluginResult(
            "ok",
            "No actionable levers detected under current heuristics/budgets",
            metrics,
            [],
            artifacts,
            None,
        )
    return PluginResult(
        "ok",
        f"Generated {len(findings)} actionable lever recommendation(s)",
        metrics,
        findings,
        artifacts,
        None,
    )


HANDLERS: dict[str, Callable[..., PluginResult]] = {
    # Family A
    "analysis_tda_persistent_homology": _tda_persistent_homology,
    "analysis_tda_persistence_landscapes": _tda_persistence_landscapes,
    "analysis_tda_mapper_graph": _tda_mapper_graph,
    "analysis_tda_betti_curve_changepoint": _tda_betti_curve_changepoint,
    # Family B
    "analysis_topographic_similarity_angle_projection": _topographic_similarity_angle_projection,
    "analysis_topographic_angle_dynamics": _topographic_angle_dynamics,
    "analysis_topographic_tanova_permutation": _topographic_tanova_permutation,
    "analysis_map_permutation_test_karniski": _map_permutation_test_karniski,
    # Family C
    "analysis_surface_multiscale_wavelet_curvature": _surface_multiscale_wavelet_curvature,
    "analysis_surface_fractal_dimension_variogram": _surface_fractal_dimension_variogram,
    "analysis_surface_rugosity_index": _surface_rugosity_index,
    "analysis_surface_terrain_position_index": _surface_terrain_position_index,
    "analysis_surface_fabric_sso_eigen": _surface_fabric_sso_eigen,
    "analysis_surface_hydrology_flow_watershed": _surface_hydrology_flow_watershed,
    # Family D
    "analysis_ttests_auto": _ttests_auto,
    "analysis_chi_square_association": _chi_square_association,
    "analysis_anova_auto": _anova_auto,
    "analysis_regression_auto": _regression_auto,
    "analysis_time_series_analysis_auto": _time_series_analysis_auto,
    "analysis_survival_time_to_event": _survival_time_to_event,
    "analysis_factor_analysis_auto": _factor_analysis_auto,
    "analysis_cluster_analysis_auto": _cluster_analysis_auto,
    "analysis_pca_auto": _pca_auto,
    # Family E
    "analysis_bayesian_point_displacement": _bayesian_point_displacement,
    "analysis_monte_carlo_surface_uncertainty": _monte_carlo_surface_uncertainty,
    "analysis_surface_roughness_metrics": _surface_roughness_metrics,
    # Family F
    "analysis_actionable_ops_levers_v1": _actionable_ops_levers_v1,
}
