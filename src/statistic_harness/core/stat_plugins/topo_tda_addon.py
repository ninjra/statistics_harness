from __future__ import annotations

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
from statistic_harness.core.utils import quote_identifier, write_json
from statistic_harness.core.process_matcher import (
    compile_patterns,
    default_exclude_process_patterns,
    merge_patterns,
    parse_exclude_patterns_env,
)

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
        try:
            chi2, p, _, _ = scipy_stats.chi2_contingency(table)
            return float(p), float(chi2)
        except ValueError:
            # Some sparse categorical cross-tabs can trigger SciPy's
            # "expected frequencies has a zero element" path. Treat this
            # deterministically as non-significant instead of failing plugin.
            return 1.0, 0.0
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
    max_levels_per_col = int(config.get("max_unique_levels_per_column", 2000))
    max_cells = int(config.get("max_contingency_cells", 500_000))
    pvals: list[float] = []
    rows: list[dict[str, Any]] = []
    eligible_level_counts: dict[str, int] = {}
    skipped_large_pairs = 0
    skipped_crosstab_alloc = 0
    for col in cols:
        try:
            vc = df[col].value_counts(dropna=False)
            eligible = int((vc >= k_min).sum())
        except Exception:
            eligible = 0
        if eligible > 0:
            eligible_level_counts[col] = eligible
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if timer.exceeded() or len(rows) >= max_pairs:
                break
            a, b = cols[i], cols[j]
            levels_a = int(eligible_level_counts.get(a, 0))
            levels_b = int(eligible_level_counts.get(b, 0))
            if levels_a > max_levels_per_col or levels_b > max_levels_per_col:
                skipped_large_pairs += 1
                continue
            if levels_a > 0 and levels_b > 0 and (levels_a * levels_b) > max_cells:
                skipped_large_pairs += 1
                continue
            try:
                tab = pd.crosstab(df[a], df[b], dropna=False)
            except MemoryError:
                skipped_crosstab_alloc += 1
                continue
            except ValueError as exc:
                if "Unable to allocate" in str(exc):
                    skipped_crosstab_alloc += 1
                    continue
                raise
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
        metrics = _basic_metrics(df, sample_meta)
        metrics["skipped_large_pairs"] = int(skipped_large_pairs)
        metrics["skipped_crosstab_alloc"] = int(skipped_crosstab_alloc)
        return PluginResult(
            "ok",
            "No chi-square pairs eligible under k-min/budget",
            metrics,
            [],
            [],
            None,
        )

    qvals, _ = bh_fdr(pvals)
    findings: list[dict[str, Any]] = []
    for row, q in zip(rows, qvals):
        row["q_value"] = float(q)
        if float(q) > float(config.get("fdr_q", 0.1)):
            continue
        findings.append({"kind": "chi_square_association", "measurement_type": "measured", **row})
    artifacts = [_artifact(ctx, plugin_id, "chi_square_association.json", {"rows": rows}, "json")]
    summary = (
        f"Chi-square: computed={len(rows)} significant={len(findings)} "
        f"skipped_large_pairs={skipped_large_pairs} skipped_crosstab_alloc={skipped_crosstab_alloc}"
    )
    metrics = _basic_metrics(df, sample_meta)
    metrics["skipped_large_pairs"] = int(skipped_large_pairs)
    metrics["skipped_crosstab_alloc"] = int(skipped_crosstab_alloc)
    return PluginResult("ok", summary, metrics, findings, artifacts, None)


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

    measurement_type = "modeled" if HAS_SCIPY else "measured"
    finding: dict[str, Any] = {
        "kind": "regression_auto_drivers",
        "measurement_type": measurement_type,
        "metric": value_col,
        "drivers": top,
    }
    if measurement_type == "modeled":
        finding["scope"] = {"plugin_id": plugin_id, "metric": value_col}
        finding["assumptions"] = [
            "Predictor effects are approximately linear in the selected feature space.",
            "Observed historical relationships remain directionally stable over the modeled horizon.",
        ]
    findings = [finding]
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
        pca.fit(X)
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
                "levels": [_safe_group_value(level_value, redactor) for level_value in levels],
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
# Family A/C/E: topology + surface + bayesian methods
# ---------------------------


def _heavy_disabled(plugin_id: str, config: dict[str, Any]) -> bool:
    # Backward-compatible switch:
    # - If enable_heavy is present, honor it.
    # - Otherwise run methods by default unless disable_heavy=true.
    if "enable_heavy" in config:
        return not bool(config.get("enable_heavy"))
    return bool(config.get("disable_heavy", False))


def _heavy_placeholder(kind: str, msg: str) -> list[dict[str, Any]]:
    return [{"kind": kind, "measurement_type": "degraded", "reason": msg}]


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _knn_edges(X: np.ndarray, k: int, timer: BudgetTimer) -> list[tuple[int, int, float]]:
    n = int(X.shape[0])
    if n <= 1:
        return []
    k = max(1, min(int(k), n - 1))
    edges_map: dict[tuple[int, int], float] = {}

    if HAS_SKLEARN and NearestNeighbors is not None:
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
        nn.fit(X)
        dists, neigh = nn.kneighbors(X, return_distance=True)
        for i in range(n):
            if timer.exceeded():
                break
            for j_idx in range(1, neigh.shape[1]):
                j = int(neigh[i, j_idx])
                if i == j:
                    continue
                a, b = (i, j) if i < j else (j, i)
                dist = float(dists[i, j_idx])
                prev = edges_map.get((a, b))
                if prev is None or dist < prev:
                    edges_map[(a, b)] = dist
    else:
        # Fallback without sklearn: bounded brute-force distance matrix.
        gram = np.sum(X * X, axis=1)
        d2 = gram[:, None] + gram[None, :] - (2.0 * np.dot(X, X.T))
        d2 = np.maximum(d2, 0.0)
        np.fill_diagonal(d2, np.inf)
        for i in range(n):
            if timer.exceeded():
                break
            idx = np.argpartition(d2[i], k)[:k]
            for j in idx:
                jj = int(j)
                if i == jj:
                    continue
                a, b = (i, jj) if i < jj else (jj, i)
                dist = float(math.sqrt(float(d2[i, jj])))
                prev = edges_map.get((a, b))
                if prev is None or dist < prev:
                    edges_map[(a, b)] = dist

    edges = [(a, b, d) for (a, b), d in edges_map.items()]
    edges.sort(key=lambda row: row[2])
    return edges


def _union_find(n: int) -> tuple[list[int], list[int]]:
    return list(range(n)), [0] * n


def _uf_find(parent: list[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _uf_union(parent: list[int], rank: list[int], a: int, b: int) -> bool:
    ra = _uf_find(parent, a)
    rb = _uf_find(parent, b)
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


def _h0_from_edges(n: int, edges: list[tuple[int, int, float]], timer: BudgetTimer) -> tuple[list[float], float]:
    parent, rank = _union_find(n)
    lifetimes: list[float] = []
    mst_len = 0.0
    merges = 0
    for i, j, dist in edges:
        if timer.exceeded():
            break
        if _uf_union(parent, rank, int(i), int(j)):
            lifetimes.append(float(dist))
            mst_len += float(dist)
            merges += 1
            if merges >= n - 1:
                break
    return lifetimes, mst_len


def _connected_components_from_edges(n: int, edges: list[tuple[int, int, float]]) -> int:
    parent, rank = _union_find(n)
    for i, j, _ in edges:
        _uf_union(parent, rank, int(i), int(j))
    roots = {_uf_find(parent, i) for i in range(n)}
    return int(len(roots))


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
        return PluginResult("degraded", "Heavy TDA disabled by config", _basic_metrics(df, sample_meta), _heavy_placeholder("tda_persistent_homology", "disabled_by_config"), [], None)
    numeric_cols = inferred.get("numeric_columns") or []
    if len(numeric_cols) < 2:
        return PluginResult("skipped", "Need multi-numeric columns", {}, [], [], None)
    max_points = int(config.get("max_points", 1200))
    X, cols = _numeric_matrix(df.head(max_points), numeric_cols, max_cols=int(config.get("max_cols", 20)))
    if X.shape[0] < 50:
        return PluginResult("skipped", "Insufficient points for TDA proxy", {}, [], [], None)
    edges = _knn_edges(X, k=int(config.get("knn_k", 10)), timer=timer)
    if not edges:
        return PluginResult("skipped", "No graph edges produced", {}, [], [], None)
    lifetimes, mst_len = _h0_from_edges(int(X.shape[0]), edges, timer)
    comps = _connected_components_from_edges(int(X.shape[0]), edges)
    V = int(X.shape[0])
    E = int(len(edges))
    cycle_rank = max(0, E - V + comps)
    life = np.array(lifetimes, dtype=float) if lifetimes else np.array([], dtype=float)

    finding = {
        "kind": "tda_persistent_homology",
        "measurement_type": "measured",
        "max_points": int(X.shape[0]),
        "mst_total_length": float(mst_len),
        "cycle_rank_proxy": int(cycle_rank),
        "h0_lifetime_p50": float(np.nanmedian(life)) if life.size else 0.0,
        "h0_lifetime_p90": float(np.nanquantile(life, 0.9)) if life.size else 0.0,
        "features_used": cols,
    }
    artifacts = [_artifact(ctx, plugin_id, "tda_proxy.json", finding, "json")]
    return PluginResult("ok", "Persistent homology proxy metrics computed", _basic_metrics(df, sample_meta), [finding], artifacts, None)


def _tda_persistence_landscapes(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    if _heavy_disabled(plugin_id, config):
        return PluginResult("degraded", "Heavy TDA disabled by config", _basic_metrics(df, sample_meta), _heavy_placeholder("tda_persistence_landscapes", "disabled_by_config"), [], None)
    numeric_cols = inferred.get("numeric_columns") or []
    if len(numeric_cols) < 2:
        return PluginResult("skipped", "Need multi-numeric columns", {}, [], [], None)
    X, cols = _numeric_matrix(
        df.head(int(config.get("max_points", 1200))),
        numeric_cols,
        max_cols=int(config.get("max_cols", 20)),
    )
    if X.shape[0] < 50:
        return PluginResult("skipped", "Insufficient points for landscapes proxy", {}, [], [], None)
    edges = _knn_edges(X, k=int(config.get("knn_k", 10)), timer=timer)
    lifetimes, _ = _h0_from_edges(int(X.shape[0]), edges, timer)
    if not lifetimes:
        return PluginResult("skipped", "Insufficient topology signal", {}, [], [], None)
    life = np.array(lifetimes, dtype=float)
    t_max = float(np.nanmax(life))
    grid_n = max(10, int(config.get("landscape_grid", 40)))
    t_grid = np.linspace(0.0, t_max, grid_n)
    l1 = []
    l2 = []
    for t in t_grid:
        tri = np.minimum(t, life - t)
        tri = np.where(tri > 0.0, tri, 0.0)
        tri.sort()
        l1.append(float(tri[-1]) if tri.size else 0.0)
        l2.append(float(tri[-2]) if tri.size > 1 else 0.0)
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    finding = {
        "kind": "tda_persistence_landscapes",
        "measurement_type": "measured",
        "max_points": int(X.shape[0]),
        "features_used": cols,
        "l1_peak": float(max(l1) if l1 else 0.0),
        "l2_peak": float(max(l2) if l2 else 0.0),
        "l1_area": float(integrate(np.array(l1, dtype=float), t_grid)),
        "l2_area": float(integrate(np.array(l2, dtype=float), t_grid)),
        "h0_lifetime_p95": float(np.nanquantile(life, 0.95)),
    }
    payload = {"t_grid": t_grid.tolist(), "lambda_1": l1, "lambda_2": l2, "summary": finding}
    artifacts = [_artifact(ctx, plugin_id, "persistence_landscapes.json", payload, "json")]
    return PluginResult("ok", "Persistence landscape proxy computed", _basic_metrics(df, sample_meta), [finding], artifacts, None)


def _tda_mapper_graph(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    if _heavy_disabled(plugin_id, config):
        return PluginResult("degraded", "Heavy mapper disabled by config", _basic_metrics(df, sample_meta), _heavy_placeholder("tda_mapper_graph", "disabled_by_config"), [], None)
    numeric_cols = inferred.get("numeric_columns") or []
    if len(numeric_cols) < 2:
        return PluginResult("skipped", "Need multi-numeric columns", {}, [], [], None)
    X, cols = _numeric_matrix(
        df.head(int(config.get("max_points", 1200))),
        numeric_cols,
        max_cols=int(config.get("max_cols", 20)),
    )
    if X.shape[0] < 80:
        return PluginResult("skipped", "Insufficient points for mapper graph", {}, [], [], None)

    seed = int(config.get("seed", 1337))
    n_intervals = max(4, int(config.get("n_intervals", 8)))
    overlap = float(config.get("overlap", 0.3))
    overlap = min(0.75, max(0.0, overlap))
    min_cluster_size = max(10, int(config.get("min_cluster_size", 30)))
    max_clusters = max(1, int(config.get("max_clusters", 3)))

    if HAS_SKLEARN and PCA is not None:
        lens = PCA(n_components=1, random_state=seed).fit_transform(X)[:, 0]
    else:
        lens = X[:, 0]
    lo = float(np.nanmin(lens))
    hi = float(np.nanmax(lens))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return PluginResult("skipped", "Lens is degenerate", {}, [], [], None)

    span = hi - lo
    width = span / float(n_intervals)
    pad = width * overlap
    intervals: list[tuple[float, float]] = []
    for i in range(n_intervals):
        start = lo + (i * width) - pad
        end = lo + ((i + 1) * width) + pad
        intervals.append((start, end))

    nodes: list[dict[str, Any]] = []
    node_points: list[set[int]] = []
    point_to_nodes: dict[int, list[int]] = {}
    for interval_id, (start, end) in enumerate(intervals):
        if timer.exceeded():
            break
        mask = np.logical_and(lens >= start, lens <= end)
        idx = np.where(mask)[0]
        if idx.size < min_cluster_size:
            continue
        sub_x = X[idx]
        k = min(max_clusters, max(1, int(idx.size // min_cluster_size)))
        if k <= 1:
            labels = np.zeros(idx.size, dtype=int)
        elif HAS_SKLEARN and KMeans is not None:
            km = KMeans(n_clusters=k, random_state=seed, n_init=10)
            labels = km.fit_predict(sub_x)
        else:
            rank = np.argsort(lens[idx])
            labels = np.zeros(idx.size, dtype=int)
            chunk = max(1, int(math.ceil(idx.size / k)))
            for ridx, pos in enumerate(rank):
                labels[pos] = min(k - 1, ridx // chunk)
        for cluster_id in sorted(set(int(v) for v in labels.tolist())):
            cluster_points = idx[labels == cluster_id]
            if cluster_points.size < min_cluster_size:
                continue
            node_id = len(nodes)
            node_set = {int(v) for v in cluster_points.tolist()}
            node_points.append(node_set)
            for p in node_set:
                point_to_nodes.setdefault(int(p), []).append(node_id)
            nodes.append(
                {
                    "node_id": node_id,
                    "interval_id": int(interval_id),
                    "cluster_id": int(cluster_id),
                    "n_points": int(cluster_points.size),
                    "lens_min": float(np.nanmin(lens[cluster_points])),
                    "lens_max": float(np.nanmax(lens[cluster_points])),
                }
            )

    if not nodes:
        return PluginResult("skipped", "No mapper nodes survived thresholds", {}, [], [], None)

    edges_set: set[tuple[int, int]] = set()
    for node_ids in point_to_nodes.values():
        if len(node_ids) < 2:
            continue
        uniq = sorted(set(node_ids))
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                edges_set.add((uniq[i], uniq[j]))
    edges = sorted(list(edges_set))
    degree = [0] * len(nodes)
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1

    parent, rank = _union_find(len(nodes))
    for a, b in edges:
        _uf_union(parent, rank, int(a), int(b))
    n_components = len({_uf_find(parent, i) for i in range(len(nodes))})
    top_nodes = sorted(
        [
            {"node_id": int(i), "degree": int(degree[i]), "n_points": int(nodes[i]["n_points"])}
            for i in range(len(nodes))
        ],
        key=lambda row: (row["degree"], row["n_points"]),
        reverse=True,
    )[:5]

    finding = {
        "kind": "tda_mapper_graph",
        "measurement_type": "measured",
        "nodes": int(len(nodes)),
        "edges": int(len(edges)),
        "components": int(n_components),
        "max_degree": int(max(degree) if degree else 0),
        "features_used": cols,
        "top_nodes": top_nodes,
    }
    payload = {"nodes": nodes, "edges": [{"a": int(a), "b": int(b)} for a, b in edges], "summary": finding}
    artifacts = [_artifact(ctx, plugin_id, "mapper_graph.json", payload, "json")]
    return PluginResult("ok", "Mapper-style graph computed", _basic_metrics(df, sample_meta), [finding], artifacts, None)


def _tda_betti_curve_changepoint(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    if _heavy_disabled(plugin_id, config):
        return PluginResult("degraded", "Heavy betti curves disabled by config", _basic_metrics(df, sample_meta), _heavy_placeholder("tda_betti_curve_changepoint", "disabled_by_config"), [], None)
    numeric_cols = inferred.get("numeric_columns") or []
    time_col = inferred.get("time_column")
    if time_col is None:
        time_col = _pick_column_by_tokens(df, ("time", "date", "timestamp", "created", "start"))
    if len(numeric_cols) < 2 or not isinstance(time_col, str) or time_col not in df.columns:
        return PluginResult("skipped", "Need multi-numeric columns and time column", {}, [], [], None)
    ts = pd.to_datetime(df[time_col], errors="coerce")
    valid = ts.notna()
    if int(valid.sum()) < 120:
        return PluginResult("skipped", "Insufficient valid timestamps", {}, [], [], None)
    order = np.argsort(ts[valid].to_numpy())
    dff = df.loc[valid].iloc[order]
    tsv = ts[valid].to_numpy()[order]

    window = max(60, int(config.get("window_rows", 200)))
    step = max(20, int(config.get("step_rows", max(40, window // 4))))
    if len(dff) < window:
        return PluginResult("skipped", "Insufficient rows for betti windows", {}, [], [], None)
    k = int(config.get("knn_k", 8))
    rows = []
    for start in range(0, len(dff) - window + 1, step):
        if timer.exceeded():
            break
        seg = dff.iloc[start : start + window]
        X, _ = _numeric_matrix(seg, numeric_cols, max_cols=int(config.get("max_cols", 15)))
        if X.shape[0] < 30:
            continue
        edges = _knn_edges(X, k=k, timer=timer)
        if not edges:
            continue
        comps = _connected_components_from_edges(int(X.shape[0]), edges)
        cycle_rank = max(0, int(len(edges)) - int(X.shape[0]) + comps)
        rows.append(
            {
                "t_mid": str(pd.Timestamp(tsv[start + (window // 2)]).to_pydatetime()),
                "betti0_proxy": int(comps),
                "betti1_proxy": int(cycle_rank),
                "window_rows": int(X.shape[0]),
            }
        )
    if len(rows) < 5:
        return PluginResult("skipped", "Insufficient betti windows", _basic_metrics(df, sample_meta), [], [], None)

    b1 = np.array([float(r["betti1_proxy"]) for r in rows], dtype=float)
    jump = np.abs(np.diff(b1))
    if jump.size == 0:
        return PluginResult("ok", "Betti curve stable", _basic_metrics(df, sample_meta), [], [], None)
    med = float(np.nanmedian(jump))
    mad = float(np.nanmedian(np.abs(jump - med)))
    scale = max(1e-6, 1.4826 * mad)
    z = (jump - med) / scale
    z_threshold = float(config.get("changepoint_z", 3.0))
    findings = []
    for i, score in enumerate(z):
        if float(score) < z_threshold:
            continue
        findings.append(
            {
                "kind": "tda_betti_curve_changepoint",
                "measurement_type": "measured",
                "t_at": rows[i + 1]["t_mid"],
                "jump_betti1": float(jump[i]),
                "z_score": float(score),
                "betti1_before": float(b1[i]),
                "betti1_after": float(b1[i + 1]),
            }
        )
    artifacts = [_artifact(ctx, plugin_id, "betti_curve_windows.json", {"rows": rows, "jump_z": z.tolist()}, "json")]
    return PluginResult("ok", f"Betti windows={len(rows)} changepoints={len(findings)}", _basic_metrics(df, sample_meta), findings, artifacts, None)


def _surface_multiscale_wavelet_curvature(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    if _heavy_disabled(plugin_id, config):
        return PluginResult("degraded", "Surface complexity disabled by config", _basic_metrics(df, sample_meta), _heavy_placeholder("surface_multiscale_wavelet_curvature", "disabled_by_config"), [], None)
    numeric_cols = inferred.get("numeric_columns") or []
    if not numeric_cols:
        return PluginResult("skipped", "Need numeric columns", {}, [], [], None)
    cols = [c for c in numeric_cols if c in df.columns][: int(config.get("max_cols", 12))]
    if not cols:
        return PluginResult("skipped", "No numeric columns selected", {}, [], [], None)
    scales = [int(v) for v in config.get("scales", [1, 2, 4, 8, 16]) if int(v) > 0]
    rows = []
    for col in cols:
        if timer.exceeded():
            break
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(x).sum() < 40:
            continue
        med = float(np.nanmedian(x))
        x = np.where(np.isfinite(x), x, med)
        scale_rows = []
        for s in scales:
            if (2 * s + 1) >= x.size:
                continue
            d2 = x[2 * s :] - (2.0 * x[s:-s]) + x[: -2 * s]
            curv = float(np.nanmean(np.abs(d2)) / max(1.0, float(s * s)))
            energy = float(np.nanmean(d2 * d2))
            scale_rows.append({"scale": int(s), "curvature_mean_abs": curv, "energy": energy})
        if not scale_rows:
            continue
        mean_curv = float(np.nanmean([r["curvature_mean_abs"] for r in scale_rows]))
        rows.append({"column": str(col), "mean_multiscale_curvature": mean_curv, "scales": scale_rows})
    if not rows:
        return PluginResult("skipped", "No valid surface curvature rows", _basic_metrics(df, sample_meta), [], [], None)
    rows.sort(key=lambda r: float(r["mean_multiscale_curvature"]), reverse=True)
    findings = [
        {
            "kind": "surface_multiscale_wavelet_curvature",
            "measurement_type": "measured",
            "column": row["column"],
            "mean_multiscale_curvature": row["mean_multiscale_curvature"],
        }
        for row in rows[:5]
    ]
    artifacts = [_artifact(ctx, plugin_id, "surface_multiscale_curvature.json", {"rows": rows}, "json")]
    return PluginResult("ok", f"Surface curvature computed for {len(rows)} columns", _basic_metrics(df, sample_meta), findings, artifacts, None)


def _surface_fractal_dimension_variogram(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    if _heavy_disabled(plugin_id, config):
        return PluginResult("degraded", "Surface complexity disabled by config", _basic_metrics(df, sample_meta), _heavy_placeholder("surface_fractal_dimension_variogram", "disabled_by_config"), [], None)
    numeric_cols = inferred.get("numeric_columns") or []
    cols = [c for c in numeric_cols if c in df.columns][: int(config.get("max_cols", 10))]
    if not cols:
        return PluginResult("skipped", "Need numeric columns", {}, [], [], None)
    max_lag = max(4, int(config.get("max_lag", 24)))
    rows = []
    for col in cols:
        if timer.exceeded():
            break
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(x).sum() < 50:
            continue
        med = float(np.nanmedian(x))
        x = np.where(np.isfinite(x), x, med)
        lags = []
        semivars = []
        cap = min(max_lag, max(4, (x.size // 4)))
        for h in range(1, cap + 1):
            diff = x[h:] - x[:-h]
            gamma = 0.5 * float(np.nanmean(diff * diff))
            if gamma > 0.0 and math.isfinite(gamma):
                lags.append(float(h))
                semivars.append(float(gamma))
        if len(lags) < 4:
            continue
        lx = np.log(np.array(lags, dtype=float))
        ly = np.log(np.array(semivars, dtype=float))
        slope, intercept = np.polyfit(lx, ly, 1)
        hurst = float(max(0.0, min(1.0, slope / 2.0)))
        fractal_dim = float(2.0 - hurst)
        rows.append(
            {
                "column": str(col),
                "fractal_dimension": fractal_dim,
                "hurst_proxy": hurst,
                "slope": float(slope),
                "intercept": float(intercept),
            }
        )
    if not rows:
        return PluginResult("skipped", "Insufficient data for variogram dimension", _basic_metrics(df, sample_meta), [], [], None)
    rows.sort(key=lambda r: float(r["fractal_dimension"]), reverse=True)
    findings = [
        {
            "kind": "surface_fractal_dimension_variogram",
            "measurement_type": "measured",
            "column": row["column"],
            "fractal_dimension": row["fractal_dimension"],
            "hurst_proxy": row["hurst_proxy"],
        }
        for row in rows[:5]
    ]
    artifacts = [_artifact(ctx, plugin_id, "surface_variogram_dimension.json", {"rows": rows}, "json")]
    return PluginResult("ok", f"Variogram fractal metrics columns={len(rows)}", _basic_metrics(df, sample_meta), findings, artifacts, None)


def _surface_rugosity_index(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    if _heavy_disabled(plugin_id, config):
        return PluginResult("degraded", "Surface complexity disabled by config", _basic_metrics(df, sample_meta), _heavy_placeholder("surface_rugosity_index", "disabled_by_config"), [], None)
    numeric_cols = inferred.get("numeric_columns") or []
    cols = [c for c in numeric_cols if c in df.columns][: int(config.get("max_cols", 12))]
    if not cols:
        return PluginResult("skipped", "Need numeric columns", {}, [], [], None)
    rows = []
    for col in cols:
        if timer.exceeded():
            break
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(x).sum() < 30:
            continue
        med = float(np.nanmedian(x))
        x = np.where(np.isfinite(x), x, med)
        diff = np.diff(x)
        if diff.size == 0:
            continue
        robust_range = float(np.nanquantile(x, 0.95) - np.nanquantile(x, 0.05))
        robust_range = robust_range if robust_range > 1e-9 else 1.0
        rugosity = float(np.sum(np.abs(diff)) / (robust_range * max(1, x.size - 1)))
        local = float(np.nanmedian(np.abs(diff)) / robust_range)
        rows.append({"column": str(col), "rugosity_index": rugosity, "local_roughness": local})
    if not rows:
        return PluginResult("skipped", "No valid rugosity metrics", _basic_metrics(df, sample_meta), [], [], None)
    rows.sort(key=lambda r: float(r["rugosity_index"]), reverse=True)
    findings = [
        {
            "kind": "surface_rugosity_index",
            "measurement_type": "measured",
            "column": row["column"],
            "rugosity_index": row["rugosity_index"],
            "local_roughness": row["local_roughness"],
        }
        for row in rows[:5]
    ]
    artifacts = [_artifact(ctx, plugin_id, "surface_rugosity.json", {"rows": rows}, "json")]
    return PluginResult("ok", f"Rugosity computed for {len(rows)} columns", _basic_metrics(df, sample_meta), findings, artifacts, None)


def _surface_terrain_position_index(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    if _heavy_disabled(plugin_id, config):
        return PluginResult("degraded", "Surface complexity disabled by config", _basic_metrics(df, sample_meta), _heavy_placeholder("surface_terrain_position_index", "disabled_by_config"), [], None)
    numeric_cols = inferred.get("numeric_columns") or []
    cols = [c for c in numeric_cols if c in df.columns][: int(config.get("max_cols", 12))]
    if not cols:
        return PluginResult("skipped", "Need numeric columns", {}, [], [], None)
    window = max(5, int(config.get("window", 15)))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / float(window)
    rows = []
    for col in cols:
        if timer.exceeded():
            break
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(x).sum() < window + 5:
            continue
        med = float(np.nanmedian(x))
        x = np.where(np.isfinite(x), x, med)
        smooth = np.convolve(x, kernel, mode="same")
        tpi = x - smooth
        abs_tpi = np.abs(tpi)
        q90 = float(np.nanquantile(abs_tpi, 0.9))
        rows.append(
            {
                "column": str(col),
                "mean_abs_tpi": float(np.nanmean(abs_tpi)),
                "peak_share": float(np.nanmean(abs_tpi >= q90)) if q90 > 0 else 0.0,
                "q90_abs_tpi": q90,
            }
        )
    if not rows:
        return PluginResult("skipped", "No terrain position metrics produced", _basic_metrics(df, sample_meta), [], [], None)
    rows.sort(key=lambda r: float(r["mean_abs_tpi"]), reverse=True)
    findings = [
        {
            "kind": "surface_terrain_position_index",
            "measurement_type": "measured",
            "column": row["column"],
            "mean_abs_tpi": row["mean_abs_tpi"],
            "peak_share": row["peak_share"],
        }
        for row in rows[:5]
    ]
    artifacts = [_artifact(ctx, plugin_id, "surface_tpi.json", {"rows": rows}, "json")]
    return PluginResult("ok", f"Terrain position computed for {len(rows)} columns", _basic_metrics(df, sample_meta), findings, artifacts, None)


def _surface_fabric_sso_eigen(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    if _heavy_disabled(plugin_id, config):
        return PluginResult("degraded", "Surface fabric disabled by config", _basic_metrics(df, sample_meta), _heavy_placeholder("surface_fabric_sso_eigen", "disabled_by_config"), [], None)
    numeric_cols = inferred.get("numeric_columns") or []
    X, cols = _numeric_matrix(
        df.head(int(config.get("max_points", 2500))),
        numeric_cols,
        max_cols=int(config.get("max_cols", 12)),
    )
    if X.shape[0] < 60 or X.shape[1] < 2:
        return PluginResult("skipped", "Need at least 60 rows and 2 numeric columns", {}, [], [], None)
    cov = np.cov(X, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    eigvals = np.maximum(eigvals, 0.0)
    total = float(np.sum(eigvals))
    if total <= 0.0:
        return PluginResult("skipped", "Degenerate covariance for fabric eigen analysis", {}, [], [], None)
    lam = eigvals / total
    anisotropy = float((lam[0] - lam[1]) / max(1e-9, float(np.sum(lam[:2]))))
    planarity = float((lam[1] - lam[2]) / max(1e-9, float(np.sum(lam[:3])))) if lam.size >= 3 else 0.0
    principal = eigvecs[:, 0]
    top_load = sorted(
        [{"column": cols[i], "loading_abs": float(abs(principal[i]))} for i in range(len(cols))],
        key=lambda row: row["loading_abs"],
        reverse=True,
    )[:5]
    finding = {
        "kind": "surface_fabric_sso_eigen",
        "measurement_type": "measured",
        "anisotropy_index": anisotropy,
        "planarity_index": planarity,
        "eigenvalue_ratio_1": float(lam[0]),
        "eigenvalue_ratio_2": float(lam[1]) if lam.size > 1 else 0.0,
        "top_principal_loadings": top_load,
    }
    artifacts = [_artifact(ctx, plugin_id, "surface_fabric_eigen.json", finding, "json")]
    return PluginResult("ok", "Surface fabric eigen-analysis computed", _basic_metrics(df, sample_meta), [finding], artifacts, None)


def _surface_hydrology_flow_watershed(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    if _heavy_disabled(plugin_id, config):
        return PluginResult("degraded", "Hydrology operations disabled by config", _basic_metrics(df, sample_meta), _heavy_placeholder("surface_hydrology_flow_watershed", "disabled_by_config"), [], None)
    numeric_cols = inferred.get("numeric_columns") or []
    cols = [c for c in numeric_cols if c in df.columns][: int(config.get("max_cols", 6))]
    if len(cols) < 3:
        return PluginResult("skipped", "Need at least 3 numeric columns for flow/watershed proxy", {}, [], [], None)
    frame = df[cols].head(int(config.get("max_points", 2000))).apply(pd.to_numeric, errors="coerce")
    frame = frame.fillna(frame.median(numeric_only=True))
    arr = frame.to_numpy(dtype=float)
    if arr.shape[0] < 80:
        return PluginResult("skipped", "Insufficient rows for watershed proxy", {}, [], [], None)
    elev = arr[:, 2]
    k = max(3, int(config.get("knn_k", 8)))
    edges = _knn_edges(arr[:, :2], k=k, timer=timer)
    nbrs: dict[int, list[tuple[int, float]]] = {}
    for a, b, d in edges:
        nbrs.setdefault(int(a), []).append((int(b), float(d)))
        nbrs.setdefault(int(b), []).append((int(a), float(d)))

    downstream = [-1] * arr.shape[0]
    for i in range(arr.shape[0]):
        if timer.exceeded():
            break
        best_j = i
        best_drop = 0.0
        for j, dist in nbrs.get(i, []):
            if dist <= 0:
                continue
            drop = float((elev[i] - elev[j]) / dist)
            if drop > best_drop:
                best_drop = drop
                best_j = int(j)
        downstream[i] = int(best_j)

    sink_cache: dict[int, int] = {}
    for i in range(arr.shape[0]):
        path = []
        cur = i
        seen = set()
        while True:
            if cur in sink_cache:
                sink = sink_cache[cur]
                break
            nxt = downstream[cur]
            if nxt == cur or nxt in seen:
                sink = cur
                break
            seen.add(cur)
            path.append(cur)
            cur = nxt
        sink_cache[i] = sink
        for p in path:
            sink_cache[p] = sink
    basin_counts = pd.Series(list(sink_cache.values())).value_counts(dropna=False)
    largest_share = float(float(basin_counts.iloc[0]) / float(arr.shape[0])) if not basin_counts.empty else 0.0
    finding = {
        "kind": "surface_hydrology_flow_watershed",
        "measurement_type": "measured",
        "n_points": int(arr.shape[0]),
        "n_watersheds": int(len(basin_counts)),
        "largest_watershed_share": largest_share,
        "flow_edge_count": int(len(edges)),
        "features_used": cols[:3],
    }
    artifacts = [_artifact(ctx, plugin_id, "surface_hydrology_watershed.json", {"basin_counts": basin_counts.to_dict(), "summary": finding}, "json")]
    return PluginResult("ok", "Hydrology/watershed proxy computed", _basic_metrics(df, sample_meta), [finding], artifacts, None)


def _bayesian_point_displacement(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    if _heavy_disabled(plugin_id, config):
        return PluginResult("degraded", "Bayesian displacement disabled by config", _basic_metrics(df, sample_meta), _heavy_placeholder("bayesian_point_displacement", "disabled_by_config"), [], None)
    numeric_cols = inferred.get("numeric_columns") or []
    cols = [c for c in numeric_cols if c in df.columns][: int(config.get("max_cols", 12))]
    if not cols:
        return PluginResult("skipped", "Need numeric columns", {}, [], [], None)
    time_col = inferred.get("time_column")
    if isinstance(time_col, str) and time_col in df.columns:
        ts = pd.to_datetime(df[time_col], errors="coerce")
        order = np.argsort(ts.fillna(pd.Timestamp("1970-01-01")).to_numpy())
        dff = df.iloc[order]
    else:
        dff = df
    if len(dff) < 80:
        return PluginResult("skipped", "Insufficient rows for displacement", {}, [], [], None)

    split = int(len(dff) * 0.5)
    left = dff.iloc[:split]
    right = dff.iloc[split:]
    findings = []
    rows = []
    prior_mult = float(config.get("prior_var_multiplier", 1.0))
    min_prob = float(config.get("posterior_prob_min", 0.95))
    for col in cols:
        if timer.exceeded():
            break
        x0 = pd.to_numeric(left[col], errors="coerce").to_numpy(dtype=float)
        x1 = pd.to_numeric(right[col], errors="coerce").to_numpy(dtype=float)
        x0 = x0[np.isfinite(x0)]
        x1 = x1[np.isfinite(x1)]
        if x0.size < 20 or x1.size < 20:
            continue
        m0 = float(np.nanmean(x0))
        m1 = float(np.nanmean(x1))
        d = m1 - m0
        v0 = float(np.nanvar(x0, ddof=1)) if x0.size > 1 else 1.0
        v1 = float(np.nanvar(x1, ddof=1)) if x1.size > 1 else 1.0
        se2 = (v0 / max(1, x0.size)) + (v1 / max(1, x1.size))
        se2 = se2 if se2 > 1e-12 else 1e-12
        tau2 = max(1e-9, prior_mult * max(v0, v1, 1e-9))
        post_var = 1.0 / ((1.0 / tau2) + (1.0 / se2))
        post_mean = post_var * (d / se2)
        post_sd = math.sqrt(max(1e-12, post_var))
        z = post_mean / post_sd
        p_gt_0 = _normal_cdf(z)
        lo95 = float(post_mean - 1.96 * post_sd)
        hi95 = float(post_mean + 1.96 * post_sd)
        row = {
            "column": str(col),
            "posterior_mean_delta": float(post_mean),
            "posterior_sd": float(post_sd),
            "posterior_p_delta_gt_0": float(p_gt_0),
            "posterior_ci95_low": lo95,
            "posterior_ci95_high": hi95,
            "n_left": int(x0.size),
            "n_right": int(x1.size),
        }
        rows.append(row)
        if (p_gt_0 >= min_prob) or (p_gt_0 <= (1.0 - min_prob)):
            findings.append(
                {
                    "kind": "bayesian_point_displacement",
                    "measurement_type": "measured",
                    **row,
                }
            )
    if not rows:
        return PluginResult("skipped", "No valid numeric columns for displacement", _basic_metrics(df, sample_meta), [], [], None)
    rows.sort(key=lambda r: abs(float(r["posterior_mean_delta"])), reverse=True)
    findings.sort(key=lambda r: abs(float(r["posterior_mean_delta"])), reverse=True)
    artifacts = [_artifact(ctx, plugin_id, "bayesian_point_displacement.json", {"rows": rows}, "json")]
    return PluginResult("ok", f"Bayesian displacement columns={len(rows)} strong={len(findings)}", _basic_metrics(df, sample_meta), findings[:10], artifacts, None)


def _monte_carlo_surface_uncertainty(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    if _heavy_disabled(plugin_id, config):
        return PluginResult("degraded", "Monte Carlo uncertainty disabled by config", _basic_metrics(df, sample_meta), _heavy_placeholder("monte_carlo_surface_uncertainty", "disabled_by_config"), [], None)
    numeric_cols = inferred.get("numeric_columns") or []
    cols = [c for c in numeric_cols if c in df.columns][: int(config.get("max_cols", 8))]
    if not cols:
        return PluginResult("skipped", "Need numeric columns", {}, [], [], None)
    frame = df[cols].head(int(config.get("max_points", 3000))).apply(pd.to_numeric, errors="coerce")
    frame = frame.fillna(frame.median(numeric_only=True))
    arr = frame.to_numpy(dtype=float)
    if arr.shape[0] < 80:
        return PluginResult("skipped", "Insufficient rows for Monte Carlo uncertainty", {}, [], [], None)
    n_iter = max(40, int(config.get("n_iter", 200)))
    seed = int(config.get("seed", 1337))
    rng = np.random.RandomState(seed)
    metrics = []
    for _ in range(n_iter):
        if timer.exceeded():
            break
        idx = rng.randint(0, arr.shape[0], size=arr.shape[0])
        sample = arr[idx, :]
        col_std = np.nanstd(sample, axis=0)
        metrics.append(float(np.nanmean(col_std)))
    if not metrics:
        return PluginResult("skipped", "Monte Carlo halted by time budget", _basic_metrics(df, sample_meta), [], [], None)
    m = np.array(metrics, dtype=float)
    finding = {
        "kind": "monte_carlo_surface_uncertainty",
        "measurement_type": "measured",
        "n_iter_completed": int(m.size),
        "uncertainty_mean": float(np.nanmean(m)),
        "uncertainty_std": float(np.nanstd(m)),
        "uncertainty_ci95_low": float(np.nanquantile(m, 0.025)),
        "uncertainty_ci95_high": float(np.nanquantile(m, 0.975)),
        "features_used": cols,
    }
    artifacts = [_artifact(ctx, plugin_id, "monte_carlo_surface_uncertainty.json", {"samples": metrics, "summary": finding}, "json")]
    return PluginResult("ok", f"Monte Carlo uncertainty iterations={int(m.size)}", _basic_metrics(df, sample_meta), [finding], artifacts, None)


def _surface_roughness_metrics(
    plugin_id: str,
    ctx,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    if _heavy_disabled(plugin_id, config):
        return PluginResult("degraded", "Surface roughness disabled by config", _basic_metrics(df, sample_meta), _heavy_placeholder("surface_roughness_metrics", "disabled_by_config"), [], None)
    numeric_cols = inferred.get("numeric_columns") or []
    cols = [c for c in numeric_cols if c in df.columns][: int(config.get("max_cols", 12))]
    if not cols:
        return PluginResult("skipped", "Need numeric columns", {}, [], [], None)
    rows = []
    for col in cols:
        if timer.exceeded():
            break
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(x).sum() < 40:
            continue
        med = float(np.nanmedian(x))
        x = np.where(np.isfinite(x), x, med)
        d = np.diff(x)
        if d.size == 0:
            continue
        std_diff = float(np.nanstd(d))
        mad_diff = float(np.nanmedian(np.abs(d - np.nanmedian(d))))
        sign = np.sign(d)
        zc = float(np.nanmean(sign[1:] * sign[:-1] < 0.0)) if sign.size > 1 else 0.0
        spec = np.abs(np.fft.rfft(d))
        spec = spec[1:] if spec.size > 1 else spec
        if spec.size > 0:
            gm = float(np.exp(np.nanmean(np.log(np.maximum(spec, 1e-12)))))
            am = float(np.nanmean(spec))
            flatness = gm / am if am > 0 else 0.0
        else:
            flatness = 0.0
        rough_score = float(std_diff + mad_diff + zc + (1.0 - flatness))
        rows.append(
            {
                "column": str(col),
                "roughness_score": rough_score,
                "std_diff": std_diff,
                "mad_diff": mad_diff,
                "zero_cross_rate": zc,
                "spectral_flatness": float(flatness),
            }
        )
    if not rows:
        return PluginResult("skipped", "No valid roughness rows", _basic_metrics(df, sample_meta), [], [], None)
    rows.sort(key=lambda r: float(r["roughness_score"]), reverse=True)
    findings = [
        {
            "kind": "surface_roughness_metrics",
            "measurement_type": "measured",
            "column": row["column"],
            "roughness_score": row["roughness_score"],
            "zero_cross_rate": row["zero_cross_rate"],
        }
        for row in rows[:5]
    ]
    artifacts = [_artifact(ctx, plugin_id, "surface_roughness_metrics.json", {"rows": rows}, "json")]
    return PluginResult("ok", f"Surface roughness computed for {len(rows)} columns", _basic_metrics(df, sample_meta), findings, artifacts, None)


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

    # Process exclusions: interpret as patterns (glob/SQL-like/regex), with conservative defaults.
    exclude_list = config.get("exclude_processes")
    if isinstance(exclude_list, str):
        exclude_list = [s.strip() for s in exclude_list.split(",") if s.strip()]
    if not isinstance(exclude_list, (list, tuple, set)):
        exclude_list = []
    exclude_patterns = merge_patterns(
        default_exclude_process_patterns(),
        parse_exclude_patterns_env(),
        [str(x) for x in exclude_list],
    )
    exclude_match = compile_patterns(exclude_patterns)

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
        if proc_norm and exclude_match(proc_norm):
            return
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
        elif action_type == "orchestrate_chain":
            parent_raw = str((extra or {}).get("parent_process") or evidence.get("parent_process") or "").strip()
            child_raw = str((extra or {}).get("child_process") or evidence.get("child_process") or "").strip()
            parent = redactor(parent_raw) if parent_raw else label
            child = redactor(child_raw) if child_raw else "(downstream child)"
            chain_ratio = evidence.get("child_chain_ratio")
            ratio_txt = (
                f"{float(chain_ratio) * 100.0:.1f}%"
                if isinstance(chain_ratio, (int, float))
                else "high"
            )
            recommendation = (
                f"{child} behaves like a chain child (dependency/master linkage non-null ratio {ratio_txt}), "
                f"so target {parent} or the dependency rule instead of changing {child} directly. "
                f"Upper bound: {_fmt_hours(impact_hours)} of wait time above {thr:.0f}s tied to this parent->child linkage. "
                "Next: shorten parent runtime, enable overlap/parallelism where safe, or remove unnecessary dependency edges."
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
        elif action_type == "batch_input":
            key = str((extra or {}).get("key") or "").strip() or "parameter"
            cm = str((extra or {}).get("best_close_month") or "").strip()
            runs = (extra or {}).get("runs_with_key")
            uniq = (extra or {}).get("unique_values")
            coverage = (extra or {}).get("coverage")
            unique_ratio = (extra or {}).get("unique_ratio")
            reducible = (extra or {}).get("estimated_calls_reduced")
            top_user = str((extra or {}).get("top_user_redacted") or "").strip()
            top_user_runs = (extra or {}).get("top_user_runs")
            top_user_share = (extra or {}).get("top_user_run_share")
            distinct_users = (extra or {}).get("distinct_users")
            user_runs_total = (extra or {}).get("user_runs_total")
            cm_txt = f" in close month {cm}" if cm else ""
            runs_txt = f"{int(runs):,}" if isinstance(runs, (int, float)) else "many"
            uniq_txt = f"{int(uniq):,}" if isinstance(uniq, (int, float)) else "many"
            coverage_txt = (
                f"{float(coverage) * 100.0:.1f}%"
                if isinstance(coverage, (int, float))
                else "n/a"
            )
            unique_ratio_txt = (
                f"{float(unique_ratio) * 100.0:.1f}%"
                if isinstance(unique_ratio, (int, float))
                else "n/a"
            )
            reducible_txt = f"{int(reducible):,}" if isinstance(reducible, (int, float)) else ""
            reducible_clause = f" (reducing job launches by ~{reducible_txt})" if reducible_txt else ""
            user_loop_clause = ""
            if (
                isinstance(top_user_runs, (int, float))
                and isinstance(top_user_share, (int, float))
                and isinstance(user_runs_total, (int, float))
                and float(top_user_runs) > 1.0
                and float(top_user_share) >= 0.5
            ):
                user_label = top_user if top_user else "single user"
                user_loop_clause = (
                    f" Manual-loop signal: {user_label} executed {int(top_user_runs):,} of "
                    f"{int(user_runs_total):,} close-window runs "
                    f"({float(top_user_share) * 100.0:.1f}% concentration"
                )
                if isinstance(distinct_users, (int, float)):
                    user_loop_clause += f", {int(distinct_users):,} distinct users overall"
                user_loop_clause += ")."
            recommendation = (
                f"Convert specific process_id `{label}` to batch-input mode{cm_txt}. "
                f"Why: {runs_txt} runs handled {uniq_txt} unique `{key}` values "
                f"(coverage {coverage_txt}, uniqueness {unique_ratio_txt}), which indicates one-call-per-value sweep behavior. "
                f"Change `{label}` to accept a list of `{key}` values and process one close-month cohort per run{reducible_clause}. "
                f"Upper bound: {_fmt_hours(impact_hours)} of over-threshold wait time above {thr:.0f}s associated with this process over the observation window (not guaranteed)."
                f"{user_loop_clause}"
            )
        elif action_type == "batch_group_candidate":
            key = str((extra or {}).get("key") or evidence.get("key") or "").strip() or "parameter"
            cm = str((extra or {}).get("best_close_month") or evidence.get("close_month") or "").strip()
            target_ids_raw = (extra or {}).get("target_process_ids") or evidence.get("target_process_ids")
            target_ids = [
                str(v).strip().lower()
                for v in (target_ids_raw if isinstance(target_ids_raw, list) else [])
                if isinstance(v, str) and str(v).strip()
            ]
            target_ids = list(dict.fromkeys(target_ids))
            target_preview = ", ".join(f"`{redactor(pid)}`" for pid in target_ids[:8])
            if len(target_ids) > 8:
                target_preview += f", +{len(target_ids) - 8} more"
            cm_txt = f" for close month {cm}" if cm else ""
            reducible = (extra or {}).get("estimated_calls_reduced") or evidence.get("estimated_calls_reduced")
            reducible_txt = f"{int(reducible):,}" if isinstance(reducible, (int, float)) else "multiple"
            recommendation = (
                f"Convert the payout cohort sweep{cm_txt} to explicit multi-input runs by `{key}`. "
                f"Specific process_id targets: {target_preview if target_preview else f'`{label}`'}. "
                f"Change each listed process to accept a list of `{key}` values, then run one cohort execution instead of many single-value launches "
                f"(estimated launch reduction ~{reducible_txt}). "
                f"Upper bound: {_fmt_hours(impact_hours)} of over-threshold wait above {thr:.0f}s across this grouped workload."
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

        if not isinstance(recommendation, str) or not recommendation.strip():
            # Fail-closed for operator UX: do not emit empty recommendations.
            recommendation = (
                f"{title}. "
                "Validate the supporting evidence artifact, make a targeted change, and re-run the harness to confirm the expected impact."
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
                total_over = None
                try:
                    total_over = float(_over_sum_by_proc.get(str(row["process"]), 0.0))
                except Exception:
                    total_over = None
                # Volume-aware estimate: median delta times a conservative "swappable" count,
                # capped by total over-threshold contribution for the process.
                est_total = float(row["delta_seconds"]) * float(min(row["n_worst"], row["n_best"]))
                if isinstance(total_over, (int, float)) and math.isfinite(float(total_over)):
                    est_total = min(est_total, float(total_over))
                emit(
                    title=f"Route {row['process']} from {row['server_worst']} to {row['server_best']}",
                    action_type="route_process",
                    process_value=row["process"],
                    expected_delta_seconds=float(est_total),
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
                        "median_delta_seconds": float(row["delta_seconds"]),
                        "over_threshold_seconds_total_upper_bound": float(total_over) if isinstance(total_over, (int, float)) else None,
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
            masterq = (
                pd.to_numeric(df["MASTER_PROCESS_QUEUE_ID"], errors="coerce")
                if "MASTER_PROCESS_QUEUE_ID" in df.columns
                else None
            )
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
                        child_str = str(child).strip()
                        child_mask = proc.astype(str) == child_str
                        child_rows_total = int(child_mask.sum())
                        dep_non_null_ratio = None
                        dep_null_ratio = None
                        master_non_null_ratio = None
                        master_null_ratio = None
                        if child_rows_total > 0:
                            dep_non_null = int(depq[child_mask].notna().sum())
                            dep_non_null_ratio = float(dep_non_null) / float(child_rows_total)
                            dep_null_ratio = 1.0 - dep_non_null_ratio
                            if isinstance(masterq, pd.Series):
                                master_non_null = int(masterq[child_mask].notna().sum())
                                master_non_null_ratio = float(master_non_null) / float(child_rows_total)
                                master_null_ratio = 1.0 - master_non_null_ratio
                        distinct_parent_count = int(
                            grp[grp["_child"] == child]["_parent"].nunique(dropna=True)
                        )
                        chain_ratio_candidates = [
                            float(v)
                            for v in (dep_non_null_ratio, master_non_null_ratio)
                            if isinstance(v, (int, float))
                        ]
                        chain_ratio = max(chain_ratio_candidates) if chain_ratio_candidates else 0.0
                        chain_child_ratio_min = float(
                            config.get("dependency_chain_child_ratio_min", 0.95)
                        )
                        likely_chain_child = bool(
                            child_rows_total > 0 and chain_ratio >= chain_child_ratio_min
                        )
                        target_process = parent if likely_chain_child else child_str
                        target_action = (
                            "orchestrate_chain" if likely_chain_child else "unblock_dependency_chain"
                        )
                        title = (
                            f"Reduce blocker {parent} to unblock {child}"
                            if likely_chain_child
                            else f"Unblock {child} when preceded by {parent}"
                        )
                        feasibility = "indirect_only" if likely_chain_child else "direct_child_actionable"
                        emit(
                            title=title,
                            action_type=target_action,
                            process_value=target_process,
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
                                "child_process": child_str,
                                "subset_runs": n,
                                "subset_over_threshold_wait_sec_total": sum_over,
                                "child_rows_total": child_rows_total,
                                "child_dependency_non_null_ratio": dep_non_null_ratio,
                                "child_dependency_null_ratio": dep_null_ratio,
                                "child_master_queue_non_null_ratio": master_non_null_ratio,
                                "child_master_queue_null_ratio": master_null_ratio,
                                "child_distinct_parent_count": distinct_parent_count,
                                "child_chain_ratio": chain_ratio,
                                "chain_child_ratio_min": chain_child_ratio_min,
                                "likely_chain_child": likely_chain_child,
                                "actionability_mode": feasibility,
                                "dependency_hotspots_artifact": str(artifacts[-1].path) if artifacts else None,
                            },
                            extra={
                                "parent_process": redactor(parent),
                                "child_process": redactor(child_str),
                                "actionability_mode": feasibility,
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

    # ---- Batch input (parameter sweeps; multi-input candidates) ----
    # Detect cases where a process runs many times while sweeping across mostly-unique values
    # of a parameter key within a close-month cohort (classic "list-of-IDs" batch endpoint).
    try:
        baseline_start_day = int(config.get("close_cycle_start_day", 20))
        baseline_end_day = int(config.get("close_cycle_end_day", 5))
        min_runs_total = int(config.get("batch_input_min_runs_total", 50))
        min_runs_with_key = int(config.get("batch_input_min_runs_with_key", 50))
        min_coverage = float(config.get("batch_input_min_coverage", 0.9))
        min_unique_ratio = float(config.get("batch_input_min_unique_ratio", 0.8))
        max_batch_inputs = int(config.get("max_batch_input_findings", 8))
    except Exception:
        baseline_start_day = 20
        baseline_end_day = 5
        min_runs_total = 50
        min_runs_with_key = 50
        min_coverage = 0.9
        min_unique_ratio = 0.8
        max_batch_inputs = 8

    def _key_blacklisted(key: str) -> bool:
        k = key.strip().lower()
        if not k or k in {"raw"}:
            return True
        # Reject obvious volatile/correlation identifiers; allow business IDs like "payout id".
        for bad in ("request", "trace", "span", "session", "uuid", "guid", "timestamp"):
            if bad in k:
                return True
        for bad in ("started", "ended", "created", "updated"):
            if bad in k:
                return True
        return False

    def _detect_batch_input_candidates() -> list[dict[str, Any]]:
        if not getattr(ctx, "dataset_version_id", None):
            return []
        dataset_version_id = str(getattr(ctx, "dataset_version_id"))
        tmpl = None
        try:
            tmpl = ctx.storage.fetch_dataset_template(dataset_version_id)
        except Exception:
            tmpl = None
        if not isinstance(tmpl, dict) or tmpl.get("status") != "ready":
            return []
        table_name = str(tmpl.get("table_name") or "").strip()
        template_id = tmpl.get("template_id")
        if not table_name or not isinstance(template_id, (int, float)):
            return []
        fields = ctx.storage.fetch_template_fields(int(template_id))
        name_to_safe = {f.get("name"): f.get("safe_name") for f in fields if f.get("name") and f.get("safe_name")}

        proc_safe = name_to_safe.get(process_col)
        # Prefer queue timestamp for close-month logic; else start; else inferred time.
        ts_name = queue_col or start_col or time_col
        ts_safe = name_to_safe.get(ts_name) if isinstance(ts_name, str) else None
        if not proc_safe or not ts_safe:
            return []

        safe_table = quote_identifier(table_name)
        proc_expr = f"LOWER(TRIM(COALESCE({quote_identifier(str(proc_safe))}, '')))"
        ts_expr = quote_identifier(str(ts_safe))
        day_expr = f"CAST(SUBSTR({ts_expr}, 9, 2) AS INTEGER)"
        # Baseline close window mask (wrap-aware).
        if baseline_start_day <= baseline_end_day:
            day_pred = f"({day_expr} BETWEEN ? AND ?)"
            day_params = [int(baseline_start_day), int(baseline_end_day)]
        else:
            day_pred = f"(({day_expr} >= ?) OR ({day_expr} <= ?))"
            day_params = [int(baseline_start_day), int(baseline_end_day)]
        # Close-month cohort label (YYYY-MM). Day <= end_day belongs to previous month.
        close_month_expr = (
            f"CASE WHEN {day_expr} <= ? "
            f"THEN STRFTIME('%Y-%m', DATE(SUBSTR({ts_expr}, 1, 10), '-1 month')) "
            f"ELSE STRFTIME('%Y-%m', DATE(SUBSTR({ts_expr}, 1, 10))) END"
        )

        sql = f"""
        WITH base AS (
          SELECT row_index,
                 {proc_expr} AS process_norm,
                 {close_month_expr} AS close_month
          FROM {safe_table}
          WHERE dataset_version_id = ?
            AND LENGTH({ts_expr}) >= 10
            AND {proc_expr} <> ''
            AND {day_pred}
        ),
        totals AS (
          SELECT process_norm, close_month, COUNT(*) AS runs_total
          FROM base
          GROUP BY process_norm, close_month
        ),
        agg AS (
          SELECT b.process_norm,
                 b.close_month,
                 pk.key AS key,
                 COUNT(DISTINCT b.row_index) AS runs_with_key,
                 COUNT(DISTINCT pk.value) AS unique_values
          FROM base b
          JOIN row_parameter_link rpl
            ON rpl.dataset_version_id = ? AND rpl.row_index = b.row_index
          JOIN parameter_kv pk
            ON pk.entity_id = rpl.entity_id
          GROUP BY b.process_norm, b.close_month, pk.key
        )
        SELECT a.process_norm,
               a.close_month,
               a.key,
               t.runs_total,
               a.runs_with_key,
               a.unique_values,
               (a.runs_with_key * 1.0 / t.runs_total) AS coverage,
               (a.unique_values * 1.0 / a.runs_with_key) AS unique_ratio
        FROM agg a
        JOIN totals t
          ON t.process_norm = a.process_norm AND t.close_month = a.close_month
        WHERE t.runs_total >= ?
          AND a.runs_with_key >= ?
        ORDER BY unique_ratio DESC, coverage DESC, runs_with_key DESC
        """
        params: list[Any] = []
        params.append(int(baseline_end_day))  # for close_month_expr
        params.append(dataset_version_id)
        params.extend(day_params)
        params.append(dataset_version_id)  # row_parameter_link join
        params.append(int(min_runs_total))
        params.append(int(min_runs_with_key))

        rows: list[dict[str, Any]] = []
        with ctx.storage.connection() as conn:
            cur = conn.execute(sql, params)
            rows = [dict(r) for r in cur.fetchall()]
        if not rows:
            return []

        # Filter + pick best key per process by a simple score.
        best: dict[str, dict[str, Any]] = {}
        for r in rows:
            proc_norm = str(r.get("process_norm") or "").strip().lower()
            if not proc_norm or exclude_match(proc_norm):
                continue
            key = str(r.get("key") or "").strip()
            if _key_blacklisted(key):
                continue
            try:
                coverage = float(r.get("coverage") or 0.0)
                uniq_ratio = float(r.get("unique_ratio") or 0.0)
            except Exception:
                continue
            if coverage < min_coverage or uniq_ratio < min_unique_ratio:
                continue
            score = float(r.get("runs_with_key") or 0.0) * coverage * uniq_ratio
            prev = best.get(proc_norm)
            if prev is None or float(prev.get("_score") or 0.0) < score:
                r["_score"] = score
                best[proc_norm] = r
        out = list(best.values())
        out.sort(key=lambda rr: float(rr.get("_score") or 0.0), reverse=True)
        return out[: max(1, max_batch_inputs * 3)]

    candidates = _detect_batch_input_candidates()
    if candidates:
        # Persist evidence table for citation.
        artifacts.append(_artifact(ctx, plugin_id, "batch_input_candidates.json", candidates[:200], "json"))

        # Precompute close-window median target per process for a volume-aware estimate.
        proc_norm_series = df[process_col].astype(str).str.strip().str.lower()
        _over_sum_by_proc_norm = over.groupby(proc_norm_series, dropna=False).sum() if not over.empty else pd.Series(dtype=float)
        median_close_by_proc: dict[str, float] = {}
        user_repeat_by_proc: dict[str, dict[str, Any]] = {}
        if (queue_col or start_col or time_col) and (queue_col or start_col or time_col) in df.columns:
            ts_local = pd.to_datetime(df[queue_col or start_col or time_col], errors="coerce", utc=False)
            if ts_local.notna().any():
                days = ts_local.dt.day
                if baseline_start_day <= baseline_end_day:
                    close_mask = (days >= baseline_start_day) & (days <= baseline_end_day)
                else:
                    close_mask = (days >= baseline_start_day) | (days <= baseline_end_day)
                tmp_y = y.where(close_mask)
                try:
                    med = tmp_y.groupby(proc_norm_series, dropna=False).median()
                    median_close_by_proc = {str(k): float(v) for k, v in med.items() if isinstance(v, (int, float)) and math.isfinite(float(v))}
                except Exception:
                    median_close_by_proc = {}
        user_col = None
        for candidate in (
            "USER_ID",
            "ASSIGNED_USER_ID",
            "OWNER_ID",
            "OPERATOR_ID",
            "REQUEST_USER_ID",
            "USER_ID_RQST_TERMINATE",
        ):
            if candidate in df.columns:
                user_col = candidate
                break
        if not isinstance(user_col, str) or user_col not in df.columns:
            inferred_user_col = _pick_from_all(("user", "owner", "operator", "request_user"))
            if isinstance(inferred_user_col, str) and inferred_user_col in df.columns:
                user_col = inferred_user_col
        if isinstance(user_col, str) and user_col in df.columns:
            try:
                user_series = df[user_col].astype(str).str.strip()
                proc_series = proc_norm_series.astype(str).str.strip().str.lower()
                valid = (
                    proc_series.ne("")
                    & user_series.ne("")
                    & user_series.str.lower().ne("nan")
                    & user_series.str.lower().ne("none")
                )
                if bool(valid.any()):
                    tmp_users = pd.DataFrame(
                        {"process_norm": proc_series[valid], "user_norm": user_series[valid]}
                    )
                    pair_counts = (
                        tmp_users.groupby(["process_norm", "user_norm"], dropna=False)
                        .size()
                        .rename("runs")
                        .reset_index()
                    )
                    proc_totals = tmp_users.groupby("process_norm", dropna=False).size()
                    for process_norm, sub in pair_counts.groupby("process_norm", dropna=False):
                        if sub.empty:
                            continue
                        ordered = sub.sort_values(["runs", "user_norm"], ascending=[False, True])
                        top = ordered.iloc[0]
                        top_runs = int(top.get("runs") or 0)
                        total_runs = int(proc_totals.get(process_norm, 0))
                        if top_runs <= 0 or total_runs <= 0:
                            continue
                        user_repeat_by_proc[str(process_norm)] = {
                            "user_column": str(user_col),
                            "top_user_redacted": redactor(str(top.get("user_norm") or "")),
                            "top_user_runs": top_runs,
                            "top_user_run_share": float(top_runs) / float(total_runs),
                            "distinct_users": int(sub["user_norm"].nunique(dropna=True)),
                            "user_runs_total": total_runs,
                        }
            except Exception:
                user_repeat_by_proc = {}

        emitted = 0
        emitted_batch_rows: list[dict[str, Any]] = []
        for row in candidates:
            if emitted >= max_batch_inputs:
                break
            proc_norm = str(row.get("process_norm") or "").strip().lower()
            if not proc_norm or exclude_match(proc_norm):
                continue
            key = str(row.get("key") or "").strip()
            if not key or _key_blacklisted(key):
                continue
            runs_with_key = row.get("runs_with_key")
            unique_values = row.get("unique_values")
            runs_total = row.get("runs_total")
            close_month = str(row.get("close_month") or "").strip()

            try:
                n_runs = int(float(runs_with_key))
            except Exception:
                n_runs = 0
            if n_runs <= 1:
                continue
            estimated_calls_reduced = max(0, n_runs - 1)

            # Upper-bound delta: min(over-threshold sum, (calls_reduced * median_close_target)).
            total_over = None
            try:
                total_over = float(_over_sum_by_proc_norm.get(proc_norm, 0.0))
            except Exception:
                total_over = None
            median_close = median_close_by_proc.get(proc_norm)
            user_repeat = user_repeat_by_proc.get(proc_norm, {})
            est_total = None
            if median_close is not None and math.isfinite(float(median_close)):
                est_total = float(estimated_calls_reduced) * float(median_close)
                if (
                    isinstance(total_over, (int, float))
                    and math.isfinite(float(total_over))
                    and float(total_over) > 0.0
                ):
                    est_total = min(est_total, float(total_over))
            elif isinstance(total_over, (int, float)) and math.isfinite(float(total_over)):
                est_total = float(total_over)

            emit(
                title=f"Batch input: refactor {proc_norm} to accept list of {key}",
                action_type="batch_input",
                process_value=proc_norm,
                expected_delta_seconds=float(est_total) if isinstance(est_total, (int, float)) else None,
                expected_delta_percent=None,
                confidence=0.75,
                assumptions=[
                    "Close-month cohorts are defined by day<=end_day belonging to previous month.",
                    "High unique_ratio implies a sweep over single-item calls (list-of-IDs opportunity).",
                    "Savings estimate is an upper bound; service work may still dominate.",
                ],
                evidence={
                    "method": "parameter_sweep_batch_input",
                    "baseline_close_start_day": baseline_start_day,
                    "baseline_close_end_day": baseline_end_day,
                    "close_month": close_month or None,
                    "runs_total_in_cohort": int(runs_total) if isinstance(runs_total, (int, float)) else runs_total,
                    "runs_with_key": int(runs_with_key) if isinstance(runs_with_key, (int, float)) else runs_with_key,
                    "unique_values": int(unique_values) if isinstance(unique_values, (int, float)) else unique_values,
                    "coverage": float(row.get("coverage")) if isinstance(row.get("coverage"), (int, float)) else row.get("coverage"),
                    "unique_ratio": float(row.get("unique_ratio")) if isinstance(row.get("unique_ratio"), (int, float)) else row.get("unique_ratio"),
                    "batch_input_candidates_artifact": str(artifacts[-1].path) if artifacts else None,
                    "over_threshold_seconds_total_upper_bound": float(total_over) if isinstance(total_over, (int, float)) else None,
                    "median_close_target_seconds": float(median_close) if isinstance(median_close, (int, float)) else None,
                    "user_column": user_repeat.get("user_column"),
                    "top_user_redacted": user_repeat.get("top_user_redacted"),
                    "top_user_runs": user_repeat.get("top_user_runs"),
                    "top_user_run_share": user_repeat.get("top_user_run_share"),
                    "distinct_users": user_repeat.get("distinct_users"),
                    "user_runs_total": user_repeat.get("user_runs_total"),
                },
                extra={
                    "key": key,
                    "best_close_month": close_month or None,
                    "runs_with_key": int(runs_with_key) if isinstance(runs_with_key, (int, float)) else runs_with_key,
                    "unique_values": int(unique_values) if isinstance(unique_values, (int, float)) else unique_values,
                    "coverage": float(row.get("coverage")) if isinstance(row.get("coverage"), (int, float)) else row.get("coverage"),
                    "unique_ratio": float(row.get("unique_ratio")) if isinstance(row.get("unique_ratio"), (int, float)) else row.get("unique_ratio"),
                    "estimated_calls_reduced": int(estimated_calls_reduced),
                    "top_user_redacted": user_repeat.get("top_user_redacted"),
                    "top_user_runs": user_repeat.get("top_user_runs"),
                    "top_user_run_share": user_repeat.get("top_user_run_share"),
                    "distinct_users": user_repeat.get("distinct_users"),
                    "user_runs_total": user_repeat.get("user_runs_total"),
                    "validation_steps": [
                        "Pick 5-10 sample values from the artifact and confirm they differ only by this key.",
                        "Add/validate a multi-input endpoint that accepts a list of these values.",
                        "Re-run the harness and confirm job launches and queue wait drop in the close-month cohort.",
                    ],
                },
            )
            emitted_batch_rows.append(
                {
                    "process_norm": proc_norm,
                    "key": key,
                    "close_month": close_month or "",
                    "runs_with_key": int(runs_with_key) if isinstance(runs_with_key, (int, float)) else 0,
                    "unique_values": int(unique_values) if isinstance(unique_values, (int, float)) else 0,
                    "coverage": float(row.get("coverage")) if isinstance(row.get("coverage"), (int, float)) else 0.0,
                    "unique_ratio": float(row.get("unique_ratio")) if isinstance(row.get("unique_ratio"), (int, float)) else 0.0,
                    "estimated_delta_seconds_upper": float(est_total) if isinstance(est_total, (int, float)) else 0.0,
                }
            )
            emitted += 1

        # Payout-report chain synthesis: group payout-key sweeps by close month and
        # emit one explicit "convert these process_ids" recommendation.
        payout_rows = [
            row
            for row in emitted_batch_rows
            if "payout" in str(row.get("key") or "").strip().lower()
        ]
        if payout_rows:
            grouped: dict[str, list[dict[str, Any]]] = {}
            for row in payout_rows:
                cm = str(row.get("close_month") or "").strip() or "unknown"
                grouped.setdefault(cm, []).append(row)
            for close_month, rows_for_month in sorted(grouped.items()):
                if not rows_for_month:
                    continue
                rows_for_month.sort(
                    key=lambda rr: (
                        float(rr.get("estimated_delta_seconds_upper") or 0.0),
                        float(rr.get("runs_with_key") or 0.0),
                        float(rr.get("unique_ratio") or 0.0),
                    ),
                    reverse=True,
                )
                top_rows = rows_for_month[:6]
                target_process_ids = [str(r.get("process_norm") or "").strip() for r in top_rows if str(r.get("process_norm") or "").strip()]
                if not target_process_ids:
                    continue
                strongest_key = str(top_rows[0].get("key") or "payout_id")
                total_runs = int(sum(int(r.get("runs_with_key") or 0) for r in top_rows))
                total_delta = float(sum(float(r.get("estimated_delta_seconds_upper") or 0.0) for r in top_rows))
                mean_unique_ratio = float(
                    sum(float(r.get("unique_ratio") or 0.0) for r in top_rows) / float(len(top_rows))
                )
                mean_coverage = float(
                    sum(float(r.get("coverage") or 0.0) for r in top_rows) / float(len(top_rows))
                )
                sequence_id = stable_id(
                    [
                        "batch_group_candidate",
                        close_month,
                        strongest_key,
                        ",".join(target_process_ids),
                    ],
                    prefix="batchseq",
                )
                primary_process = target_process_ids[0]
                emit(
                    title=f"Batch payout-report chain for close month {close_month}",
                    action_type="batch_group_candidate",
                    process_value=primary_process,
                    expected_delta_seconds=total_delta if total_delta > 0.0 else None,
                    expected_delta_percent=None,
                    confidence=0.8,
                    assumptions=[
                        "These process_ids are part of the same payout close-month sweep workload.",
                        "Single-input calls can be replaced by list/batch input without changing business correctness.",
                        "Savings are an upper bound based on observed over-threshold wait and launch count.",
                    ],
                    evidence={
                        "method": "payout_chain_batch_group",
                        "sequence_id": sequence_id,
                        "close_month": close_month,
                        "key": strongest_key,
                        "target_process_ids": target_process_ids,
                        "runs_with_key_total": total_runs,
                        "estimated_calls_reduced": max(total_runs - 1, 0),
                        "estimated_delta_seconds_upper": total_delta if total_delta > 0.0 else None,
                        "mean_unique_ratio": mean_unique_ratio,
                        "mean_coverage": mean_coverage,
                        "batch_input_candidates_artifact": str(artifacts[-1].path) if artifacts else None,
                    },
                    extra={
                        "sequence_id": sequence_id,
                        "key": strongest_key,
                        "best_close_month": close_month,
                        "target_process_ids": target_process_ids,
                        "runs_with_key": total_runs,
                        "estimated_calls_reduced": max(total_runs - 1, 0),
                        "estimated_delta_seconds_upper": total_delta if total_delta > 0.0 else None,
                        "unique_ratio": mean_unique_ratio,
                        "coverage": mean_coverage,
                        "validation_steps": [
                            "Convert each listed process_id to accept a list of payout IDs.",
                            "Run the chain once per close-month cohort instead of once per payout ID.",
                            "Re-run and verify launch count and queue-wait drop for these process_ids.",
                        ],
                    },
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
