"""Cross-domain plugins: social sciences, sports & linguistics (plugins 73-94)."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Callable

import numpy as np
import pandas as pd

from statistic_harness.core.stat_plugins import (
    BudgetTimer, deterministic_sample, robust_center_scale, stable_id, standardized_median_diff,
)
from statistic_harness.core.types import PluginArtifact, PluginResult

try:
    from scipy import stats as scipy_stats
    from scipy import optimize as scipy_optimize
    HAS_SCIPY = True
except Exception:
    scipy_stats = scipy_optimize = None
    HAS_SCIPY = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except Exception:
    nx = None
    HAS_NETWORKX = False

try:
    import nashpy
    HAS_NASHPY = True
except Exception:
    nashpy = None
    HAS_NASHPY = False

try:
    import pingouin
    HAS_PINGOUIN = True
except Exception:
    pingouin = None
    HAS_PINGOUIN = False

try:
    from rapidfuzz import distance as rf_distance
    HAS_RAPIDFUZZ = True
except Exception:
    rf_distance = None
    HAS_RAPIDFUZZ = False

try:
    import powerlaw
    HAS_POWERLAW = True
except Exception:
    powerlaw = None
    HAS_POWERLAW = False


# ---------------------------------------------------------------------------
# Plugin ID tuple
# ---------------------------------------------------------------------------

CROSS_DOMAIN_IDS: tuple[str, ...] = (
    "analysis_war_value_above_replacement_v1",
    "analysis_win_probability_added_v1",
    "analysis_leverage_index_dag_v1",
    "analysis_pythagorean_expectation_v1",
    "analysis_zipf_law_frequency_v1",
    "analysis_entropy_rate_step_sequences_v1",
    "analysis_levenshtein_workflow_dist_v1",
    "analysis_ngram_step_transitions_v1",
    "analysis_weber_fechner_threshold_v1",
    "analysis_stevens_power_law_severity_v1",
    "analysis_signal_detection_dprime_v1",
    "analysis_irt_step_difficulty_v1",
    "analysis_rasch_step_host_scale_v1",
    "analysis_cronbach_alpha_consistency_v1",
    "analysis_structural_holes_broker_v1",
    "analysis_weak_ties_bridge_v1",
    "analysis_assortativity_scheduling_bias_v1",
    "analysis_nash_equilibrium_contention_v1",
    "analysis_mechanism_design_scheduling_v1",
    "analysis_vickrey_priority_truth_v1",
    "analysis_benford_second_digit_v1",
)


# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------

def _safe_id(plugin_id: str, key: str) -> str:
    try:
        return stable_id((plugin_id, key))
    except Exception:
        return hashlib.sha256(f"{plugin_id}:{key}".encode()).hexdigest()[:16]


def _basic_metrics(df: pd.DataFrame, sample_meta: dict[str, Any]) -> dict[str, Any]:
    m: dict[str, Any] = {
        "rows_seen": int(sample_meta.get("rows_total", len(df))),
        "rows_used": int(sample_meta.get("rows_used", len(df))),
        "cols_used": int(len(df.columns)),
    }
    m.update(sample_meta or {})
    return m


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
    kind: str | None = None,
) -> dict[str, Any]:
    f: dict[str, Any] = {
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
    if kind:
        f["kind"] = kind
    return f


def _ok_with_reason(
    plugin_id: str,
    ctx: Any,
    df: pd.DataFrame,
    sample_meta: dict[str, Any],
    reason: str,
    *,
    debug: dict[str, Any] | None = None,
) -> PluginResult:
    ctx.logger(f"SKIP reason={reason}")
    p = dict(debug or {})
    p.setdefault("gating_reason", reason)
    return PluginResult(
        "ok",
        f"No actionable result: {reason}",
        _basic_metrics(df, sample_meta),
        [],
        [],
        None,
        debug=p,
    )


def _finalize(
    plugin_id: str,
    ctx: Any,
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
    rt = int(metrics.pop("runtime_ms", 0))
    ctx.logger(f"END runtime_ms={rt} findings={len(findings)}")
    d = dict(debug or {})
    d["runtime_ms"] = rt
    return PluginResult("ok", summary, metrics, findings, artifacts, None, debug=d)


def _log_start(ctx: Any, plugin_id: str, df: pd.DataFrame, config: dict[str, Any], inferred: dict[str, Any]) -> None:
    ctx.logger(
        f"START plugin_id={plugin_id} rows={len(df)} cols={len(df.columns)} seed={int(config.get('seed', 0))}"
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
    tc = inferred.get("time_column")
    if isinstance(tc, str) and tc in df.columns:
        p = pd.to_datetime(df[tc], errors="coerce")
        if p.notna().sum() >= 10:
            return tc, p
    for col in df.columns:
        l = str(col).lower()
        if "time" not in l and "date" not in l:
            continue
        p = pd.to_datetime(df[col], errors="coerce")
        if p.notna().sum() >= 10:
            return str(col), p
    return None, None


def _duration_column(df: pd.DataFrame, inferred: dict[str, Any]) -> str | None:
    hints = ("duration", "latency", "wait", "elapsed", "runtime", "service", "time")
    for col in _numeric_columns(df, inferred, max_cols=100):
        if any(h in col.lower() for h in hints):
            return col
    cols = _numeric_columns(df, inferred, max_cols=100)
    return cols[0] if cols else None


def _variance_sorted_numeric(df: pd.DataFrame, inferred: dict[str, Any], limit: int = 8) -> list[str]:
    cols = _numeric_columns(df, inferred, max_cols=80)
    scored = [(float(np.nanvar(pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float))), c) for c in cols]
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [c for _, c in scored[:limit]]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        x = float(value)
        return default if not math.isfinite(x) else x
    except Exception:
        return default


def _find_parent_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        cl = str(col).lower().replace(" ", "_")
        if any(h in cl for h in ("parent", "ppid", "parent_id", "parent_process", "caller", "upstream")):
            nf = float(df[col].isna().mean())
            if 0.01 < nf < 0.99:
                return str(col)
    return None


def _build_dag(df: pd.DataFrame, parent_col: str, id_col: str | None = None) -> Any:
    if not HAS_NETWORKX:
        return None
    G = nx.DiGraph()
    ids = df.index.tolist() if id_col is None else df[id_col].tolist()
    for idx, par in zip(ids, df[parent_col].tolist()):
        G.add_node(idx)
        if pd.notna(par):
            G.add_edge(par, idx)
    return G


def _find_graph_columns(df: pd.DataFrame, inferred: dict[str, Any]) -> tuple[str | None, str | None]:
    cols = list(df.columns)
    low = {c: str(c).lower() for c in cols}
    src = dst = None
    for c in cols:
        if any(h in low[c] for h in ("src", "from", "parent", "source", "caller")):
            src = c
            break
    for c in cols:
        if c == src:
            continue
        if any(h in low[c] for h in ("dst", "to", "child", "target", "callee")):
            dst = c
            break
    if src and dst:
        return str(src), str(dst)
    cats = _categorical_columns(df, inferred, max_cols=6)
    return (cats[0], cats[1]) if len(cats) >= 2 else (None, None)


def _build_edges(df: pd.DataFrame, inferred: dict[str, Any], max_edges: int = 20000) -> tuple[pd.DataFrame, str, str]:
    sc, dc = _find_graph_columns(df, inferred)
    if sc is None or dc is None:
        return pd.DataFrame(columns=["src", "dst"]), "", ""
    edges = pd.DataFrame({"src": df[sc].astype(str), "dst": df[dc].astype(str)})
    edges = edges[(edges["src"] != "") & (edges["dst"] != "") & (edges["src"] != "nan") & (edges["dst"] != "nan")]
    edges = edges[edges["src"] != edges["dst"]]
    if len(edges) > max_edges:
        edges = edges.iloc[:max_edges]
    return edges, sc, dc


def _artifact_json(ctx: Any, plugin_id: str, filename: str, payload: Any, description: str) -> PluginArtifact:
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


def _event_type_column(df: pd.DataFrame, inferred: dict[str, Any]) -> str | None:
    hints = ("status", "type", "outcome", "event", "result", "step", "activity")
    for col in _categorical_columns(df, inferred, max_cols=20):
        if any(h in col.lower() for h in hints):
            return col
    cols = _categorical_columns(df, inferred, max_cols=20)
    return cols[0] if cols else None


def _host_column(df: pd.DataFrame, inferred: dict[str, Any]) -> str | None:
    hints = ("host", "server", "machine", "node", "agent", "worker", "instance")
    for col in _categorical_columns(df, inferred, max_cols=20):
        if any(h in col.lower() for h in hints):
            return col
    return None


def _success_column(df: pd.DataFrame, inferred: dict[str, Any]) -> str | None:
    hints = ("success", "pass", "ok", "completed", "win", "good")
    for col in df.columns:
        cl = str(col).lower()
        if any(h in cl for h in hints):
            return str(col)
    return None


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


def _levenshtein_dp(a: list[str], b: list[str]) -> int:
    """Manual Levenshtein distance via DP on string sequences."""
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    return dp[m]


# ---------------------------------------------------------------------------
# Handler 73: WAR (Value Above Replacement)
# ---------------------------------------------------------------------------

def _handler_war_value_above_replacement_v1(
    plugin_id: str,
    ctx: Any,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    host_col = _host_column(df, inferred)
    if host_col is None:
        cats = _categorical_columns(df, inferred, max_cols=6)
        host_col = cats[0] if cats else None
    if host_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_categorical_column_for_host")
    dur_col = _duration_column(df, inferred)
    if dur_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_column")
    vals = pd.to_numeric(df[dur_col], errors="coerce")
    mask = vals.notna()
    if mask.sum() < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_numeric_data")
    work = df.loc[mask].copy()
    work["_val"] = vals[mask].to_numpy(dtype=float)
    # Replacement level = overall median
    replacement = float(np.nanmedian(work["_val"].to_numpy(dtype=float)))
    groups = work.groupby(host_col, sort=False)
    rows = []
    for name, grp in groups:
        if len(grp) < 3:
            continue
        arr = grp["_val"].to_numpy(dtype=float)
        mean_val = float(np.nanmean(arr))
        war = float(mean_val - replacement)
        rows.append({"host": str(name), "mean": mean_val, "war": war, "n": int(len(grp))})
        _ensure_budget(timer)
    if not rows:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_valid_groups")
    rows.sort(key=lambda r: (-r["war"], r["host"]))
    findings = []
    top = rows[0]
    bottom = rows[-1]
    spread = top["war"] - bottom["war"]
    if spread > 0:
        findings.append(
            _make_finding(
                plugin_id,
                "war_spread",
                "Value-above-replacement spread detected",
                f"Top host contributes WAR={top['war']:.3f}, bottom WAR={bottom['war']:.3f} (spread={spread:.3f}).",
                "Large WAR spread indicates uneven component quality; low-WAR hosts may be candidates for replacement.",
                {"metrics": {"top": top, "bottom": bottom, "spread": spread, "replacement_level": replacement, "n_hosts": len(rows)}},
                recommendation="Prioritize replacement or remediation of bottom-WAR hosts to raise overall throughput.",
                severity="warn" if spread < float(np.nanstd(work["_val"].to_numpy(dtype=float))) else "critical",
                confidence=min(0.90, 0.50 + min(0.35, len(rows) / 50.0)),
            )
        )
    artifacts = [
        _artifact_json(ctx, plugin_id, "war_rankings.json", {"replacement_level": replacement, "hosts": rows[:30]}, "WAR rankings per host"),
    ]
    return _finalize(
        plugin_id, ctx, df, sample_meta,
        "Computed value-above-replacement rankings.",
        findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "n_hosts": len(rows), "spread": spread},
    )


# ---------------------------------------------------------------------------
# Handler 74: Win Probability Added (WPA)
# ---------------------------------------------------------------------------

def _handler_win_probability_added_v1(
    plugin_id: str,
    ctx: Any,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    step_col = _event_type_column(df, inferred)
    if step_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_step_column")
    succ_col = _success_column(df, inferred)
    if succ_col is None:
        # Fall back: use duration column and define "win" as below-median
        dur_col = _duration_column(df, inferred)
        if dur_col is None:
            return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_success_or_duration_column")
        vals = pd.to_numeric(df[dur_col], errors="coerce")
        med = float(np.nanmedian(vals.dropna().to_numpy(dtype=float)))
        outcome = (vals <= med).astype(float)
    else:
        outcome = pd.to_numeric(df[succ_col], errors="coerce").fillna(0.0)
        # Coerce boolean-like to 0/1
        if set(outcome.dropna().unique().tolist()) - {0.0, 1.0}:
            outcome = (outcome > 0).astype(float)
    overall_win = float(outcome.mean())
    if overall_win <= 0 or overall_win >= 1.0:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "degenerate_outcome")
    steps = df[step_col].astype(str)
    unique_steps = [s for s in steps.unique().tolist() if s and s != "nan"]
    if len(unique_steps) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "too_few_steps")
    rows = []
    for step in unique_steps[:50]:
        mask = steps == step
        if mask.sum() < 5:
            continue
        p_win_given_step = float(outcome[mask].mean())
        p_win_given_not = float(outcome[~mask].mean())
        wpa = p_win_given_step - p_win_given_not
        rows.append({"step": step, "wpa": wpa, "p_win_step": p_win_given_step, "p_win_not_step": p_win_given_not, "n": int(mask.sum())})
        _ensure_budget(timer)
    if not rows:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_valid_steps")
    rows.sort(key=lambda r: (-abs(r["wpa"]), r["step"]))
    findings = []
    top = rows[0]
    if abs(top["wpa"]) > 0.05:
        findings.append(
            _make_finding(
                plugin_id,
                f"wpa:{top['step']}",
                "High win-probability-added step",
                f"Step '{top['step']}' has WPA={top['wpa']:.3f} (P(win|step)={top['p_win_step']:.3f} vs P(win|~step)={top['p_win_not_step']:.3f}).",
                "High WPA steps disproportionately influence overall success rates.",
                {"metrics": top},
                recommendation="Focus reliability investment on highest-WPA steps to protect success probability.",
                severity="warn" if abs(top["wpa"]) < 0.15 else "critical",
                confidence=min(0.88, 0.50 + min(0.30, top["n"] / 200.0)),
            )
        )
    artifacts = [_artifact_json(ctx, plugin_id, "wpa_steps.json", {"overall_win_rate": overall_win, "steps": rows[:30]}, "WPA per step")]
    return _finalize(
        plugin_id, ctx, df, sample_meta,
        "Computed win-probability-added per step.",
        findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "n_steps": len(rows)},
    )


# ---------------------------------------------------------------------------
# Handler 75: Leverage Index (DAG)
# ---------------------------------------------------------------------------

def _handler_leverage_index_dag_v1(
    plugin_id: str,
    ctx: Any,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    step_col = _event_type_column(df, inferred)
    if step_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_step_column")
    dur_col = _duration_column(df, inferred)
    if dur_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_column")
    vals = pd.to_numeric(df[dur_col], errors="coerce")
    med = float(np.nanmedian(vals.dropna().to_numpy(dtype=float)))
    outcome = (vals <= med).astype(float)
    overall_win = float(outcome.mean())
    if overall_win <= 0 or overall_win >= 1.0:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "degenerate_outcome")
    steps = df[step_col].astype(str)
    unique_steps = [s for s in steps.unique().tolist() if s and s != "nan"]
    if len(unique_steps) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "too_few_steps")
    wpa_vals = []
    step_info = []
    for step in unique_steps[:50]:
        mask = steps == step
        if mask.sum() < 5:
            continue
        p_ok = float(outcome[mask].mean())
        p_not = float(outcome[~mask].mean())
        wpa = abs(p_ok - p_not)
        wpa_vals.append(wpa)
        step_info.append({"step": step, "wpa_swing": wpa, "n": int(mask.sum())})
        _ensure_budget(timer)
    if not wpa_vals:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_valid_steps")
    mean_swing = float(np.mean(wpa_vals))
    for row in step_info:
        row["leverage_index"] = float(row["wpa_swing"] / max(1e-9, mean_swing))
    step_info.sort(key=lambda r: (-r["leverage_index"], r["step"]))
    findings = []
    top = step_info[0]
    if top["leverage_index"] > 1.5:
        findings.append(
            _make_finding(
                plugin_id,
                f"li:{top['step']}",
                "High-leverage step identified",
                f"Step '{top['step']}' has leverage index {top['leverage_index']:.2f} (mean swing={mean_swing:.3f}).",
                "High-leverage steps have outsized impact on outcome variance; failures here are disproportionately costly.",
                {"metrics": top},
                recommendation="Add redundancy, monitoring, and fallback paths around high-leverage steps.",
                severity="warn" if top["leverage_index"] < 3.0 else "critical",
                confidence=min(0.85, 0.50 + min(0.30, len(step_info) / 30.0)),
            )
        )
    artifacts = [_artifact_json(ctx, plugin_id, "leverage_index.json", {"mean_swing": mean_swing, "steps": step_info[:30]}, "Leverage index per step")]
    return _finalize(
        plugin_id, ctx, df, sample_meta,
        "Computed leverage index per step.",
        findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "n_steps": len(step_info), "mean_swing": mean_swing},
    )


# ---------------------------------------------------------------------------
# Handler 76: Pythagorean Expectation
# ---------------------------------------------------------------------------

def _handler_pythagorean_expectation_v1(
    plugin_id: str,
    ctx: Any,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    host_col = _host_column(df, inferred)
    if host_col is None:
        cats = _categorical_columns(df, inferred, max_cols=6)
        host_col = cats[0] if cats else None
    if host_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_host_column")
    dur_col = _duration_column(df, inferred)
    if dur_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_column")
    vals = pd.to_numeric(df[dur_col], errors="coerce")
    mask = vals.notna()
    if mask.sum() < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_data")
    work = df.loc[mask].copy()
    work["_val"] = vals[mask].to_numpy(dtype=float)
    med = float(np.nanmedian(work["_val"].to_numpy(dtype=float)))
    alpha = float(config.get("plugin", {}).get("alpha", 2.0))
    groups = work.groupby(host_col, sort=False)
    rows = []
    for name, grp in groups:
        if len(grp) < 5:
            continue
        arr = grp["_val"].to_numpy(dtype=float)
        good = float(np.sum(arr <= med))
        bad = float(np.sum(arr > med))
        total = good + bad
        if total < 1:
            continue
        actual_rate = good / total
        denom = good ** alpha + bad ** alpha
        expected_rate = (good ** alpha / denom) if denom > 0 else 0.5
        luck = actual_rate - expected_rate
        rows.append({"host": str(name), "good": int(good), "bad": int(bad), "actual_rate": actual_rate, "expected_rate": expected_rate, "luck_factor": luck})
        _ensure_budget(timer)
    if not rows:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_valid_groups")
    rows.sort(key=lambda r: (-abs(r["luck_factor"]), r["host"]))
    findings = []
    top = rows[0]
    if abs(top["luck_factor"]) > 0.05:
        direction = "over-performing" if top["luck_factor"] > 0 else "under-performing"
        findings.append(
            _make_finding(
                plugin_id,
                f"pyth:{top['host']}",
                "Pythagorean expectation deviation",
                f"Host '{top['host']}' is {direction}: actual={top['actual_rate']:.3f} vs expected={top['expected_rate']:.3f} (luck={top['luck_factor']:.3f}).",
                "Sustained deviation from pythagorean expectation suggests hidden quality factors or measurement artifacts.",
                {"metrics": top},
                recommendation="Investigate root causes of deviation from expected performance; over-performers may mask fragility.",
                severity="info" if abs(top["luck_factor"]) < 0.10 else "warn",
                confidence=min(0.80, 0.45 + min(0.30, len(rows) / 40.0)),
            )
        )
    artifacts = [_artifact_json(ctx, plugin_id, "pythagorean.json", {"alpha": alpha, "hosts": rows[:30]}, "Pythagorean expectation per host")]
    return _finalize(
        plugin_id, ctx, df, sample_meta,
        "Computed pythagorean expectation analysis.",
        findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "n_hosts": len(rows), "alpha": alpha},
    )


# ---------------------------------------------------------------------------
# Handler 77: Zipf's Law Frequency
# ---------------------------------------------------------------------------

def _handler_zipf_law_frequency_v1(
    plugin_id: str,
    ctx: Any,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    step_col = _event_type_column(df, inferred)
    if step_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_step_column")
    counts = Counter(df[step_col].astype(str).tolist())
    # Remove nan
    counts.pop("nan", None)
    counts.pop("", None)
    if len(counts) < 5:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "too_few_categories")
    ranked = sorted(counts.values(), reverse=True)
    ranks = np.arange(1, len(ranked) + 1, dtype=float)
    freqs = np.array(ranked, dtype=float)
    log_rank = np.log(ranks)
    log_freq = np.log(np.maximum(freqs, 1.0))
    beta, sse = _fit_linear(log_rank, log_freq)
    slope = float(beta[1]) if len(beta) > 1 else 0.0
    # R^2
    yvar = float(np.var(log_freq))
    r2 = 1.0 - sse / max(1e-9, yvar * len(log_freq)) if yvar > 0 else 0.0
    r2 = max(-1.0, min(1.0, r2))
    # Zipf's law: slope should be approximately -1
    zipf_deviation = abs(slope + 1.0)
    findings = []
    if r2 > 0.7:
        if zipf_deviation > 0.3:
            findings.append(
                _make_finding(
                    plugin_id,
                    "zipf_deviation",
                    "Zipf's law deviation in event frequencies",
                    f"Rank-frequency slope={slope:.3f} (Zipf expects -1.0), deviation={zipf_deviation:.3f}, R2={r2:.3f}.",
                    "Non-Zipfian event distributions may indicate artificial throttling, missing categories, or unusual load patterns.",
                    {"metrics": {"slope": slope, "r2": r2, "zipf_deviation": zipf_deviation, "n_categories": len(counts)}},
                    recommendation="Investigate whether event frequency distribution reflects natural process or artificial constraints.",
                    severity="info" if zipf_deviation < 0.5 else "warn",
                    confidence=min(0.85, 0.50 + min(0.30, r2)),
                )
            )
        else:
            findings.append(
                _make_finding(
                    plugin_id,
                    "zipf_confirmed",
                    "Event frequencies follow Zipf's law",
                    f"Rank-frequency slope={slope:.3f} closely matches Zipf expectation (R2={r2:.3f}).",
                    "Zipfian distribution is typical of natural, organic event generation processes.",
                    {"metrics": {"slope": slope, "r2": r2, "zipf_deviation": zipf_deviation}},
                    recommendation="Distribution is consistent with natural patterns; no action required.",
                    severity="info",
                    confidence=min(0.85, 0.55 + min(0.25, r2)),
                )
            )
    artifacts = [_artifact_json(ctx, plugin_id, "zipf_analysis.json", {"slope": slope, "r2": r2, "zipf_deviation": zipf_deviation, "n_categories": len(counts), "top_10": [{"rank": int(i + 1), "freq": int(ranked[i])} for i in range(min(10, len(ranked)))]}, "Zipf's law analysis")]
    return _finalize(
        plugin_id, ctx, df, sample_meta,
        "Computed Zipf's law rank-frequency analysis.",
        findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "slope": slope, "r2": r2},
    )


# ---------------------------------------------------------------------------
# Handler 78: Entropy Rate of Step Sequences
# ---------------------------------------------------------------------------

def _handler_entropy_rate_step_sequences_v1(
    plugin_id: str,
    ctx: Any,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    step_col = _event_type_column(df, inferred)
    if step_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_step_column")
    tc, ts = _time_series(df, inferred)
    if ts is not None:
        order = ts.argsort(kind="mergesort")
        seq = df[step_col].astype(str).iloc[order].tolist()
    else:
        seq = df[step_col].astype(str).tolist()
    seq = [s for s in seq if s and s != "nan"]
    if len(seq) < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "too_few_events")
    # Unigram entropy
    unigram = Counter(seq)
    total = sum(unigram.values())
    h1 = -sum((c / total) * math.log2(max(c / total, 1e-12)) for c in unigram.values())
    # Bigram conditional entropy
    bigrams: dict[str, Counter] = defaultdict(Counter)
    for i in range(len(seq) - 1):
        bigrams[seq[i]][seq[i + 1]] += 1
        if i % 500 == 0:
            _ensure_budget(timer)
    h_cond = 0.0
    for prev, nexts in bigrams.items():
        p_prev = unigram[prev] / total
        total_next = sum(nexts.values())
        for cnt in nexts.values():
            p_next = cnt / total_next
            h_cond -= p_prev * p_next * math.log2(max(p_next, 1e-12))
    # Trigram for comparison
    trigrams: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for i in range(len(seq) - 2):
        trigrams[(seq[i], seq[i + 1])][seq[i + 2]] += 1
    bigram_total = Counter()
    for i in range(len(seq) - 1):
        bigram_total[(seq[i], seq[i + 1])] += 1
    h_cond2 = 0.0
    total_bigrams = sum(bigram_total.values())
    for bg, nexts in trigrams.items():
        p_bg = bigram_total[bg] / max(total_bigrams, 1)
        total_next = sum(nexts.values())
        for cnt in nexts.values():
            p_next = cnt / total_next
            h_cond2 -= p_bg * p_next * math.log2(max(p_next, 1e-12))
    findings = []
    # Low entropy rate suggests high predictability (automatable)
    if h_cond < 1.0 and len(unigram) > 2:
        findings.append(
            _make_finding(
                plugin_id,
                "low_entropy_rate",
                "Low entropy rate: highly predictable sequences",
                f"Bigram conditional entropy H={h_cond:.3f} bits (unigram H={h1:.3f}). Sequences are highly predictable.",
                "Low entropy rate indicates step sequences follow rigid patterns; candidates for automation.",
                {"metrics": {"h_unigram": h1, "h_cond_bigram": h_cond, "h_cond_trigram": h_cond2, "n_events": len(seq), "n_types": len(unigram)}},
                recommendation="Evaluate whether predictable step patterns can be automated or streamlined.",
                severity="info" if h_cond > 0.5 else "warn",
                confidence=min(0.85, 0.55 + min(0.25, len(seq) / 1000.0)),
            )
        )
    elif h_cond > 3.0:
        findings.append(
            _make_finding(
                plugin_id,
                "high_entropy_rate",
                "High entropy rate: chaotic sequences",
                f"Bigram conditional entropy H={h_cond:.3f} bits. Sequences are unpredictable.",
                "High entropy rate means step transitions lack pattern; may indicate process instability.",
                {"metrics": {"h_unigram": h1, "h_cond_bigram": h_cond, "h_cond_trigram": h_cond2, "n_events": len(seq), "n_types": len(unigram)}},
                recommendation="Investigate sources of unpredictable step transitions; consider standardizing workflows.",
                severity="warn",
                confidence=min(0.80, 0.50 + min(0.25, len(seq) / 1000.0)),
            )
        )
    artifacts = [_artifact_json(ctx, plugin_id, "entropy_rate.json", {"h_unigram": h1, "h_cond_bigram": h_cond, "h_cond_trigram": h_cond2, "n_events": len(seq), "n_types": len(unigram)}, "Step sequence entropy rate")]
    return _finalize(
        plugin_id, ctx, df, sample_meta,
        "Computed entropy rate of step sequences.",
        findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "h_unigram": h1, "h_cond_bigram": h_cond},
    )


# ---------------------------------------------------------------------------
# Handler 79: Levenshtein Workflow Distance
# ---------------------------------------------------------------------------

def _handler_levenshtein_workflow_dist_v1(
    plugin_id: str,
    ctx: Any,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    step_col = _event_type_column(df, inferred)
    if step_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_step_column")
    # Try to group by a case/job column
    case_col = None
    for col in _categorical_columns(df, inferred, max_cols=20):
        cl = str(col).lower()
        if any(h in cl for h in ("case", "job", "trace", "session", "ticket", "id")):
            case_col = str(col)
            break
    if case_col is None:
        # Without case grouping, chunk sequence by fixed windows
        seq = df[step_col].astype(str).tolist()
        seq = [s for s in seq if s and s != "nan"]
        chunk_size = max(5, len(seq) // 20)
        workflows = [seq[i:i + chunk_size] for i in range(0, len(seq), chunk_size) if len(seq[i:i + chunk_size]) >= 3]
    else:
        groups = df.groupby(case_col, sort=False)
        workflows = []
        for _, grp in groups:
            wf = grp[step_col].astype(str).tolist()
            wf = [s for s in wf if s and s != "nan"]
            if len(wf) >= 2:
                workflows.append(wf)
            if len(workflows) >= 200:
                break
    if len(workflows) < 3:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "too_few_workflows")
    # Compute pairwise distances (capped)
    max_pairs = min(500, len(workflows) * (len(workflows) - 1) // 2)
    dists = []
    pair_count = 0
    for i, j in combinations(range(len(workflows)), 2):
        if pair_count >= max_pairs:
            break
        a, b = workflows[i], workflows[j]
        if HAS_RAPIDFUZZ:
            d = rf_distance.Levenshtein.distance(a, b)
        else:
            d = _levenshtein_dp(a, b)
        norm_d = d / max(len(a), len(b), 1)
        dists.append(norm_d)
        pair_count += 1
        if pair_count % 50 == 0:
            _ensure_budget(timer)
    if not dists:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_distances_computed")
    arr = np.array(dists, dtype=float)
    mean_dist = float(np.mean(arr))
    std_dist = float(np.std(arr))
    median_dist = float(np.median(arr))
    findings = []
    if mean_dist < 0.2:
        findings.append(
            _make_finding(
                plugin_id,
                "similar_workflows",
                "Workflows are highly similar (low edit distance)",
                f"Mean normalized edit distance={mean_dist:.3f} across {len(dists)} pairs. Workflows are nearly identical.",
                "Highly similar workflows are good candidates for templating or standardization.",
                {"metrics": {"mean_dist": mean_dist, "std_dist": std_dist, "median_dist": median_dist, "n_workflows": len(workflows), "n_pairs": len(dists)}},
                recommendation="Consider templating the dominant workflow pattern to reduce variation.",
                severity="info",
                confidence=0.75,
            )
        )
    elif mean_dist > 0.6:
        findings.append(
            _make_finding(
                plugin_id,
                "diverse_workflows",
                "Workflows are highly diverse (high edit distance)",
                f"Mean normalized edit distance={mean_dist:.3f}. Workflows vary substantially.",
                "High workflow diversity may indicate lack of standardization or multiple process variants.",
                {"metrics": {"mean_dist": mean_dist, "std_dist": std_dist, "median_dist": median_dist, "n_workflows": len(workflows), "n_pairs": len(dists)}},
                recommendation="Investigate whether workflow diversity is intentional or reflects process drift.",
                severity="warn",
                confidence=0.70,
            )
        )
    artifacts = [_artifact_json(ctx, plugin_id, "levenshtein.json", {"mean_dist": mean_dist, "std_dist": std_dist, "median_dist": median_dist, "n_workflows": len(workflows), "n_pairs": len(dists)}, "Levenshtein workflow distance summary")]
    return _finalize(
        plugin_id, ctx, df, sample_meta,
        "Computed Levenshtein workflow distances.",
        findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "mean_dist": mean_dist, "n_workflows": len(workflows)},
    )


# ---------------------------------------------------------------------------
# Handler 80: N-gram Step Transitions
# ---------------------------------------------------------------------------

def _handler_ngram_step_transitions_v1(
    plugin_id: str,
    ctx: Any,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    step_col = _event_type_column(df, inferred)
    if step_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_step_column")
    tc, ts = _time_series(df, inferred)
    if ts is not None:
        order = ts.argsort(kind="mergesort")
        seq = df[step_col].astype(str).iloc[order].tolist()
    else:
        seq = df[step_col].astype(str).tolist()
    seq = [s for s in seq if s and s != "nan"]
    if len(seq) < 10:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "too_few_events")
    # Bigrams
    bigrams: Counter = Counter()
    for i in range(len(seq) - 1):
        bigrams[(seq[i], seq[i + 1])] += 1
    # Trigrams
    trigrams: Counter = Counter()
    for i in range(len(seq) - 2):
        trigrams[(seq[i], seq[i + 1], seq[i + 2])] += 1
    total_bi = sum(bigrams.values())
    total_tri = sum(trigrams.values())
    # Find rare/surprising transitions (bottom 10% frequency)
    if not bigrams:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_transitions")
    bi_sorted = bigrams.most_common()
    threshold = max(1, total_bi // (10 * len(bigrams))) if len(bigrams) > 0 else 1
    rare_bi = [(bg, cnt) for bg, cnt in bi_sorted if cnt <= threshold]
    common_bi = bi_sorted[:10]
    findings = []
    if rare_bi:
        n_rare = len(rare_bi)
        findings.append(
            _make_finding(
                plugin_id,
                "rare_transitions",
                "Rare step transitions detected",
                f"Found {n_rare} rare bigram transitions (frequency <= {threshold}).",
                "Rare transitions may indicate error paths, manual overrides, or process anomalies.",
                {"metrics": {"n_rare_bigrams": n_rare, "total_bigrams": total_bi, "threshold": threshold, "examples": [{"transition": list(bg), "count": int(cnt)} for bg, cnt in rare_bi[:5]]}},
                recommendation="Review rare transitions for error paths or unauthorized workflow deviations.",
                severity="info" if n_rare < 5 else "warn",
                confidence=min(0.80, 0.50 + min(0.25, total_bi / 500.0)),
            )
        )
    artifacts = [_artifact_json(ctx, plugin_id, "ngram_transitions.json", {
        "n_events": len(seq),
        "n_unique_bigrams": len(bigrams),
        "n_unique_trigrams": len(trigrams),
        "top_bigrams": [{"transition": list(bg), "count": int(cnt)} for bg, cnt in common_bi],
        "rare_bigrams": [{"transition": list(bg), "count": int(cnt)} for bg, cnt in rare_bi[:10]],
    }, "N-gram step transition analysis")]
    return _finalize(
        plugin_id, ctx, df, sample_meta,
        "Computed n-gram step transition analysis.",
        findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "n_bigrams": len(bigrams), "n_trigrams": len(trigrams)},
    )


# ---------------------------------------------------------------------------
# Handler 81: Weber-Fechner Threshold
# ---------------------------------------------------------------------------

def _handler_weber_fechner_threshold_v1(
    plugin_id: str,
    ctx: Any,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    tc, ts = _time_series(df, inferred)
    dur_col = _duration_column(df, inferred)
    if dur_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_column")
    vals = pd.to_numeric(df[dur_col], errors="coerce").dropna().to_numpy(dtype=float)
    if len(vals) < 30:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_data")
    # If time-ordered, compute consecutive deltas
    if ts is not None:
        order = ts.dropna().argsort(kind="mergesort")
        ordered = pd.to_numeric(df[dur_col], errors="coerce").iloc[order].dropna().to_numpy(dtype=float)
    else:
        ordered = vals
    if len(ordered) < 30:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_ordered_data")
    # Weber fraction: delta_I / I for consecutive observations
    I = ordered[:-1]
    delta_I = np.abs(np.diff(ordered))
    valid = I > 0
    if valid.sum() < 10:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_positive_values")
    weber_fractions = delta_I[valid] / I[valid]
    k = float(np.median(weber_fractions))
    k_std = float(np.std(weber_fractions))
    # Just-noticeable-difference at various intensity levels
    intensity_levels = np.percentile(vals[vals > 0], [25, 50, 75, 90])
    jnd_table = [{"intensity": float(il), "jnd": float(k * il)} for il in intensity_levels]
    findings = []
    if k > 0 and k < 2.0:
        findings.append(
            _make_finding(
                plugin_id,
                "weber_fraction",
                "Weber-Fechner just-noticeable-difference estimated",
                f"Weber fraction k={k:.4f} (std={k_std:.4f}). Minimum detectable change scales with intensity.",
                "Alert thresholds should scale with baseline intensity: a fixed threshold over-alerts at low levels and under-alerts at high levels.",
                {"metrics": {"weber_k": k, "weber_k_std": k_std, "jnd_table": jnd_table, "n_observations": len(ordered)}},
                recommendation=f"Set adaptive alert thresholds at {k:.1%} of baseline intensity rather than fixed values.",
                severity="info" if k < 0.3 else "warn",
                confidence=min(0.82, 0.50 + min(0.28, len(ordered) / 500.0)),
            )
        )
    artifacts = [_artifact_json(ctx, plugin_id, "weber_fechner.json", {"weber_k": k, "weber_k_std": k_std, "jnd_table": jnd_table, "n_observations": len(ordered)}, "Weber-Fechner threshold analysis")]
    return _finalize(
        plugin_id, ctx, df, sample_meta,
        "Computed Weber-Fechner threshold analysis.",
        findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "weber_k": k},
    )


# ---------------------------------------------------------------------------
# Handler 82: Stevens' Power Law Severity
# ---------------------------------------------------------------------------

def _handler_stevens_power_law_severity_v1(
    plugin_id: str,
    ctx: Any,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    num_cols = _variance_sorted_numeric(df, inferred, limit=6)
    if len(num_cols) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_numeric_columns")
    results = []
    for col in num_cols:
        vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
        positive = vals[vals > 0]
        if len(positive) < 20:
            continue
        log_x = np.log(positive)
        # Use rank as proxy for "perceived severity"
        ranks = np.argsort(np.argsort(positive)).astype(float) + 1.0
        log_y = np.log(ranks)
        beta, sse = _fit_linear(log_x, log_y)
        n_exp = float(beta[1]) if len(beta) > 1 else 1.0
        k = float(np.exp(beta[0])) if len(beta) > 0 else 1.0
        yvar = float(np.var(log_y))
        r2 = 1.0 - sse / max(1e-9, yvar * len(log_y)) if yvar > 0 else 0.0
        r2 = max(-1.0, min(1.0, r2))
        results.append({"column": col, "exponent_n": n_exp, "scale_k": k, "r2": r2, "n_positive": int(len(positive))})
        _ensure_budget(timer)
    if not results:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_fittable_columns")
    results.sort(key=lambda r: (-r["r2"], r["column"]))
    findings = []
    top = results[0]
    if top["r2"] > 0.5:
        findings.append(
            _make_finding(
                plugin_id,
                f"power_law:{top['column']}",
                "Stevens' power law fit for severity mapping",
                f"Column '{top['column']}': perceived_severity ~ {top['scale_k']:.3f} * actual^{top['exponent_n']:.3f} (R2={top['r2']:.3f}).",
                "Power-law severity mapping enables unified severity scores across metrics with different scales.",
                {"metrics": top},
                recommendation="Use the fitted power-law exponent to build a unified severity scale across heterogeneous metrics.",
                severity="info",
                confidence=min(0.82, 0.50 + min(0.28, top["r2"])),
            )
        )
    artifacts = [_artifact_json(ctx, plugin_id, "stevens_power_law.json", {"columns": results}, "Stevens' power law severity fit")]
    return _finalize(
        plugin_id, ctx, df, sample_meta,
        "Computed Stevens' power law severity mapping.",
        findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "n_columns_fitted": len(results)},
    )


# ---------------------------------------------------------------------------
# Handler 83: Signal Detection d-prime
# ---------------------------------------------------------------------------

def _handler_signal_detection_dprime_v1(
    plugin_id: str,
    ctx: Any,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    # Look for hit/miss/false-alarm columns or binary outcome columns
    cats = _categorical_columns(df, inferred, max_cols=20)
    alert_col = None
    truth_col = None
    for col in cats:
        cl = str(col).lower()
        if any(h in cl for h in ("alert", "predicted", "flagged", "detected")):
            alert_col = col
        elif any(h in cl for h in ("actual", "truth", "real", "ground", "label")):
            truth_col = col
    if alert_col is None or truth_col is None:
        # Try numeric binary columns
        num_cols = _numeric_columns(df, inferred, max_cols=20)
        binary = []
        for col in num_cols:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            uniq = set(vals.unique().tolist())
            if uniq.issubset({0.0, 1.0}) and len(uniq) == 2:
                binary.append(col)
        if len(binary) >= 2:
            alert_col = binary[0]
            truth_col = binary[1]
        elif alert_col and not truth_col and binary:
            truth_col = binary[0]
        elif truth_col and not alert_col and binary:
            alert_col = binary[0]
    if alert_col is None or truth_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_alert_truth_columns")
    pred = pd.to_numeric(df[alert_col], errors="coerce").fillna(0)
    pred = (pred > 0).astype(int)
    truth = pd.to_numeric(df[truth_col], errors="coerce").fillna(0)
    truth = (truth > 0).astype(int)
    n = len(pred)
    if n < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_data")
    tp = int(((pred == 1) & (truth == 1)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())
    fn = int(((pred == 0) & (truth == 1)).sum())
    tn = int(((pred == 0) & (truth == 0)).sum())
    # Hit rate and false-alarm rate with Hautus (1995) correction
    hit_rate = (tp + 0.5) / (tp + fn + 1.0)
    fa_rate = (fp + 0.5) / (fp + tn + 1.0)
    if HAS_SCIPY:
        z_hit = float(scipy_stats.norm.ppf(hit_rate))
        z_fa = float(scipy_stats.norm.ppf(fa_rate))
    else:
        # Probit approximation
        def _probit(p: float) -> float:
            p = max(1e-6, min(1 - 1e-6, p))
            # Rational approximation (Abramowitz & Stegun 26.2.23)
            t = math.sqrt(-2.0 * math.log(p if p < 0.5 else 1 - p))
            c = [2.515517, 0.802853, 0.010328]
            d = [1.432788, 0.189269, 0.001308]
            z = t - (c[0] + c[1] * t + c[2] * t * t) / (1 + d[0] * t + d[1] * t * t + d[2] * t * t * t)
            return z if p >= 0.5 else -z
        z_hit = _probit(hit_rate)
        z_fa = _probit(fa_rate)
    dprime = z_hit - z_fa
    criterion = -0.5 * (z_hit + z_fa)
    findings = []
    if abs(dprime) < 0.5:
        findings.append(
            _make_finding(
                plugin_id,
                "low_dprime",
                "Low detection sensitivity (d-prime near zero)",
                f"d'={dprime:.3f}, criterion c={criterion:.3f}. Detection is near chance.",
                "Low d-prime means the detection system cannot reliably distinguish signal from noise.",
                {"metrics": {"dprime": dprime, "criterion": criterion, "hit_rate": hit_rate, "fa_rate": fa_rate, "tp": tp, "fp": fp, "fn": fn, "tn": tn}},
                recommendation="Redesign detection rules or thresholds; current system adds noise rather than signal.",
                severity="critical",
                confidence=min(0.90, 0.55 + min(0.30, n / 200.0)),
            )
        )
    elif dprime > 2.0:
        findings.append(
            _make_finding(
                plugin_id,
                "high_dprime",
                "Strong detection sensitivity",
                f"d'={dprime:.3f}, criterion c={criterion:.3f}. Detection system performs well.",
                "High d-prime indicates reliable discrimination between signal and noise.",
                {"metrics": {"dprime": dprime, "criterion": criterion, "hit_rate": hit_rate, "fa_rate": fa_rate}},
                recommendation="Detection quality is strong; focus on criterion calibration to balance precision vs recall.",
                severity="info",
                confidence=min(0.88, 0.60 + min(0.25, n / 200.0)),
            )
        )
    else:
        findings.append(
            _make_finding(
                plugin_id,
                "moderate_dprime",
                "Moderate detection sensitivity",
                f"d'={dprime:.3f}, criterion c={criterion:.3f}.",
                "Moderate sensitivity; detection works but has room for improvement.",
                {"metrics": {"dprime": dprime, "criterion": criterion, "hit_rate": hit_rate, "fa_rate": fa_rate}},
                recommendation="Consider feature engineering or threshold tuning to improve discriminability.",
                severity="info",
                confidence=min(0.80, 0.50 + min(0.25, n / 200.0)),
            )
        )
    artifacts = [_artifact_json(ctx, plugin_id, "dprime.json", {"dprime": dprime, "criterion": criterion, "hit_rate": hit_rate, "fa_rate": fa_rate, "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}}, "Signal detection d-prime analysis")]
    return _finalize(
        plugin_id, ctx, df, sample_meta,
        "Computed signal detection d-prime analysis.",
        findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "dprime": dprime, "criterion": criterion},
    )


# ---------------------------------------------------------------------------
# Handler 84: IRT Step Difficulty
# ---------------------------------------------------------------------------

def _handler_irt_step_difficulty_v1(
    plugin_id: str,
    ctx: Any,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    step_col = _event_type_column(df, inferred)
    host_col = _host_column(df, inferred)
    if host_col is None:
        cats = _categorical_columns(df, inferred, max_cols=6)
        host_col = cats[0] if cats else None
    if step_col is None or host_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "need_step_and_host_columns")
    # Build success matrix: host x step
    dur_col = _duration_column(df, inferred)
    if dur_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_column")
    vals = pd.to_numeric(df[dur_col], errors="coerce")
    med = float(np.nanmedian(vals.dropna().to_numpy(dtype=float)))
    work = df.copy()
    work["_success"] = (vals <= med).astype(float)
    work["_step"] = work[step_col].astype(str)
    work["_host"] = work[host_col].astype(str)
    # Compute pass rates per step (difficulty) and per host (ability)
    step_rates = work.groupby("_step", sort=False)["_success"].agg(["mean", "count"])
    step_rates = step_rates[step_rates["count"] >= 5]
    host_rates = work.groupby("_host", sort=False)["_success"].agg(["mean", "count"])
    host_rates = host_rates[host_rates["count"] >= 5]
    if len(step_rates) < 2 or len(host_rates) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_groups")
    # 2PL logistic approximation: difficulty = -logit(pass_rate)
    def _logit(p: float) -> float:
        p = max(0.01, min(0.99, p))
        return math.log(p / (1 - p))
    step_items = []
    for step, row in step_rates.iterrows():
        diff = -_logit(float(row["mean"]))
        step_items.append({"step": str(step), "difficulty": diff, "pass_rate": float(row["mean"]), "n": int(row["count"])})
    host_items = []
    for host, row in host_rates.iterrows():
        ability = _logit(float(row["mean"]))
        host_items.append({"host": str(host), "ability": ability, "pass_rate": float(row["mean"]), "n": int(row["count"])})
    step_items.sort(key=lambda r: (-r["difficulty"], r["step"]))
    host_items.sort(key=lambda r: (-r["ability"], r["host"]))
    findings = []
    hardest = step_items[0]
    if hardest["difficulty"] > 1.0:
        findings.append(
            _make_finding(
                plugin_id,
                f"irt_hard:{hardest['step']}",
                "IRT: high-difficulty step identified",
                f"Step '{hardest['step']}' has difficulty={hardest['difficulty']:.3f} (pass_rate={hardest['pass_rate']:.3f}).",
                "Steps with high IRT difficulty consistently defeat most hosts; they are process bottlenecks.",
                {"metrics": {"hardest_step": hardest, "n_steps": len(step_items), "n_hosts": len(host_items)}},
                recommendation="Simplify or add support for highest-difficulty steps to improve overall throughput.",
                severity="warn" if hardest["difficulty"] < 2.0 else "critical",
                confidence=min(0.85, 0.50 + min(0.30, hardest["n"] / 100.0)),
            )
        )
    artifacts = [_artifact_json(ctx, plugin_id, "irt_difficulty.json", {"steps": step_items[:30], "hosts": host_items[:30]}, "IRT step difficulty and host ability")]
    return _finalize(
        plugin_id, ctx, df, sample_meta,
        "Computed IRT step difficulty and host ability parameters.",
        findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "n_steps": len(step_items), "n_hosts": len(host_items)},
    )


# ---------------------------------------------------------------------------
# Handler 85: Rasch Step-Host Scale
# ---------------------------------------------------------------------------

def _handler_rasch_step_host_scale_v1(
    plugin_id: str,
    ctx: Any,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    step_col = _event_type_column(df, inferred)
    host_col = _host_column(df, inferred)
    if host_col is None:
        cats = _categorical_columns(df, inferred, max_cols=6)
        host_col = cats[0] if cats else None
    if step_col is None or host_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "need_step_and_host_columns")
    dur_col = _duration_column(df, inferred)
    if dur_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_column")
    vals = pd.to_numeric(df[dur_col], errors="coerce")
    med = float(np.nanmedian(vals.dropna().to_numpy(dtype=float)))
    work = df.copy()
    work["_success"] = (vals <= med).astype(float)
    work["_step"] = work[step_col].astype(str)
    work["_host"] = work[host_col].astype(str)
    # Cross-tabulate success rates
    ct = work.groupby(["_host", "_step"], sort=False)["_success"].agg(["mean", "count"])
    ct = ct[ct["count"] >= 3].reset_index()
    if len(ct) < 5:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_cross_tab")
    # JMLE-like iteration (simplified): alternate between estimating ability and difficulty
    steps_uniq = ct["_step"].unique().tolist()
    hosts_uniq = ct["_host"].unique().tolist()
    diff = {s: 0.0 for s in steps_uniq}
    abil = {h: 0.0 for h in hosts_uniq}

    def _logit(p: float) -> float:
        p = max(0.01, min(0.99, p))
        return math.log(p / (1 - p))

    for _iteration in range(10):
        # Update difficulty given ability
        for s in steps_uniq:
            mask = ct["_step"] == s
            sub = ct[mask]
            residuals = []
            for _, row in sub.iterrows():
                h = row["_host"]
                residuals.append(float(row["mean"]) - 1.0 / (1.0 + math.exp(-(abil.get(h, 0.0) - diff[s]))))
            if residuals:
                diff[s] -= 0.5 * float(np.mean(residuals))
        # Update ability given difficulty
        for h in hosts_uniq:
            mask = ct["_host"] == h
            sub = ct[mask]
            residuals = []
            for _, row in sub.iterrows():
                s = row["_step"]
                residuals.append(float(row["mean"]) - 1.0 / (1.0 + math.exp(-(abil[h] - diff.get(s, 0.0)))))
            if residuals:
                abil[h] += 0.5 * float(np.mean(residuals))
        _ensure_budget(timer)
    step_items = [{"step": s, "difficulty": diff[s]} for s in steps_uniq]
    host_items = [{"host": h, "ability": abil[h]} for h in hosts_uniq]
    step_items.sort(key=lambda r: (-r["difficulty"], r["step"]))
    host_items.sort(key=lambda r: (-r["ability"], r["host"]))
    # Separation reliability (analogous to Rasch reliability)
    step_diffs = np.array([r["difficulty"] for r in step_items], dtype=float)
    host_abils = np.array([r["ability"] for r in host_items], dtype=float)
    step_var = float(np.var(step_diffs))
    host_var = float(np.var(host_abils))
    findings = []
    if step_var > 0.5:
        findings.append(
            _make_finding(
                plugin_id,
                "rasch_separation",
                "Rasch model: significant step difficulty separation",
                f"Step difficulty variance={step_var:.3f}; steps are not interchangeable in difficulty.",
                "Well-separated difficulty levels indicate genuine heterogeneity in step demands.",
                {"metrics": {"step_difficulty_var": step_var, "host_ability_var": host_var, "n_steps": len(step_items), "n_hosts": len(host_items), "hardest": step_items[0], "easiest": step_items[-1]}},
                recommendation="Route work to hosts with ability matched to step difficulty for optimal outcomes.",
                severity="info" if step_var < 1.0 else "warn",
                confidence=min(0.82, 0.50 + min(0.28, len(ct) / 100.0)),
            )
        )
    artifacts = [_artifact_json(ctx, plugin_id, "rasch_scale.json", {"steps": step_items[:30], "hosts": host_items[:30], "step_var": step_var, "host_var": host_var}, "Rasch step-host scale")]
    return _finalize(
        plugin_id, ctx, df, sample_meta,
        "Computed Rasch step-host scale.",
        findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "step_var": step_var, "host_var": host_var},
    )


# ---------------------------------------------------------------------------
# Handler 86: Cronbach's Alpha Consistency
# ---------------------------------------------------------------------------

def _handler_cronbach_alpha_consistency_v1(
    plugin_id: str,
    ctx: Any,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    num_cols = _numeric_columns(df, inferred, max_cols=30)
    if len(num_cols) < 3:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "need_at_least_3_numeric_columns")
    mat = df[num_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(mat) < 10:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_complete_rows")
    k = len(num_cols)
    if HAS_PINGOUIN:
        try:
            result = pingouin.cronbach_alpha(mat)
            alpha = float(result[0])
            ci = (float(result[1][0]), float(result[1][1]))
        except Exception:
            alpha, ci = _manual_cronbach(mat, k)
    else:
        alpha, ci = _manual_cronbach(mat, k)
    findings = []
    if alpha < 0.7:
        findings.append(
            _make_finding(
                plugin_id,
                "low_alpha",
                "Low internal consistency (Cronbach alpha < 0.7)",
                f"Cronbach's alpha={alpha:.3f} across {k} numeric columns. Metrics lack internal consistency.",
                "Low alpha means the metrics do not measure the same construct; combining them into a composite score would be unreliable.",
                {"metrics": {"alpha": alpha, "ci_low": ci[0], "ci_high": ci[1], "n_items": k, "n_rows": len(mat)}},
                recommendation="Review whether these metrics should be combined; consider dropping inconsistent items or creating separate indices.",
                severity="warn" if alpha > 0.5 else "critical",
                confidence=min(0.85, 0.55 + min(0.25, len(mat) / 200.0)),
            )
        )
    elif alpha > 0.9:
        findings.append(
            _make_finding(
                plugin_id,
                "high_alpha",
                "High internal consistency (Cronbach alpha > 0.9)",
                f"Cronbach's alpha={alpha:.3f}. Metrics are highly consistent; some may be redundant.",
                "Very high alpha can indicate item redundancy; consider dropping near-duplicate metrics.",
                {"metrics": {"alpha": alpha, "ci_low": ci[0], "ci_high": ci[1], "n_items": k}},
                recommendation="Check for redundant metrics that could be dropped to simplify without losing information.",
                severity="info",
                confidence=min(0.85, 0.55 + min(0.25, len(mat) / 200.0)),
            )
        )
    artifacts = [_artifact_json(ctx, plugin_id, "cronbach_alpha.json", {"alpha": alpha, "ci_low": ci[0], "ci_high": ci[1], "n_items": k, "n_rows": len(mat), "columns": num_cols}, "Cronbach's alpha consistency analysis")]
    return _finalize(
        plugin_id, ctx, df, sample_meta,
        "Computed Cronbach's alpha internal consistency.",
        findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "alpha": alpha, "n_items": k},
    )


def _manual_cronbach(mat: pd.DataFrame, k: int) -> tuple[float, tuple[float, float]]:
    """Manual Cronbach's alpha computation."""
    item_vars = mat.var(ddof=1).to_numpy(dtype=float)
    total_var = float(mat.sum(axis=1).var(ddof=1))
    if total_var <= 0:
        return 0.0, (0.0, 0.0)
    alpha = (k / (k - 1)) * (1 - float(np.sum(item_vars)) / total_var)
    alpha = max(-1.0, min(1.0, alpha))
    # Approximate CI (Feldt 1965)
    n = len(mat)
    if n > 3:
        se = math.sqrt(2.0 * k * (1 - alpha) ** 2 / ((n - 1) * (k - 1)))
        ci = (alpha - 1.96 * se, alpha + 1.96 * se)
    else:
        ci = (alpha, alpha)
    return alpha, ci


# ---------------------------------------------------------------------------
# Handler 87: Structural Holes Broker
# ---------------------------------------------------------------------------

def _handler_structural_holes_broker_v1(
    plugin_id: str,
    ctx: Any,
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
    if not HAS_NETWORKX:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "networkx_not_available")
    G = nx.Graph()
    for row in edges.itertuples(index=False):
        G.add_edge(str(row.src), str(row.dst))
    if G.number_of_nodes() < 5:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "too_few_nodes")
    # Burt's constraint metric per node
    constraint = {}
    for node in list(G.nodes())[:200]:
        neighbors = set(G.neighbors(node))
        if not neighbors:
            continue
        c = 0.0
        for j in neighbors:
            # p_ij = proportion of i's network invested in j
            p_ij = 1.0 / len(neighbors)
            # indirect constraint through shared neighbors
            indirect = 0.0
            neighbors_j = set(G.neighbors(j))
            for q in neighbors:
                if q == j:
                    continue
                if q in neighbors_j:
                    indirect += 1.0 / len(neighbors) * 1.0 / max(len(neighbors_j), 1)
            c += (p_ij + indirect) ** 2
        constraint[node] = c
        _ensure_budget(timer)
    if not constraint:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_constraint_computed")
    # Low constraint = structural hole spanner (broker)
    ranked = sorted(constraint.items(), key=lambda kv: (kv[1], str(kv[0])))
    brokers = [{"node_hash": _hash_node(n), "constraint": float(c)} for n, c in ranked[:10]]
    non_brokers = [{"node_hash": _hash_node(n), "constraint": float(c)} for n, c in ranked[-5:]]
    mean_constraint = float(np.mean(list(constraint.values())))
    findings = []
    top_broker = ranked[0]
    if top_broker[1] < mean_constraint * 0.5:
        findings.append(
            _make_finding(
                plugin_id,
                "broker_node",
                "Structural hole broker identified",
                f"Node {_hash_node(top_broker[0])} has constraint={top_broker[1]:.4f} (mean={mean_constraint:.4f}). Acts as a bridge across clusters.",
                "Broker nodes span structural holes; they are critical for information flow but are single points of failure.",
                {"metrics": {"top_broker_constraint": top_broker[1], "mean_constraint": mean_constraint, "n_nodes": len(constraint), "brokers": brokers}},
                recommendation="Add redundant paths around broker nodes to reduce single-point-of-failure risk.",
                severity="warn",
                confidence=min(0.82, 0.50 + min(0.28, G.number_of_nodes() / 100.0)),
            )
        )
    artifacts = [_artifact_json(ctx, plugin_id, "structural_holes.json", {"mean_constraint": mean_constraint, "brokers": brokers, "non_brokers": non_brokers, "n_nodes": len(constraint)}, "Structural holes broker analysis")]
    return _finalize(
        plugin_id, ctx, df, sample_meta,
        "Computed structural holes broker analysis.",
        findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "mean_constraint": mean_constraint, "n_nodes": len(constraint)},
    )


# ---------------------------------------------------------------------------
# Handler 88: Weak Ties Bridge
# ---------------------------------------------------------------------------

def _handler_weak_ties_bridge_v1(
    plugin_id: str,
    ctx: Any,
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
    if not HAS_NETWORKX:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "networkx_not_available")
    G = nx.Graph()
    for row in edges.itertuples(index=False):
        G.add_edge(str(row.src), str(row.dst))
    if G.number_of_edges() < 5:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "too_few_edges")
    # Bridge coefficient: edges whose removal increases connected components
    bridges = list(nx.bridges(G))
    n_bridges = len(bridges)
    n_edges = G.number_of_edges()
    bridge_ratio = n_bridges / max(n_edges, 1)
    # Compute edge betweenness for bridge edges
    bridge_details = []
    if n_bridges > 0:
        # Only compute betweenness if reasonable size
        if G.number_of_nodes() <= 500:
            eb = nx.edge_betweenness_centrality(G)
            for u, v in bridges[:20]:
                bc = eb.get((u, v), eb.get((v, u), 0.0))
                bridge_details.append({"src_hash": _hash_node(u), "dst_hash": _hash_node(v), "betweenness": float(bc)})
            bridge_details.sort(key=lambda r: (-r["betweenness"], r["src_hash"]))
        else:
            for u, v in bridges[:20]:
                bridge_details.append({"src_hash": _hash_node(u), "dst_hash": _hash_node(v), "betweenness": 0.0})
    findings = []
    if n_bridges > 0:
        findings.append(
            _make_finding(
                plugin_id,
                "bridges",
                "Weak-tie bridges found in dependency graph",
                f"Found {n_bridges} bridge edges ({bridge_ratio:.1%} of all edges). Removal would disconnect the graph.",
                "Bridge edges (weak ties) are critical links; if they fail, graph connectivity is lost.",
                {"metrics": {"n_bridges": n_bridges, "n_edges": n_edges, "bridge_ratio": bridge_ratio, "top_bridges": bridge_details[:5]}},
                recommendation="Add redundant connections parallel to bridge edges to prevent graph fragmentation.",
                severity="warn" if bridge_ratio < 0.1 else "critical",
                confidence=min(0.88, 0.55 + min(0.28, n_edges / 200.0)),
            )
        )
    artifacts = [_artifact_json(ctx, plugin_id, "weak_ties.json", {"n_bridges": n_bridges, "n_edges": n_edges, "bridge_ratio": bridge_ratio, "bridges": bridge_details}, "Weak ties bridge analysis")]
    return _finalize(
        plugin_id, ctx, df, sample_meta,
        "Computed weak ties bridge analysis.",
        findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "n_bridges": n_bridges, "bridge_ratio": bridge_ratio},
    )


# ---------------------------------------------------------------------------
# Handler 89: Assortativity Scheduling Bias
# ---------------------------------------------------------------------------

def _handler_assortativity_scheduling_bias_v1(
    plugin_id: str,
    ctx: Any,
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
    # Degree assortativity: do high-degree nodes connect to high-degree nodes?
    src_deg = Counter(edges["src"].tolist())
    dst_deg = Counter(edges["dst"].tolist())
    all_deg: dict[str, int] = defaultdict(int)
    for n, c in src_deg.items():
        all_deg[n] += c
    for n, c in dst_deg.items():
        all_deg[n] += c
    x_vals = []
    y_vals = []
    for row in edges.itertuples(index=False):
        x_vals.append(float(all_deg[row.src]))
        y_vals.append(float(all_deg[row.dst]))
    if len(x_vals) < 10:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "too_few_edges")
    x_arr = np.array(x_vals, dtype=float)
    y_arr = np.array(y_vals, dtype=float)
    if np.std(x_arr) <= 0 or np.std(y_arr) <= 0:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "zero_degree_variance")
    assort = float(np.corrcoef(x_arr, y_arr)[0, 1])
    findings = []
    if assort > 0.3:
        findings.append(
            _make_finding(
                plugin_id,
                "assortative",
                "Assortative scheduling bias: high-load nodes connect to high-load nodes",
                f"Degree assortativity r={assort:.3f}. High-load nodes disproportionately interact with other high-load nodes.",
                "Assortative bias can create load concentration hotspots and cascade failure risk.",
                {"metrics": {"assortativity": assort, "n_edges": len(edges), "n_nodes": len(all_deg)}},
                recommendation="Rebalance scheduling to distribute load across nodes of varying capacity.",
                severity="warn" if assort < 0.5 else "critical",
                confidence=min(0.82, 0.50 + min(0.28, len(edges) / 500.0)),
            )
        )
    elif assort < -0.3:
        findings.append(
            _make_finding(
                plugin_id,
                "disassortative",
                "Disassortative scheduling: load balancing detected",
                f"Degree assortativity r={assort:.3f}. High-load nodes connect to low-load nodes.",
                "Disassortative mixing indicates effective load balancing across the topology.",
                {"metrics": {"assortativity": assort, "n_edges": len(edges)}},
                recommendation="Current scheduling shows effective load distribution; maintain this pattern.",
                severity="info",
                confidence=min(0.80, 0.50 + min(0.25, len(edges) / 500.0)),
            )
        )
    artifacts = [_artifact_json(ctx, plugin_id, "assortativity.json", {"assortativity": assort, "n_edges": len(edges), "n_nodes": len(all_deg)}, "Assortativity scheduling bias analysis")]
    return _finalize(
        plugin_id, ctx, df, sample_meta,
        "Computed degree assortativity scheduling bias.",
        findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "assortativity": assort},
    )


# ---------------------------------------------------------------------------
# Handler 90: Nash Equilibrium Contention
# ---------------------------------------------------------------------------

def _handler_nash_equilibrium_contention_v1(
    plugin_id: str,
    ctx: Any,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    num_cols = _numeric_columns(df, inferred, max_cols=10)
    if len(num_cols) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "need_at_least_2_numeric_columns")
    # Model as 2-player game: use first two numeric columns as payoff proxies
    col_a, col_b = num_cols[0], num_cols[1]
    a = pd.to_numeric(df[col_a], errors="coerce").dropna().to_numpy(dtype=float)
    b = pd.to_numeric(df[col_b], errors="coerce").dropna().to_numpy(dtype=float)
    n = min(len(a), len(b))
    if n < 10:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_data")
    a, b = a[:n], b[:n]
    # Build 2x2 payoff matrix from median splits
    med_a, med_b = float(np.median(a)), float(np.median(b))
    # Quadrants
    hh = float(np.mean(a[(a > med_a) & (b > med_b)])) if np.sum((a > med_a) & (b > med_b)) > 0 else 0.0
    hl = float(np.mean(a[(a > med_a) & (b <= med_b)])) if np.sum((a > med_a) & (b <= med_b)) > 0 else 0.0
    lh = float(np.mean(a[(a <= med_a) & (b > med_b)])) if np.sum((a <= med_a) & (b > med_b)) > 0 else 0.0
    ll = float(np.mean(a[(a <= med_a) & (b <= med_b)])) if np.sum((a <= med_a) & (b <= med_b)) > 0 else 0.0
    payoff_a = np.array([[hh, hl], [lh, ll]], dtype=float)
    hh_b = float(np.mean(b[(a > med_a) & (b > med_b)])) if np.sum((a > med_a) & (b > med_b)) > 0 else 0.0
    hl_b = float(np.mean(b[(a > med_a) & (b <= med_b)])) if np.sum((a > med_a) & (b <= med_b)) > 0 else 0.0
    lh_b = float(np.mean(b[(a <= med_a) & (b > med_b)])) if np.sum((a <= med_a) & (b > med_b)) > 0 else 0.0
    ll_b = float(np.mean(b[(a <= med_a) & (b <= med_b)])) if np.sum((a <= med_a) & (b <= med_b)) > 0 else 0.0
    payoff_b = np.array([[hh_b, hl_b], [lh_b, ll_b]], dtype=float)
    equilibria = []
    if HAS_NASHPY:
        try:
            game = nashpy.Game(payoff_a, payoff_b)
            for eq in game.support_enumeration():
                equilibria.append({"p1_strategy": eq[0].tolist(), "p2_strategy": eq[1].tolist()})
                if len(equilibria) >= 5:
                    break
        except Exception:
            pass
    if not equilibria:
        # Pure strategy check
        for i in range(2):
            for j in range(2):
                is_eq = True
                for ii in range(2):
                    if payoff_a[ii, j] > payoff_a[i, j]:
                        is_eq = False
                for jj in range(2):
                    if payoff_b[i, jj] > payoff_b[i, j]:
                        is_eq = False
                if is_eq:
                    strat_1 = [1.0, 0.0] if i == 0 else [0.0, 1.0]
                    strat_2 = [1.0, 0.0] if j == 0 else [0.0, 1.0]
                    equilibria.append({"p1_strategy": strat_1, "p2_strategy": strat_2})
    findings = []
    n_eq = len(equilibria)
    if n_eq == 0:
        findings.append(
            _make_finding(
                plugin_id,
                "no_equilibrium",
                "No Nash equilibrium found in contention game",
                "The modeled resource contention game has no pure or mixed Nash equilibrium.",
                "Absence of equilibrium suggests unstable allocation that may oscillate.",
                {"metrics": {"payoff_a": payoff_a.tolist(), "payoff_b": payoff_b.tolist(), "n_equilibria": 0}},
                recommendation="Introduce coordination mechanisms (e.g., scheduling locks) to stabilize allocation.",
                severity="warn",
                confidence=0.60,
            )
        )
    elif n_eq > 1:
        findings.append(
            _make_finding(
                plugin_id,
                "multiple_equilibria",
                "Multiple Nash equilibria in resource contention",
                f"Found {n_eq} equilibria. Multiple stable states exist; coordination needed to select optimal one.",
                "Multiple equilibria can cause coordination failures if agents settle on suboptimal states.",
                {"metrics": {"n_equilibria": n_eq, "equilibria": equilibria[:3]}},
                recommendation="Implement explicit coordination or priority rules to guide agents to the Pareto-optimal equilibrium.",
                severity="info",
                confidence=0.65,
            )
        )
    artifacts = [_artifact_json(ctx, plugin_id, "nash_equilibrium.json", {"payoff_a": payoff_a.tolist(), "payoff_b": payoff_b.tolist(), "equilibria": equilibria, "columns": [col_a, col_b]}, "Nash equilibrium contention analysis")]
    return _finalize(
        plugin_id, ctx, df, sample_meta,
        "Computed Nash equilibrium contention analysis.",
        findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "n_equilibria": n_eq},
    )


# ---------------------------------------------------------------------------
# Handler 91: Mechanism Design Scheduling
# ---------------------------------------------------------------------------

def _handler_mechanism_design_scheduling_v1(
    plugin_id: str,
    ctx: Any,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    # Need: resource request column and actual allocation column
    num_cols = _numeric_columns(df, inferred, max_cols=20)
    request_col = alloc_col = None
    for col in num_cols:
        cl = col.lower()
        if any(h in cl for h in ("request", "asked", "demand", "need", "estimate")):
            request_col = col
        elif any(h in cl for h in ("alloc", "grant", "given", "actual", "received")):
            alloc_col = col
    if request_col is None or alloc_col is None:
        # Fallback: use two highest-variance numeric columns
        var_cols = _variance_sorted_numeric(df, inferred, limit=4)
        if len(var_cols) >= 2:
            request_col, alloc_col = var_cols[0], var_cols[1]
        else:
            return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_request_alloc_columns")
    req = pd.to_numeric(df[request_col], errors="coerce").dropna().to_numpy(dtype=float)
    alloc = pd.to_numeric(df[alloc_col], errors="coerce").dropna().to_numpy(dtype=float)
    n = min(len(req), len(alloc))
    if n < 10:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_data")
    req, alloc = req[:n], alloc[:n]
    # Truthfulness test: does over-requesting lead to proportionally more allocation?
    # If allocation is proportional to request, truth-telling is not dominant
    corr = float(np.corrcoef(req, alloc)[0, 1]) if np.std(req) > 0 and np.std(alloc) > 0 else 0.0
    # Check for overbidding: requests >> allocations
    ratio = req / np.maximum(alloc, 1e-9)
    mean_ratio = float(np.mean(ratio))
    overbid_frac = float(np.mean(ratio > 1.5))
    # Incentive compatibility: is the mechanism strategy-proof?
    # If high-requesters get proportionally more, gaming is rewarded
    findings = []
    if corr > 0.8 and overbid_frac > 0.3:
        findings.append(
            _make_finding(
                plugin_id,
                "gaming_incentive",
                "Scheduling mechanism rewards over-requesting",
                f"Request-allocation correlation={corr:.3f}, overbid fraction={overbid_frac:.1%}. Over-requesting is rewarded.",
                "When allocation tracks request proportionally, agents are incentivized to inflate demands (not strategy-proof).",
                {"metrics": {"correlation": corr, "mean_ratio": mean_ratio, "overbid_fraction": overbid_frac, "request_col": request_col, "alloc_col": alloc_col}},
                recommendation="Redesign allocation mechanism (e.g., VCG, proportional fairness) to make truth-telling dominant.",
                severity="warn" if overbid_frac < 0.5 else "critical",
                confidence=min(0.82, 0.50 + min(0.28, n / 200.0)),
            )
        )
    elif corr < 0.3:
        findings.append(
            _make_finding(
                plugin_id,
                "weak_tracking",
                "Weak request-allocation tracking",
                f"Request-allocation correlation={corr:.3f}. Allocation does not strongly track requests.",
                "Weak correlation may indicate randomized or capacity-based allocation independent of stated needs.",
                {"metrics": {"correlation": corr, "mean_ratio": mean_ratio}},
                recommendation="Verify that allocation mechanism correctly reflects actual resource needs.",
                severity="info",
                confidence=0.60,
            )
        )
    artifacts = [_artifact_json(ctx, plugin_id, "mechanism_design.json", {"correlation": corr, "mean_ratio": mean_ratio, "overbid_fraction": overbid_frac, "request_col": request_col, "alloc_col": alloc_col, "n": n}, "Mechanism design scheduling analysis")]
    return _finalize(
        plugin_id, ctx, df, sample_meta,
        "Computed mechanism design scheduling analysis.",
        findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "correlation": corr, "overbid_fraction": overbid_frac},
    )


# ---------------------------------------------------------------------------
# Handler 92: Vickrey Priority Truth
# ---------------------------------------------------------------------------

def _handler_vickrey_priority_truth_v1(
    plugin_id: str,
    ctx: Any,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    # Look for priority and value/cost columns
    num_cols = _numeric_columns(df, inferred, max_cols=20)
    priority_col = value_col = None
    for col in num_cols:
        cl = col.lower()
        if any(h in cl for h in ("priority", "rank", "urgency", "weight")):
            priority_col = col
        elif any(h in cl for h in ("value", "cost", "price", "bid", "duration", "latency")):
            value_col = col
    if priority_col is None:
        # Use first numeric as priority proxy
        priority_col = num_cols[0] if num_cols else None
    if value_col is None:
        dur_col = _duration_column(df, inferred)
        value_col = dur_col
    if priority_col is None or value_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_priority_value_columns")
    pri = pd.to_numeric(df[priority_col], errors="coerce").dropna()
    val = pd.to_numeric(df[value_col], errors="coerce").dropna()
    common = pri.index.intersection(val.index)
    if len(common) < 10:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_data")
    pri = pri.loc[common].to_numpy(dtype=float)
    val = val.loc[common].to_numpy(dtype=float)
    # Vickrey (second-price) analysis: in truthful mechanisms, the price paid
    # should be the second-highest bid, not the first
    # Sort by priority (descending = highest priority first)
    order = np.argsort(-pri)
    pri_sorted = pri[order]
    val_sorted = val[order]
    # Check if highest-priority items pay proportional to second-highest value
    # Truthfulness metric: correlation between priority rank and value rank
    pri_ranks = np.argsort(np.argsort(-pri)).astype(float)
    val_ranks = np.argsort(np.argsort(-val)).astype(float)
    rank_corr = float(np.corrcoef(pri_ranks, val_ranks)[0, 1]) if np.std(pri_ranks) > 0 and np.std(val_ranks) > 0 else 0.0
    # Overbid detection: items with high priority but low actual value
    pri_z = (pri - np.mean(pri)) / max(np.std(pri), 1e-9)
    val_z = (val - np.mean(val)) / max(np.std(val), 1e-9)
    overbid_mask = (pri_z > 1.0) & (val_z < 0.0)
    overbid_frac = float(np.mean(overbid_mask))
    findings = []
    if overbid_frac > 0.1:
        findings.append(
            _make_finding(
                plugin_id,
                "vickrey_overbid",
                "Priority inflation detected (Vickrey analysis)",
                f"Overbid fraction={overbid_frac:.1%}: items claim high priority but show low actual value.",
                "Priority inflation distorts scheduling and starves genuinely urgent work.",
                {"metrics": {"overbid_fraction": overbid_frac, "rank_correlation": rank_corr, "priority_col": priority_col, "value_col": value_col}},
                recommendation="Implement second-price (Vickrey) scheduling: penalize inflated priority claims to incentivize truthful reporting.",
                severity="warn" if overbid_frac < 0.25 else "critical",
                confidence=min(0.82, 0.50 + min(0.28, len(common) / 200.0)),
            )
        )
    elif abs(rank_corr) > 0.7:
        findings.append(
            _make_finding(
                plugin_id,
                "vickrey_aligned",
                "Priority ordering is well-aligned with value",
                f"Rank correlation={rank_corr:.3f}. Priority claims match actual value.",
                "Good priority-value alignment indicates truthful reporting or well-calibrated scheduling.",
                {"metrics": {"rank_correlation": rank_corr, "overbid_fraction": overbid_frac}},
                recommendation="Priority mechanism appears truthful; maintain current incentive structure.",
                severity="info",
                confidence=min(0.80, 0.50 + min(0.25, len(common) / 200.0)),
            )
        )
    artifacts = [_artifact_json(ctx, plugin_id, "vickrey_priority.json", {"rank_correlation": rank_corr, "overbid_fraction": overbid_frac, "priority_col": priority_col, "value_col": value_col, "n": len(common)}, "Vickrey priority truth analysis")]
    return _finalize(
        plugin_id, ctx, df, sample_meta,
        "Computed Vickrey priority truth analysis.",
        findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "rank_correlation": rank_corr, "overbid_fraction": overbid_frac},
    )


# ---------------------------------------------------------------------------
# Handler 94: Benford's Second Digit Test
# ---------------------------------------------------------------------------

def _handler_benford_second_digit_v1(
    plugin_id: str,
    ctx: Any,
    df: pd.DataFrame,
    config: dict[str, Any],
    inferred: dict[str, Any],
    timer: BudgetTimer,
    sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    num_cols = _numeric_columns(df, inferred, max_cols=10)
    if not num_cols:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_columns")
    # Expected second-digit distribution (Benford)
    expected = np.zeros(10, dtype=float)
    for d2 in range(10):
        for d1 in range(1, 10):
            expected[d2] += math.log10(1 + 1.0 / (10 * d1 + d2))
    expected /= expected.sum()
    results = []
    for col in num_cols:
        vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
        vals = np.abs(vals[vals != 0])
        if len(vals) < 50:
            continue
        # Extract second digit
        digits = []
        for v in vals:
            s = f"{abs(v):.10e}"
            # Find significant digits
            sig = s.replace(".", "").lstrip("0").split("e")[0]
            if len(sig) >= 2:
                digits.append(int(sig[1]))
        if len(digits) < 50:
            continue
        observed = np.zeros(10, dtype=float)
        for d in digits:
            observed[d] += 1
        observed /= observed.sum()
        # Chi-square test
        n_obs = len(digits)
        chi2 = float(np.sum((observed - expected) ** 2 / np.maximum(expected, 1e-9)) * n_obs)
        if HAS_SCIPY:
            p_val = float(1 - scipy_stats.chi2.cdf(chi2, df=9))
        else:
            # Rough approximation: chi2 with 9 df, critical value at 0.05 is ~16.92
            p_val = 0.01 if chi2 > 16.92 else 0.5
        mad_stat = float(np.mean(np.abs(observed - expected)))
        results.append({"column": col, "chi2": chi2, "p_value": p_val, "mad": mad_stat, "n_digits": n_obs, "observed": observed.tolist(), "expected": expected.tolist()})
        _ensure_budget(timer)
    if not results:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_suitable_columns")
    results.sort(key=lambda r: (r["p_value"], r["column"]))
    findings = []
    for res in results[:3]:
        if res["p_value"] < 0.05:
            findings.append(
                _make_finding(
                    plugin_id,
                    f"benford2:{res['column']}",
                    "Second-digit Benford's law deviation",
                    f"Column '{res['column']}': chi2={res['chi2']:.2f}, p={res['p_value']:.4f}, MAD={res['mad']:.4f}. Second digits deviate from Benford expectation.",
                    "Second-digit Benford deviations can indicate data manipulation, rounding artifacts, or synthetic data generation.",
                    {"metrics": res},
                    recommendation="Audit data provenance for columns with Benford anomalies; check for manual rounding or fabrication.",
                    severity="warn" if res["p_value"] > 0.01 else "critical",
                    confidence=min(0.85, 0.55 + min(0.25, res["n_digits"] / 500.0)),
                )
            )
    artifacts = [_artifact_json(ctx, plugin_id, "benford_second_digit.json", {"columns": results}, "Benford second-digit analysis")]
    return _finalize(
        plugin_id, ctx, df, sample_meta,
        "Computed Benford's second-digit test.",
        findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "n_columns_tested": len(results)},
    )


# ---------------------------------------------------------------------------
# HANDLERS registry
# ---------------------------------------------------------------------------

HANDLERS: dict[str, Callable[..., PluginResult]] = {
    "analysis_war_value_above_replacement_v1": _handler_war_value_above_replacement_v1,
    "analysis_win_probability_added_v1": _handler_win_probability_added_v1,
    "analysis_leverage_index_dag_v1": _handler_leverage_index_dag_v1,
    "analysis_pythagorean_expectation_v1": _handler_pythagorean_expectation_v1,
    "analysis_zipf_law_frequency_v1": _handler_zipf_law_frequency_v1,
    "analysis_entropy_rate_step_sequences_v1": _handler_entropy_rate_step_sequences_v1,
    "analysis_levenshtein_workflow_dist_v1": _handler_levenshtein_workflow_dist_v1,
    "analysis_ngram_step_transitions_v1": _handler_ngram_step_transitions_v1,
    "analysis_weber_fechner_threshold_v1": _handler_weber_fechner_threshold_v1,
    "analysis_stevens_power_law_severity_v1": _handler_stevens_power_law_severity_v1,
    "analysis_signal_detection_dprime_v1": _handler_signal_detection_dprime_v1,
    "analysis_irt_step_difficulty_v1": _handler_irt_step_difficulty_v1,
    "analysis_rasch_step_host_scale_v1": _handler_rasch_step_host_scale_v1,
    "analysis_cronbach_alpha_consistency_v1": _handler_cronbach_alpha_consistency_v1,
    "analysis_structural_holes_broker_v1": _handler_structural_holes_broker_v1,
    "analysis_weak_ties_bridge_v1": _handler_weak_ties_bridge_v1,
    "analysis_assortativity_scheduling_bias_v1": _handler_assortativity_scheduling_bias_v1,
    "analysis_nash_equilibrium_contention_v1": _handler_nash_equilibrium_contention_v1,
    "analysis_mechanism_design_scheduling_v1": _handler_mechanism_design_scheduling_v1,
    "analysis_vickrey_priority_truth_v1": _handler_vickrey_priority_truth_v1,
    "analysis_benford_second_digit_v1": _handler_benford_second_digit_v1,
}
