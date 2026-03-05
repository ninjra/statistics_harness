"""Cross-domain plugins: unconventional domains (plugins 149-182, skip 170)."""
from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
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

try:
    from scipy import stats as scipy_stats
    from scipy import optimize as scipy_optimize
    from scipy import spatial as scipy_spatial
    HAS_SCIPY = True
except Exception:
    scipy_stats = scipy_optimize = scipy_spatial = None
    HAS_SCIPY = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except Exception:
    nx = None
    HAS_NETWORKX = False

try:
    import lifelines as lifelines_pkg
    HAS_LIFELINES = True
except Exception:
    lifelines_pkg = None
    HAS_LIFELINES = False

try:
    from rapidfuzz import fuzz as rapidfuzz_fuzz
    HAS_RAPIDFUZZ = True
except Exception:
    rapidfuzz_fuzz = None
    HAS_RAPIDFUZZ = False


# ---------------------------------------------------------------------------
# Module-private helpers (duplicated per addon by convention)
# ---------------------------------------------------------------------------

def _safe_id(plugin_id, key):
    try: return stable_id((plugin_id, key))
    except Exception: return hashlib.sha256(f"{plugin_id}:{key}".encode()).hexdigest()[:16]

def _basic_metrics(df, sample_meta):
    m = {"rows_seen": int(sample_meta.get("rows_total", len(df))), "rows_used": int(sample_meta.get("rows_used", len(df))), "cols_used": int(len(df.columns))}
    m.update(sample_meta or {})
    return m

def _make_finding(plugin_id, key, title, what, why, evidence, *, recommendation, severity="info", confidence=0.5, where=None, measurement_type="measured", kind=None):
    f = {"id": _safe_id(plugin_id, key), "severity": severity, "confidence": float(max(0.0, min(1.0, confidence))), "title": title, "what": what, "why": why, "evidence": evidence, "where": where or {}, "recommendation": recommendation, "measurement_type": measurement_type}
    if kind: f["kind"] = kind
    return f

def _ok_with_reason(plugin_id, ctx, df, sample_meta, reason, *, debug=None):
    ctx.logger(f"SKIP reason={reason}")
    p = dict(debug or {}); p.setdefault("gating_reason", reason)
    return PluginResult("ok", f"No actionable result: {reason}", _basic_metrics(df, sample_meta), [], [], None, debug=p)

def _finalize(plugin_id, ctx, df, sample_meta, summary, findings, artifacts, *, extra_metrics=None, debug=None):
    metrics = _basic_metrics(df, sample_meta)
    if extra_metrics: metrics.update(extra_metrics)
    rt = int(metrics.pop("runtime_ms", 0))
    ctx.logger(f"END runtime_ms={rt} findings={len(findings)}")
    d = dict(debug or {}); d["runtime_ms"] = rt
    return PluginResult("ok", summary, metrics, findings, artifacts, None, debug=d)

def _log_start(ctx, plugin_id, df, config, inferred):
    ctx.logger(f"START plugin_id={plugin_id} rows={len(df)} cols={len(df.columns)} seed={int(config.get('seed', 0))}")

def _ensure_budget(timer): timer.ensure("time_budget_exceeded")
def _runtime_ms(timer): return int(max(0.0, timer.elapsed_ms()))

def _numeric_columns(df, inferred, max_cols=None):
    cols = [str(c) for c in (inferred.get("numeric_columns") or []) if str(c) in df.columns]
    if not cols: cols = [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if max_cols is not None: cols = cols[:max(1, int(max_cols))]
    return cols

def _categorical_columns(df, inferred, max_cols=None):
    cols = [str(c) for c in (inferred.get("categorical_columns") or []) if str(c) in df.columns]
    if not cols: cols = [str(c) for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if max_cols is not None: cols = cols[:max(1, int(max_cols))]
    return cols

def _time_series(df, inferred):
    tc = inferred.get("time_column")
    if isinstance(tc, str) and tc in df.columns:
        p = pd.to_datetime(df[tc], errors="coerce")
        if p.notna().sum() >= 10: return tc, p
    for col in df.columns:
        l = str(col).lower()
        if "time" not in l and "date" not in l: continue
        p = pd.to_datetime(df[col], errors="coerce")
        if p.notna().sum() >= 10: return str(col), p
    return None, None

def _duration_column(df, inferred):
    hints = ("duration", "latency", "wait", "elapsed", "runtime", "service", "time")
    for col in _numeric_columns(df, inferred, max_cols=100):
        if any(h in col.lower() for h in hints): return col
    cols = _numeric_columns(df, inferred, max_cols=100)
    return cols[0] if cols else None

def _variance_sorted_numeric(df, inferred, limit=8):
    cols = _numeric_columns(df, inferred, max_cols=80)
    scored = [(float(np.nanvar(pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float))), c) for c in cols]
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [c for _, c in scored[:limit]]

def _safe_float(value, default=0.0):
    try:
        if value is None: return default
        x = float(value)
        return default if not math.isfinite(x) else x
    except Exception: return default

def _find_parent_column(df):
    hints = ("parent", "ppid", "parent_id", "parent_process", "caller", "upstream", "predecessor")
    for col in df.columns:
        cl = str(col).lower().replace(" ", "_")
        if any(h in cl for h in hints):
            nf = float(df[col].isna().mean())
            if 0.01 < nf < 0.99: return str(col)
    return None

def _build_dag(df, parent_col, id_col=None):
    if not HAS_NETWORKX: return None
    G = nx.DiGraph()
    ids = df.index.tolist() if id_col is None else df[id_col].tolist()
    for idx, parent in zip(ids, df[parent_col].tolist()):
        G.add_node(idx)
        if pd.notna(parent): G.add_edge(parent, idx)
    return G

def _find_graph_columns(df, inferred):
    cols = list(df.columns); low = {c: str(c).lower() for c in cols}
    src = dst = None
    for c in cols:
        if any(h in low[c] for h in ("src","from","parent","source","caller")): src = c; break
    for c in cols:
        if c == src: continue
        if any(h in low[c] for h in ("dst","to","child","target","callee")): dst = c; break
    if src and dst: return str(src), str(dst)
    cats = _categorical_columns(df, inferred, max_cols=6)
    return (cats[0], cats[1]) if len(cats) >= 2 else (None, None)

def _build_edges(df, inferred, max_edges=20000):
    sc, dc = _find_graph_columns(df, inferred)
    if sc is None or dc is None: return pd.DataFrame(columns=["src","dst"]), "", ""
    edges = pd.DataFrame({"src": df[sc].astype(str), "dst": df[dc].astype(str)})
    edges = edges[(edges["src"]!="") & (edges["dst"]!="") & (edges["src"]!="nan") & (edges["dst"]!="nan")]
    edges = edges[edges["src"] != edges["dst"]]
    return (edges.iloc[:max_edges], sc, dc) if len(edges) > max_edges else (edges, sc, dc)

def _build_nx_graph_from_edges(edges_df):
    if not HAS_NETWORKX: return None
    G = nx.DiGraph()
    for _, row in edges_df.iterrows():
        G.add_edge(row["src"], row["dst"])
    return G


# ---------------------------------------------------------------------------
# Plugin handlers (149-182, skip 170)
# ---------------------------------------------------------------------------


def _seriation_ordering(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Spectral seriation on similarity matrix to find optimal row ordering."""
    _log_start(ctx, plugin_id, df, config, inferred)
    num_cols = _variance_sorted_numeric(df, inferred, limit=8)
    if len(num_cols) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_numeric_columns")
    mat = df[num_cols].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(dtype=float)
    if len(mat) < 10:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    mat = mat[:min(2000, len(mat))]
    _ensure_budget(timer)
    # Build similarity matrix using correlation
    corr = np.corrcoef(mat)
    corr = np.nan_to_num(corr, nan=0.0)
    # Laplacian for spectral ordering
    D = np.diag(corr.sum(axis=1))
    L = D - corr
    _ensure_budget(timer)
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        fiedler = eigvecs[:, 1]  # second smallest eigenvector
        order = np.argsort(fiedler)
    except Exception:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "eigen_failed")
    # Measure how much seriation improves adjacency smoothness
    orig_jumps = float(np.mean(np.abs(np.diff(mat, axis=0)).sum(axis=1)))
    reordered = mat[order]
    new_jumps = float(np.mean(np.abs(np.diff(reordered, axis=0)).sum(axis=1)))
    improvement = 1.0 - new_jumps / max(1e-9, orig_jumps)
    findings = []
    if improvement > 0.1:
        findings.append(_make_finding(
            plugin_id, "seriation", "Seriation reveals hidden ordering",
            f"Spectral seriation reduces row-to-row jumps by {improvement*100:.1f}%.",
            "A latent ordering exists in the data that is obscured by the current row arrangement.",
            {"metrics": {"improvement_pct": round(improvement * 100, 2), "orig_jumps": round(orig_jumps, 4), "new_jumps": round(new_jumps, 4)}},
            recommendation="Reorder rows by seriation index to reveal gradients; investigate what drives the latent ordering.",
            severity="warn" if improvement > 0.3 else "info", confidence=min(0.9, 0.5 + improvement),
        ))
    return _finalize(plugin_id, ctx, df, sample_meta, "Spectral seriation analysis.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "improvement_pct": round(improvement * 100, 2)})


def _stratigraphic_layers(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Detect layer boundaries where characteristics change abruptly."""
    _log_start(ctx, plugin_id, df, config, inferred)
    time_col, ts = _time_series(df, inferred)
    if ts is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_time_column")
    num_cols = _variance_sorted_numeric(df, inferred, limit=6)
    if not num_cols:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_columns")
    frame = df[num_cols].apply(pd.to_numeric, errors="coerce").copy()
    frame["_ts"] = ts
    frame = frame.sort_values("_ts").dropna(subset=["_ts"]).reset_index(drop=True)
    if len(frame) < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    _ensure_budget(timer)
    # Compute rolling mean difference to detect change points
    w = max(5, len(frame) // 20)
    vals = frame[num_cols].to_numpy(dtype=float)
    vals = np.nan_to_num(vals, nan=0.0)
    diffs = np.zeros(len(vals))
    for i in range(w, len(vals) - w):
        before = vals[i - w:i].mean(axis=0)
        after = vals[i:i + w].mean(axis=0)
        diffs[i] = float(np.linalg.norm(after - before))
    threshold = float(np.mean(diffs) + 2.0 * np.std(diffs))
    boundaries = [i for i in range(w, len(vals) - w) if diffs[i] > threshold]
    # Merge nearby boundaries
    merged = []
    for b in boundaries:
        if not merged or b - merged[-1] > w:
            merged.append(b)
    n_layers = len(merged) + 1
    findings = []
    if merged:
        findings.append(_make_finding(
            plugin_id, "layers", f"{n_layers} stratigraphic layers detected",
            f"Found {len(merged)} boundary points dividing data into {n_layers} distinct layers.",
            "Abrupt characteristic changes suggest regime shifts over time.",
            {"metrics": {"n_layers": n_layers, "boundaries": merged[:20], "threshold": round(threshold, 4)}},
            recommendation="Investigate what changed at each boundary; consider treating each layer as a separate regime.",
            severity="warn" if n_layers >= 4 else "info", confidence=min(0.85, 0.5 + 0.05 * len(merged)),
        ))
    return _finalize(plugin_id, ctx, df, sample_meta, "Stratigraphic layer analysis.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "n_layers": n_layers})


def _harris_matrix_ordering(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Partial ordering from DAG. Compute minimum number of layers via longest path."""
    _log_start(ctx, plugin_id, df, config, inferred)
    if not HAS_NETWORKX:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "networkx_unavailable")
    parent_col = _find_parent_column(df)
    if parent_col is None:
        edges, sc, dc = _build_edges(df, inferred)
        if edges.empty:
            return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_dag_structure")
        G = _build_nx_graph_from_edges(edges)
    else:
        G = _build_dag(df, parent_col)
    if G is None or G.number_of_nodes() < 5:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "graph_too_small")
    _ensure_budget(timer)
    if not nx.is_directed_acyclic_graph(G):
        try:
            G = nx.DiGraph([(u, v) for u, v in G.edges() if u != v])
            cycles = list(nx.simple_cycles(G))
            if cycles:
                for cycle in cycles[:100]:
                    G.remove_edge(cycle[-1], cycle[0])
        except Exception:
            return _ok_with_reason(plugin_id, ctx, df, sample_meta, "cyclic_graph")
        if not nx.is_directed_acyclic_graph(G):
            return _ok_with_reason(plugin_id, ctx, df, sample_meta, "cyclic_graph")
    longest = nx.dag_longest_path_length(G)
    n_layers = longest + 1
    width = max(len(list(gen)) for gen in nx.topological_generations(G))
    findings = []
    findings.append(_make_finding(
        plugin_id, "harris", f"DAG has {n_layers} layers (depth={longest})",
        f"Harris matrix ordering: {n_layers} minimum layers, max width={width}.",
        "Deep DAGs indicate long dependency chains; wide layers indicate parallelism opportunities.",
        {"metrics": {"n_layers": n_layers, "longest_path": longest, "max_width": width, "nodes": G.number_of_nodes()}},
        recommendation="Reduce depth to shorten critical path; exploit wide layers for parallel execution.",
        severity="warn" if longest > 10 else "info", confidence=0.75,
    ))
    return _finalize(plugin_id, ctx, df, sample_meta, "Harris matrix ordering.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "n_layers": n_layers, "max_width": width})


def _fitness_landscape_mapping(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Map parameter space to performance. Identify local optima."""
    _log_start(ctx, plugin_id, df, config, inferred)
    num_cols = _variance_sorted_numeric(df, inferred, limit=6)
    if len(num_cols) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_numeric_columns")
    perf_col = _duration_column(df, inferred) or num_cols[-1]
    param_cols = [c for c in num_cols if c != perf_col][:4]
    if not param_cols:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_param_columns")
    frame = df[param_cols + [perf_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(frame) < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    _ensure_budget(timer)
    perf = frame[perf_col].to_numpy(dtype=float)
    params = frame[param_cols].to_numpy(dtype=float)
    # Normalize params
    pmin, pmax = params.min(axis=0), params.max(axis=0)
    rng = pmax - pmin; rng[rng == 0] = 1.0
    params_n = (params - pmin) / rng
    # Find local optima: points better than all k-nearest neighbors
    from scipy.spatial import cKDTree
    tree = cKDTree(params_n)
    k = min(10, len(params_n) - 1)
    _, idx = tree.query(params_n, k=k + 1)
    local_optima = []
    for i in range(len(perf)):
        neighbors = idx[i, 1:]
        if all(perf[i] <= perf[j] for j in neighbors if j < len(perf)):
            local_optima.append(i)
    n_optima = len(local_optima)
    ruggedness = n_optima / max(1, len(perf))
    findings = []
    if n_optima > 1:
        findings.append(_make_finding(
            plugin_id, "landscape", f"{n_optima} local optima in fitness landscape",
            f"Found {n_optima} local optima (ruggedness={ruggedness:.3f}) across {len(param_cols)} parameters.",
            "Multiple local optima indicate a rugged fitness landscape where greedy optimization may get stuck.",
            {"metrics": {"n_optima": n_optima, "ruggedness": round(ruggedness, 4), "perf_col": perf_col, "param_cols": param_cols}},
            recommendation="Use global search strategies; the landscape has multiple traps that greedy approaches will miss.",
            severity="warn" if ruggedness > 0.05 else "info", confidence=min(0.85, 0.5 + ruggedness * 5),
        ))
    return _finalize(plugin_id, ctx, df, sample_meta, "Fitness landscape mapping.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "n_optima": n_optima, "ruggedness": round(ruggedness, 4)})


def _punctuated_equilibrium(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Detect long stasis + rapid change pattern in time series."""
    _log_start(ctx, plugin_id, df, config, inferred)
    time_col, ts = _time_series(df, inferred)
    if ts is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_time_column")
    dur_col = _duration_column(df, inferred)
    if dur_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_metric_column")
    vals = pd.to_numeric(df[dur_col], errors="coerce").to_numpy(dtype=float)
    order = ts.argsort()
    vals = vals[order]
    vals = vals[~np.isnan(vals)]
    if len(vals) < 30:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    _ensure_budget(timer)
    # Compute rolling std in windows
    w = max(5, len(vals) // 20)
    rolling_std = np.array([np.std(vals[max(0, i - w):i + 1]) for i in range(len(vals))])
    median_std = float(np.median(rolling_std[rolling_std > 0])) if np.any(rolling_std > 0) else 1.0
    bursts = rolling_std > 3.0 * median_std
    n_bursts = int(np.sum(np.diff(bursts.astype(int)) == 1))
    stasis_pct = 1.0 - float(bursts.mean())
    findings = []
    if n_bursts >= 1 and stasis_pct > 0.7:
        findings.append(_make_finding(
            plugin_id, "punctuated", f"Punctuated equilibrium: {n_bursts} burst(s), {stasis_pct*100:.0f}% stasis",
            f"Series shows {n_bursts} rapid-change bursts amid {stasis_pct*100:.0f}% stasis periods.",
            "Punctuated equilibrium suggests the system resists change until a threshold triggers rapid shifts.",
            {"metrics": {"n_bursts": n_bursts, "stasis_pct": round(stasis_pct, 3), "metric": dur_col}},
            recommendation="Identify triggers for burst periods; prepare for rapid shifts after prolonged stability.",
            severity="warn" if n_bursts >= 3 else "info", confidence=min(0.85, 0.5 + 0.1 * n_bursts),
        ))
    return _finalize(plugin_id, ctx, df, sample_meta, "Punctuated equilibrium analysis.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "n_bursts": n_bursts, "stasis_pct": round(stasis_pct, 3)})


def _neutral_drift_noise(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Variance/mean ratio test for neutral drift vs selection."""
    _log_start(ctx, plugin_id, df, config, inferred)
    time_col, ts = _time_series(df, inferred)
    if ts is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_time_column")
    num_cols = _numeric_columns(df, inferred, max_cols=10)
    if not num_cols:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_columns")
    _ensure_budget(timer)
    findings = []
    for col in num_cols[:5]:
        vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
        vals = vals[vals > 0]
        if len(vals) < 20: continue
        vmr = float(np.var(vals) / np.mean(vals))
        # VMR ~ 1 for Poisson (neutral drift); >1 overdispersed (selection); <1 underdispersed
        classification = "neutral_drift" if 0.8 <= vmr <= 1.2 else ("overdispersed" if vmr > 1.2 else "underdispersed")
        if classification != "neutral_drift":
            findings.append(_make_finding(
                plugin_id, f"vmr_{col}", f"Non-neutral drift in {col} (VMR={vmr:.2f})",
                f"Variance/mean ratio={vmr:.2f} ({classification}), deviating from neutral expectation.",
                "Departure from neutral drift suggests active selection or structural forces shaping the distribution.",
                {"metrics": {"vmr": round(vmr, 4), "classification": classification, "column": col}},
                recommendation=f"Investigate drivers of {classification} pattern in {col}; not random noise.",
                severity="warn" if abs(vmr - 1.0) > 1.0 else "info", confidence=min(0.8, 0.5 + abs(vmr - 1.0) * 0.2),
            ))
        if len(findings) >= 3: break
    return _finalize(plugin_id, ctx, df, sample_meta, "Neutral drift noise analysis.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer)})


def _red_queen_arms_race(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Detect co-escalation of competing metrics over time."""
    _log_start(ctx, plugin_id, df, config, inferred)
    time_col, ts = _time_series(df, inferred)
    if ts is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_time_column")
    num_cols = _variance_sorted_numeric(df, inferred, limit=8)
    if len(num_cols) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_numeric_columns")
    frame = df[num_cols].apply(pd.to_numeric, errors="coerce").copy()
    frame["_ts"] = ts
    frame = frame.sort_values("_ts").dropna(subset=["_ts"]).reset_index(drop=True)
    if len(frame) < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    _ensure_budget(timer)
    # Check pairs for co-escalation: both trending up with positive correlation of differences
    findings = []
    checked = 0
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            a = frame[num_cols[i]].to_numpy(dtype=float)
            b = frame[num_cols[j]].to_numpy(dtype=float)
            da = np.diff(a); db = np.diff(b)
            da = da[~np.isnan(da)]; db = db[~np.isnan(db)]
            mlen = min(len(da), len(db))
            if mlen < 10: continue
            da, db = da[:mlen], db[:mlen]
            corr = float(np.corrcoef(da, db)[0, 1]) if np.std(da) > 0 and np.std(db) > 0 else 0.0
            both_up = float(np.mean(da)) > 0 and float(np.mean(db)) > 0
            if corr > 0.4 and both_up:
                findings.append(_make_finding(
                    plugin_id, f"arms_race_{num_cols[i]}_{num_cols[j]}",
                    f"Red Queen co-escalation: {num_cols[i]} vs {num_cols[j]}",
                    f"Both metrics trend upward with correlated changes (r={corr:.2f}).",
                    "Co-escalation suggests an arms-race dynamic where improvements in one metric drive increases in another.",
                    {"metrics": {"col_a": num_cols[i], "col_b": num_cols[j], "corr": round(corr, 3)}},
                    recommendation=f"Break the co-escalation cycle between {num_cols[i]} and {num_cols[j]}.",
                    severity="warn", confidence=min(0.85, 0.4 + corr * 0.5),
                ))
            checked += 1
            if checked >= 10 or len(findings) >= 3: break
        if checked >= 10 or len(findings) >= 3: break
    return _finalize(plugin_id, ctx, df, sample_meta, "Red Queen arms race analysis.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer)})


def _hick_law_pool_sizing(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Hick's Law: RT = a + b*log2(n+1). Find optimal pool size."""
    _log_start(ctx, plugin_id, df, config, inferred)
    dur_col = _duration_column(df, inferred)
    if dur_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_duration_column")
    cat_cols = _categorical_columns(df, inferred, max_cols=10)
    if not cat_cols:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_categorical_columns")
    _ensure_budget(timer)
    # Use categorical cardinality as 'n' (number of choices)
    best_r2, best_col, best_a, best_b = -1.0, "", 0.0, 0.0
    for cc in cat_cols[:5]:
        groups = df.groupby(cc)[dur_col].agg(["mean", "count"]).dropna()
        groups = groups[groups["count"] >= 3]
        if len(groups) < 3: continue
        n_choices = np.array([float(len(groups))] * len(groups))
        # Across different group sizes, fit Hick's law
        rt = groups["mean"].to_numpy(dtype=float)
        log_n = np.log2(n_choices + 1)
        if np.std(rt) < 1e-9: continue
        corr = float(np.corrcoef(log_n, rt)[0, 1]) if np.std(log_n) > 0 else 0.0
        if corr > best_r2:
            best_r2 = corr
            best_col = cc
            m = float(np.mean(rt)); best_a = m * 0.3; best_b = m * 0.7 / max(1.0, float(np.mean(log_n)))
    # Estimate optimal pool: where marginal RT increase < threshold
    n_unique = df[best_col].nunique() if best_col else 0
    optimal_n = max(2, int(2 ** (1.0 / max(0.01, best_b)) - 1)) if best_b > 0 else n_unique
    optimal_n = min(optimal_n, 100)
    findings = []
    if best_col and n_unique > 3:
        findings.append(_make_finding(
            plugin_id, "hick", f"Hick's Law: optimal pool ~{optimal_n} for {best_col}",
            f"RT increases logarithmically with pool size (a={best_a:.2f}, b={best_b:.2f}). Current n={n_unique}.",
            "Hick's Law predicts diminishing returns from adding more choices; oversized pools slow decisions.",
            {"metrics": {"optimal_n": optimal_n, "current_n": n_unique, "a": round(best_a, 3), "b": round(best_b, 3), "column": best_col}},
            recommendation=f"Consider reducing {best_col} pool from {n_unique} to ~{optimal_n} choices.",
            severity="warn" if n_unique > optimal_n * 2 else "info", confidence=0.6,
        ))
    return _finalize(plugin_id, ctx, df, sample_meta, "Hick's Law pool sizing.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "optimal_n": optimal_n})


def _fitts_law_precision(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Fitts' Law: MT = a + b*log2(2D/W). Speed-accuracy tradeoff."""
    _log_start(ctx, plugin_id, df, config, inferred)
    dur_col = _duration_column(df, inferred)
    num_cols = _numeric_columns(df, inferred, max_cols=10)
    if dur_col is None or len(num_cols) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_columns")
    _ensure_budget(timer)
    mt = pd.to_numeric(df[dur_col], errors="coerce").dropna().to_numpy(dtype=float)
    # Try to find distance and width analogues
    other = [c for c in num_cols if c != dur_col][:4]
    best_r2, best_pair = 0.0, ("", "")
    for i, d_col in enumerate(other):
        for w_col in other[i + 1:]:
            d = pd.to_numeric(df[d_col], errors="coerce").to_numpy(dtype=float)
            w = pd.to_numeric(df[w_col], errors="coerce").to_numpy(dtype=float)
            mask = ~(np.isnan(d) | np.isnan(w) | np.isnan(mt[:len(d)]) | (w <= 0) | (d <= 0))
            if mask.sum() < 20: continue
            id_fitts = np.log2(2.0 * d[mask] / w[mask])
            mt_sub = mt[:len(d)][mask]
            if np.std(id_fitts) < 1e-9: continue
            r = float(np.corrcoef(id_fitts, mt_sub)[0, 1])
            if r > best_r2: best_r2 = r; best_pair = (d_col, w_col)
    findings = []
    if best_r2 > 0.3:
        findings.append(_make_finding(
            plugin_id, "fitts", f"Fitts' Law fit (r={best_r2:.2f}): {best_pair[0]}/{best_pair[1]}",
            f"Movement time correlates with log2(2D/W) index of difficulty (r={best_r2:.2f}).",
            "Fitts' Law governs speed-accuracy tradeoffs; high ID tasks need more time or larger targets.",
            {"metrics": {"r": round(best_r2, 3), "distance_col": best_pair[0], "width_col": best_pair[1]}},
            recommendation="Reduce index of difficulty for slow tasks by increasing target size or reducing distance.",
            severity="warn" if best_r2 > 0.6 else "info", confidence=min(0.85, 0.4 + best_r2 * 0.5),
        ))
    return _finalize(plugin_id, ctx, df, sample_meta, "Fitts' Law precision analysis.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "best_r2": round(best_r2, 3)})


def _cognitive_load_concurrency(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Error rate vs concurrency to find cognitive load threshold."""
    _log_start(ctx, plugin_id, df, config, inferred)
    num_cols = _numeric_columns(df, inferred, max_cols=20)
    # Look for error/failure column
    err_col = None
    for c in num_cols:
        if any(h in c.lower() for h in ("error", "fail", "reject", "defect", "fault")):
            err_col = c; break
    conc_col = None
    for c in num_cols:
        if any(h in c.lower() for h in ("concurrent", "parallel", "active", "open", "wip", "queue")):
            conc_col = c; break
    if err_col is None or conc_col is None:
        # Fall back to first two numeric columns
        if len(num_cols) >= 2:
            conc_col = conc_col or num_cols[0]; err_col = err_col or num_cols[1]
        else:
            return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_columns")
    _ensure_budget(timer)
    conc = pd.to_numeric(df[conc_col], errors="coerce").to_numpy(dtype=float)
    err = pd.to_numeric(df[err_col], errors="coerce").to_numpy(dtype=float)
    mask = ~(np.isnan(conc) | np.isnan(err))
    if mask.sum() < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    conc, err = conc[mask], err[mask]
    # Bin by concurrency and compute error rate
    n_bins = min(10, len(np.unique(conc)))
    if n_bins < 3:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "too_few_concurrency_levels")
    bins = np.percentile(conc, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)
    if len(bins) < 3:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "too_few_bins")
    bin_idx = np.digitize(conc, bins[1:-1])
    means = [float(np.mean(err[bin_idx == i])) for i in range(len(bins) - 1) if np.sum(bin_idx == i) > 0]
    # Find elbow/threshold
    threshold_idx = 0
    max_jump = 0.0
    for i in range(1, len(means)):
        jump = means[i] - means[i - 1]
        if jump > max_jump: max_jump = jump; threshold_idx = i
    threshold_conc = float(bins[min(threshold_idx + 1, len(bins) - 1)])
    findings = []
    if max_jump > 0 and len(means) >= 3:
        findings.append(_make_finding(
            plugin_id, "cog_load", f"Cognitive load threshold at {conc_col}~{threshold_conc:.1f}",
            f"Error rate jumps at {conc_col}~{threshold_conc:.1f}; max increase={max_jump:.3f}.",
            "Beyond the threshold, concurrent load degrades quality, consistent with cognitive overload.",
            {"metrics": {"threshold": round(threshold_conc, 2), "max_jump": round(max_jump, 4), "conc_col": conc_col, "err_col": err_col}},
            recommendation=f"Cap {conc_col} near {threshold_conc:.0f} to prevent error escalation.",
            severity="warn" if max_jump > np.mean(means) * 0.5 else "info", confidence=0.6,
        ))
    return _finalize(plugin_id, ctx, df, sample_meta, "Cognitive load concurrency analysis.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "threshold": round(threshold_conc, 2)})


def _bullwhip_effect_detection(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Variance amplification ratio across pipeline stages."""
    _log_start(ctx, plugin_id, df, config, inferred)
    time_col, ts = _time_series(df, inferred)
    if ts is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_time_column")
    num_cols = _variance_sorted_numeric(df, inferred, limit=8)
    if len(num_cols) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_numeric_columns")
    _ensure_budget(timer)
    # Compute variance for each column and check amplification along column order
    variances = []
    for c in num_cols:
        v = pd.to_numeric(df[c], errors="coerce").dropna()
        variances.append((c, float(np.var(v)) if len(v) > 1 else 0.0))
    ratios = []
    for i in range(1, len(variances)):
        if variances[i - 1][1] > 1e-9:
            ratios.append((variances[i][0], variances[i][1] / variances[i - 1][1]))
    max_ratio = max((r for _, r in ratios), default=1.0)
    amplifying = [(c, r) for c, r in ratios if r > 1.5]
    findings = []
    if amplifying:
        worst = max(amplifying, key=lambda x: x[1])
        findings.append(_make_finding(
            plugin_id, "bullwhip", f"Bullwhip effect: {worst[0]} amplifies variance {worst[1]:.1f}x",
            f"{len(amplifying)} stage(s) amplify variance; worst={worst[0]} at {worst[1]:.1f}x.",
            "Variance amplification across stages is the bullwhip effect, causing overreaction to small signals.",
            {"metrics": {"max_ratio": round(max_ratio, 2), "amplifying_stages": [(c, round(r, 2)) for c, r in amplifying]}},
            recommendation="Add dampening at amplifying stages; share upstream demand signals to reduce oscillation.",
            severity="warn" if max_ratio > 3.0 else "info", confidence=min(0.8, 0.5 + 0.1 * len(amplifying)),
        ))
    return _finalize(plugin_id, ctx, df, sample_meta, "Bullwhip effect detection.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "max_ratio": round(max_ratio, 2)})


def _eoq_batch_sizing(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """EOQ: Q* = sqrt(2DS/H) optimal batch size."""
    _log_start(ctx, plugin_id, df, config, inferred)
    num_cols = _numeric_columns(df, inferred, max_cols=20)
    if len(num_cols) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_numeric_columns")
    _ensure_budget(timer)
    # Heuristic: use demand-like and cost-like columns
    demand_col = cost_col = None
    for c in num_cols:
        cl = c.lower()
        if any(h in cl for h in ("demand", "volume", "count", "quantity", "order")): demand_col = c
        elif any(h in cl for h in ("cost", "price", "fee", "rate", "hold")): cost_col = c
    if demand_col is None: demand_col = num_cols[0]
    if cost_col is None: cost_col = num_cols[1] if len(num_cols) > 1 else num_cols[0]
    D = pd.to_numeric(df[demand_col], errors="coerce").dropna()
    H = pd.to_numeric(df[cost_col], errors="coerce").dropna()
    if len(D) < 5 or len(H) < 5:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    D_mean = _safe_float(D.mean(), 1.0)
    H_mean = _safe_float(H.mean(), 1.0)
    S = H_mean * 2.0  # Setup cost heuristic
    if D_mean <= 0 or H_mean <= 0:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "non_positive_values")
    Q_star = math.sqrt(2.0 * D_mean * S / H_mean)
    findings = []
    findings.append(_make_finding(
        plugin_id, "eoq", f"Optimal batch size Q*={Q_star:.1f}",
        f"EOQ model: Q*={Q_star:.1f} (D={D_mean:.1f}, S={S:.1f}, H={H_mean:.1f}).",
        "Economic Order Quantity balances setup costs against holding costs for optimal batch sizing.",
        {"metrics": {"Q_star": round(Q_star, 2), "D": round(D_mean, 2), "S": round(S, 2), "H": round(H_mean, 2)}},
        recommendation=f"Target batch sizes near {Q_star:.0f} to minimize total cost.",
        severity="info", confidence=0.55,
    ))
    return _finalize(plugin_id, ctx, df, sample_meta, "EOQ batch sizing.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "Q_star": round(Q_star, 2)})


def _toc_binding_constraint(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Theory of Constraints: identify binding bottleneck in DAG."""
    _log_start(ctx, plugin_id, df, config, inferred)
    if not HAS_NETWORKX:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "networkx_unavailable")
    dur_col = _duration_column(df, inferred)
    edges, sc, dc = _build_edges(df, inferred)
    if edges.empty:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_dag_structure")
    G = _build_nx_graph_from_edges(edges)
    if G is None or G.number_of_nodes() < 3:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "graph_too_small")
    _ensure_budget(timer)
    # Compute utilization per node (degree * optional duration)
    util = {}
    for node in G.nodes():
        deg = G.in_degree(node) + G.out_degree(node)
        util[node] = float(deg)
    bottleneck = max(util, key=util.get)
    max_util = util[bottleneck]
    mean_util = float(np.mean(list(util.values())))
    ratio = max_util / max(1e-9, mean_util)
    findings = []
    if ratio > 2.0:
        findings.append(_make_finding(
            plugin_id, "toc", f"TOC bottleneck: node '{bottleneck}' ({ratio:.1f}x avg utilization)",
            f"Node '{bottleneck}' has {ratio:.1f}x average utilization (degree={max_util:.0f}).",
            "Theory of Constraints: system throughput is limited by its binding constraint.",
            {"metrics": {"bottleneck": str(bottleneck), "ratio": round(ratio, 2), "max_util": max_util, "mean_util": round(mean_util, 2)}},
            recommendation=f"Elevate the constraint at '{bottleneck}': add capacity or offload work.",
            severity="warn" if ratio > 4.0 else "info", confidence=min(0.8, 0.5 + ratio * 0.05),
        ))
    return _finalize(plugin_id, ctx, df, sample_meta, "TOC binding constraint.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "bottleneck": str(bottleneck), "ratio": round(ratio, 2)})


def _safety_stock_buffer(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Safety stock: SS = z*sigma*sqrt(lead_time) per stage."""
    _log_start(ctx, plugin_id, df, config, inferred)
    num_cols = _numeric_columns(df, inferred, max_cols=10)
    if not num_cols:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_columns")
    _ensure_budget(timer)
    z = 1.96  # 95% service level
    findings = []
    for col in num_cols[:5]:
        vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals) < 10: continue
        sigma = float(np.std(vals))
        mean_val = float(np.mean(vals))
        if sigma < 1e-9 or mean_val <= 0: continue
        # Assume lead_time ~ mean/sigma ratio as proxy
        lt_proxy = max(1.0, mean_val / sigma)
        ss = z * sigma * math.sqrt(lt_proxy)
        cv = sigma / mean_val
        if cv > 0.3:
            findings.append(_make_finding(
                plugin_id, f"ss_{col}", f"Safety stock for {col}: SS={ss:.1f} (CV={cv:.2f})",
                f"High variability (CV={cv:.2f}) in {col} requires safety buffer SS={ss:.1f}.",
                "High coefficient of variation demands larger buffers to maintain service levels.",
                {"metrics": {"ss": round(ss, 2), "cv": round(cv, 3), "sigma": round(sigma, 3), "column": col}},
                recommendation=f"Maintain buffer of {ss:.0f} for {col} or reduce variability (CV={cv:.2f}).",
                severity="warn" if cv > 0.6 else "info", confidence=0.6,
            ))
        if len(findings) >= 3: break
    return _finalize(plugin_id, ctx, df, sample_meta, "Safety stock buffer analysis.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer)})


def _spc_cpk_capability(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Cp, Cpk process capability per numeric column."""
    _log_start(ctx, plugin_id, df, config, inferred)
    num_cols = _numeric_columns(df, inferred, max_cols=10)
    if not num_cols:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_columns")
    _ensure_budget(timer)
    findings = []
    for col in num_cols[:6]:
        vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals) < 20: continue
        mu, sigma = float(np.mean(vals)), float(np.std(vals))
        if sigma < 1e-9: continue
        # Use +-3sigma as spec limits if not provided
        USL = mu + 3 * sigma
        LSL = mu - 3 * sigma
        Cp = (USL - LSL) / (6 * sigma)
        Cpk = min((USL - mu) / (3 * sigma), (mu - LSL) / (3 * sigma))
        if Cpk < 1.0:
            findings.append(_make_finding(
                plugin_id, f"cpk_{col}", f"Low capability Cpk={Cpk:.2f} for {col}",
                f"Process capability Cp={Cp:.2f}, Cpk={Cpk:.2f} for {col} (Cpk<1.0 = not capable).",
                "Cpk below 1.0 means the process does not consistently meet specification limits.",
                {"metrics": {"Cp": round(Cp, 3), "Cpk": round(Cpk, 3), "mu": round(mu, 4), "sigma": round(sigma, 4), "column": col}},
                recommendation=f"Center {col} on target and reduce variation to achieve Cpk >= 1.33.",
                severity="warn" if Cpk < 0.67 else "info", confidence=0.7,
            ))
        if len(findings) >= 3: break
    return _finalize(plugin_id, ctx, df, sample_meta, "SPC Cpk capability analysis.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer)})


def _poisson_yield_complexity(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Poisson yield model: Y = exp(-D*A). Estimate defect density and yield."""
    _log_start(ctx, plugin_id, df, config, inferred)
    num_cols = _numeric_columns(df, inferred, max_cols=10)
    if not num_cols:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_columns")
    _ensure_budget(timer)
    # Find defect-like column and area/complexity-like column
    defect_col = area_col = None
    for c in num_cols:
        cl = c.lower()
        if any(h in cl for h in ("defect", "error", "fault", "bug", "fail")): defect_col = c
        elif any(h in cl for h in ("area", "size", "complex", "length", "loc", "step")): area_col = c
    if defect_col is None: defect_col = num_cols[0]
    if area_col is None and len(num_cols) > 1: area_col = num_cols[1]
    if area_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "need_two_columns")
    d_vals = pd.to_numeric(df[defect_col], errors="coerce").dropna().to_numpy(dtype=float)
    a_vals = pd.to_numeric(df[area_col], errors="coerce").dropna().to_numpy(dtype=float)
    n = min(len(d_vals), len(a_vals))
    if n < 10:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    d_vals, a_vals = d_vals[:n], a_vals[:n]
    a_vals = np.abs(a_vals); a_vals[a_vals == 0] = 1.0
    D_density = float(np.mean(d_vals / a_vals))
    A_mean = float(np.mean(a_vals))
    Y = math.exp(-D_density * A_mean) if D_density * A_mean < 50 else 0.0
    findings = []
    findings.append(_make_finding(
        plugin_id, "yield", f"Poisson yield Y={Y*100:.1f}% (D={D_density:.4f})",
        f"Estimated yield={Y*100:.1f}% from defect density D={D_density:.4f}, mean area={A_mean:.1f}.",
        "Poisson yield model predicts output quality from defect density and complexity.",
        {"metrics": {"yield_pct": round(Y * 100, 2), "D_density": round(D_density, 5), "A_mean": round(A_mean, 2)}},
        recommendation=f"Reduce defect density below {D_density/2:.4f} to double yield.",
        severity="warn" if Y < 0.8 else "info", confidence=0.6,
    ))
    return _finalize(plugin_id, ctx, df, sample_meta, "Poisson yield complexity.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "yield_pct": round(Y * 100, 2)})


def _virtual_metrology_early(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Predict final quality from early-stage metrics using simple regression."""
    _log_start(ctx, plugin_id, df, config, inferred)
    time_col, ts = _time_series(df, inferred)
    num_cols = _variance_sorted_numeric(df, inferred, limit=8)
    if len(num_cols) < 3:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_numeric_columns")
    _ensure_budget(timer)
    # Last column as target (final quality), first half as early predictors
    target = num_cols[-1]
    predictors = num_cols[:len(num_cols) // 2 + 1]
    if target in predictors: predictors = [c for c in predictors if c != target]
    if not predictors:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_predictors")
    frame = df[predictors + [target]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(frame) < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    X = frame[predictors].to_numpy(dtype=float)
    y = frame[target].to_numpy(dtype=float)
    # Simple OLS
    X_aug = np.column_stack([np.ones(len(X)), X])
    try:
        beta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
        y_pred = X_aug @ beta
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / max(1e-9, ss_tot)
    except Exception:
        r2 = 0.0
    findings = []
    if r2 > 0.3:
        findings.append(_make_finding(
            plugin_id, "vm", f"Early metrics predict final quality (R²={r2:.2f})",
            f"Early predictors {predictors} explain {r2*100:.0f}% variance in {target}.",
            "Virtual metrology enables early quality prediction, catching issues before final stage.",
            {"metrics": {"r2": round(r2, 3), "target": target, "predictors": predictors}},
            recommendation=f"Use early-stage metrics to predict {target} and intervene before completion.",
            severity="warn" if r2 > 0.7 else "info", confidence=min(0.85, 0.4 + r2 * 0.5),
        ))
    return _finalize(plugin_id, ctx, df, sample_meta, "Virtual metrology analysis.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "r2": round(r2, 3)})


def _r2r_ewma_control(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Run-to-run EWMA control limits."""
    _log_start(ctx, plugin_id, df, config, inferred)
    time_col, ts = _time_series(df, inferred)
    if ts is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_time_column")
    dur_col = _duration_column(df, inferred)
    if dur_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_metric_column")
    vals = pd.to_numeric(df[dur_col], errors="coerce")
    if ts is not None:
        vals = vals[ts.argsort()]
    vals = vals.dropna().to_numpy(dtype=float)
    if len(vals) < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    _ensure_budget(timer)
    lam = 0.2
    ewma = np.zeros(len(vals))
    ewma[0] = vals[0]
    for i in range(1, len(vals)):
        ewma[i] = lam * vals[i] + (1 - lam) * ewma[i - 1]
    sigma = float(np.std(vals))
    mu = float(np.mean(vals))
    # Control limits for EWMA
    cl_factor = 3.0 * sigma * math.sqrt(lam / (2.0 - lam))
    ucl = mu + cl_factor
    lcl = mu - cl_factor
    ooc = int(np.sum((ewma > ucl) | (ewma < lcl)))
    ooc_pct = ooc / len(ewma)
    findings = []
    if ooc > 0:
        findings.append(_make_finding(
            plugin_id, "ewma", f"EWMA out-of-control: {ooc} points ({ooc_pct*100:.1f}%)",
            f"{ooc}/{len(ewma)} EWMA values exceed control limits for {dur_col}.",
            "Out-of-control EWMA signals indicate process drift requiring run-to-run adjustment.",
            {"metrics": {"ooc": ooc, "ooc_pct": round(ooc_pct, 3), "ucl": round(ucl, 3), "lcl": round(lcl, 3), "lambda": lam}},
            recommendation="Implement run-to-run control to adjust process parameters after OOC signals.",
            severity="warn" if ooc_pct > 0.1 else "info", confidence=min(0.8, 0.5 + ooc_pct * 2),
        ))
    return _finalize(plugin_id, ctx, df, sample_meta, "R2R EWMA control analysis.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "ooc": ooc, "ooc_pct": round(ooc_pct, 3)})


def _tfidf_step_importance(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """TF-IDF for step/category uniqueness across entities."""
    _log_start(ctx, plugin_id, df, config, inferred)
    cat_cols = _categorical_columns(df, inferred, max_cols=10)
    if not cat_cols:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_categorical_columns")
    _ensure_budget(timer)
    # Treat first cat col as "term" and second (or index) as "document"
    term_col = cat_cols[0]
    doc_col = cat_cols[1] if len(cat_cols) > 1 else None
    terms = df[term_col].astype(str)
    if doc_col:
        docs = df[doc_col].astype(str)
    else:
        # Use row index bins as documents
        n_docs = min(20, len(df) // 5)
        if n_docs < 2:
            return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
        docs = pd.Series(np.digitize(np.arange(len(df)), np.linspace(0, len(df), n_docs + 1)[1:-1]).astype(str))
    unique_docs = docs.unique()
    N = len(unique_docs)
    if N < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "single_document")
    # Compute IDF
    doc_freq = Counter()
    for d in unique_docs:
        mask = docs == d
        doc_freq.update(set(terms[mask].tolist()))
    tfidf_scores = {}
    tf = Counter(terms.tolist())
    total = sum(tf.values())
    for t, count in tf.items():
        tf_val = count / total
        idf_val = math.log(N / max(1, doc_freq.get(t, 1)))
        tfidf_scores[t] = tf_val * idf_val
    top = sorted(tfidf_scores.items(), key=lambda x: -x[1])[:10]
    findings = []
    if top and top[0][1] > 0:
        findings.append(_make_finding(
            plugin_id, "tfidf", f"Top distinctive terms in {term_col}",
            f"TF-IDF identifies {len(top)} distinctive values; top='{top[0][0]}' (score={top[0][1]:.4f}).",
            "High TF-IDF values indicate steps/categories that are unusually concentrated in specific contexts.",
            {"metrics": {"top_terms": [{"term": t, "score": round(s, 4)} for t, s in top[:5]], "column": term_col}},
            recommendation="Investigate high-TF-IDF steps for specialization opportunities or anomalies.",
            severity="info", confidence=0.6,
        ))
    return _finalize(plugin_id, ctx, df, sample_meta, "TF-IDF step importance.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer)})


def _bradford_law_core_hosts(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Bradford's Law: identify core vs peripheral hosts/sources."""
    _log_start(ctx, plugin_id, df, config, inferred)
    cat_cols = _categorical_columns(df, inferred, max_cols=10)
    if not cat_cols:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_categorical_columns")
    _ensure_budget(timer)
    col = cat_cols[0]
    counts = df[col].value_counts().sort_values(ascending=False)
    if len(counts) < 5:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "too_few_categories")
    total = int(counts.sum())
    cumsum = counts.cumsum()
    # Bradford zones: core produces ~1/3 of output
    third = total / 3.0
    core = int((cumsum <= third).sum()) + 1
    core = min(core, len(counts))
    core_pct = core / len(counts) * 100
    concentration = 1.0 - core / len(counts)
    findings = []
    if core_pct < 30:
        findings.append(_make_finding(
            plugin_id, "bradford", f"Bradford's Law: {core} core sources ({core_pct:.0f}%) produce 33% of output",
            f"{core}/{len(counts)} sources ({core_pct:.1f}%) produce one-third of all activity.",
            "Bradford's Law concentration means a small core dominates; peripheral sources contribute marginally.",
            {"metrics": {"core_count": core, "total_sources": len(counts), "core_pct": round(core_pct, 1), "concentration": round(concentration, 3)}},
            recommendation=f"Focus resources on the {core} core sources; evaluate whether peripheral sources justify their overhead.",
            severity="info", confidence=0.65,
        ))
    return _finalize(plugin_id, ctx, df, sample_meta, "Bradford's Law core hosts.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "core_count": core, "concentration": round(concentration, 3)})


def _lotka_law_concentration(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Lotka's Law: workload concentration 1/n^alpha."""
    _log_start(ctx, plugin_id, df, config, inferred)
    cat_cols = _categorical_columns(df, inferred, max_cols=10)
    if not cat_cols:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_categorical_columns")
    _ensure_budget(timer)
    col = cat_cols[0]
    counts = df[col].value_counts()
    if len(counts) < 10:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "too_few_categories")
    freq_of_freq = Counter(counts.values)
    x = np.array(sorted(freq_of_freq.keys()), dtype=float)
    y = np.array([freq_of_freq[int(k)] for k in x], dtype=float)
    mask = (x > 0) & (y > 0)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_frequency_data")
    # Fit log-log: log(y) = log(C) - alpha*log(x)
    log_x, log_y = np.log(x), np.log(y)
    coeffs = np.polyfit(log_x, log_y, 1)
    alpha = -coeffs[0]
    r2 = 1.0 - np.sum((log_y - np.polyval(coeffs, log_x)) ** 2) / max(1e-9, np.sum((log_y - np.mean(log_y)) ** 2))
    findings = []
    if r2 > 0.5:
        findings.append(_make_finding(
            plugin_id, "lotka", f"Lotka's Law: alpha={alpha:.2f} (R²={r2:.2f}) for {col}",
            f"Workload follows power law 1/n^{alpha:.2f} (R²={r2:.2f}).",
            "Lotka's Law concentration means few entities do most work; alpha~2 is typical.",
            {"metrics": {"alpha": round(alpha, 3), "r2": round(r2, 3), "column": col}},
            recommendation=f"Alpha={alpha:.1f}: {'typical' if 1.5 < alpha < 2.5 else 'atypical'} concentration. Balance workload if alpha > 2.5.",
            severity="warn" if alpha > 2.5 else "info", confidence=min(0.8, 0.4 + r2 * 0.4),
        ))
    return _finalize(plugin_id, ctx, df, sample_meta, "Lotka's Law concentration.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "alpha": round(alpha, 3)})


def _life_table_hazard(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Age-specific hazard rates using life table approach."""
    _log_start(ctx, plugin_id, df, config, inferred)
    dur_col = _duration_column(df, inferred)
    if dur_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_duration_column")
    vals = pd.to_numeric(df[dur_col], errors="coerce").dropna().to_numpy(dtype=float)
    vals = vals[vals > 0]
    if len(vals) < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    _ensure_budget(timer)
    # Build life table with equal-width bins
    n_bins = min(10, max(3, len(vals) // 20))
    bins = np.linspace(vals.min(), vals.max() + 1e-9, n_bins + 1)
    table = []
    alive = len(vals)
    for i in range(n_bins):
        in_bin = int(np.sum((vals >= bins[i]) & (vals < bins[i + 1])))
        hazard = in_bin / max(1, alive)
        table.append({"interval": f"[{bins[i]:.1f}, {bins[i+1]:.1f})", "deaths": in_bin, "alive": alive, "hazard": round(hazard, 4)})
        alive -= in_bin
    # Check for increasing hazard (aging) vs bathtub
    hazards = [t["hazard"] for t in table]
    trend = "flat"
    if len(hazards) >= 3:
        first_third = np.mean(hazards[:len(hazards) // 3])
        last_third = np.mean(hazards[-(len(hazards) // 3):])
        if last_third > first_third * 1.5: trend = "increasing"
        elif first_third > last_third * 1.5: trend = "decreasing"
    findings = []
    findings.append(_make_finding(
        plugin_id, "hazard", f"Life table hazard: {trend} pattern in {dur_col}",
        f"Hazard rate pattern: {trend}. Peak hazard={max(hazards):.4f}.",
        "Hazard rate shape reveals whether failure risk increases, decreases, or stays constant over time.",
        {"metrics": {"trend": trend, "peak_hazard": round(max(hazards), 4), "table": table[:5]}},
        recommendation=f"{'Preventive maintenance needed for aging items.' if trend == 'increasing' else 'Early-life screening may help.' if trend == 'decreasing' else 'Constant risk; random failure model applies.'}",
        severity="warn" if trend == "increasing" else "info", confidence=0.65,
    ))
    return _finalize(plugin_id, ctx, df, sample_meta, "Life table hazard analysis.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "trend": trend})


def _population_pyramid_shape(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Queue age distribution shape classification."""
    _log_start(ctx, plugin_id, df, config, inferred)
    dur_col = _duration_column(df, inferred)
    if dur_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_duration_column")
    vals = pd.to_numeric(df[dur_col], errors="coerce").dropna().to_numpy(dtype=float)
    if len(vals) < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    _ensure_budget(timer)
    skew = float(scipy_stats.skew(vals)) if HAS_SCIPY else float(np.mean(((vals - np.mean(vals)) / max(1e-9, np.std(vals))) ** 3))
    kurt = float(scipy_stats.kurtosis(vals)) if HAS_SCIPY else float(np.mean(((vals - np.mean(vals)) / max(1e-9, np.std(vals))) ** 4) - 3)
    # Classify shape
    if skew > 1.0: shape = "expansive"  # young-heavy, right-skewed durations
    elif skew < -0.5: shape = "constrictive"  # old-heavy
    elif abs(kurt) > 2: shape = "bimodal"
    else: shape = "stationary"
    findings = []
    findings.append(_make_finding(
        plugin_id, "pyramid", f"Age pyramid shape: {shape} (skew={skew:.2f})",
        f"Distribution shape={shape}, skewness={skew:.2f}, kurtosis={kurt:.2f}.",
        "Population pyramid shape indicates whether the queue/cohort is growing, shrinking, or stable.",
        {"metrics": {"shape": shape, "skew": round(skew, 3), "kurtosis": round(kurt, 3), "column": dur_col}},
        recommendation=f"{'Queue is young-heavy: expect growth.' if shape == 'expansive' else 'Queue is aging: plan for turnover.' if shape == 'constrictive' else 'Bimodal: two distinct cohorts.' if shape == 'bimodal' else 'Stable age distribution.'}",
        severity="warn" if shape in ("expansive", "constrictive") else "info", confidence=0.6,
    ))
    return _finalize(plugin_id, ctx, df, sample_meta, "Population pyramid shape.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "shape": shape})


def _demographic_transition(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Track birth/death rate analogue over time."""
    _log_start(ctx, plugin_id, df, config, inferred)
    time_col, ts = _time_series(df, inferred)
    if ts is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_time_column")
    _ensure_budget(timer)
    order = ts.argsort()
    ts_sorted = ts.iloc[order]
    n = len(ts_sorted)
    n_windows = min(10, max(3, n // 30))
    chunks = np.array_split(np.arange(n), n_windows)
    # Birth rate = new unique values; death rate = values that stop appearing
    cat_cols = _categorical_columns(df, inferred, max_cols=3)
    col = cat_cols[0] if cat_cols else None
    if col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_categorical_column")
    df_sorted = df.iloc[order]
    seen = set()
    birth_rates, death_rates = [], []
    prev_set = set()
    for chunk in chunks:
        chunk_vals = set(df_sorted[col].iloc[chunk].dropna().astype(str).tolist())
        births = len(chunk_vals - seen)
        deaths = len(prev_set - chunk_vals) if prev_set else 0
        total = max(1, len(chunk_vals))
        birth_rates.append(births / total)
        death_rates.append(deaths / max(1, len(prev_set)) if prev_set else 0.0)
        seen.update(chunk_vals)
        prev_set = chunk_vals
    # Classify transition stage
    avg_birth = float(np.mean(birth_rates))
    avg_death = float(np.mean(death_rates))
    if avg_birth > 0.3 and avg_death < 0.1: stage = "pre-transition"
    elif avg_birth > avg_death * 1.5: stage = "early_transition"
    elif abs(avg_birth - avg_death) < 0.1: stage = "late_transition"
    else: stage = "post-transition"
    findings = []
    findings.append(_make_finding(
        plugin_id, "demo_transition", f"Demographic transition: {stage}",
        f"Birth rate={avg_birth:.2f}, death rate={avg_death:.2f}. Stage: {stage}.",
        "Demographic transition model tracks entity creation/removal dynamics over time.",
        {"metrics": {"stage": stage, "avg_birth": round(avg_birth, 3), "avg_death": round(avg_death, 3), "column": col}},
        recommendation=f"{'Rapid growth phase: prepare capacity.' if 'pre' in stage or 'early' in stage else 'Stable or declining: optimize retention.'}",
        severity="warn" if "pre" in stage else "info", confidence=0.6,
    ))
    return _finalize(plugin_id, ctx, df, sample_meta, "Demographic transition analysis.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "stage": stage})


def _condorcet_host_ranking(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Condorcet pairwise comparison matrix for ranking entities."""
    _log_start(ctx, plugin_id, df, config, inferred)
    num_cols = _numeric_columns(df, inferred, max_cols=10)
    cat_cols = _categorical_columns(df, inferred, max_cols=5)
    if not cat_cols or not num_cols:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_columns")
    _ensure_budget(timer)
    entity_col = cat_cols[0]
    metric_col = num_cols[0]
    groups = df.groupby(entity_col)[metric_col].mean().dropna()
    entities = list(groups.index)
    if len(entities) < 3 or len(entities) > 200:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "wrong_entity_count")
    # Build pairwise wins matrix
    n = len(entities)
    wins = np.zeros((n, n))
    vals = groups.to_numpy(dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            if vals[i] < vals[j]: wins[i, j] += 1  # lower is better (duration-like)
            elif vals[j] < vals[i]: wins[j, i] += 1
    # Condorcet winner: beats all others
    total_wins = wins.sum(axis=1)
    ranking = np.argsort(-total_wins)
    condorcet_winner = entities[ranking[0]] if total_wins[ranking[0]] == n - 1 else None
    findings = []
    top3 = [(str(entities[i]), int(total_wins[i])) for i in ranking[:3]]
    findings.append(_make_finding(
        plugin_id, "condorcet", f"Condorcet ranking: top={top3[0][0]}" + (" (Condorcet winner)" if condorcet_winner else ""),
        f"Pairwise ranking of {n} entities by {metric_col}. Top: {top3}.",
        "Condorcet ranking aggregates pairwise comparisons for robust ordering.",
        {"metrics": {"top3": top3, "condorcet_winner": str(condorcet_winner), "n_entities": n}},
        recommendation=f"{'Condorcet winner exists: clear best performer.' if condorcet_winner else 'No Condorcet winner: cyclical preferences exist.'}",
        severity="info", confidence=0.7,
    ))
    return _finalize(plugin_id, ctx, df, sample_meta, "Condorcet host ranking.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "n_entities": n})


def _borda_count_multicriteria(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Borda count: sum-of-ranks across multiple criteria."""
    _log_start(ctx, plugin_id, df, config, inferred)
    num_cols = _numeric_columns(df, inferred, max_cols=10)
    cat_cols = _categorical_columns(df, inferred, max_cols=5)
    if not cat_cols or len(num_cols) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_columns")
    _ensure_budget(timer)
    entity_col = cat_cols[0]
    criteria = num_cols[:5]
    groups = df.groupby(entity_col)[criteria].mean().dropna()
    if len(groups) < 3:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "too_few_entities")
    # Rank each criterion (lower rank = better, assuming lower values are better)
    ranks = groups.rank(ascending=True)
    borda = ranks.sum(axis=1).sort_values()
    top3 = [(str(idx), round(float(val), 1)) for idx, val in borda.head(3).items()]
    spread = float(borda.max() - borda.min())
    findings = []
    findings.append(_make_finding(
        plugin_id, "borda", f"Borda ranking: best={top3[0][0]} across {len(criteria)} criteria",
        f"Borda count across {len(criteria)} criteria for {len(groups)} entities. Top: {top3}.",
        "Borda count provides fair multi-criteria ranking by summing ranks across dimensions.",
        {"metrics": {"top3": top3, "n_criteria": len(criteria), "n_entities": len(groups), "spread": round(spread, 1)}},
        recommendation=f"Top performer '{top3[0][0]}' is best across criteria; study its practices.",
        severity="info", confidence=0.65,
    ))
    return _finalize(plugin_id, ctx, df, sample_meta, "Borda count multicriteria.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer)})


def _arrow_impossibility_pareto(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Pareto frontier computation for multi-objective optimization."""
    _log_start(ctx, plugin_id, df, config, inferred)
    num_cols = _numeric_columns(df, inferred, max_cols=10)
    if len(num_cols) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_numeric_columns")
    _ensure_budget(timer)
    cols = num_cols[:4]
    mat = df[cols].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(dtype=float)
    if len(mat) < 10:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    # Find Pareto-optimal points (minimize all objectives)
    is_pareto = np.ones(len(mat), dtype=bool)
    for i in range(len(mat)):
        if not is_pareto[i]: continue
        for j in range(len(mat)):
            if i == j or not is_pareto[j]: continue
            if np.all(mat[j] <= mat[i]) and np.any(mat[j] < mat[i]):
                is_pareto[i] = False; break
    n_pareto = int(is_pareto.sum())
    pareto_pct = n_pareto / len(mat) * 100
    findings = []
    findings.append(_make_finding(
        plugin_id, "pareto", f"Pareto frontier: {n_pareto} optimal points ({pareto_pct:.1f}%)",
        f"{n_pareto}/{len(mat)} points ({pareto_pct:.1f}%) are Pareto-optimal across {len(cols)} objectives.",
        "Large Pareto frontiers indicate genuine tradeoffs (Arrow impossibility); small ones suggest dominant solutions.",
        {"metrics": {"n_pareto": n_pareto, "pareto_pct": round(pareto_pct, 1), "n_objectives": len(cols), "objectives": cols}},
        recommendation=f"{'Genuine tradeoffs exist; no single solution dominates.' if pareto_pct > 10 else 'Few Pareto points: dominant solutions exist. Focus on those.'}",
        severity="warn" if pareto_pct > 30 else "info", confidence=0.7,
    ))
    return _finalize(plugin_id, ctx, df, sample_meta, "Arrow impossibility / Pareto analysis.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "n_pareto": n_pareto, "pareto_pct": round(pareto_pct, 1)})


def _hrv_intercompletion(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """RMSSD, SDNN of inter-completion intervals (HRV analogue)."""
    _log_start(ctx, plugin_id, df, config, inferred)
    time_col, ts = _time_series(df, inferred)
    if ts is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_time_column")
    ts_sorted = ts.dropna().sort_values()
    if len(ts_sorted) < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    _ensure_budget(timer)
    # Inter-completion intervals in seconds
    diffs = np.diff(ts_sorted.astype(np.int64) / 1e9)  # nanoseconds to seconds
    diffs = diffs[diffs > 0]
    if len(diffs) < 10:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_intervals")
    sdnn = float(np.std(diffs))
    rmssd = float(np.sqrt(np.mean(np.diff(diffs) ** 2)))
    mean_rr = float(np.mean(diffs))
    cv = sdnn / max(1e-9, mean_rr)
    findings = []
    if cv > 0.3:
        findings.append(_make_finding(
            plugin_id, "hrv", f"High variability: SDNN={sdnn:.1f}s, RMSSD={rmssd:.1f}s (CV={cv:.2f})",
            f"Inter-completion HRV: SDNN={sdnn:.1f}s, RMSSD={rmssd:.1f}s, CV={cv:.2f}.",
            "High HRV indicates irregular completion cadence, analogous to cardiac variability.",
            {"metrics": {"sdnn": round(sdnn, 2), "rmssd": round(rmssd, 2), "mean_rr": round(mean_rr, 2), "cv": round(cv, 3)}},
            recommendation="Reduce inter-completion variability for more predictable throughput.",
            severity="warn" if cv > 0.6 else "info", confidence=min(0.8, 0.5 + cv * 0.3),
        ))
    return _finalize(plugin_id, ctx, df, sample_meta, "HRV inter-completion analysis.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "sdnn": round(sdnn, 2), "rmssd": round(rmssd, 2)})


def _vo2max_capacity_ceiling(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Max sustainable throughput (VO2max analogue)."""
    _log_start(ctx, plugin_id, df, config, inferred)
    time_col, ts = _time_series(df, inferred)
    if ts is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_time_column")
    dur_col = _duration_column(df, inferred)
    num_cols = _numeric_columns(df, inferred, max_cols=5)
    metric_col = dur_col or (num_cols[0] if num_cols else None)
    if metric_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_metric_column")
    _ensure_budget(timer)
    frame = pd.DataFrame({"ts": ts, "val": pd.to_numeric(df[metric_col], errors="coerce")}).dropna().sort_values("ts")
    if len(frame) < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    # Rolling throughput: count per time window
    vals = frame["val"].to_numpy(dtype=float)
    n_windows = min(20, len(vals) // 5)
    chunks = np.array_split(vals, n_windows)
    throughputs = [float(len(c)) / max(1.0, float(np.ptp(c))) if np.ptp(c) > 0 else float(len(c)) for c in chunks]
    vo2max = float(np.max(throughputs))
    sustained = float(np.percentile(throughputs, 75))
    utilization = sustained / max(1e-9, vo2max)
    findings = []
    findings.append(_make_finding(
        plugin_id, "vo2max", f"Capacity ceiling: peak={vo2max:.2f}, sustained={sustained:.2f} ({utilization*100:.0f}% util)",
        f"Peak throughput={vo2max:.2f}, sustained (p75)={sustained:.2f}, utilization={utilization*100:.0f}%.",
        "VO2max analogue shows maximum vs sustained throughput; high utilization means little headroom.",
        {"metrics": {"vo2max": round(vo2max, 3), "sustained": round(sustained, 3), "utilization": round(utilization, 3)}},
        recommendation=f"{'At capacity ceiling; add capacity or reduce load.' if utilization > 0.85 else 'Headroom available; current load is sustainable.'}",
        severity="warn" if utilization > 0.85 else "info", confidence=0.6,
    ))
    return _finalize(plugin_id, ctx, df, sample_meta, "VO2max capacity ceiling.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "vo2max": round(vo2max, 3)})


def _banister_fitness_fatigue(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Dual exponential fitness-fatigue model."""
    _log_start(ctx, plugin_id, df, config, inferred)
    time_col, ts = _time_series(df, inferred)
    if ts is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_time_column")
    dur_col = _duration_column(df, inferred)
    if dur_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_metric_column")
    vals = pd.to_numeric(df[dur_col], errors="coerce")
    frame = pd.DataFrame({"ts": ts, "val": vals}).dropna().sort_values("ts").reset_index(drop=True)
    if len(frame) < 30:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    _ensure_budget(timer)
    y = frame["val"].to_numpy(dtype=float)
    # Banister model: performance = baseline + fitness - fatigue
    # fitness decays with tau1, fatigue decays with tau2
    tau1, tau2 = 10.0, 5.0  # fitness and fatigue time constants
    k1, k2 = 1.0, 1.5  # gain factors
    fitness = np.zeros(len(y))
    fatigue = np.zeros(len(y))
    for i in range(1, len(y)):
        fitness[i] = fitness[i - 1] * math.exp(-1.0 / tau1) + k1 * y[i - 1]
        fatigue[i] = fatigue[i - 1] * math.exp(-1.0 / tau2) + k2 * y[i - 1]
    performance = fitness - fatigue
    # Find where fatigue dominates
    fatigue_dominant_pct = float(np.mean(fatigue > fitness))
    trend_perf = float(np.corrcoef(np.arange(len(performance)), performance)[0, 1]) if np.std(performance) > 0 else 0.0
    findings = []
    if fatigue_dominant_pct > 0.3:
        findings.append(_make_finding(
            plugin_id, "banister", f"Fatigue dominates {fatigue_dominant_pct*100:.0f}% of time",
            f"Banister model: fatigue exceeds fitness {fatigue_dominant_pct*100:.0f}% of periods. Performance trend r={trend_perf:.2f}.",
            "When fatigue exceeds fitness, performance degrades; recovery periods are needed.",
            {"metrics": {"fatigue_dominant_pct": round(fatigue_dominant_pct, 3), "trend": round(trend_perf, 3)}},
            recommendation="Schedule recovery periods when fatigue dominates; reduce load intensity.",
            severity="warn" if fatigue_dominant_pct > 0.5 else "info", confidence=0.6,
        ))
    return _finalize(plugin_id, ctx, df, sample_meta, "Banister fitness-fatigue model.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "fatigue_dominant_pct": round(fatigue_dominant_pct, 3)})


def _voronoi_resource_partition(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Voronoi diagram for resource coverage analysis."""
    _log_start(ctx, plugin_id, df, config, inferred)
    if not HAS_SCIPY:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "scipy_unavailable")
    num_cols = _numeric_columns(df, inferred, max_cols=10)
    if len(num_cols) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_numeric_columns")
    _ensure_budget(timer)
    cols = num_cols[:2]
    points = df[cols].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(dtype=float)
    if len(points) < 5:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    points = points[:min(5000, len(points))]
    try:
        vor = scipy_spatial.Voronoi(points)
        # Compute region areas for finite regions
        areas = []
        for region_idx in vor.point_region:
            region = vor.regions[region_idx]
            if -1 in region or not region: continue
            verts = vor.vertices[region]
            # Shoelace formula
            n = len(verts)
            area = 0.5 * abs(sum(verts[i][0] * verts[(i + 1) % n][1] - verts[(i + 1) % n][0] * verts[i][1] for i in range(n)))
            if area > 0: areas.append(area)
    except Exception:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "voronoi_failed")
    if not areas:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_finite_regions")
    cv_area = float(np.std(areas) / max(1e-9, np.mean(areas)))
    max_min_ratio = float(max(areas) / max(1e-9, min(areas)))
    findings = []
    if cv_area > 0.5:
        findings.append(_make_finding(
            plugin_id, "voronoi", f"Uneven Voronoi partition: CV={cv_area:.2f}, max/min={max_min_ratio:.1f}x",
            f"Voronoi cell area CV={cv_area:.2f}; largest cell is {max_min_ratio:.1f}x smallest.",
            "Uneven Voronoi cells indicate unbalanced resource coverage in the 2D parameter space.",
            {"metrics": {"cv_area": round(cv_area, 3), "max_min_ratio": round(max_min_ratio, 1), "n_regions": len(areas)}},
            recommendation="Redistribute resources to equalize coverage; large cells are underserved.",
            severity="warn" if cv_area > 1.0 else "info", confidence=min(0.8, 0.5 + cv_area * 0.2),
        ))
    return _finalize(plugin_id, ctx, df, sample_meta, "Voronoi resource partition.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "cv_area": round(cv_area, 3)})


def _delaunay_interpolation(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Delaunay triangulation for interpolation quality assessment."""
    _log_start(ctx, plugin_id, df, config, inferred)
    if not HAS_SCIPY:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "scipy_unavailable")
    num_cols = _numeric_columns(df, inferred, max_cols=10)
    if len(num_cols) < 3:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_numeric_columns")
    _ensure_budget(timer)
    xy_cols = num_cols[:2]
    z_col = num_cols[2]
    frame = df[xy_cols + [z_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(frame) < 10:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    points = frame[xy_cols].to_numpy(dtype=float)[:min(5000, len(frame))]
    z_vals = frame[z_col].to_numpy(dtype=float)[:len(points)]
    try:
        tri = scipy_spatial.Delaunay(points)
        n_simplices = len(tri.simplices)
    except Exception:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "delaunay_failed")
    # Assess interpolation quality via leave-one-out CV on small sample
    sample_n = min(100, len(points))
    rng = np.random.RandomState(int(config.get("seed", 42)))
    sample_idx = rng.choice(len(points), sample_n, replace=False)
    errors = []
    for idx in sample_idx:
        simplex = tri.find_simplex(points[idx])
        if simplex >= 0:
            verts = tri.simplices[simplex]
            neighbor_z = z_vals[verts]
            pred = float(np.mean(neighbor_z))
            errors.append(abs(z_vals[idx] - pred))
    mae = float(np.mean(errors)) if errors else 0.0
    z_range = float(np.ptp(z_vals))
    rel_error = mae / max(1e-9, z_range)
    findings = []
    findings.append(_make_finding(
        plugin_id, "delaunay", f"Delaunay interpolation: MAE={mae:.3f} ({rel_error*100:.1f}% of range)",
        f"{n_simplices} triangles; interpolation MAE={mae:.3f} ({rel_error*100:.1f}% relative error).",
        "Delaunay triangulation quality indicates how well sparse samples represent the continuous surface.",
        {"metrics": {"mae": round(mae, 4), "rel_error": round(rel_error, 4), "n_simplices": n_simplices}},
        recommendation=f"{'Interpolation is poor; add sampling points in sparse regions.' if rel_error > 0.2 else 'Coverage is adequate for interpolation.'}",
        severity="warn" if rel_error > 0.2 else "info", confidence=0.6,
    ))
    return _finalize(plugin_id, ctx, df, sample_meta, "Delaunay interpolation.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "mae": round(mae, 4), "n_simplices": n_simplices})


def _convex_hull_operating_envelope(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Convex hull as safety/operating envelope boundary."""
    _log_start(ctx, plugin_id, df, config, inferred)
    if not HAS_SCIPY:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "scipy_unavailable")
    num_cols = _numeric_columns(df, inferred, max_cols=10)
    if len(num_cols) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_numeric_columns")
    _ensure_budget(timer)
    cols = num_cols[:2]
    points = df[cols].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(dtype=float)
    if len(points) < 5:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    points = points[:min(10000, len(points))]
    try:
        hull = scipy_spatial.ConvexHull(points)
        hull_area = float(hull.volume)  # In 2D, volume = area
        n_hull = len(hull.vertices)
    except Exception:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "hull_failed")
    # Bounding box area
    ranges = np.ptp(points, axis=0)
    bbox_area = float(np.prod(ranges)) if np.all(ranges > 0) else 1.0
    fill_ratio = hull_area / max(1e-9, bbox_area)
    hull_pct = n_hull / len(points) * 100
    findings = []
    findings.append(_make_finding(
        plugin_id, "hull", f"Operating envelope: {fill_ratio*100:.0f}% fill, {n_hull} boundary points",
        f"Convex hull uses {n_hull} boundary points ({hull_pct:.1f}%), fill ratio={fill_ratio*100:.0f}%.",
        "Convex hull defines the operating envelope; points near the boundary are at operational limits.",
        {"metrics": {"fill_ratio": round(fill_ratio, 3), "n_hull": n_hull, "hull_pct": round(hull_pct, 1), "hull_area": round(hull_area, 2)}},
        recommendation=f"{'Sparse envelope; many configurations are unexplored.' if fill_ratio < 0.5 else 'Dense coverage of the operating space.'}",
        severity="warn" if fill_ratio < 0.3 else "info", confidence=0.65,
    ))
    return _finalize(plugin_id, ctx, df, sample_meta, "Convex hull operating envelope.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "fill_ratio": round(fill_ratio, 3), "n_hull": n_hull})


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

HANDLERS: dict[str, Callable[..., PluginResult]] = {
    "analysis_seriation_ordering_v1": _seriation_ordering,
    "analysis_stratigraphic_layers_v1": _stratigraphic_layers,
    "analysis_harris_matrix_ordering_v1": _harris_matrix_ordering,
    "analysis_fitness_landscape_mapping_v1": _fitness_landscape_mapping,
    "analysis_punctuated_equilibrium_v1": _punctuated_equilibrium,
    "analysis_neutral_drift_noise_v1": _neutral_drift_noise,
    "analysis_red_queen_arms_race_v1": _red_queen_arms_race,
    "analysis_hick_law_pool_sizing_v1": _hick_law_pool_sizing,
    "analysis_fitts_law_precision_v1": _fitts_law_precision,
    "analysis_cognitive_load_concurrency_v1": _cognitive_load_concurrency,
    "analysis_bullwhip_effect_detection_v1": _bullwhip_effect_detection,
    "analysis_eoq_batch_sizing_v1": _eoq_batch_sizing,
    "analysis_toc_binding_constraint_v1": _toc_binding_constraint,
    "analysis_safety_stock_buffer_v1": _safety_stock_buffer,
    "analysis_spc_cpk_capability_v1": _spc_cpk_capability,
    "analysis_poisson_yield_complexity_v1": _poisson_yield_complexity,
    "analysis_virtual_metrology_early_v1": _virtual_metrology_early,
    "analysis_r2r_ewma_control_v1": _r2r_ewma_control,
    "analysis_tfidf_step_importance_v1": _tfidf_step_importance,
    "analysis_bradford_law_core_hosts_v1": _bradford_law_core_hosts,
    "analysis_lotka_law_concentration_v1": _lotka_law_concentration,
    "analysis_life_table_hazard_v1": _life_table_hazard,
    "analysis_population_pyramid_shape_v1": _population_pyramid_shape,
    "analysis_demographic_transition_v1": _demographic_transition,
    "analysis_condorcet_host_ranking_v1": _condorcet_host_ranking,
    "analysis_borda_count_multicriteria_v1": _borda_count_multicriteria,
    "analysis_arrow_impossibility_pareto_v1": _arrow_impossibility_pareto,
    "analysis_hrv_intercompletion_v1": _hrv_intercompletion,
    "analysis_vo2max_capacity_ceiling_v1": _vo2max_capacity_ceiling,
    "analysis_banister_fitness_fatigue_v1": _banister_fitness_fatigue,
    "analysis_voronoi_resource_partition_v1": _voronoi_resource_partition,
    "analysis_delaunay_interpolation_v1": _delaunay_interpolation,
    "analysis_convex_hull_operating_envelope_v1": _convex_hull_operating_envelope,
}
