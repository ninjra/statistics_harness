"""Cross-domain plugins: physics & engineering (plugins 25-47)."""
from __future__ import annotations

import hashlib
import json
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
    from scipy import signal as scipy_signal
    from scipy import integrate as scipy_integrate
    from scipy import optimize as scipy_optimize
    HAS_SCIPY = True
except Exception:
    scipy_stats = scipy_signal = scipy_integrate = scipy_optimize = None
    HAS_SCIPY = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except Exception:
    nx = None
    HAS_NETWORKX = False

try:
    import reliability as reliability_pkg
    HAS_RELIABILITY = True
except Exception:
    reliability_pkg = None
    HAS_RELIABILITY = False

try:
    import control as control_pkg
    HAS_CONTROL = True
except Exception:
    control_pkg = None
    HAS_CONTROL = False


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


def _binary_columns(df, inferred, max_cols=20):
    """Return columns that are binary (0/1) or can be thresholded to binary."""
    cols = []
    for col in _numeric_columns(df, inferred, max_cols=80):
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if vals.empty:
            continue
        uniq = set(np.unique(vals.to_numpy(dtype=float)).tolist())
        if uniq.issubset({0.0, 1.0}):
            cols.append(col)
    return cols[:max_cols]


def _threshold_to_binary(series, threshold=None):
    """Threshold a numeric series to binary (0/1) around its median."""
    vals = pd.to_numeric(series, errors="coerce")
    if threshold is None:
        threshold = float(np.nanmedian(vals.to_numpy(dtype=float)))
    return (vals >= threshold).astype(float)


def _build_nx_graph_from_edges(edges_df):
    """Build a networkx DiGraph from an edges dataframe with src/dst columns."""
    if not HAS_NETWORKX:
        return None
    G = nx.DiGraph()
    for _, row in edges_df.iterrows():
        G.add_edge(row["src"], row["dst"])
    return G


def _time_windows(ts, n_windows=5):
    """Split a sorted timestamp series into n equal-sized windows."""
    valid = ts.dropna()
    if len(valid) < n_windows * 5:
        return []
    indices = np.array_split(np.arange(len(valid)), n_windows)
    return [(valid.iloc[idx[0]], valid.iloc[idx[-1]], idx) for idx in indices if len(idx) > 0]


# ---------------------------------------------------------------------------
# Plugin handlers (25-47)
# ---------------------------------------------------------------------------


def _handler_ising_model_correlation_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Binary state interactions on DAG lattice. Compute magnetization and critical temperature analogue."""
    _log_start(ctx, plugin_id, df, config, inferred)
    if not HAS_NETWORKX:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "networkx_unavailable")
    edges, sc, dc = _build_edges(df, inferred)
    if edges.empty:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_graph_edges")
    bin_cols = _binary_columns(df, inferred, max_cols=10)
    num_cols = _numeric_columns(df, inferred, max_cols=10)
    if not bin_cols and not num_cols:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_usable_columns")
    # Build spin array: use binary columns directly or threshold numeric ones
    if bin_cols:
        spin_col = bin_cols[0]
        spins = pd.to_numeric(df[spin_col], errors="coerce").fillna(0).to_numpy(dtype=float)
        spins = 2.0 * spins - 1.0  # map 0/1 -> -1/+1
    else:
        spin_col = num_cols[0]
        spins = _threshold_to_binary(df[spin_col]).to_numpy(dtype=float)
        spins = 2.0 * spins - 1.0
    _ensure_budget(timer)
    # Build interaction graph
    G = _build_nx_graph_from_edges(edges)
    if G is None or G.number_of_edges() == 0:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "empty_interaction_graph")
    # Map spins to node indices
    idx_map = {str(i): i for i in range(len(df))}
    node_spins = {}
    for node in G.nodes():
        idx = idx_map.get(str(node))
        if idx is not None and idx < len(spins):
            node_spins[node] = spins[idx]
    # Compute energy H = -sum(J_ij * s_i * s_j) and magnetization
    energy = 0.0
    n_interactions = 0
    for u, v in G.edges():
        si = node_spins.get(u, 0.0)
        sv = node_spins.get(v, 0.0)
        energy -= si * sv
        n_interactions += 1
    _ensure_budget(timer)
    magnetization = float(np.mean(list(node_spins.values()))) if node_spins else 0.0
    abs_magnetization = abs(magnetization)
    energy_per_edge = energy / max(1, n_interactions)
    # Critical temperature analogue: T_c ~ |E| / (k * N * ln(N))
    n_nodes = max(1, len(node_spins))
    t_critical = abs(energy) / max(1e-9, n_nodes * max(1.0, math.log(n_nodes)))
    findings = []
    if abs_magnetization > 0.3:
        findings.append(_make_finding(
            plugin_id, "magnetization",
            "Strong binary alignment detected (Ising magnetization)",
            f"Mean magnetization |M|={abs_magnetization:.3f} indicates coordinated binary states across the interaction graph.",
            "High magnetization suggests systemic bias or correlated binary outcomes across connected entities.",
            {"metrics": {"magnetization": magnetization, "abs_magnetization": abs_magnetization,
                         "energy": energy, "energy_per_edge": energy_per_edge,
                         "n_interactions": n_interactions, "spin_column": spin_col}},
            recommendation="Investigate drivers of correlated binary outcomes across connected entities; consider decoupling mechanisms.",
            severity="warn" if abs_magnetization < 0.6 else "critical",
            confidence=min(0.9, 0.5 + abs_magnetization * 0.4),
        ))
    if energy_per_edge < -0.5:
        findings.append(_make_finding(
            plugin_id, "low_energy",
            "Ordered state detected (low Ising energy)",
            f"Energy per edge={energy_per_edge:.3f} indicates strong alignment between neighbors.",
            "Low energy states correspond to ordered, potentially rigid configurations resistant to change.",
            {"metrics": {"energy_per_edge": energy_per_edge, "t_critical": t_critical}},
            recommendation="Assess whether observed ordering is desirable or represents unhealthy lock-in.",
            severity="info",
            confidence=0.6,
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        "Computed Ising model correlation on interaction graph.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "magnetization": magnetization,
                       "energy": energy, "t_critical": t_critical, "n_interactions": n_interactions})


def _handler_percolation_fragility_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Build DAG. Randomly remove nodes and track giant component to find percolation threshold."""
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
    if G is None or G.number_of_nodes() < 10:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "graph_too_small")
    seed = int(config.get("seed", 42))
    rng = np.random.RandomState(seed)
    nodes = list(G.nodes())
    n = len(nodes)
    n_trials = int(config.get("percolation_trials", 20))
    fractions = np.linspace(0.0, 0.9, n_trials)
    giant_sizes = []
    for frac in fractions:
        _ensure_budget(timer)
        k = int(frac * n)
        remove = set(rng.choice(nodes, size=min(k, n - 1), replace=False).tolist())
        sub = G.subgraph([nd for nd in nodes if nd not in remove])
        if sub.number_of_nodes() == 0:
            giant_sizes.append(0.0)
            continue
        uG = sub.to_undirected()
        components = sorted(nx.connected_components(uG), key=len, reverse=True)
        giant = len(components[0]) / max(1, n) if components else 0.0
        giant_sizes.append(giant)
    # Find critical threshold: fraction where giant component drops below 0.5 of original
    p_c = 1.0
    for i, gs in enumerate(giant_sizes):
        if gs < 0.5:
            p_c = float(fractions[i])
            break
    fragility = 1.0 - p_c  # higher = more fragile
    findings = []
    if p_c < 0.3:
        findings.append(_make_finding(
            plugin_id, "percolation_fragile",
            "High percolation fragility detected",
            f"Giant component collapses at p_c={p_c:.2f} (only {p_c*100:.0f}% node removal needed).",
            "Low percolation threshold means the network is fragile to random failures.",
            {"metrics": {"p_c": p_c, "fragility": fragility, "nodes": n, "edges": G.number_of_edges(),
                         "giant_curve": [{"frac": float(f), "giant": float(g)} for f, g in zip(fractions.tolist(), giant_sizes)]}},
            recommendation="Add redundant paths and reduce single-point-of-failure nodes to improve network resilience.",
            severity="critical" if p_c < 0.15 else "warn",
            confidence=min(0.9, 0.55 + fragility * 0.3),
        ))
    elif p_c < 0.5:
        findings.append(_make_finding(
            plugin_id, "percolation_moderate",
            "Moderate percolation fragility",
            f"Giant component collapses at p_c={p_c:.2f}.",
            "Network shows moderate resilience to random node removal.",
            {"metrics": {"p_c": p_c, "fragility": fragility, "nodes": n}},
            recommendation="Monitor key hub nodes and consider targeted redundancy.",
            severity="warn", confidence=0.6,
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        "Computed percolation fragility analysis.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "p_c": p_c, "fragility": fragility, "nodes": n})


def _handler_renormalization_multiscale_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Coarse-grain DAG at multiple scales. Track parameter relevance across scales."""
    _log_start(ctx, plugin_id, df, config, inferred)
    num_cols = _variance_sorted_numeric(df, inferred, limit=8)
    if len(num_cols) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_numeric_columns")
    time_col, ts = _time_series(df, inferred)
    if ts is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_time_column")
    frame = df[num_cols].apply(pd.to_numeric, errors="coerce").copy()
    frame["_ts"] = ts
    frame = frame.sort_values("_ts").dropna(subset=["_ts"])
    if len(frame) < 30:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    scales = [1, 2, 4, 8, 16]
    relevance_by_scale: dict[int, dict[str, float]] = {}
    for scale in scales:
        _ensure_budget(timer)
        if scale > 1:
            n_groups = max(1, len(frame) // scale)
            groups = np.array_split(np.arange(len(frame)), n_groups)
            coarse = pd.DataFrame({
                col: [float(np.nanmean(frame[col].iloc[g].to_numpy(dtype=float))) for g in groups]
                for col in num_cols
            })
        else:
            coarse = frame[num_cols].copy()
        if len(coarse) < 5:
            continue
        variances = {}
        for col in num_cols:
            vals = coarse[col].to_numpy(dtype=float)
            variances[col] = float(np.nanvar(vals))
        total_var = sum(variances.values())
        if total_var > 0:
            relevance_by_scale[scale] = {col: v / total_var for col, v in variances.items()}
        else:
            relevance_by_scale[scale] = {col: 1.0 / len(num_cols) for col in num_cols}
    if not relevance_by_scale:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "coarsening_failed")
    # Identify relevant vs irrelevant: parameters whose share grows with scale are relevant
    scale_keys = sorted(relevance_by_scale.keys())
    relevant = []
    irrelevant = []
    for col in num_cols:
        shares = [relevance_by_scale[s].get(col, 0.0) for s in scale_keys if s in relevance_by_scale]
        if len(shares) >= 2:
            trend = shares[-1] - shares[0]
            if trend > 0.05:
                relevant.append({"column": col, "trend": trend, "shares": shares})
            elif trend < -0.05:
                irrelevant.append({"column": col, "trend": trend, "shares": shares})
    findings = []
    if relevant:
        findings.append(_make_finding(
            plugin_id, "relevant_params",
            "Scale-relevant parameters identified via renormalization",
            f"{len(relevant)} parameter(s) grow in relative importance at coarser scales.",
            "Parameters that remain relevant under coarse-graining drive macro-level behavior.",
            {"metrics": {"relevant": [r["column"] for r in relevant], "irrelevant": [r["column"] for r in irrelevant],
                         "top_relevant": relevant[0]}},
            recommendation=f"Focus monitoring and optimization on scale-relevant parameters: {', '.join(r['column'] for r in relevant[:3])}.",
            severity="warn" if len(relevant) <= 2 else "info",
            confidence=min(0.85, 0.5 + 0.1 * len(relevant)),
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        "Computed renormalization multiscale analysis.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "scales_computed": len(relevance_by_scale),
                       "n_relevant": len(relevant), "n_irrelevant": len(irrelevant)})


def _handler_partition_function_landscape_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Enumerate parameter configs weighted by Boltzmann factors. Map energy landscape."""
    _log_start(ctx, plugin_id, df, config, inferred)
    num_cols = _variance_sorted_numeric(df, inferred, limit=6)
    if len(num_cols) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_numeric_columns")
    mat = df[num_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(mat) < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    # Standardize
    arr = mat.to_numpy(dtype=float)
    for j in range(arr.shape[1]):
        center, scale = robust_center_scale(arr[:, j])
        if not math.isfinite(scale) or scale <= 0:
            scale = 1.0
        arr[:, j] = (arr[:, j] - center) / scale
    _ensure_budget(timer)
    # Define energy as squared distance from centroid
    centroid = np.mean(arr, axis=0)
    energies = np.sum((arr - centroid) ** 2, axis=1)
    # Boltzmann weights at various temperatures
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
    landscape = {}
    for T in temperatures:
        _ensure_budget(timer)
        boltzmann = np.exp(-energies / max(T, 1e-9))
        Z = float(np.sum(boltzmann))
        probs = boltzmann / max(Z, 1e-15)
        effective_states = float(np.exp(-np.sum(probs * np.log(np.maximum(probs, 1e-15)))))
        landscape[T] = {"Z": Z, "effective_states": effective_states, "entropy": float(math.log(max(1.0, effective_states)))}
    # Find local minima: rows with lowest energy
    sorted_idx = np.argsort(energies)
    n_minima = min(5, len(sorted_idx))
    minima_energies = energies[sorted_idx[:n_minima]].tolist()
    # Energy barriers: gap between consecutive sorted minima
    barriers = []
    for i in range(1, n_minima):
        barriers.append(float(minima_energies[i] - minima_energies[i - 1]))
    mean_barrier = float(np.mean(barriers)) if barriers else 0.0
    findings = []
    low_T = landscape.get(0.1, {})
    high_T = landscape.get(5.0, {})
    ratio = low_T.get("effective_states", 1.0) / max(1.0, high_T.get("effective_states", 1.0))
    if ratio < 0.3:
        findings.append(_make_finding(
            plugin_id, "landscape_funnel",
            "Energy landscape shows funneling (few dominant configurations)",
            f"At low temperature, only {low_T.get('effective_states', 0):.1f} effective states vs {high_T.get('effective_states', 0):.1f} at high T.",
            "Strong funneling suggests the system is dominated by a few parameter configurations.",
            {"metrics": {"low_T_states": low_T.get("effective_states"), "high_T_states": high_T.get("effective_states"),
                         "ratio": ratio, "mean_barrier": mean_barrier}},
            recommendation="Investigate dominant parameter configurations for rigidity or optimization opportunities.",
            severity="warn", confidence=0.65,
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        "Computed partition function energy landscape.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "n_minima": n_minima,
                       "mean_barrier": mean_barrier, "state_ratio": ratio})


def _handler_reynolds_number_analogue_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Re = (throughput x pipeline_depth) / resource_capacity. Classify laminar vs turbulent."""
    _log_start(ctx, plugin_id, df, config, inferred)
    time_col, ts = _time_series(df, inferred)
    num_cols = _numeric_columns(df, inferred, max_cols=20)
    if not num_cols:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_columns")
    dur_col = _duration_column(df, inferred)
    if dur_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_duration_column")
    durations = pd.to_numeric(df[dur_col], errors="coerce").dropna().to_numpy(dtype=float)
    if len(durations) < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_duration_data")
    _ensure_budget(timer)
    # Throughput proxy: rows per unit time or inverse mean duration
    throughput = 1.0 / max(float(np.nanmean(durations)), 1e-9)
    # Pipeline depth proxy: coefficient of variation of durations (variability = depth complexity)
    mean_dur = float(np.nanmean(durations))
    std_dur = float(np.nanstd(durations))
    pipeline_depth = std_dur / max(mean_dur, 1e-9)
    # Resource capacity proxy: inverse of p95 utilization
    p95 = float(np.nanpercentile(durations, 95))
    resource_cap = 1.0 / max(p95, 1e-9)
    Re = (throughput * max(pipeline_depth, 0.01)) / max(resource_cap, 1e-9)
    # Classify regime
    if Re < 2300:
        regime = "laminar"
    elif Re < 4000:
        regime = "transitional"
    else:
        regime = "turbulent"
    findings = []
    if regime == "turbulent":
        findings.append(_make_finding(
            plugin_id, "turbulent_flow",
            "Turbulent flow regime detected (high Reynolds analogue)",
            f"Re={Re:.1f} classifies as turbulent (>4000). High variability relative to capacity.",
            "Turbulent regimes indicate unpredictable throughput and potential congestion cascades.",
            {"metrics": {"Re": Re, "regime": regime, "throughput": throughput,
                         "pipeline_depth": pipeline_depth, "resource_capacity": resource_cap, "duration_col": dur_col}},
            recommendation="Reduce pipeline depth or increase resource capacity to move toward laminar flow.",
            severity="critical", confidence=0.7,
        ))
    elif regime == "transitional":
        findings.append(_make_finding(
            plugin_id, "transitional_flow",
            "Transitional flow regime (Reynolds analogue near critical)",
            f"Re={Re:.1f} is in transitional zone (2300-4000).",
            "System is near the laminar-turbulent boundary; small perturbations may cause instability.",
            {"metrics": {"Re": Re, "regime": regime}},
            recommendation="Monitor for regime transitions and consider buffering to stay in laminar zone.",
            severity="warn", confidence=0.6,
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Computed Reynolds number analogue: regime={regime}.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "Re": Re, "regime": regime})


def _handler_navier_stokes_backpressure_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Model job flow as fluid through DAG. Identify backpressure propagation points."""
    _log_start(ctx, plugin_id, df, config, inferred)
    if not HAS_NETWORKX:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "networkx_unavailable")
    edges, sc, dc = _build_edges(df, inferred)
    if edges.empty:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_graph_edges")
    dur_col = _duration_column(df, inferred)
    if dur_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_duration_column")
    G = _build_nx_graph_from_edges(edges)
    if G is None or G.number_of_nodes() < 5:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "graph_too_small")
    _ensure_budget(timer)
    # Compute mean duration per node (entity)
    node_dur = {}
    if sc in df.columns:
        for node, group in df.groupby(sc):
            vals = pd.to_numeric(group[dur_col], errors="coerce").dropna()
            if len(vals) > 0:
                node_dur[str(node)] = float(np.nanmean(vals.to_numpy(dtype=float)))
    # Backpressure: downstream congestion that propagates upstream
    # For each node, backpressure = sum of successor durations / own duration
    backpressure = {}
    for node in G.nodes():
        my_dur = node_dur.get(str(node), 0.0)
        if my_dur <= 0:
            continue
        successors = list(G.successors(node))
        succ_dur = sum(node_dur.get(str(s), 0.0) for s in successors)
        if successors:
            bp = succ_dur / (len(successors) * max(my_dur, 1e-9))
            backpressure[str(node)] = bp
    _ensure_budget(timer)
    if not backpressure:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_backpressure_data")
    ranked = sorted(backpressure.items(), key=lambda kv: -kv[1])
    top_bp = ranked[:10]
    max_bp = top_bp[0][1] if top_bp else 0.0
    findings = []
    if max_bp > 2.0:
        findings.append(_make_finding(
            plugin_id, "backpressure_hotspot",
            "Backpressure propagation hotspot detected",
            f"Top node has backpressure ratio={max_bp:.2f}x (downstream duration >> upstream).",
            "High backpressure nodes are congestion origins where downstream slowness propagates upstream.",
            {"metrics": {"top_backpressure": [{"node_hash": hashlib.sha256(n.encode()).hexdigest()[:12],
                                                "bp_ratio": float(bp)} for n, bp in top_bp[:5]],
                         "max_bp": max_bp}},
            recommendation="Add buffering or capacity at top backpressure nodes to prevent upstream congestion.",
            severity="warn" if max_bp < 5.0 else "critical",
            confidence=min(0.85, 0.5 + max_bp * 0.05),
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        "Computed Navier-Stokes backpressure analysis.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "max_bp": max_bp, "nodes_scored": len(backpressure)})


def _handler_vorticity_thrashing_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Detect circular flow patterns (retry storms, thrashing). Quantify recirculation rate."""
    _log_start(ctx, plugin_id, df, config, inferred)
    if not HAS_NETWORKX:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "networkx_unavailable")
    edges, sc, dc = _build_edges(df, inferred)
    if edges.empty:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_graph_edges")
    G = _build_nx_graph_from_edges(edges)
    if G is None or G.number_of_nodes() < 3:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "graph_too_small")
    _ensure_budget(timer)
    # Find cycles (limit to short cycles for performance)
    cycles = []
    try:
        for cycle in nx.simple_cycles(G):
            if len(cycle) <= 10:
                cycles.append(cycle)
            if len(cycles) >= 100:
                break
    except Exception:
        pass
    _ensure_budget(timer)
    # Count edges involved in cycles
    cycle_edges = set()
    for c in cycles:
        for i in range(len(c)):
            cycle_edges.add((str(c[i]), str(c[(i + 1) % len(c)])))
    total_edges = G.number_of_edges()
    recirculation_rate = len(cycle_edges) / max(1, total_edges)
    avg_cycle_len = float(np.mean([len(c) for c in cycles])) if cycles else 0.0
    findings = []
    if recirculation_rate > 0.1:
        findings.append(_make_finding(
            plugin_id, "vorticity_thrashing",
            "Circular flow / thrashing detected (vorticity)",
            f"Recirculation rate={recirculation_rate:.2%}; {len(cycles)} cycles found with avg length {avg_cycle_len:.1f}.",
            "Cycles in state transitions indicate retry storms or thrashing that waste resources.",
            {"metrics": {"recirculation_rate": recirculation_rate, "n_cycles": len(cycles),
                         "avg_cycle_length": avg_cycle_len, "cycle_edge_count": len(cycle_edges)}},
            recommendation="Break retry loops with exponential backoff, circuit breakers, or dead-letter mechanisms.",
            severity="warn" if recirculation_rate < 0.3 else "critical",
            confidence=min(0.88, 0.5 + recirculation_rate),
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        "Computed vorticity/thrashing analysis.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "n_cycles": len(cycles),
                       "recirculation_rate": recirculation_rate})


def _handler_entropy_production_rate_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """dH/dt of duration distribution across time windows. Rising = degradation warning."""
    _log_start(ctx, plugin_id, df, config, inferred)
    time_col, ts = _time_series(df, inferred)
    if ts is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_time_column")
    dur_col = _duration_column(df, inferred)
    if dur_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_duration_column")
    frame = df[[dur_col]].copy()
    frame["_ts"] = ts
    frame = frame.dropna(subset=["_ts", dur_col]).sort_values("_ts")
    if len(frame) < 50:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    durations = pd.to_numeric(frame[dur_col], errors="coerce").to_numpy(dtype=float)
    n_windows = int(config.get("n_windows", 5))
    windows = np.array_split(np.arange(len(durations)), n_windows)
    entropies = []
    for win_idx in windows:
        _ensure_budget(timer)
        vals = durations[win_idx]
        vals = vals[np.isfinite(vals)]
        if len(vals) < 5:
            entropies.append(0.0)
            continue
        # Compute Shannon entropy of histogram
        counts, _ = np.histogram(vals, bins=min(20, len(vals) // 3 + 1))
        probs = counts / max(float(np.sum(counts)), 1e-9)
        probs = probs[probs > 0]
        H = float(-np.sum(probs * np.log(probs)))
        entropies.append(H)
    # Compute dH/dt trend
    if len(entropies) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_windows")
    x = np.arange(len(entropies), dtype=float)
    y = np.array(entropies, dtype=float)
    slope = float(np.polyfit(x, y, 1)[0]) if len(x) >= 2 else 0.0
    findings = []
    if slope > 0.1:
        findings.append(_make_finding(
            plugin_id, "rising_entropy",
            "Rising entropy production rate (early degradation signal)",
            f"Duration distribution entropy increasing at slope={slope:.3f}/window.",
            "Rising entropy production indicates growing disorder, often an early degradation warning.",
            {"metrics": {"slope": slope, "entropies": entropies, "n_windows": n_windows, "duration_col": dur_col}},
            recommendation="Investigate root causes of increasing process variability before failure occurs.",
            severity="warn" if slope < 0.3 else "critical",
            confidence=min(0.85, 0.5 + slope),
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        "Computed entropy production rate.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "entropy_slope": slope, "n_windows": n_windows})


def _handler_carnot_efficiency_bound_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """eta = useful_work / total_effort. Compute theoretical efficiency ceiling."""
    _log_start(ctx, plugin_id, df, config, inferred)
    dur_col = _duration_column(df, inferred)
    if dur_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_duration_column")
    durations = pd.to_numeric(df[dur_col], errors="coerce").dropna().to_numpy(dtype=float)
    if len(durations) < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_data")
    _ensure_budget(timer)
    total_effort = float(np.nansum(durations))
    # Useful work proxy: effort in [p5, p95] range (non-extreme durations)
    p5 = float(np.nanpercentile(durations, 5))
    p95 = float(np.nanpercentile(durations, 95))
    useful_mask = (durations >= p5) & (durations <= p95)
    useful_work = float(np.nansum(durations[useful_mask]))
    # Waste = extremes (very fast = idle overhead, very slow = rework/wait)
    idle_waste = float(np.nansum(durations[durations < p5]))
    slow_waste = float(np.nansum(durations[durations > p95]))
    eta = useful_work / max(total_effort, 1e-9)
    # Carnot bound: theoretical maximum given temperature differential
    T_hot = float(np.nanmax(durations))
    T_cold = float(np.nanmin(durations[durations > 0])) if np.any(durations > 0) else 1.0
    carnot_bound = 1.0 - T_cold / max(T_hot, 1e-9) if T_hot > T_cold else 0.0
    findings = []
    if eta < 0.7:
        findings.append(_make_finding(
            plugin_id, "low_efficiency",
            "Low Carnot efficiency bound detected",
            f"Efficiency eta={eta:.2%} with Carnot bound={carnot_bound:.2%}. Waste ratio={(1-eta):.2%}.",
            "Low efficiency means significant effort goes to extremes (idle overhead or rework).",
            {"metrics": {"eta": eta, "carnot_bound": carnot_bound, "total_effort": total_effort,
                         "useful_work": useful_work, "idle_waste": idle_waste, "slow_waste": slow_waste,
                         "duration_col": dur_col}},
            recommendation="Reduce idle overhead and tail latency to improve system efficiency toward theoretical bound.",
            severity="warn" if eta > 0.5 else "critical",
            confidence=0.7,
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Computed Carnot efficiency analysis: eta={eta:.2%}.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "eta": eta, "carnot_bound": carnot_bound})


def _handler_exergy_waste_analysis_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Decompose total effort into useful compute vs waste (retries, idle, context-switch)."""
    _log_start(ctx, plugin_id, df, config, inferred)
    if not HAS_NETWORKX:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "networkx_unavailable")
    dur_col = _duration_column(df, inferred)
    num_cols = _numeric_columns(df, inferred, max_cols=20)
    if dur_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_duration_column")
    durations = pd.to_numeric(df[dur_col], errors="coerce").dropna().to_numpy(dtype=float)
    if len(durations) < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_data")
    _ensure_budget(timer)
    total = float(np.nansum(durations))
    median_dur = float(np.nanmedian(durations))
    # Classify waste buckets
    # Retry waste: items that appear multiple times (duplicate effort)
    edges, sc, dc = _build_edges(df, inferred)
    retry_waste = 0.0
    if not edges.empty:
        edge_counts = edges.groupby(["src", "dst"]).size().reset_index(name="count")
        retries = edge_counts[edge_counts["count"] > 1]
        retry_waste = float(retries["count"].sum() - len(retries)) * median_dur
    # Idle waste: durations below p10 (likely idle/overhead)
    p10 = float(np.nanpercentile(durations, 10))
    idle_mask = durations < p10
    idle_waste = float(np.nansum(durations[idle_mask]))
    # Context-switch overhead: rapid alternation between entities
    cat_cols = _categorical_columns(df, inferred, max_cols=3)
    switch_waste = 0.0
    if cat_cols:
        cat_vals = df[cat_cols[0]].astype(str).tolist()
        switches = sum(1 for i in range(1, len(cat_vals)) if cat_vals[i] != cat_vals[i - 1])
        switch_rate = switches / max(1, len(cat_vals) - 1)
        switch_waste = switch_rate * total * 0.05  # assume 5% overhead per switch
    useful = max(0.0, total - retry_waste - idle_waste - switch_waste)
    exergy_ratio = useful / max(total, 1e-9)
    findings = []
    if exergy_ratio < 0.75:
        waste_breakdown = {"retry_waste": retry_waste, "idle_waste": idle_waste,
                           "switch_waste": switch_waste, "useful": useful, "total": total}
        findings.append(_make_finding(
            plugin_id, "exergy_waste",
            "Significant exergy waste detected",
            f"Only {exergy_ratio:.1%} of total effort is useful compute. {(1-exergy_ratio):.1%} is waste.",
            "Waste categories: retries, idle overhead, and context-switching reduce effective throughput.",
            {"metrics": {**waste_breakdown, "exergy_ratio": exergy_ratio}},
            recommendation="Target the largest waste bucket for reduction: " +
                           ("retries" if retry_waste >= idle_waste and retry_waste >= switch_waste
                            else "idle overhead" if idle_waste >= switch_waste else "context switches") + ".",
            severity="warn" if exergy_ratio > 0.5 else "critical",
            confidence=0.65,
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        "Computed exergy waste decomposition.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "exergy_ratio": exergy_ratio})


def _handler_maxwell_demon_monitoring_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Information cost of scheduling decisions vs entropy reduction achieved."""
    _log_start(ctx, plugin_id, df, config, inferred)
    time_col, ts = _time_series(df, inferred)
    num_cols = _numeric_columns(df, inferred, max_cols=10)
    if not num_cols:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_columns")
    dur_col = _duration_column(df, inferred)
    if dur_col is None:
        dur_col = num_cols[0]
    durations = pd.to_numeric(df[dur_col], errors="coerce").dropna().to_numpy(dtype=float)
    if len(durations) < 30:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_data")
    _ensure_budget(timer)
    # Entropy of unsorted (natural order) durations
    counts_nat, _ = np.histogram(durations, bins=min(20, len(durations) // 5 + 1))
    p_nat = counts_nat / max(float(np.sum(counts_nat)), 1e-9)
    p_nat = p_nat[p_nat > 0]
    H_natural = float(-np.sum(p_nat * np.log(p_nat)))
    # Entropy of sorted (perfectly scheduled) durations - lower bound
    sorted_dur = np.sort(durations)
    half = len(sorted_dur) // 2
    H_parts = []
    for part in [sorted_dur[:half], sorted_dur[half:]]:
        if len(part) < 3:
            continue
        c, _ = np.histogram(part, bins=min(10, len(part) // 3 + 1))
        p = c / max(float(np.sum(c)), 1e-9)
        p = p[p > 0]
        H_parts.append(float(-np.sum(p * np.log(p))))
    H_sorted = float(np.mean(H_parts)) if H_parts else H_natural
    entropy_reduction = H_natural - H_sorted
    # Information cost proxy: number of categorical columns * log2(unique values)
    cat_cols = _categorical_columns(df, inferred, max_cols=5)
    info_cost = 0.0
    for col in cat_cols:
        nuniq = float(df[col].nunique())
        if nuniq > 1:
            info_cost += math.log2(nuniq)
    net_value = entropy_reduction - info_cost * 0.1  # scaled info cost
    findings = []
    if net_value < 0:
        findings.append(_make_finding(
            plugin_id, "demon_negative_value",
            "Monitoring overhead exceeds scheduling benefit (Maxwell demon)",
            f"Entropy reduction={entropy_reduction:.3f} but information cost proxy={info_cost:.2f}. Net value={net_value:.3f}.",
            "When monitoring/scheduling overhead exceeds the entropy reduction achieved, simplification may help.",
            {"metrics": {"H_natural": H_natural, "H_sorted": H_sorted, "entropy_reduction": entropy_reduction,
                         "info_cost": info_cost, "net_value": net_value}},
            recommendation="Simplify scheduling rules or reduce monitoring dimensions to improve net benefit.",
            severity="warn", confidence=0.55,
        ))
    elif entropy_reduction > 0.5:
        findings.append(_make_finding(
            plugin_id, "demon_positive_value",
            "Scheduling provides significant entropy reduction",
            f"Entropy reduction={entropy_reduction:.3f} with positive net value={net_value:.3f}.",
            "Active scheduling/monitoring is providing measurable disorder reduction.",
            {"metrics": {"entropy_reduction": entropy_reduction, "net_value": net_value}},
            recommendation="Maintain current scheduling approach; consider optimizing information channels.",
            severity="info", confidence=0.6,
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        "Computed Maxwell demon monitoring analysis.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "entropy_reduction": entropy_reduction, "net_value": net_value})


def _handler_periodic_schedule_fft_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """FFT on start timestamps to detect hidden periodicities."""
    _log_start(ctx, plugin_id, df, config, inferred)
    time_col, ts = _time_series(df, inferred)
    if ts is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_time_column")
    valid_ts = ts.dropna().sort_values()
    if len(valid_ts) < 30:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_timestamps")
    _ensure_budget(timer)
    # Convert to inter-arrival times in seconds
    deltas = np.diff(valid_ts.astype(np.int64) / 1e9)  # nanoseconds to seconds
    deltas = deltas[np.isfinite(deltas) & (deltas > 0)]
    if len(deltas) < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_inter_arrivals")
    # FFT on inter-arrival times
    n = len(deltas)
    fft_vals = np.fft.rfft(deltas - np.mean(deltas))
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(n, d=float(np.median(deltas)))
    _ensure_budget(timer)
    # Find dominant frequencies (exclude DC component)
    if len(power) > 1:
        power_no_dc = power[1:]
        freqs_no_dc = freqs[1:]
        if len(power_no_dc) > 0:
            top_idx = np.argsort(power_no_dc)[-5:][::-1]
            dominant = [{"freq": float(freqs_no_dc[i]), "power": float(power_no_dc[i]),
                         "period_s": 1.0 / max(float(freqs_no_dc[i]), 1e-15)}
                        for i in top_idx if freqs_no_dc[i] > 0]
        else:
            dominant = []
    else:
        dominant = []
    # Signal-to-noise: ratio of top frequency power to median power
    median_power = float(np.median(power[1:])) if len(power) > 1 else 1.0
    snr = float(dominant[0]["power"] / max(median_power, 1e-9)) if dominant else 0.0
    findings = []
    if dominant and snr > 5.0:
        period_hours = dominant[0]["period_s"] / 3600.0
        findings.append(_make_finding(
            plugin_id, "fft_periodicity",
            "Hidden periodicity detected via FFT",
            f"Dominant period={period_hours:.2f}h (SNR={snr:.1f}x above noise).",
            "Hidden periodicities can indicate batch schedules, external drivers, or cyclic resource contention.",
            {"metrics": {"dominant_periods": dominant[:3], "snr": snr, "n_arrivals": n}},
            recommendation="Investigate whether detected periodicity aligns with known schedules or reveals unknown cyclic behavior.",
            severity="warn" if snr < 10 else "critical",
            confidence=min(0.9, 0.5 + snr * 0.02),
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        "Computed periodic schedule FFT analysis.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "snr": snr, "n_dominant": len(dominant)})


def _handler_defect_density_classification_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Classify anomalies as point, line, or planar defects."""
    _log_start(ctx, plugin_id, df, config, inferred)
    num_cols = _variance_sorted_numeric(df, inferred, limit=6)
    if not num_cols:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_columns")
    dur_col = _duration_column(df, inferred)
    target_col = dur_col if dur_col else num_cols[0]
    vals = pd.to_numeric(df[target_col], errors="coerce").dropna().to_numpy(dtype=float)
    if len(vals) < 30:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_data")
    _ensure_budget(timer)
    # Detect anomalies via IQR
    q1, q3 = float(np.nanpercentile(vals, 25)), float(np.nanpercentile(vals, 75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    anomaly_mask = (vals < lower) | (vals > upper)
    anomaly_indices = np.where(anomaly_mask)[0]
    n_anomalies = len(anomaly_indices)
    if n_anomalies < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "too_few_anomalies")
    # Classify defect types by spatial clustering of anomaly indices
    gaps = np.diff(anomaly_indices)
    point_defects = 0  # isolated
    line_defects = 0   # sequential chain (gap == 1)
    planar_defects = 0 # correlated clusters (multiple consecutive)
    i = 0
    while i < len(anomaly_indices):
        chain_len = 1
        while i + chain_len < len(anomaly_indices) and anomaly_indices[i + chain_len] - anomaly_indices[i + chain_len - 1] == 1:
            chain_len += 1
        if chain_len == 1:
            point_defects += 1
        elif chain_len <= 3:
            line_defects += 1
        else:
            planar_defects += 1
        i += chain_len
    _ensure_budget(timer)
    total_defects = point_defects + line_defects + planar_defects
    defect_density = n_anomalies / max(1, len(vals))
    findings = []
    dominant_type = "point" if point_defects >= line_defects and point_defects >= planar_defects \
        else "line" if line_defects >= planar_defects else "planar"
    if defect_density > 0.05:
        findings.append(_make_finding(
            plugin_id, "defect_density",
            f"High defect density with {dominant_type}-type dominance",
            f"Defect density={defect_density:.2%}. Distribution: point={point_defects}, line={line_defects}, planar={planar_defects}.",
            "Defect type indicates failure pattern: point=random, line=sequential cascade, planar=systemic cluster.",
            {"metrics": {"defect_density": defect_density, "point_defects": point_defects,
                         "line_defects": line_defects, "planar_defects": planar_defects,
                         "n_anomalies": n_anomalies, "dominant_type": dominant_type, "column": target_col}},
            recommendation="Point: improve individual item quality. Line: break sequential failure chains. Planar: address systemic root causes.",
            severity="warn" if defect_density < 0.15 else "critical",
            confidence=min(0.85, 0.5 + defect_density * 2),
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        "Computed defect density classification.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "defect_density": defect_density,
                       "dominant_type": dominant_type, "total_defects": total_defects})


def _handler_stress_strain_capacity_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Load-latency curve. Identify elastic limit, yield point, and fracture point."""
    _log_start(ctx, plugin_id, df, config, inferred)
    time_col, ts = _time_series(df, inferred)
    dur_col = _duration_column(df, inferred)
    if dur_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_duration_column")
    num_cols = _numeric_columns(df, inferred, max_cols=20)
    # Find a load/count proxy column
    load_col = None
    for col in num_cols:
        if any(h in col.lower() for h in ("count", "load", "queue", "volume", "rate", "throughput", "concurrency")):
            load_col = col
            break
    if load_col is None and ts is not None:
        # Use arrival rate as load proxy
        frame = df[[dur_col]].copy()
        frame["_ts"] = ts
        frame = frame.dropna(subset=["_ts"]).sort_values("_ts")
        if len(frame) < 30:
            return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
        # Bin by time and compute arrival rate
        n_bins = min(50, len(frame) // 5)
        bins = np.array_split(np.arange(len(frame)), max(1, n_bins))
        load_vals = np.array([len(b) for b in bins], dtype=float)
        lat_vals = np.array([float(np.nanmean(pd.to_numeric(frame[dur_col].iloc[b], errors="coerce").to_numpy(dtype=float))) for b in bins])
    else:
        if load_col is None:
            return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_load_proxy")
        load_vals = pd.to_numeric(df[load_col], errors="coerce").dropna().to_numpy(dtype=float)
        lat_vals = pd.to_numeric(df[dur_col], errors="coerce").dropna().to_numpy(dtype=float)
        min_len = min(len(load_vals), len(lat_vals))
        load_vals, lat_vals = load_vals[:min_len], lat_vals[:min_len]
    if len(load_vals) < 10 or len(lat_vals) < 10:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_data")
    _ensure_budget(timer)
    # Sort by load
    sort_idx = np.argsort(load_vals)
    load_sorted = load_vals[sort_idx]
    lat_sorted = lat_vals[sort_idx]
    # Fit piecewise: find elastic limit (end of linear region) and yield point
    n = len(load_sorted)
    best_break = n // 2
    best_mse = float("inf")
    for bp in range(n // 5, 4 * n // 5):
        x1, y1 = load_sorted[:bp], lat_sorted[:bp]
        x2, y2 = load_sorted[bp:], lat_sorted[bp:]
        if len(x1) < 3 or len(x2) < 3:
            continue
        p1 = np.polyfit(x1, y1, 1)
        p2 = np.polyfit(x2, y2, 1)
        r1 = y1 - np.polyval(p1, x1)
        r2 = y2 - np.polyval(p2, x2)
        mse = float(np.mean(r1 ** 2) + np.mean(r2 ** 2))
        if mse < best_mse:
            best_mse = mse
            best_break = bp
    _ensure_budget(timer)
    elastic_limit_load = float(load_sorted[best_break])
    # Fracture point: where latency exceeds 3x median
    median_lat = float(np.nanmedian(lat_sorted))
    fracture_idx = np.where(lat_sorted > 3 * median_lat)[0]
    fracture_load = float(load_sorted[fracture_idx[0]]) if len(fracture_idx) > 0 else float(load_sorted[-1])
    # Slopes
    pre_slope = float(np.polyfit(load_sorted[:best_break], lat_sorted[:best_break], 1)[0]) if best_break > 2 else 0.0
    post_slope = float(np.polyfit(load_sorted[best_break:], lat_sorted[best_break:], 1)[0]) if n - best_break > 2 else 0.0
    nonlinearity_ratio = post_slope / max(abs(pre_slope), 1e-9)
    findings = []
    if nonlinearity_ratio > 2.0:
        findings.append(_make_finding(
            plugin_id, "yield_point",
            "Stress-strain yield point detected in load-latency curve",
            f"Nonlinearity ratio={nonlinearity_ratio:.1f}x at elastic limit load={elastic_limit_load:.1f}.",
            "Beyond the yield point, latency increases nonlinearly with load, indicating capacity saturation.",
            {"metrics": {"elastic_limit_load": elastic_limit_load, "fracture_load": fracture_load,
                         "pre_slope": pre_slope, "post_slope": post_slope, "nonlinearity_ratio": nonlinearity_ratio,
                         "duration_col": dur_col}},
            recommendation=f"Keep load below elastic limit ({elastic_limit_load:.1f}) or add capacity to shift the yield point.",
            severity="warn" if nonlinearity_ratio < 5.0 else "critical",
            confidence=min(0.85, 0.5 + nonlinearity_ratio * 0.05),
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        "Computed stress-strain capacity analysis.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "elastic_limit": elastic_limit_load,
                       "fracture_load": fracture_load, "nonlinearity_ratio": nonlinearity_ratio})


def _handler_creep_fatigue_prediction_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Norton creep law + S-N fatigue curve. Predict remaining useful life under sustained load."""
    _log_start(ctx, plugin_id, df, config, inferred)
    time_col, ts = _time_series(df, inferred)
    if ts is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_time_column")
    dur_col = _duration_column(df, inferred)
    if dur_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_duration_column")
    frame = df[[dur_col]].copy()
    frame["_ts"] = ts
    frame = frame.dropna(subset=["_ts", dur_col]).sort_values("_ts")
    if len(frame) < 40:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_data")
    durations = pd.to_numeric(frame[dur_col], errors="coerce").to_numpy(dtype=float)
    _ensure_budget(timer)
    # Norton's creep: strain_rate = A * sigma^n
    # Proxy: compute rolling mean of durations (creep = gradual increase under sustained load)
    window = max(5, len(durations) // 10)
    rolling_mean = np.convolve(durations, np.ones(window) / window, mode="valid")
    if len(rolling_mean) < 10:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rolling_data")
    # Creep rate: slope of rolling mean over time
    x = np.arange(len(rolling_mean), dtype=float)
    creep_slope = float(np.polyfit(x, rolling_mean, 1)[0])
    # S-N fatigue: estimate cycles to failure
    # Amplitude of duration oscillations
    amplitude = float(np.nanstd(durations))
    mean_dur = float(np.nanmean(durations))
    stress_ratio = amplitude / max(mean_dur, 1e-9)
    # Basquin law: N_f = C * (stress_amplitude)^(-b), use b~3 as typical
    b = 3.0
    C = 1e6  # normalization constant
    if stress_ratio > 0.01:
        cycles_to_failure = C * (stress_ratio ** (-b))
    else:
        cycles_to_failure = float("inf")
    current_cycles = float(len(durations))
    remaining_life = max(0.0, cycles_to_failure - current_cycles) if math.isfinite(cycles_to_failure) else float("inf")
    life_fraction = current_cycles / max(cycles_to_failure, 1e-9) if math.isfinite(cycles_to_failure) else 0.0
    _ensure_budget(timer)
    findings = []
    if creep_slope > 0 and life_fraction > 0.5:
        findings.append(_make_finding(
            plugin_id, "creep_fatigue",
            "Creep-fatigue degradation: approaching predicted end of useful life",
            f"Creep slope={creep_slope:.4f}/cycle, life fraction consumed={life_fraction:.1%}.",
            "Combined creep (gradual increase) and fatigue (cyclic stress) predict remaining useful life.",
            {"metrics": {"creep_slope": creep_slope, "stress_ratio": stress_ratio,
                         "cycles_to_failure": _safe_float(cycles_to_failure, 1e9),
                         "remaining_life_cycles": _safe_float(remaining_life, 1e9),
                         "life_fraction": life_fraction, "duration_col": dur_col}},
            recommendation="Plan capacity refresh or load reduction before estimated end of useful life.",
            severity="warn" if life_fraction < 0.8 else "critical",
            confidence=min(0.8, 0.4 + life_fraction * 0.4),
        ))
    elif creep_slope > 0.001:
        findings.append(_make_finding(
            plugin_id, "creep_detected",
            "Creep trend detected in durations",
            f"Duration creep slope={creep_slope:.4f}/cycle indicates gradual degradation.",
            "Positive creep slope under sustained load is an early warning of capacity erosion.",
            {"metrics": {"creep_slope": creep_slope, "life_fraction": life_fraction}},
            recommendation="Monitor creep trend and plan proactive maintenance.",
            severity="info", confidence=0.55,
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        "Computed creep-fatigue prediction.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "creep_slope": creep_slope, "life_fraction": life_fraction})


def _handler_bathtub_curve_lifecycle_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Three-phase hazard: infant mortality, useful life, wearout. Classify current phase."""
    _log_start(ctx, plugin_id, df, config, inferred)
    time_col, ts = _time_series(df, inferred)
    if ts is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_time_column")
    dur_col = _duration_column(df, inferred)
    if dur_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_duration_column")
    frame = df[[dur_col]].copy()
    frame["_ts"] = ts
    frame = frame.dropna(subset=["_ts", dur_col]).sort_values("_ts")
    if len(frame) < 30:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_data")
    durations = pd.to_numeric(frame[dur_col], errors="coerce").to_numpy(dtype=float)
    _ensure_budget(timer)
    # Split into thirds for three-phase analysis
    n = len(durations)
    third = max(1, n // 3)
    early = durations[:third]
    middle = durations[third:2 * third]
    late = durations[2 * third:]
    # Compute failure rates (proxy: fraction above p90 threshold)
    threshold = float(np.nanpercentile(durations, 90))
    rate_early = float(np.mean(early > threshold))
    rate_middle = float(np.mean(middle > threshold))
    rate_late = float(np.mean(late > threshold))
    # Classify phase
    if rate_early > rate_middle and rate_late > rate_middle:
        shape = "bathtub"
        if rate_late > rate_early:
            current_phase = "wearout"
        elif rate_early > rate_late:
            current_phase = "infant_mortality"
        else:
            current_phase = "useful_life"
    elif rate_early > rate_middle:
        shape = "decreasing_hazard"
        current_phase = "infant_mortality"
    elif rate_late > rate_middle:
        shape = "increasing_hazard"
        current_phase = "wearout"
    else:
        shape = "constant_hazard"
        current_phase = "useful_life"
    findings = []
    if current_phase == "wearout":
        findings.append(_make_finding(
            plugin_id, "wearout_phase",
            "System in wearout phase (bathtub curve)",
            f"Hazard rate pattern: early={rate_early:.2%}, middle={rate_middle:.2%}, late={rate_late:.2%}. Shape={shape}.",
            "Increasing failure rate in the late phase indicates aging/wearout requiring proactive intervention.",
            {"metrics": {"rate_early": rate_early, "rate_middle": rate_middle, "rate_late": rate_late,
                         "shape": shape, "current_phase": current_phase, "duration_col": dur_col}},
            recommendation="Plan replacement, refactoring, or capacity refresh to address wearout degradation.",
            severity="critical" if rate_late > 0.3 else "warn",
            confidence=0.7,
        ))
    elif current_phase == "infant_mortality":
        findings.append(_make_finding(
            plugin_id, "infant_mortality",
            "System in infant mortality phase (bathtub curve)",
            f"Hazard rate: early={rate_early:.2%} > middle={rate_middle:.2%}. Decreasing hazard detected.",
            "High early failure rate suggests deployment issues, configuration errors, or burn-in needs.",
            {"metrics": {"rate_early": rate_early, "rate_middle": rate_middle, "rate_late": rate_late,
                         "shape": shape, "current_phase": current_phase}},
            recommendation="Improve deployment validation and burn-in procedures to reduce early failures.",
            severity="warn", confidence=0.65,
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Computed bathtub curve lifecycle analysis: phase={current_phase}.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "current_phase": current_phase, "shape": shape})


def _handler_fmea_rpn_scoring_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Risk Priority Number = Severity x Occurrence x Detection per failure mode."""
    _log_start(ctx, plugin_id, df, config, inferred)
    dur_col = _duration_column(df, inferred)
    cat_cols = _categorical_columns(df, inferred, max_cols=5)
    num_cols = _numeric_columns(df, inferred, max_cols=10)
    if dur_col is None or not cat_cols:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "missing_required_columns")
    durations = pd.to_numeric(df[dur_col], errors="coerce")
    if durations.notna().sum() < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_duration_data")
    _ensure_budget(timer)
    # Define failure: duration above p90
    threshold = float(np.nanpercentile(durations.dropna().to_numpy(dtype=float), 90))
    failure_mask = durations > threshold
    mode_col = cat_cols[0]
    modes = df[mode_col].astype(str)
    rpn_rows = []
    for mode_val, group in df.groupby(modes):
        if len(group) < 3:
            continue
        mode_durs = pd.to_numeric(group[dur_col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(mode_durs) < 2:
            continue
        # Severity: normalized mean duration for failure cases
        mode_fail = mode_durs[mode_durs > threshold]
        severity_score = min(10, max(1, int(float(np.nanmean(mode_fail)) / max(threshold, 1e-9) * 5))) if len(mode_fail) > 0 else 1
        # Occurrence: failure frequency
        occurrence_rate = float(len(mode_fail)) / max(1, len(mode_durs))
        occurrence_score = min(10, max(1, int(occurrence_rate * 10)))
        # Detection: inverse of detectability (high variance = hard to detect)
        cv = float(np.nanstd(mode_durs) / max(float(np.nanmean(mode_durs)), 1e-9))
        detection_score = min(10, max(1, int(cv * 5)))
        rpn = severity_score * occurrence_score * detection_score
        rpn_rows.append({"mode": str(mode_val), "severity": severity_score, "occurrence": occurrence_score,
                         "detection": detection_score, "rpn": rpn, "count": len(mode_durs)})
        _ensure_budget(timer)
    if not rpn_rows:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_failure_modes")
    rpn_rows.sort(key=lambda r: (-r["rpn"], r["mode"]))
    findings = []
    top = rpn_rows[0]
    if top["rpn"] > 100:
        findings.append(_make_finding(
            plugin_id, f"rpn:{top['mode']}",
            f"High RPN failure mode: {top['mode']}",
            f"RPN={top['rpn']} (S={top['severity']} x O={top['occurrence']} x D={top['detection']}).",
            "High Risk Priority Number indicates a failure mode needing priority corrective action.",
            {"metrics": {"top_modes": rpn_rows[:5]}},
            recommendation=f"Address failure mode '{top['mode']}': reduce severity, occurrence, or improve detection.",
            severity="warn" if top["rpn"] < 300 else "critical",
            confidence=min(0.85, 0.5 + top["rpn"] / 1000),
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        "Computed FMEA RPN scoring.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "n_modes": len(rpn_rows), "max_rpn": top["rpn"]})


def _handler_fault_tree_minimal_cuts_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Build fault tree from DAG. Find minimal cut sets."""
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
    # Find root nodes (no predecessors) and leaf nodes (no successors)
    roots = [n for n in G.nodes() if G.in_degree(n) == 0]
    leaves = [n for n in G.nodes() if G.out_degree(n) == 0]
    if not roots or not leaves:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_root_or_leaf_nodes")
    # Find minimal cut sets: minimal node sets whose removal disconnects all roots from all leaves
    # Use min-cut approximation: for each root-leaf pair, find minimum vertex cut
    cut_sets = []
    pairs_checked = 0
    for root in roots[:5]:
        for leaf in leaves[:5]:
            _ensure_budget(timer)
            if root == leaf:
                continue
            try:
                uG = G.to_undirected()
                if nx.has_path(uG, root, leaf):
                    cut = nx.minimum_node_cut(uG, root, leaf)
                    if cut:
                        cut_sets.append({"root": str(root), "leaf": str(leaf),
                                         "cut_size": len(cut),
                                         "cut_nodes": [hashlib.sha256(str(n).encode()).hexdigest()[:12] for n in list(cut)[:10]]})
            except Exception:
                pass
            pairs_checked += 1
            if pairs_checked >= 20:
                break
        if pairs_checked >= 20:
            break
    if not cut_sets:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_cut_sets_found")
    cut_sets.sort(key=lambda c: (c["cut_size"], c["root"]))
    min_cut_size = cut_sets[0]["cut_size"]
    findings = []
    if min_cut_size <= 2:
        findings.append(_make_finding(
            plugin_id, "minimal_cut",
            "Small minimal cut set detected (single/double point of failure)",
            f"Minimum cut set size={min_cut_size}. Removing {min_cut_size} node(s) disconnects root from leaf.",
            "Small cut sets indicate critical single points of failure in the dependency graph.",
            {"metrics": {"min_cut_size": min_cut_size, "cut_sets": cut_sets[:5], "nodes": G.number_of_nodes()}},
            recommendation="Add redundancy at minimal cut set nodes to prevent single-point failures.",
            severity="critical" if min_cut_size == 1 else "warn",
            confidence=0.75,
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        "Computed fault tree minimal cut sets.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "min_cut_size": min_cut_size, "n_cut_sets": len(cut_sets)})


def _handler_reliability_block_diagram_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Compute series/parallel system reliability from component reliabilities."""
    _log_start(ctx, plugin_id, df, config, inferred)
    if not HAS_NETWORKX:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "networkx_unavailable")
    edges, sc, dc = _build_edges(df, inferred)
    if edges.empty:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_graph_edges")
    dur_col = _duration_column(df, inferred)
    G = _build_nx_graph_from_edges(edges)
    if G is None or G.number_of_nodes() < 3:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "graph_too_small")
    _ensure_budget(timer)
    # Compute per-node reliability: fraction of non-anomalous durations
    node_reliability = {}
    if dur_col and sc in df.columns:
        overall_p90 = float(np.nanpercentile(pd.to_numeric(df[dur_col], errors="coerce").dropna().to_numpy(dtype=float), 90))
        for node, group in df.groupby(sc):
            vals = pd.to_numeric(group[dur_col], errors="coerce").dropna().to_numpy(dtype=float)
            if len(vals) >= 3:
                r = float(np.mean(vals <= overall_p90))
                node_reliability[str(node)] = max(0.01, min(0.999, r))
    if not node_reliability:
        # Assign uniform reliability
        for node in G.nodes():
            node_reliability[str(node)] = 0.95
    # Identify series paths (sequential chains) and parallel groups
    roots = [n for n in G.nodes() if G.in_degree(n) == 0]
    leaves = [n for n in G.nodes() if G.out_degree(n) == 0]
    if not roots:
        roots = [list(G.nodes())[0]]
    if not leaves:
        leaves = [list(G.nodes())[-1]]
    # Series reliability: product along longest path
    series_paths = []
    for root in roots[:3]:
        for leaf in leaves[:3]:
            try:
                for path in list(nx.all_simple_paths(G, root, leaf))[:10]:
                    r_series = 1.0
                    for node in path:
                        r_series *= node_reliability.get(str(node), 0.95)
                    series_paths.append({"path_len": len(path), "reliability": r_series})
            except Exception:
                pass
    _ensure_budget(timer)
    # Parallel reliability: 1 - product(1 - R_i) for parallel branches
    all_reliabilities = list(node_reliability.values())
    r_parallel = 1.0 - float(np.prod([1.0 - r for r in all_reliabilities[:20]]))
    min_series = min((p["reliability"] for p in series_paths), default=0.95)
    system_reliability = min_series  # conservative: weakest series path
    findings = []
    if system_reliability < 0.9:
        findings.append(_make_finding(
            plugin_id, "low_system_reliability",
            "Low system reliability detected via block diagram",
            f"System reliability={system_reliability:.3f} (weakest series path). Parallel bound={r_parallel:.3f}.",
            "Low series reliability means the weakest component chain dominates system failure probability.",
            {"metrics": {"system_reliability": system_reliability, "parallel_reliability": r_parallel,
                         "n_series_paths": len(series_paths), "weakest_path": series_paths[0] if series_paths else {}}},
            recommendation="Add parallel redundancy to the weakest series path or improve component reliability.",
            severity="warn" if system_reliability > 0.7 else "critical",
            confidence=0.7,
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        "Computed reliability block diagram.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "system_reliability": system_reliability,
                       "parallel_reliability": r_parallel})


def _handler_pid_controller_analogue_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """P/I/D decomposition of error signal (target - actual)."""
    _log_start(ctx, plugin_id, df, config, inferred)
    time_col, ts = _time_series(df, inferred)
    if ts is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_time_column")
    dur_col = _duration_column(df, inferred)
    if dur_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_duration_column")
    frame = df[[dur_col]].copy()
    frame["_ts"] = ts
    frame = frame.dropna(subset=["_ts", dur_col]).sort_values("_ts")
    if len(frame) < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_data")
    durations = pd.to_numeric(frame[dur_col], errors="coerce").to_numpy(dtype=float)
    _ensure_budget(timer)
    # Target = median duration (setpoint)
    target = float(np.nanmedian(durations))
    error = durations - target
    # Proportional component: current error
    P = error
    # Integral component: cumulative error
    I = np.cumsum(error)
    # Derivative component: rate of change of error
    D = np.zeros_like(error)
    D[1:] = np.diff(error)
    # Compute component energies (variance contribution)
    total_var = float(np.nanvar(error))
    p_energy = float(np.nanvar(P)) / max(total_var, 1e-9)
    i_energy = float(np.nanvar(I)) / max(total_var, 1e-9)
    d_energy = float(np.nanvar(D)) / max(total_var, 1e-9)
    # Dominant component
    dominant = "P" if p_energy >= i_energy and p_energy >= d_energy else "I" if i_energy >= d_energy else "D"
    # Integral windup detection
    i_final = float(I[-1])
    i_max = float(np.nanmax(np.abs(I)))
    windup = abs(i_final) / max(i_max, 1e-9) > 0.7 and i_max > target * len(durations) * 0.1
    findings = []
    if windup:
        findings.append(_make_finding(
            plugin_id, "integral_windup",
            "Integral windup detected (PID analogue)",
            f"Cumulative error grows monotonically: I_final={i_final:.1f}, I_max={i_max:.1f}. Dominant={dominant}.",
            "Integral windup indicates persistent one-directional error accumulation without correction.",
            {"metrics": {"P_energy": p_energy, "I_energy": i_energy, "D_energy": d_energy,
                         "dominant": dominant, "i_final": i_final, "windup": True, "duration_col": dur_col}},
            recommendation="Implement anti-windup: add feedback limits, reset integral term, or adjust setpoint.",
            severity="warn", confidence=0.65,
        ))
    elif d_energy > 0.5:
        findings.append(_make_finding(
            plugin_id, "derivative_dominated",
            "Derivative-dominated error signal (high jitter)",
            f"D-component accounts for {d_energy:.1%} of error variance. System is jittery.",
            "Derivative dominance means rapid fluctuations outweigh steady-state or cumulative error.",
            {"metrics": {"P_energy": p_energy, "I_energy": i_energy, "D_energy": d_energy, "dominant": dominant}},
            recommendation="Apply smoothing or damping to reduce high-frequency error oscillations.",
            severity="warn", confidence=0.6,
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Computed PID controller analogue: dominant={dominant}.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "dominant": dominant,
                       "P_energy": p_energy, "I_energy": i_energy, "D_energy": d_energy})


def _handler_lyapunov_stability_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Estimate Lyapunov exponent from duration time series. Positive = unstable divergence."""
    _log_start(ctx, plugin_id, df, config, inferred)
    time_col, ts = _time_series(df, inferred)
    if ts is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_time_column")
    dur_col = _duration_column(df, inferred)
    if dur_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_duration_column")
    frame = df[[dur_col]].copy()
    frame["_ts"] = ts
    frame = frame.dropna(subset=["_ts", dur_col]).sort_values("_ts")
    if len(frame) < 50:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_data")
    vals = pd.to_numeric(frame[dur_col], errors="coerce").to_numpy(dtype=float)
    _ensure_budget(timer)
    # Estimate largest Lyapunov exponent via Rosenstein's method (simplified)
    n = len(vals)
    embedding_dim = 3
    tau = 1  # delay
    # Build delay vectors
    m = n - (embedding_dim - 1) * tau
    if m < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_embedding_length")
    vectors = np.array([vals[i:i + embedding_dim * tau:tau] for i in range(m)])
    # For each point, find nearest neighbor (excluding temporal neighbors)
    min_sep = max(1, embedding_dim * tau)
    divergences = []
    max_check = min(m, 200)  # limit for performance
    for i in range(0, m, max(1, m // max_check)):
        _ensure_budget(timer)
        best_dist = float("inf")
        best_j = -1
        for j in range(m):
            if abs(i - j) < min_sep:
                continue
            d = float(np.linalg.norm(vectors[i] - vectors[j]))
            if 0 < d < best_dist:
                best_dist = d
                best_j = j
        if best_j >= 0 and best_dist > 0:
            # Track divergence over time
            steps = min(10, m - max(i, best_j) - 1)
            for k in range(1, steps + 1):
                if i + k < m and best_j + k < m:
                    d_k = float(np.linalg.norm(vectors[i + k] - vectors[best_j + k]))
                    if d_k > 0:
                        divergences.append((k, math.log(d_k / max(best_dist, 1e-15))))
    if not divergences:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "divergence_computation_failed")
    # Estimate Lyapunov exponent as slope of log(divergence) vs time step
    steps_arr = np.array([d[0] for d in divergences], dtype=float)
    divs_arr = np.array([d[1] for d in divergences], dtype=float)
    lyapunov = float(np.polyfit(steps_arr, divs_arr, 1)[0]) if len(steps_arr) >= 2 else 0.0
    findings = []
    if lyapunov > 0.05:
        findings.append(_make_finding(
            plugin_id, "positive_lyapunov",
            "Positive Lyapunov exponent: chaotic/unstable dynamics",
            f"Largest Lyapunov exponent={lyapunov:.4f}. Nearby trajectories diverge exponentially.",
            "Positive Lyapunov exponent indicates sensitive dependence on initial conditions (chaos).",
            {"metrics": {"lyapunov_exponent": lyapunov, "embedding_dim": embedding_dim,
                         "n_divergence_pairs": len(divergences), "duration_col": dur_col}},
            recommendation="Add damping, feedback controls, or reduce system coupling to stabilize dynamics.",
            severity="warn" if lyapunov < 0.2 else "critical",
            confidence=min(0.85, 0.5 + lyapunov),
        ))
    elif lyapunov < -0.05:
        findings.append(_make_finding(
            plugin_id, "stable_attractor",
            "Negative Lyapunov exponent: stable attractor dynamics",
            f"Lyapunov exponent={lyapunov:.4f}. System converges to stable behavior.",
            "Negative exponent indicates self-correcting dynamics that return to equilibrium.",
            {"metrics": {"lyapunov_exponent": lyapunov}},
            recommendation="Current dynamics are stable; monitor for sign changes indicating destabilization.",
            severity="info", confidence=0.6,
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Computed Lyapunov stability: exponent={lyapunov:.4f}.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "lyapunov_exponent": lyapunov})


def _handler_bode_frequency_response_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Compute transfer function from arrival rate to duration via cross-spectral density."""
    _log_start(ctx, plugin_id, df, config, inferred)
    time_col, ts = _time_series(df, inferred)
    if ts is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_time_column")
    dur_col = _duration_column(df, inferred)
    if dur_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_duration_column")
    frame = df[[dur_col]].copy()
    frame["_ts"] = ts
    frame = frame.dropna(subset=["_ts", dur_col]).sort_values("_ts")
    if len(frame) < 40:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_data")
    _ensure_budget(timer)
    # Create arrival rate signal: bin by time and count arrivals
    n_bins = min(64, len(frame) // 3)
    bins = np.array_split(np.arange(len(frame)), max(1, n_bins))
    arrival_rate = np.array([len(b) for b in bins], dtype=float)
    mean_duration = np.array([
        float(np.nanmean(pd.to_numeric(frame[dur_col].iloc[b], errors="coerce").to_numpy(dtype=float)))
        for b in bins
    ])
    # Remove mean
    arrival_rate = arrival_rate - np.mean(arrival_rate)
    mean_duration = mean_duration - np.nanmean(mean_duration)
    mean_duration = np.nan_to_num(mean_duration, nan=0.0)
    if len(arrival_rate) < 8:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_bins")
    # Cross-spectral density
    fft_input = np.fft.rfft(arrival_rate)
    fft_output = np.fft.rfft(mean_duration)
    cross_spectrum = fft_output * np.conj(fft_input)
    auto_spectrum = fft_input * np.conj(fft_input)
    # Transfer function H(f) = cross / auto
    H = cross_spectrum / np.maximum(np.abs(auto_spectrum), 1e-15)
    magnitude = np.abs(H)
    phase = np.angle(H)
    freqs = np.fft.rfftfreq(len(arrival_rate))
    _ensure_budget(timer)
    # Find resonant frequency (peak magnitude, excluding DC)
    if len(magnitude) > 1:
        mag_no_dc = magnitude[1:]
        freq_no_dc = freqs[1:]
        peak_idx = int(np.argmax(mag_no_dc))
        resonant_freq = float(freq_no_dc[peak_idx])
        peak_gain = float(mag_no_dc[peak_idx])
        median_gain = float(np.median(mag_no_dc))
    else:
        resonant_freq = 0.0
        peak_gain = 0.0
        median_gain = 0.0
    gain_ratio = peak_gain / max(median_gain, 1e-9)
    findings = []
    if gain_ratio > 3.0:
        findings.append(_make_finding(
            plugin_id, "resonance",
            "Resonant frequency detected in arrival-to-duration transfer function",
            f"Peak gain={peak_gain:.2f} at freq={resonant_freq:.4f} ({gain_ratio:.1f}x above median).",
            "Resonance means specific arrival rate frequencies amplify duration, causing oscillatory congestion.",
            {"metrics": {"resonant_freq": resonant_freq, "peak_gain": peak_gain, "gain_ratio": gain_ratio,
                         "median_gain": median_gain, "n_bins": n_bins}},
            recommendation="Avoid periodic load patterns near the resonant frequency; add damping/buffering.",
            severity="warn" if gain_ratio < 6.0 else "critical",
            confidence=min(0.8, 0.5 + gain_ratio * 0.03),
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        "Computed Bode frequency response analysis.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "resonant_freq": resonant_freq, "gain_ratio": gain_ratio})


def _handler_controllability_gramian_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Compute controllability matrix rank. Identify dark states not observable from logs."""
    _log_start(ctx, plugin_id, df, config, inferred)
    time_col, ts = _time_series(df, inferred)
    num_cols = _variance_sorted_numeric(df, inferred, limit=6)
    if len(num_cols) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_numeric_columns")
    mat = df[num_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(mat) < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    _ensure_budget(timer)
    # Build state transition matrix A from consecutive observations
    X = mat.to_numpy(dtype=float)
    n_states = X.shape[1]
    # Standardize
    for j in range(n_states):
        center, scale = robust_center_scale(X[:, j])
        if not math.isfinite(scale) or scale <= 0:
            scale = 1.0
        X[:, j] = (X[:, j] - center) / scale
    # Estimate A via least squares: X[t+1] = A * X[t]
    X_t = X[:-1]
    X_t1 = X[1:]
    try:
        A, _, _, _ = np.linalg.lstsq(X_t, X_t1, rcond=None)
        A = A.T  # Now A is n_states x n_states
    except Exception:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "state_matrix_estimation_failed")
    _ensure_budget(timer)
    # Controllability matrix C = [B, AB, A^2 B, ...]
    # Assume B = I (all states are inputs)
    B = np.eye(n_states)
    controllability = B.copy()
    Ak = A.copy()
    for k in range(1, n_states):
        controllability = np.hstack([controllability, Ak @ B])
        Ak = Ak @ A
    # Rank of controllability matrix
    rank = int(np.linalg.matrix_rank(controllability, tol=1e-6))
    n_dark_states = n_states - rank
    controllability_ratio = rank / max(1, n_states)
    # Observability: eigenvalue analysis
    eigenvalues = np.linalg.eigvals(A)
    spectral_radius = float(np.max(np.abs(eigenvalues)))
    findings = []
    if n_dark_states > 0:
        findings.append(_make_finding(
            plugin_id, "dark_states",
            f"{n_dark_states} dark state(s) detected (not fully controllable)",
            f"Controllability rank={rank}/{n_states}. {n_dark_states} state dimension(s) cannot be fully controlled.",
            "Dark states are system dimensions that cannot be driven to arbitrary values from observed inputs.",
            {"metrics": {"rank": rank, "n_states": n_states, "n_dark_states": n_dark_states,
                         "controllability_ratio": controllability_ratio, "spectral_radius": spectral_radius,
                         "columns": num_cols}},
            recommendation="Add observability for dark states or introduce control inputs that span the missing dimensions.",
            severity="warn" if n_dark_states <= 1 else "critical",
            confidence=min(0.8, 0.5 + n_dark_states * 0.1),
        ))
    if spectral_radius > 1.0:
        findings.append(_make_finding(
            plugin_id, "unstable_dynamics",
            "Unstable state dynamics (spectral radius > 1)",
            f"Spectral radius={spectral_radius:.3f}. State transitions amplify perturbations.",
            "Spectral radius > 1 means the system amplifies disturbances over time.",
            {"metrics": {"spectral_radius": spectral_radius}},
            recommendation="Add stabilizing feedback to bring spectral radius below 1.",
            severity="warn" if spectral_radius < 1.5 else "critical",
            confidence=0.7,
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Computed controllability Gramian: rank={rank}/{n_states}.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "rank": rank, "n_states": n_states,
                       "n_dark_states": n_dark_states, "spectral_radius": spectral_radius})


# ---------------------------------------------------------------------------
# HANDLERS registry
# ---------------------------------------------------------------------------

HANDLERS: dict[str, Callable[..., PluginResult]] = {
    "analysis_ising_model_correlation_v1": _handler_ising_model_correlation_v1,
    "analysis_percolation_fragility_v1": _handler_percolation_fragility_v1,
    "analysis_renormalization_multiscale_v1": _handler_renormalization_multiscale_v1,
    "analysis_partition_function_landscape_v1": _handler_partition_function_landscape_v1,
    "analysis_reynolds_number_analogue_v1": _handler_reynolds_number_analogue_v1,
    "analysis_navier_stokes_backpressure_v1": _handler_navier_stokes_backpressure_v1,
    "analysis_vorticity_thrashing_v1": _handler_vorticity_thrashing_v1,
    "analysis_entropy_production_rate_v1": _handler_entropy_production_rate_v1,
    "analysis_carnot_efficiency_bound_v1": _handler_carnot_efficiency_bound_v1,
    "analysis_exergy_waste_analysis_v1": _handler_exergy_waste_analysis_v1,
    "analysis_maxwell_demon_monitoring_v1": _handler_maxwell_demon_monitoring_v1,
    "analysis_periodic_schedule_fft_v1": _handler_periodic_schedule_fft_v1,
    "analysis_defect_density_classification_v1": _handler_defect_density_classification_v1,
    "analysis_stress_strain_capacity_v1": _handler_stress_strain_capacity_v1,
    "analysis_creep_fatigue_prediction_v1": _handler_creep_fatigue_prediction_v1,
    "analysis_bathtub_curve_lifecycle_v1": _handler_bathtub_curve_lifecycle_v1,
    "analysis_fmea_rpn_scoring_v1": _handler_fmea_rpn_scoring_v1,
    "analysis_fault_tree_minimal_cuts_v1": _handler_fault_tree_minimal_cuts_v1,
    "analysis_reliability_block_diagram_v1": _handler_reliability_block_diagram_v1,
    "analysis_pid_controller_analogue_v1": _handler_pid_controller_analogue_v1,
    "analysis_lyapunov_stability_v1": _handler_lyapunov_stability_v1,
    "analysis_bode_frequency_response_v1": _handler_bode_frequency_response_v1,
    "analysis_controllability_gramian_v1": _handler_controllability_gramian_v1,
}
