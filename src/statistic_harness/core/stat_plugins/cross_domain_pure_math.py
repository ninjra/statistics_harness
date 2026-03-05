"""Cross-domain plugins: pure mathematics & topology (plugins 48-72)."""
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
    from scipy.sparse import linalg as sparse_linalg
    HAS_SCIPY = True
except Exception:
    scipy_stats = scipy_optimize = sparse_linalg = None
    HAS_SCIPY = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except Exception:
    nx = None; HAS_NETWORKX = False

try:
    import ripser
    HAS_RIPSER = True
except Exception:
    ripser = None; HAS_RIPSER = False

try:
    import kmapper
    HAS_KMAPPER = True
except Exception:
    kmapper = None; HAS_KMAPPER = False

try:
    import gudhi
    HAS_GUDHI = True
except Exception:
    gudhi = None; HAS_GUDHI = False

try:
    import nolds
    HAS_NOLDS = True
except Exception:
    nolds = None; HAS_NOLDS = False

try:
    import dit as dit_pkg
    HAS_DIT = True
except Exception:
    dit_pkg = None; HAS_DIT = False


# ---------------------------------------------------------------------------
# Module-private helpers (duplicated per addon by convention)
# ---------------------------------------------------------------------------

def _safe_id(plugin_id, key):
    try: return stable_id((plugin_id, key))
    except Exception: return hashlib.sha256(f"{plugin_id}:{key}".encode()).hexdigest()[:16]

def _basic_metrics(df, sample_meta):
    m = {"rows_seen": int(sample_meta.get("rows_total", len(df))), "rows_used": int(sample_meta.get("rows_used", len(df))), "cols_used": int(len(df.columns))}
    m.update(sample_meta or {}); return m

def _make_finding(plugin_id, key, title, what, why, evidence, *, recommendation, severity="info", confidence=0.5, where=None, measurement_type="measured", kind=None):
    f = {"id": _safe_id(plugin_id, key), "severity": severity, "confidence": float(max(0.0, min(1.0, confidence))), "title": title, "what": what, "why": why, "evidence": evidence, "where": where or {}, "recommendation": recommendation, "measurement_type": measurement_type}
    if kind: f["kind"] = kind
    return f

def _ok_with_reason(plugin_id, ctx, df, sample_meta, reason, *, debug=None):
    ctx.logger(f"SKIP reason={reason}"); p = dict(debug or {}); p.setdefault("gating_reason", reason)
    return PluginResult("ok", f"No actionable result: {reason}", _basic_metrics(df, sample_meta), [], [], None, debug=p)

def _finalize(plugin_id, ctx, df, sample_meta, summary, findings, artifacts, *, extra_metrics=None, debug=None):
    metrics = _basic_metrics(df, sample_meta)
    if extra_metrics: metrics.update(extra_metrics)
    rt = int(metrics.pop("runtime_ms", 0)); ctx.logger(f"END runtime_ms={rt} findings={len(findings)}")
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
    hints = ("duration","latency","wait","elapsed","runtime","service","time")
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
        x = float(value); return default if not math.isfinite(x) else x
    except Exception: return default

def _find_parent_column(df):
    for col in df.columns:
        cl = str(col).lower().replace(" ", "_")
        if any(h in cl for h in ("parent","ppid","parent_id","parent_process","caller","upstream")):
            nf = float(df[col].isna().mean())
            if 0.01 < nf < 0.99: return str(col)
    return None

def _build_dag(df, parent_col, id_col=None):
    if not HAS_NETWORKX: return None
    G = nx.DiGraph()
    ids = df.index.tolist() if id_col is None else df[id_col].tolist()
    for idx, par in zip(ids, df[parent_col].tolist()):
        G.add_node(idx)
        if pd.notna(par): G.add_edge(par, idx)
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


# ---------------------------------------------------------------------------
# Additional math helpers
# ---------------------------------------------------------------------------

def _numeric_matrix(df, cols, max_rows=2000):
    """Build a clean numeric matrix from selected columns."""
    frame = df[cols].apply(pd.to_numeric, errors="coerce")
    med = frame.median(numeric_only=True)
    frame = frame.fillna(med)
    mat = frame.to_numpy(dtype=float)
    for j in range(mat.shape[1]):
        center, scale = robust_center_scale(mat[:, j])
        if not math.isfinite(scale) or scale <= 0:
            scale = 1.0
        mat[:, j] = (mat[:, j] - center) / scale
    if mat.shape[0] > max_rows:
        rng = np.random.default_rng(42)
        idx = rng.choice(mat.shape[0], size=max_rows, replace=False)
        mat = mat[idx]
    return mat


def _pairwise_distances(mat):
    """Compute pairwise Euclidean distance matrix."""
    diff = mat[:, np.newaxis, :] - mat[np.newaxis, :, :]
    return np.sqrt(np.maximum(0.0, np.sum(diff * diff, axis=-1)))


def _build_nx_graph(edges):
    """Build a networkx DiGraph from an edges DataFrame."""
    if not HAS_NETWORKX or edges.empty:
        return None
    G = nx.DiGraph()
    for row in edges.itertuples(index=False):
        G.add_edge(str(row.src), str(row.dst))
    return G


def _build_nx_undirected(edges):
    """Build a networkx Graph from an edges DataFrame."""
    if not HAS_NETWORKX or edges.empty:
        return None
    G = nx.Graph()
    for row in edges.itertuples(index=False):
        G.add_edge(str(row.src), str(row.dst))
    return G


def _get_sorted_ts(df, inferred):
    """Extract a sorted numeric time series from the best duration/numeric column."""
    tc, parsed = _time_series(df, inferred)
    dur_col = _duration_column(df, inferred)
    if dur_col is None:
        return None
    vals = pd.to_numeric(df[dur_col], errors="coerce").to_numpy(dtype=float)
    if tc is not None and parsed is not None:
        order = parsed.argsort()
        vals = vals[order]
    mask = np.isfinite(vals)
    vals = vals[mask]
    return vals if len(vals) >= 30 else None


def _laplacian_eigenvalues(G, k=10):
    """Compute smallest eigenvalues of the graph Laplacian."""
    if not HAS_NETWORKX:
        return np.array([])
    L = nx.laplacian_matrix(G).astype(float)
    n = L.shape[0]
    if n < 3:
        return np.array([])
    num = min(k, n - 1)
    if HAS_SCIPY and sparse_linalg is not None and n > 50:
        try:
            vals = sparse_linalg.eigsh(L, k=num, which="SM", return_eigenvectors=False)
            return np.sort(np.real(vals))
        except Exception:
            pass
    dense = L.toarray() if hasattr(L, "toarray") else np.array(L)
    vals = np.linalg.eigvalsh(dense)
    return np.sort(vals)[:num]


# ---------------------------------------------------------------------------
# Plugin handlers (48-72, skipping 58)
# ---------------------------------------------------------------------------


def _handler_persistent_homology_regimes_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """48. Persistent homology regimes via birth-death pairs."""
    _log_start(ctx, plugin_id, df, config, inferred)
    ncols = _variance_sorted_numeric(df, inferred, limit=6)
    if len(ncols) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "need_>=2_numeric_cols")
    mat = _numeric_matrix(df, ncols, max_rows=1000)
    if mat.shape[0] < 10:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "too_few_rows")
    _ensure_budget(timer)
    dist = _pairwise_distances(mat)
    # Use ripser if available, else manual 0-dim persistence
    if HAS_RIPSER:
        dgms = ripser.ripser(dist, maxdim=1, distance_matrix=True)["dgms"]
    else:
        # Manual 0-dim: sort edges, union-find for connected components
        n = dist.shape[0]
        triu_idx = np.triu_indices(n, k=1)
        edge_weights = dist[triu_idx]
        order = np.argsort(edge_weights)
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        births_deaths = []
        for ei in order:
            u, v = int(triu_idx[0][ei]), int(triu_idx[1][ei])
            w = float(edge_weights[ei])
            ru, rv = find(u), find(v)
            if ru != rv:
                births_deaths.append((0.0, w))
                parent[ru] = rv
            _ensure_budget(timer)
        dgms = [np.array(births_deaths) if births_deaths else np.empty((0, 2))]
    _ensure_budget(timer)
    findings = []
    for dim, dgm in enumerate(dgms):
        if len(dgm) == 0:
            continue
        lifetimes = dgm[:, 1] - dgm[:, 0]
        lifetimes = lifetimes[np.isfinite(lifetimes)]
        if len(lifetimes) == 0:
            continue
        med_life = float(np.median(lifetimes))
        max_life = float(np.max(lifetimes))
        long_lived = int(np.sum(lifetimes > 2.0 * med_life))
        if long_lived > 0:
            findings.append(_make_finding(
                plugin_id, f"dim{dim}_regimes", f"Persistent H{dim} features detected",
                f"{long_lived} topological features in dimension {dim} persist well beyond median lifetime ({med_life:.3f})",
                "Long-lived topological features indicate stable structural regimes in the parameter space",
                {"dimension": dim, "n_features": int(len(lifetimes)), "median_lifetime": med_life, "max_lifetime": max_life, "long_lived_count": long_lived},
                recommendation=f"Investigate the {long_lived} persistent dim-{dim} features as potential regime boundaries for process segmentation",
                severity="info" if long_lived < 3 else "warn", confidence=min(0.8, 0.4 + 0.1 * long_lived),
            ))
    extra = {"runtime_ms": _runtime_ms(timer), "n_dims": len(dgms), "backend": "ripser" if HAS_RIPSER else "manual"}
    return _finalize(plugin_id, ctx, df, sample_meta, f"Persistent homology: {len(findings)} regime findings", findings, [], extra_metrics=extra)


def _handler_mapper_subpopulations_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """49. Mapper algorithm for subpopulation detection."""
    _log_start(ctx, plugin_id, df, config, inferred)
    ncols = _variance_sorted_numeric(df, inferred, limit=8)
    if len(ncols) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "need_>=2_numeric_cols")
    mat = _numeric_matrix(df, ncols, max_rows=1500)
    if mat.shape[0] < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "too_few_rows")
    _ensure_budget(timer)
    # Use kmapper if available, else manual cover-based clustering
    if HAS_KMAPPER:
        mapper = kmapper.KeplerMapper(verbose=0)
        lens = mapper.fit_transform(mat, projection=[0])
        graph = mapper.map(lens, mat, cover=kmapper.Cover(n_cubes=10, perc_overlap=0.3))
        n_nodes = len(graph["nodes"])
        n_edges = sum(len(v) for v in graph["links"].values()) // 2
        node_sizes = [len(members) for members in graph["nodes"].values()]
    else:
        # Manual: divide lens (first PC) into overlapping intervals, cluster each
        if mat.shape[1] >= 2:
            cov = np.cov(mat.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            lens_vals = mat @ eigvecs[:, -1]
        else:
            lens_vals = mat[:, 0]
        n_cubes = 10
        lo, hi = float(np.min(lens_vals)), float(np.max(lens_vals))
        width = (hi - lo) / n_cubes * 1.3
        step = (hi - lo) / n_cubes
        nodes = {}
        node_id = 0
        for i in range(n_cubes):
            left = lo + i * step - 0.15 * width
            right = left + width
            mask = (lens_vals >= left) & (lens_vals <= right)
            members = np.where(mask)[0].tolist()
            if len(members) >= 2:
                nodes[node_id] = members
                node_id += 1
            _ensure_budget(timer)
        n_nodes = len(nodes)
        n_edges = 0
        keys = list(nodes.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                if set(nodes[keys[i]]) & set(nodes[keys[j]]):
                    n_edges += 1
        node_sizes = [len(m) for m in nodes.values()]
    _ensure_budget(timer)
    findings = []
    if n_nodes >= 2:
        size_cv = float(np.std(node_sizes) / max(1e-9, np.mean(node_sizes))) if node_sizes else 0.0
        findings.append(_make_finding(
            plugin_id, "mapper_graph", "Mapper graph reveals subpopulations",
            f"Mapper graph has {n_nodes} nodes and {n_edges} edges; node size CV={size_cv:.2f}",
            "High node-size variation suggests distinct subpopulations with different characteristics",
            {"n_nodes": n_nodes, "n_edges": n_edges, "node_size_cv": round(size_cv, 3), "min_node": int(min(node_sizes)) if node_sizes else 0, "max_node": int(max(node_sizes)) if node_sizes else 0},
            recommendation="Examine the largest and most isolated Mapper nodes as candidate subpopulations for differentiated treatment",
            severity="warn" if size_cv > 1.0 else "info", confidence=min(0.85, 0.4 + 0.05 * n_nodes),
        ))
    extra = {"runtime_ms": _runtime_ms(timer), "backend": "kmapper" if HAS_KMAPPER else "manual"}
    return _finalize(plugin_id, ctx, df, sample_meta, f"Mapper: {n_nodes} nodes, {n_edges} edges", findings, [], extra_metrics=extra)


def _handler_persistence_landscape_features_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """50. Persistence landscape statistical features."""
    _log_start(ctx, plugin_id, df, config, inferred)
    ncols = _variance_sorted_numeric(df, inferred, limit=6)
    if len(ncols) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "need_>=2_numeric_cols")
    mat = _numeric_matrix(df, ncols, max_rows=800)
    if mat.shape[0] < 10:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "too_few_rows")
    _ensure_budget(timer)
    dist = _pairwise_distances(mat)
    # Build 0-dim persistence via union-find
    n = dist.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    edge_weights = dist[triu_idx]
    order = np.argsort(edge_weights)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    pairs = []
    for ei in order:
        u, v = int(triu_idx[0][ei]), int(triu_idx[1][ei])
        w = float(edge_weights[ei])
        ru, rv = find(u), find(v)
        if ru != rv:
            pairs.append((0.0, w))
            parent[ru] = rv
        _ensure_budget(timer)
    if not pairs:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_persistence_pairs")
    # Convert to landscape: for each filtration value, sort lifetimes
    births = np.array([p[0] for p in pairs])
    deaths = np.array([p[1] for p in pairs])
    lifetimes = deaths - births
    grid = np.linspace(0, float(np.max(deaths)) * 1.05, 100)
    landscape = np.zeros(len(grid))
    for b, d in zip(births, deaths):
        mid = (b + d) / 2.0
        for gi, g in enumerate(grid):
            if b <= g <= mid:
                landscape[gi] = max(landscape[gi], g - b)
            elif mid < g <= d:
                landscape[gi] = max(landscape[gi], d - g)
    _ensure_budget(timer)
    l_mean = float(np.mean(landscape))
    l_max = float(np.max(landscape))
    l_std = float(np.std(landscape))
    l_integral = float(np.trapz(landscape, grid))
    findings = []
    if l_max > 2.0 * l_mean and l_mean > 0:
        findings.append(_make_finding(
            plugin_id, "landscape_peak", "Persistence landscape shows dominant feature",
            f"Landscape peak ({l_max:.3f}) is {l_max/max(l_mean,1e-9):.1f}x the mean ({l_mean:.3f})",
            "A dominant landscape peak indicates a single topological feature that is much more persistent than others",
            {"landscape_mean": round(l_mean, 4), "landscape_max": round(l_max, 4), "landscape_std": round(l_std, 4), "landscape_integral": round(l_integral, 4), "n_pairs": len(pairs)},
            recommendation="The dominant persistence feature likely corresponds to a major structural boundary; investigate the filtration scale at the peak",
            severity="info", confidence=min(0.75, 0.4 + 0.1 * (l_max / max(l_mean, 1e-9))),
        ))
    extra = {"runtime_ms": _runtime_ms(timer), "n_pairs": len(pairs)}
    return _finalize(plugin_id, ctx, df, sample_meta, f"Persistence landscape: integral={l_integral:.3f}", findings, [], extra_metrics=extra)


def _handler_vietoris_rips_higher_order_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """51. Vietoris-Rips complex for higher-order dependencies."""
    _log_start(ctx, plugin_id, df, config, inferred)
    ncols = _variance_sorted_numeric(df, inferred, limit=6)
    if len(ncols) < 3:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "need_>=3_numeric_cols")
    mat = _numeric_matrix(df, ncols, max_rows=500)
    if mat.shape[0] < 10:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "too_few_rows")
    _ensure_budget(timer)
    dist = _pairwise_distances(mat)
    # Compute pairwise correlation
    corr = np.corrcoef(mat.T)
    np.fill_diagonal(corr, 0.0)
    # Build VR complex at multiple epsilon thresholds
    thresholds = np.percentile(dist[np.triu_indices(dist.shape[0], k=1)], [25, 50, 75])
    simplex_counts = []
    for eps in thresholds:
        adj = (dist <= eps).astype(int)
        np.fill_diagonal(adj, 0)
        n_edges = int(adj.sum()) // 2
        # Count triangles (2-simplices)
        n_triangles = 0
        n = adj.shape[0]
        cap = min(n, 200)
        for i in range(cap):
            for j in range(i + 1, cap):
                if adj[i, j]:
                    for k in range(j + 1, cap):
                        if adj[i, k] and adj[j, k]:
                            n_triangles += 1
            _ensure_budget(timer)
        simplex_counts.append({"epsilon": round(float(eps), 4), "edges": n_edges, "triangles": n_triangles})
    # Detect higher-order: if triangles grow faster than edges, higher-order deps exist
    findings = []
    if len(simplex_counts) >= 2:
        e0, t0 = simplex_counts[0]["edges"], simplex_counts[0]["triangles"]
        e1, t1 = simplex_counts[-1]["edges"], simplex_counts[-1]["triangles"]
        tri_ratio = (t1 - t0) / max(1, e1 - e0) if e1 > e0 else 0.0
        if tri_ratio > 0.5 and t1 > 5:
            findings.append(_make_finding(
                plugin_id, "higher_order", "Higher-order dependencies detected in VR complex",
                f"Triangle growth rate ({tri_ratio:.2f} per edge) indicates dependencies beyond pairwise correlations",
                "When triangles grow disproportionately, variables interact in groups rather than just pairs",
                {"simplex_counts": simplex_counts, "triangle_ratio": round(tri_ratio, 3)},
                recommendation="Consider interaction terms or non-linear models that capture multi-way variable dependencies",
                severity="warn" if tri_ratio > 1.5 else "info", confidence=min(0.8, 0.4 + 0.1 * tri_ratio),
            ))
    extra = {"runtime_ms": _runtime_ms(timer), "simplex_counts": simplex_counts}
    return _finalize(plugin_id, ctx, df, sample_meta, f"VR complex: {len(simplex_counts)} thresholds", findings, [], extra_metrics=extra)


def _handler_euler_characteristic_curve_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """52. Euler characteristic curve chi(eps)."""
    _log_start(ctx, plugin_id, df, config, inferred)
    ncols = _variance_sorted_numeric(df, inferred, limit=6)
    if len(ncols) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "need_>=2_numeric_cols")
    mat = _numeric_matrix(df, ncols, max_rows=500)
    n = mat.shape[0]
    if n < 10:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "too_few_rows")
    _ensure_budget(timer)
    dist = _pairwise_distances(mat)
    triu_vals = dist[np.triu_indices(n, k=1)]
    eps_range = np.linspace(float(np.min(triu_vals)), float(np.percentile(triu_vals, 90)), 30)
    chi_curve = []
    cap = min(n, 150)
    for eps in eps_range:
        adj = (dist[:cap, :cap] <= eps).astype(int)
        np.fill_diagonal(adj, 0)
        vertices = cap
        edges = int(adj.sum()) // 2
        # Approximate faces (triangles)
        faces = 0
        for i in range(cap):
            nbrs = np.where(adj[i] > 0)[0]
            for ji, j in enumerate(nbrs):
                for k in nbrs[ji + 1:]:
                    if adj[j, k]:
                        faces += 1
        faces //= 3
        chi = vertices - edges + faces
        chi_curve.append({"epsilon": round(float(eps), 4), "chi": int(chi), "V": vertices, "E": edges, "F": faces})
        _ensure_budget(timer)
    findings = []
    chi_vals = [c["chi"] for c in chi_curve]
    chi_range = max(chi_vals) - min(chi_vals)
    if chi_range > n * 0.3:
        findings.append(_make_finding(
            plugin_id, "euler_curve", "Euler characteristic varies significantly across scales",
            f"chi ranges from {min(chi_vals)} to {max(chi_vals)} (range={chi_range}) over {len(eps_range)} scales",
            "Large Euler characteristic variation reveals topological transitions at different scales",
            {"chi_min": min(chi_vals), "chi_max": max(chi_vals), "chi_range": chi_range, "n_scales": len(eps_range)},
            recommendation="Use the epsilon values where chi changes most rapidly as candidate scale parameters for clustering or thresholding",
            severity="info", confidence=min(0.7, 0.3 + 0.05 * chi_range / max(n, 1)),
        ))
    extra = {"runtime_ms": _runtime_ms(timer), "n_scales": len(chi_curve)}
    return _finalize(plugin_id, ctx, df, sample_meta, f"Euler curve: chi range={chi_range}", findings, [], extra_metrics=extra)


def _handler_sheaf_consistency_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """53. Sheaf consistency across DAG neighborhoods."""
    _log_start(ctx, plugin_id, df, config, inferred)
    edges, sc, dc = _build_edges(df, inferred)
    if edges.empty:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_graph_edges")
    G = _build_nx_undirected(edges)
    if G is None or G.number_of_nodes() < 5:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "graph_too_small")
    ncols = _numeric_columns(df, inferred, max_cols=6)
    if not ncols:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_cols")
    _ensure_budget(timer)
    # Build node->row mapping
    node_col = sc
    node_to_rows = defaultdict(list)
    for i, val in enumerate(df[node_col].astype(str).tolist()):
        if val and val != "nan":
            node_to_rows[val].append(i)
    # Compute consistency: for each node, compare its numeric values to neighbors' values
    inconsistencies = []
    for node in list(G.nodes())[:500]:
        if node not in node_to_rows:
            continue
        nbrs = list(G.neighbors(node))
        if not nbrs:
            continue
        node_rows = node_to_rows[node]
        nbr_rows = []
        for nb in nbrs:
            nbr_rows.extend(node_to_rows.get(nb, []))
        if not nbr_rows or not node_rows:
            continue
        for col in ncols[:3]:
            nv = pd.to_numeric(df.iloc[node_rows][col], errors="coerce").dropna()
            nbv = pd.to_numeric(df.iloc[nbr_rows][col], errors="coerce").dropna()
            if len(nv) < 2 or len(nbv) < 2:
                continue
            nm, ns = float(np.mean(nv)), float(np.std(nv))
            nbm, nbs = float(np.mean(nbv)), float(np.std(nbv))
            scale = max(ns, nbs, 1e-9)
            diff = abs(nm - nbm) / scale
            if diff > 2.0:
                inconsistencies.append({"node": str(node), "column": col, "diff_sigma": round(diff, 2)})
        _ensure_budget(timer)
    findings = []
    if inconsistencies:
        worst = sorted(inconsistencies, key=lambda x: -x["diff_sigma"])[:10]
        findings.append(_make_finding(
            plugin_id, "sheaf_inconsistent", "Data inconsistency across graph neighborhoods",
            f"{len(inconsistencies)} node-neighborhood inconsistencies found (>{2.0} sigma)",
            "When a node's data differs significantly from its neighbors, it may indicate data quality issues or boundary effects",
            {"n_inconsistencies": len(inconsistencies), "worst": worst},
            recommendation="Review the most inconsistent nodes for data quality issues or legitimate boundary conditions",
            severity="warn" if len(inconsistencies) > 5 else "info",
            confidence=min(0.8, 0.3 + 0.05 * len(inconsistencies)),
        ))
    extra = {"runtime_ms": _runtime_ms(timer), "n_inconsistencies": len(inconsistencies)}
    return _finalize(plugin_id, ctx, df, sample_meta, f"Sheaf consistency: {len(inconsistencies)} inconsistencies", findings, [], extra_metrics=extra)


def _handler_functorial_migration_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """54. Functorial migration pattern detection."""
    _log_start(ctx, plugin_id, df, config, inferred)
    # Need a categorical column for batch/version and numeric columns for schema
    cats = _categorical_columns(df, inferred, max_cols=10)
    ncols = _numeric_columns(df, inferred, max_cols=8)
    if not cats or len(ncols) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "need_cat_and_numeric_cols")
    # Pick batch column (version, batch, period, etc.)
    batch_col = None
    for c in cats:
        if any(h in str(c).lower() for h in ("batch", "version", "period", "cohort", "wave", "group")):
            batch_col = c; break
    if batch_col is None:
        batch_col = cats[0]
    groups = df.groupby(batch_col, dropna=True)
    if groups.ngroups < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "single_batch")
    _ensure_budget(timer)
    # Check if numeric column distributions are schema-preserving across batches
    batch_stats = {}
    for name, grp in list(groups)[:20]:
        stats = {}
        for col in ncols[:6]:
            vals = pd.to_numeric(grp[col], errors="coerce").dropna()
            if len(vals) >= 5:
                stats[col] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "null_rate": float(grp[col].isna().mean())}
        batch_stats[str(name)] = stats
        _ensure_budget(timer)
    # Detect schema drift: compare each batch to the first
    batches = list(batch_stats.keys())
    ref = batch_stats[batches[0]]
    drift_scores = []
    for b in batches[1:]:
        total_drift = 0.0
        n_compared = 0
        for col in ref:
            if col in batch_stats[b]:
                ref_m, ref_s = ref[col]["mean"], ref[col]["std"]
                cur_m, cur_s = batch_stats[b][col]["mean"], batch_stats[b][col]["std"]
                scale = max(ref_s, cur_s, 1e-9)
                total_drift += abs(ref_m - cur_m) / scale
                n_compared += 1
        if n_compared > 0:
            drift_scores.append({"batch": b, "mean_drift_sigma": round(total_drift / n_compared, 3), "cols_compared": n_compared})
    findings = []
    high_drift = [d for d in drift_scores if d["mean_drift_sigma"] > 1.5]
    if high_drift:
        findings.append(_make_finding(
            plugin_id, "schema_drift", "Schema-breaking migration detected across batches",
            f"{len(high_drift)}/{len(drift_scores)} batches show >1.5 sigma drift from reference batch",
            "Large distributional shifts between batches suggest the data generation process changed, breaking schema consistency",
            {"batch_column": batch_col, "n_drifted": len(high_drift), "drift_details": sorted(high_drift, key=lambda x: -x["mean_drift_sigma"])[:5]},
            recommendation=f"Investigate batches with highest drift in column '{batch_col}' for schema or process changes",
            severity="warn", confidence=min(0.8, 0.4 + 0.1 * len(high_drift)),
        ))
    extra = {"runtime_ms": _runtime_ms(timer), "n_batches": len(batches)}
    return _finalize(plugin_id, ctx, df, sample_meta, f"Functorial migration: {len(high_drift)} drifted batches", findings, [], extra_metrics=extra)


def _handler_operad_composition_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """55. Operad composition: test if composed sub-workflows equal monolithic."""
    _log_start(ctx, plugin_id, df, config, inferred)
    edges, sc, dc = _build_edges(df, inferred)
    if edges.empty:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_graph_edges")
    dur_col = _duration_column(df, inferred)
    if dur_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_duration_col")
    G = _build_nx_graph(edges)
    if G is None or G.number_of_nodes() < 4:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "graph_too_small")
    _ensure_budget(timer)
    # Find linear chains (A->B->C) and compare sum of parts to end-to-end
    durations = pd.to_numeric(df[dur_col], errors="coerce")
    node_dur = {}
    for node in G.nodes():
        mask = df[sc].astype(str) == str(node)
        vals = durations[mask].dropna()
        if len(vals) >= 2:
            node_dur[node] = float(np.median(vals))
    chains = []
    for node in list(G.nodes())[:200]:
        succs = list(G.successors(node))
        for s in succs:
            ss = list(G.successors(s))
            for s2 in ss:
                if node in node_dur and s in node_dur and s2 in node_dur:
                    composed = node_dur[node] + node_dur[s] + node_dur[s2]
                    # Check if there's a direct edge node->s2 (monolithic)
                    if G.has_edge(node, s2):
                        mask_direct = (df[sc].astype(str) == str(node)) & (df[dc].astype(str) == str(s2))
                        direct_vals = durations[mask_direct].dropna()
                        if len(direct_vals) >= 2:
                            monolithic = float(np.median(direct_vals))
                            ratio = composed / max(monolithic, 1e-9)
                            chains.append({"chain": f"{node}->{s}->{s2}", "composed": round(composed, 2), "monolithic": round(monolithic, 2), "ratio": round(ratio, 3)})
        _ensure_budget(timer)
    findings = []
    non_composable = [c for c in chains if abs(c["ratio"] - 1.0) > 0.3]
    if non_composable:
        findings.append(_make_finding(
            plugin_id, "non_composable", "Sub-workflow composition violates additivity",
            f"{len(non_composable)}/{len(chains)} chains show >30% deviation between composed and monolithic durations",
            "When composed sub-workflows don't add up to the end-to-end duration, there are hidden coordination costs or parallelism",
            {"n_non_composable": len(non_composable), "examples": sorted(non_composable, key=lambda x: -abs(x["ratio"] - 1.0))[:5]},
            recommendation="Investigate chains with ratio >1.3 for coordination overhead and <0.7 for hidden parallelism opportunities",
            severity="warn" if len(non_composable) > 2 else "info", confidence=min(0.7, 0.3 + 0.1 * len(non_composable)),
        ))
    extra = {"runtime_ms": _runtime_ms(timer), "n_chains": len(chains)}
    return _finalize(plugin_id, ctx, df, sample_meta, f"Operad composition: {len(chains)} chains tested", findings, [], extra_metrics=extra)


def _handler_grothendieck_coverage_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """56. Grothendieck coverage: minimal monitoring configuration."""
    _log_start(ctx, plugin_id, df, config, inferred)
    edges, sc, dc = _build_edges(df, inferred)
    if edges.empty:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_graph_edges")
    G = _build_nx_undirected(edges)
    if G is None or G.number_of_nodes() < 3:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "graph_too_small")
    _ensure_budget(timer)
    # Greedy dominating set as minimal monitoring cover
    nodes = set(G.nodes())
    covered = set()
    monitors = []
    remaining = set(nodes)
    while covered != nodes and remaining:
        best = max(remaining, key=lambda n: len(set(G.neighbors(n)) - covered) + (1 if n not in covered else 0))
        monitors.append(best)
        covered.add(best)
        covered.update(G.neighbors(best))
        remaining.discard(best)
        _ensure_budget(timer)
    coverage_ratio = len(monitors) / max(1, len(nodes))
    findings = []
    findings.append(_make_finding(
        plugin_id, "min_coverage", "Minimal monitoring configuration identified",
        f"{len(monitors)} monitors cover all {len(nodes)} nodes (ratio={coverage_ratio:.2%})",
        "A small dominating set means a few well-placed monitors can observe the entire process graph",
        {"n_monitors": len(monitors), "n_nodes": len(nodes), "coverage_ratio": round(coverage_ratio, 4), "monitors": [str(m) for m in monitors[:20]]},
        recommendation=f"Deploy monitoring on the identified {len(monitors)} nodes to achieve full coverage with minimal overhead",
        severity="info" if coverage_ratio < 0.3 else "warn", confidence=0.7,
    ))
    extra = {"runtime_ms": _runtime_ms(timer), "coverage_ratio": round(coverage_ratio, 4)}
    return _finalize(plugin_id, ctx, df, sample_meta, f"Grothendieck coverage: {len(monitors)}/{len(nodes)} monitors", findings, [], extra_metrics=extra)


def _handler_lyapunov_exponent_chaos_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """57. Lyapunov exponent for chaos detection."""
    _log_start(ctx, plugin_id, df, config, inferred)
    ts = _get_sorted_ts(df, inferred)
    if ts is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_time_series")
    ts = ts[:5000]
    _ensure_budget(timer)
    if HAS_NOLDS:
        try:
            lya = float(nolds.lyap_r(ts, emb_dim=10, lag=1, min_tsep=10))
        except Exception:
            lya = float("nan")
    else:
        # Manual estimate: track divergence of nearby trajectories
        emb_dim = 5
        lag = 1
        n = len(ts) - emb_dim * lag
        if n < 20:
            return _ok_with_reason(plugin_id, ctx, df, sample_meta, "series_too_short")
        embedded = np.array([ts[i:i + emb_dim * lag:lag] for i in range(n)])
        divergences = []
        rng = np.random.default_rng(int(config.get("seed", 42)))
        for _ in range(min(200, n)):
            i = rng.integers(0, n - 2)
            dists = np.sqrt(np.sum((embedded - embedded[i]) ** 2, axis=1))
            dists[max(0, i - 5):i + 6] = np.inf
            j = int(np.argmin(dists))
            if i + 1 < n and j + 1 < n:
                d0 = max(dists[j], 1e-12)
                d1 = np.sqrt(np.sum((embedded[i + 1] - embedded[j + 1]) ** 2))
                if d1 > 0:
                    divergences.append(math.log(d1 / d0))
            _ensure_budget(timer)
        lya = float(np.mean(divergences)) if divergences else float("nan")
    findings = []
    if math.isfinite(lya):
        is_chaotic = lya > 0.01
        findings.append(_make_finding(
            plugin_id, "lyapunov", "Chaotic dynamics detected" if is_chaotic else "Non-chaotic dynamics",
            f"Largest Lyapunov exponent = {lya:.4f} ({'positive = chaotic' if is_chaotic else 'non-positive = stable'})",
            "A positive Lyapunov exponent means nearby trajectories diverge exponentially, making long-term prediction unreliable",
            {"lyapunov_exponent": round(lya, 6), "is_chaotic": is_chaotic, "series_length": len(ts), "backend": "nolds" if HAS_NOLDS else "manual"},
            recommendation="For chaotic series, use short-horizon predictions and ensemble methods rather than single long-range forecasts" if is_chaotic else "Series is stable; standard forecasting methods should work",
            severity="warn" if is_chaotic else "info", confidence=min(0.8, 0.5 + 0.1 * abs(lya)),
        ))
    extra = {"runtime_ms": _runtime_ms(timer), "lyapunov": round(lya, 6) if math.isfinite(lya) else None}
    return _finalize(plugin_id, ctx, df, sample_meta, f"Lyapunov exponent: {lya:.4f}" if math.isfinite(lya) else "Lyapunov: could not compute", findings, [], extra_metrics=extra)


def _handler_fractal_dimension_attractor_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """59. Fractal (correlation) dimension of attractor."""
    _log_start(ctx, plugin_id, df, config, inferred)
    ts = _get_sorted_ts(df, inferred)
    if ts is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_time_series")
    ts = ts[:5000]
    _ensure_budget(timer)
    if HAS_NOLDS:
        try:
            cd = float(nolds.corr_dim(ts, emb_dim=10))
        except Exception:
            cd = float("nan")
    else:
        # Manual Grassberger-Procaccia
        emb_dim = 5
        lag = 1
        n = len(ts) - emb_dim * lag
        if n < 30:
            return _ok_with_reason(plugin_id, ctx, df, sample_meta, "series_too_short")
        embedded = np.array([ts[i:i + emb_dim * lag:lag] for i in range(n)])
        # Subsample for speed
        cap = min(n, 500)
        rng = np.random.default_rng(int(config.get("seed", 42)))
        idx = rng.choice(n, size=cap, replace=False) if n > cap else np.arange(n)
        pts = embedded[idx]
        dists = _pairwise_distances(pts)
        triu = dists[np.triu_indices(cap, k=1)]
        triu = triu[triu > 0]
        if len(triu) < 10:
            return _ok_with_reason(plugin_id, ctx, df, sample_meta, "degenerate_embedding")
        _ensure_budget(timer)
        radii = np.percentile(triu, np.linspace(5, 80, 15))
        log_r = []
        log_c = []
        total_pairs = len(triu)
        for r in radii:
            count = int(np.sum(triu < r))
            if count > 0:
                log_r.append(math.log(r))
                log_c.append(math.log(count / total_pairs))
        if len(log_r) < 3:
            cd = float("nan")
        else:
            x = np.array(log_r)
            y = np.array(log_c)
            slope = float(np.polyfit(x, y, 1)[0])
            cd = slope
    findings = []
    if math.isfinite(cd) and cd > 0:
        findings.append(_make_finding(
            plugin_id, "fractal_dim", "Attractor fractal dimension estimated",
            f"Correlation dimension = {cd:.2f} in reconstructed state space",
            "A non-integer fractal dimension indicates the system has strange attractor dynamics",
            {"correlation_dimension": round(cd, 3), "series_length": len(ts), "backend": "nolds" if HAS_NOLDS else "manual"},
            recommendation=f"The system exhibits ~{cd:.1f}-dimensional dynamics; use at least {int(math.ceil(cd)) + 1} variables for adequate modeling",
            severity="info", confidence=min(0.7, 0.4 + 0.05 * cd),
        ))
    extra = {"runtime_ms": _runtime_ms(timer), "correlation_dimension": round(cd, 3) if math.isfinite(cd) else None}
    return _finalize(plugin_id, ctx, df, sample_meta, f"Fractal dimension: {cd:.3f}" if math.isfinite(cd) else "Fractal dim: could not compute", findings, [], extra_metrics=extra)


def _handler_takens_embedding_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """60. Takens embedding: estimate embedding dimension and delay."""
    _log_start(ctx, plugin_id, df, config, inferred)
    ts = _get_sorted_ts(df, inferred)
    if ts is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_time_series")
    ts = ts[:5000]
    n = len(ts)
    _ensure_budget(timer)
    # Estimate delay via first minimum of autocorrelation
    max_lag = min(n // 4, 200)
    ts_centered = ts - np.mean(ts)
    var = float(np.var(ts_centered))
    if var <= 0:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "zero_variance")
    acf = np.correlate(ts_centered, ts_centered, mode="full")[n - 1:]
    acf = acf[:max_lag + 1] / (var * n)
    # Find first zero crossing or minimum
    delay = 1
    for i in range(1, len(acf)):
        if acf[i] <= 0:
            delay = i; break
    else:
        delay = int(np.argmin(acf[1:]) + 1)
    _ensure_budget(timer)
    # Estimate embedding dimension via false nearest neighbors
    best_dim = 1
    fnn_rates = []
    for dim in range(1, 11):
        nn = n - dim * delay
        if nn < 20:
            break
        embedded = np.array([ts[i:i + dim * delay:delay] for i in range(nn)])
        # Sample pairs for FNN test
        cap = min(nn, 300)
        rng = np.random.default_rng(int(config.get("seed", 42)))
        indices = rng.choice(nn - 1, size=cap, replace=False) if nn - 1 > cap else np.arange(nn - 1)
        false_nn = 0
        total = 0
        for idx in indices:
            dists = np.sqrt(np.sum((embedded - embedded[idx]) ** 2, axis=1))
            dists[idx] = np.inf
            j = int(np.argmin(dists))
            d_r = max(dists[j], 1e-12)
            if idx + 1 < nn and j + 1 < nn:
                d_next = abs(ts[idx + dim * delay] - ts[j + dim * delay]) if idx + dim * delay < n and j + dim * delay < n else 0.0
                if d_next / d_r > 10.0:
                    false_nn += 1
                total += 1
            _ensure_budget(timer)
        rate = false_nn / max(total, 1)
        fnn_rates.append({"dim": dim, "fnn_rate": round(rate, 4)})
        if rate < 0.05 and best_dim == 1:
            best_dim = dim
    if best_dim == 1 and fnn_rates:
        best_dim = min(fnn_rates, key=lambda x: x["fnn_rate"])["dim"]
    findings = []
    findings.append(_make_finding(
        plugin_id, "takens", f"Takens embedding: dim={best_dim}, delay={delay}",
        f"Optimal embedding dimension={best_dim} with time delay={delay} (series length={n})",
        "Takens' theorem guarantees state-space reconstruction from a single observable with sufficient embedding dimension",
        {"embedding_dim": best_dim, "delay": delay, "series_length": n, "fnn_rates": fnn_rates},
        recommendation=f"Use embedding dimension {best_dim} and delay {delay} for state-space reconstruction in downstream models",
        severity="info", confidence=min(0.8, 0.5 + 0.05 * len(fnn_rates)),
    ))
    extra = {"runtime_ms": _runtime_ms(timer), "embedding_dim": best_dim, "delay": delay}
    return _finalize(plugin_id, ctx, df, sample_meta, f"Takens embedding: dim={best_dim}, delay={delay}", findings, [], extra_metrics=extra)


def _handler_job_shop_scheduling_bound_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """61. Job-shop scheduling lower bound vs actual makespan."""
    _log_start(ctx, plugin_id, df, config, inferred)
    dur_col = _duration_column(df, inferred)
    if dur_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_duration_col")
    cats = _categorical_columns(df, inferred, max_cols=10)
    # Find job and machine columns
    job_col = machine_col = None
    for c in cats:
        cl = str(c).lower()
        if any(h in cl for h in ("job", "case", "order", "ticket", "request")):
            job_col = c; break
    for c in cats:
        if c == job_col:
            continue
        cl = str(c).lower()
        if any(h in cl for h in ("machine", "resource", "server", "host", "worker", "agent")):
            machine_col = c; break
    if job_col is None or machine_col is None:
        # Fallback: use first two categoricals
        if len(cats) >= 2:
            job_col, machine_col = cats[0], cats[1]
        else:
            return _ok_with_reason(plugin_id, ctx, df, sample_meta, "need_job_and_machine_cols")
    _ensure_budget(timer)
    durations = pd.to_numeric(df[dur_col], errors="coerce").fillna(0.0)
    # Lower bound 1: max job total duration
    job_totals = df.groupby(job_col, dropna=True).apply(lambda g: float(durations.loc[g.index].sum()))
    lb_job = float(job_totals.max()) if len(job_totals) > 0 else 0.0
    # Lower bound 2: max machine load
    machine_totals = df.groupby(machine_col, dropna=True).apply(lambda g: float(durations.loc[g.index].sum()))
    lb_machine = float(machine_totals.max()) if len(machine_totals) > 0 else 0.0
    theoretical_lb = max(lb_job, lb_machine)
    # Estimate actual makespan from timestamps if available
    tc, parsed = _time_series(df, inferred)
    if parsed is not None and parsed.notna().sum() >= 2:
        actual = float((parsed.max() - parsed.min()).total_seconds())
    else:
        actual = float(durations.sum())
    gap = (actual - theoretical_lb) / max(theoretical_lb, 1e-9)
    _ensure_budget(timer)
    findings = []
    if theoretical_lb > 0 and gap > 0.1:
        findings.append(_make_finding(
            plugin_id, "makespan_gap", "Scheduling gap vs theoretical lower bound",
            f"Actual makespan ({actual:.1f}) exceeds lower bound ({theoretical_lb:.1f}) by {gap:.1%}",
            "The gap between actual and theoretical minimum indicates scheduling inefficiency from idle time, sequencing, or coordination overhead",
            {"actual_makespan": round(actual, 2), "theoretical_lb": round(theoretical_lb, 2), "lb_job": round(lb_job, 2), "lb_machine": round(lb_machine, 2), "gap_pct": round(gap * 100, 1)},
            recommendation=f"Scheduling optimization could recover up to {gap:.0%} of makespan; focus on the bottleneck ({'job' if lb_job > lb_machine else 'machine'}) constraint",
            severity="warn" if gap > 0.3 else "info", confidence=min(0.75, 0.4 + 0.5 * gap),
        ))
    extra = {"runtime_ms": _runtime_ms(timer), "gap_pct": round(gap * 100, 1)}
    return _finalize(plugin_id, ctx, df, sample_meta, f"Job-shop bound: gap={gap:.1%}", findings, [], extra_metrics=extra)


def _handler_bin_packing_utilization_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """62. First-fit-decreasing bin packing utilization."""
    _log_start(ctx, plugin_id, df, config, inferred)
    dur_col = _duration_column(df, inferred)
    if dur_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_duration_col")
    vals = pd.to_numeric(df[dur_col], errors="coerce").dropna().to_numpy(dtype=float)
    vals = vals[vals > 0]
    if len(vals) < 5:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "too_few_positive_values")
    _ensure_budget(timer)
    # Bin capacity = p95 of values (typical resource slot)
    capacity = float(np.percentile(vals, 95))
    if capacity <= 0:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "zero_capacity")
    # Theoretical minimum bins
    total = float(np.sum(vals))
    min_bins = int(math.ceil(total / capacity))
    # FFD algorithm
    sorted_vals = np.sort(vals)[::-1]
    bins = []
    for v in sorted_vals:
        if v > capacity:
            bins.append(v)
            continue
        placed = False
        for i, b in enumerate(bins):
            if b + v <= capacity:
                bins[i] += v
                placed = True
                break
        if not placed:
            bins.append(v)
        _ensure_budget(timer)
    ffd_bins = len(bins)
    utilization = total / (ffd_bins * capacity) if ffd_bins > 0 else 0.0
    waste = 1.0 - utilization
    findings = []
    if waste > 0.15:
        findings.append(_make_finding(
            plugin_id, "bin_packing", "Bin packing reveals resource underutilization",
            f"FFD packing uses {ffd_bins} bins (theoretical min={min_bins}), utilization={utilization:.1%}, waste={waste:.1%}",
            "High bin-packing waste means jobs could be better consolidated onto fewer resources",
            {"ffd_bins": ffd_bins, "theoretical_min_bins": min_bins, "utilization": round(utilization, 4), "waste": round(waste, 4), "capacity": round(capacity, 2), "n_items": len(vals)},
            recommendation=f"Consolidate workloads to approach the theoretical minimum of {min_bins} bins, saving ~{waste:.0%} of resource capacity",
            severity="warn" if waste > 0.3 else "info", confidence=min(0.8, 0.4 + waste),
        ))
    extra = {"runtime_ms": _runtime_ms(timer), "utilization": round(utilization, 4)}
    return _finalize(plugin_id, ctx, df, sample_meta, f"Bin packing: util={utilization:.1%}", findings, [], extra_metrics=extra)


def _handler_csp_binding_rules_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """63. Constraint satisfaction: identify binding rules limiting throughput."""
    _log_start(ctx, plugin_id, df, config, inferred)
    ncols = _numeric_columns(df, inferred, max_cols=20)
    if len(ncols) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "need_>=2_numeric_cols")
    dur_col = _duration_column(df, inferred)
    if dur_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_duration_col")
    _ensure_budget(timer)
    # Check which columns are at their boundary values most often (binding constraints)
    target = pd.to_numeric(df[dur_col], errors="coerce").to_numpy(dtype=float)
    binding = []
    for col in ncols:
        if col == dur_col:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(vals) & np.isfinite(target)
        if mask.sum() < 10:
            continue
        v = vals[mask]
        t = target[mask]
        lo, hi = float(np.percentile(v, 5)), float(np.percentile(v, 95))
        if hi - lo < 1e-9:
            continue
        # Fraction at upper or lower boundary (within 5% of range)
        margin = (hi - lo) * 0.05
        at_upper = float(np.mean(v >= hi - margin))
        at_lower = float(np.mean(v <= lo + margin))
        boundary_frac = max(at_upper, at_lower)
        # Correlation with target when at boundary
        if boundary_frac > 0.1:
            corr = abs(float(np.corrcoef(v, t)[0, 1])) if len(v) >= 3 else 0.0
            binding.append({"column": col, "boundary_frac": round(boundary_frac, 3), "bound_side": "upper" if at_upper > at_lower else "lower", "corr_with_duration": round(corr, 3)})
        _ensure_budget(timer)
    binding.sort(key=lambda x: (-x["boundary_frac"], -x["corr_with_duration"]))
    findings = []
    if binding:
        top = binding[:5]
        findings.append(_make_finding(
            plugin_id, "binding_constraints", "Binding constraint rules identified",
            f"{len(binding)} columns frequently hit boundary values; top: {top[0]['column']} ({top[0]['boundary_frac']:.0%} at {top[0]['bound_side']} bound)",
            "Columns that frequently hit their boundary while correlated with duration represent binding constraints on throughput",
            {"n_binding": len(binding), "top_constraints": top},
            recommendation=f"Relax the {top[0]['bound_side']} bound on '{top[0]['column']}' to potentially improve throughput",
            severity="warn" if top[0]["boundary_frac"] > 0.2 else "info",
            confidence=min(0.75, 0.3 + top[0]["boundary_frac"] + top[0]["corr_with_duration"] * 0.2),
        ))
    extra = {"runtime_ms": _runtime_ms(timer), "n_binding": len(binding)}
    return _finalize(plugin_id, ctx, df, sample_meta, f"CSP binding: {len(binding)} constraints", findings, [], extra_metrics=extra)


def _handler_min_cost_flow_throughput_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """64. Min-cut / max-flow throughput bottleneck."""
    _log_start(ctx, plugin_id, df, config, inferred)
    if not HAS_NETWORKX:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "networkx_unavailable")
    edges, sc, dc = _build_edges(df, inferred)
    if edges.empty or len(edges) < 3:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_graph_edges")
    G = nx.DiGraph()
    edge_counts = Counter()
    for row in edges.itertuples(index=False):
        edge_counts[(str(row.src), str(row.dst))] += 1
    for (s, d), c in edge_counts.items():
        G.add_edge(s, d, capacity=c)
    _ensure_budget(timer)
    # Find source (no in-edges) and sink (no out-edges) candidates
    sources = [n for n in G.nodes() if G.in_degree(n) == 0]
    sinks = [n for n in G.nodes() if G.out_degree(n) == 0]
    if not sources or not sinks:
        # Fallback: highest out-degree as source, highest in-degree as sink
        sources = [max(G.nodes(), key=lambda n: G.out_degree(n))]
        sinks = [max(G.nodes(), key=lambda n: G.in_degree(n))]
    # Add super-source and super-sink
    G.add_node("__super_source__")
    G.add_node("__super_sink__")
    for s in sources[:10]:
        G.add_edge("__super_source__", s, capacity=10000)
    for t in sinks[:10]:
        G.add_edge(t, "__super_sink__", capacity=10000)
    _ensure_budget(timer)
    try:
        flow_value, flow_dict = nx.maximum_flow(G, "__super_source__", "__super_sink__")
        cut_value, (reachable, non_reachable) = nx.minimum_cut(G, "__super_source__", "__super_sink__")
    except nx.NetworkXError:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "flow_computation_failed")
    # Identify bottleneck edges (on the min-cut)
    cut_edges = []
    for u in reachable:
        for v in G.successors(u):
            if v in non_reachable:
                cut_edges.append({"from": str(u), "to": str(v), "capacity": int(edge_counts.get((str(u), str(v)), 0))})
    cut_edges = [e for e in cut_edges if e["from"] not in ("__super_source__",) and e["to"] not in ("__super_sink__",)]
    findings = []
    if cut_edges:
        findings.append(_make_finding(
            plugin_id, "min_cut", "Throughput bottleneck identified via min-cut",
            f"Max flow = {int(flow_value)}, min-cut has {len(cut_edges)} edges",
            "The min-cut edges are the throughput-limiting bottleneck; increasing their capacity directly increases max flow",
            {"max_flow": int(flow_value), "min_cut_value": int(cut_value), "cut_edges": cut_edges[:10], "n_nodes": G.number_of_nodes() - 2},
            recommendation=f"Increase capacity on the {len(cut_edges)} min-cut edges to raise throughput above {int(flow_value)}",
            severity="warn" if len(cut_edges) <= 3 else "info", confidence=0.7,
        ))
    extra = {"runtime_ms": _runtime_ms(timer), "max_flow": int(flow_value)}
    return _finalize(plugin_id, ctx, df, sample_meta, f"Min-cost flow: max_flow={int(flow_value)}", findings, [], extra_metrics=extra)


def _handler_kolmogorov_complexity_ncd_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """65. Normalized Compression Distance for anomaly detection."""
    _log_start(ctx, plugin_id, df, config, inferred)
    import zlib
    ts = _get_sorted_ts(df, inferred)
    ncols = _variance_sorted_numeric(df, inferred, limit=4)
    if ts is None and not ncols:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_data")
    _ensure_budget(timer)
    # Use time series if available, else concatenate numeric columns
    if ts is not None:
        data = ts[:3000]
    else:
        mat = _numeric_matrix(df, ncols, max_rows=3000)
        data = mat.flatten()
    # Split into windows and compute NCD between consecutive windows
    window_size = max(50, len(data) // 10)
    n_windows = len(data) // window_size
    if n_windows < 3:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "too_few_windows")
    windows = [data[i * window_size:(i + 1) * window_size] for i in range(n_windows)]
    def _compress_len(arr):
        b = arr.astype(np.float32).tobytes()
        return len(zlib.compress(b, 6))
    ncds = []
    for i in range(len(windows) - 1):
        cx = _compress_len(windows[i])
        cy = _compress_len(windows[i + 1])
        cxy = _compress_len(np.concatenate([windows[i], windows[i + 1]]))
        ncd = (cxy - min(cx, cy)) / max(cx, cy, 1)
        ncds.append({"window": i, "ncd": round(float(ncd), 4)})
        _ensure_budget(timer)
    ncd_vals = np.array([x["ncd"] for x in ncds])
    mean_ncd = float(np.mean(ncd_vals))
    std_ncd = float(np.std(ncd_vals))
    anomalies = [x for x in ncds if abs(x["ncd"] - mean_ncd) > 2.0 * max(std_ncd, 1e-6)]
    findings = []
    if anomalies:
        findings.append(_make_finding(
            plugin_id, "ncd_anomaly", "Compression-distance anomalies detected",
            f"{len(anomalies)}/{len(ncds)} windows show anomalous NCD (mean={mean_ncd:.3f}, std={std_ncd:.3f})",
            "Anomalous compression distance indicates a window's data has fundamentally different information content",
            {"mean_ncd": round(mean_ncd, 4), "std_ncd": round(std_ncd, 4), "n_anomalies": len(anomalies), "anomaly_windows": anomalies[:5]},
            recommendation="Investigate the anomalous time windows for regime changes or data quality issues",
            severity="warn" if len(anomalies) > 2 else "info", confidence=min(0.75, 0.4 + 0.1 * len(anomalies)),
        ))
    extra = {"runtime_ms": _runtime_ms(timer), "mean_ncd": round(mean_ncd, 4)}
    return _finalize(plugin_id, ctx, df, sample_meta, f"NCD: mean={mean_ncd:.3f}, {len(anomalies)} anomalies", findings, [], extra_metrics=extra)


def _handler_renyi_entropy_spectrum_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """66. Renyi entropy spectrum H_alpha for multiple alpha values."""
    _log_start(ctx, plugin_id, df, config, inferred)
    ncols = _variance_sorted_numeric(df, inferred, limit=6)
    if not ncols:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_cols")
    _ensure_budget(timer)
    alphas = [0.0, 0.5, 1.0, 2.0, float("inf")]
    results = []
    for col in ncols[:4]:
        vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals) < 20:
            continue
        # Discretize into bins
        n_bins = min(50, max(5, int(math.sqrt(len(vals)))))
        counts, _ = np.histogram(vals, bins=n_bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        spectrum = {}
        for alpha in alphas:
            if alpha == 0.0:
                h = math.log(len(probs))
            elif alpha == 1.0:
                h = -float(np.sum(probs * np.log(probs)))
            elif math.isinf(alpha):
                h = -math.log(float(np.max(probs)))
            else:
                h = float(np.log(np.sum(probs ** alpha)) / (1.0 - alpha))
            spectrum[f"H_{alpha}"] = round(h, 4)
        results.append({"column": col, **spectrum})
        _ensure_budget(timer)
    findings = []
    if results:
        # Flag columns where H_0 >> H_inf (wide spread of entropy across orders)
        for r in results:
            h0 = r.get("H_0.0", 0)
            hinf = r.get("H_inf", 0)
            spread = h0 - hinf
            if spread > 1.0:
                findings.append(_make_finding(
                    plugin_id, f"renyi_{r['column']}", f"Wide Renyi entropy spread for {r['column']}",
                    f"H_0 - H_inf = {spread:.2f} for column '{r['column']}', indicating heterogeneous probability mass",
                    "Large entropy spread means rare events contribute disproportionately; the distribution has heavy tails or outlier concentrations",
                    {"column": r["column"], "spectrum": r, "spread": round(spread, 3)},
                    recommendation=f"Column '{r['column']}' has heterogeneous mass; consider tail-aware methods or separate analysis of rare vs common events",
                    severity="info", confidence=min(0.7, 0.3 + 0.1 * spread),
                ))
    extra = {"runtime_ms": _runtime_ms(timer), "n_columns": len(results)}
    return _finalize(plugin_id, ctx, df, sample_meta, f"Renyi entropy: {len(results)} columns analyzed", findings, [], extra_metrics=extra)


def _handler_pid_synergy_redundancy_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """67. Partial Information Decomposition: synergy vs redundancy."""
    _log_start(ctx, plugin_id, df, config, inferred)
    ncols = _variance_sorted_numeric(df, inferred, limit=6)
    dur_col = _duration_column(df, inferred)
    if dur_col is None or len(ncols) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "need_duration_and_>=2_numeric")
    features = [c for c in ncols if c != dur_col][:4]
    if not features:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_feature_cols")
    _ensure_budget(timer)
    target = pd.to_numeric(df[dur_col], errors="coerce").to_numpy(dtype=float)
    # Approximate PID via interaction information: I(X;Y;Z) = I(X,Y;Z) - I(X;Z) - I(Y;Z)
    def _mi_discrete(x, y, n_bins=20):
        """Mutual information via histogram."""
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) < 20:
            return 0.0
        hxy, _, _ = np.histogram2d(x, y, bins=n_bins)
        pxy = hxy / hxy.sum()
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)
        pxy_flat = pxy.flatten()
        pxy_flat = pxy_flat[pxy_flat > 0]
        px = px[px > 0]
        py = py[py > 0]
        return float(np.sum(pxy_flat * np.log(pxy_flat)) - np.sum(px * np.log(px)) - np.sum(py * np.log(py)))
    pid_results = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            xi = pd.to_numeric(df[features[i]], errors="coerce").to_numpy(dtype=float)
            xj = pd.to_numeric(df[features[j]], errors="coerce").to_numpy(dtype=float)
            mi_i_t = _mi_discrete(xi, target)
            mi_j_t = _mi_discrete(xj, target)
            combined = xi + xj  # simple combination
            mi_ij_t = _mi_discrete(combined, target)
            synergy = mi_ij_t - mi_i_t - mi_j_t
            redundancy = mi_i_t + mi_j_t - mi_ij_t
            pid_results.append({
                "feature_1": features[i], "feature_2": features[j],
                "mi_1": round(mi_i_t, 4), "mi_2": round(mi_j_t, 4), "mi_joint": round(mi_ij_t, 4),
                "synergy": round(synergy, 4), "redundancy": round(redundancy, 4),
            })
            _ensure_budget(timer)
    findings = []
    synergistic = [p for p in pid_results if p["synergy"] > 0.05]
    redundant = [p for p in pid_results if p["redundancy"] > 0.05]
    if synergistic:
        best = max(synergistic, key=lambda x: x["synergy"])
        findings.append(_make_finding(
            plugin_id, "synergy", "Synergistic feature interactions found",
            f"{len(synergistic)} feature pairs show synergy; strongest: {best['feature_1']} + {best['feature_2']} (synergy={best['synergy']:.3f})",
            "Synergistic pairs provide information about the target that neither feature provides alone",
            {"n_synergistic": len(synergistic), "best_pair": best, "all_pairs": pid_results},
            recommendation=f"Create interaction features combining '{best['feature_1']}' and '{best['feature_2']}' for better predictions",
            severity="info", confidence=min(0.7, 0.4 + best["synergy"]),
        ))
    if redundant:
        worst = max(redundant, key=lambda x: x["redundancy"])
        findings.append(_make_finding(
            plugin_id, "redundancy", "Redundant feature pairs detected",
            f"{len(redundant)} feature pairs are redundant; worst: {worst['feature_1']} + {worst['feature_2']} (redundancy={worst['redundancy']:.3f})",
            "Redundant features provide overlapping information; removing one may simplify models without losing accuracy",
            {"n_redundant": len(redundant), "worst_pair": worst},
            recommendation=f"Consider dropping one of '{worst['feature_1']}' or '{worst['feature_2']}' to reduce redundancy",
            severity="info", confidence=min(0.7, 0.4 + worst["redundancy"]),
        ))
    extra = {"runtime_ms": _runtime_ms(timer), "n_pairs": len(pid_results)}
    return _finalize(plugin_id, ctx, df, sample_meta, f"PID: {len(synergistic)} synergistic, {len(redundant)} redundant", findings, [], extra_metrics=extra)


def _handler_excess_entropy_memory_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """68. Excess entropy: mutual information between past and future."""
    _log_start(ctx, plugin_id, df, config, inferred)
    ts = _get_sorted_ts(df, inferred)
    if ts is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_time_series")
    ts = ts[:5000]
    n = len(ts)
    if n < 60:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "series_too_short")
    _ensure_budget(timer)
    # Discretize the series
    n_bins = min(30, max(5, int(math.sqrt(n))))
    digitized = np.digitize(ts, np.linspace(float(np.min(ts)), float(np.max(ts)), n_bins + 1)[:-1]) - 1
    # Compute block entropies for increasing block lengths
    block_entropies = []
    for L in range(1, min(8, n // 10)):
        blocks = [tuple(digitized[i:i + L]) for i in range(n - L + 1)]
        counts = Counter(blocks)
        total = sum(counts.values())
        probs = np.array([c / total for c in counts.values()])
        h = -float(np.sum(probs * np.log(np.maximum(probs, 1e-12))))
        block_entropies.append({"L": L, "H_L": round(h, 4)})
        _ensure_budget(timer)
    # Excess entropy = limit of h_L * L - H_L as L grows
    # Approximate: E = H_L - L * h_mu where h_mu is entropy rate
    if len(block_entropies) >= 3:
        Ls = np.array([b["L"] for b in block_entropies], dtype=float)
        Hs = np.array([b["H_L"] for b in block_entropies], dtype=float)
        # Entropy rate from slope of H_L vs L
        coeffs = np.polyfit(Ls, Hs, 1)
        h_mu = float(coeffs[0])
        excess = float(Hs[-1] - h_mu * Ls[-1])
        # Memory depth: where block entropy starts being linear
        residuals = Hs - (h_mu * Ls + coeffs[1])
        memory_depth = 1
        for i, r in enumerate(residuals):
            if abs(r) > 0.1:
                memory_depth = i + 1
    else:
        h_mu = 0.0
        excess = 0.0
        memory_depth = 1
    findings = []
    if excess > 0.1:
        findings.append(_make_finding(
            plugin_id, "excess_entropy", "Significant process memory detected",
            f"Excess entropy = {excess:.3f}, entropy rate = {h_mu:.3f}, memory depth ~ {memory_depth}",
            "Excess entropy quantifies how much the past informs the future; high values mean history matters for prediction",
            {"excess_entropy": round(excess, 4), "entropy_rate": round(h_mu, 4), "memory_depth": memory_depth, "block_entropies": block_entropies},
            recommendation=f"Use at least {memory_depth} time steps of history in predictive models for this series",
            severity="info" if memory_depth <= 3 else "warn", confidence=min(0.75, 0.4 + 0.1 * excess),
        ))
    extra = {"runtime_ms": _runtime_ms(timer), "excess_entropy": round(excess, 4)}
    return _finalize(plugin_id, ctx, df, sample_meta, f"Excess entropy: {excess:.3f}, depth={memory_depth}", findings, [], extra_metrics=extra)


def _handler_channel_capacity_bound_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """69. Information-theoretic channel capacity bound."""
    _log_start(ctx, plugin_id, df, config, inferred)
    ncols = _numeric_columns(df, inferred, max_cols=10)
    dur_col = _duration_column(df, inferred)
    if dur_col is None or not ncols:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "need_duration_and_numeric")
    features = [c for c in ncols if c != dur_col][:6]
    if not features:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_feature_cols")
    _ensure_budget(timer)
    target = pd.to_numeric(df[dur_col], errors="coerce").to_numpy(dtype=float)
    # Compute MI between each feature and target as channel capacity bound
    capacities = []
    n_bins = min(30, max(5, int(math.sqrt(len(df)))))
    for col in features:
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(vals) & np.isfinite(target)
        if mask.sum() < 20:
            continue
        v, t = vals[mask], target[mask]
        hxy, _, _ = np.histogram2d(v, t, bins=n_bins)
        pxy = hxy / hxy.sum()
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)
        mi = 0.0
        for i in range(pxy.shape[0]):
            for j in range(pxy.shape[1]):
                if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += pxy[i, j] * math.log(pxy[i, j] / (px[i] * py[j]))
        # Channel capacity = max MI over input distributions (approximate with uniform)
        capacities.append({"column": col, "mutual_info_bits": round(mi / math.log(2), 4), "mutual_info_nats": round(mi, 4)})
        _ensure_budget(timer)
    capacities.sort(key=lambda x: -x["mutual_info_bits"])
    findings = []
    if capacities:
        total_cap = sum(c["mutual_info_bits"] for c in capacities)
        best = capacities[0]
        findings.append(_make_finding(
            plugin_id, "channel_capacity", "Information-theoretic throughput bounds",
            f"Best single-channel capacity: {best['column']} at {best['mutual_info_bits']:.3f} bits; total across {len(capacities)} channels: {total_cap:.3f} bits",
            "Channel capacity bounds the maximum predictive information available from each input feature",
            {"best_channel": best, "total_capacity_bits": round(total_cap, 4), "all_channels": capacities},
            recommendation=f"Feature '{best['column']}' carries the most information ({best['mutual_info_bits']:.3f} bits); prioritize it in models",
            severity="info", confidence=min(0.7, 0.3 + 0.1 * total_cap),
        ))
    extra = {"runtime_ms": _runtime_ms(timer), "n_channels": len(capacities)}
    return _finalize(plugin_id, ctx, df, sample_meta, f"Channel capacity: {len(capacities)} channels", findings, [], extra_metrics=extra)


def _handler_spectral_graph_fiedler_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """70. Fiedler value (algebraic connectivity) of workflow graph."""
    _log_start(ctx, plugin_id, df, config, inferred)
    if not HAS_NETWORKX:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "networkx_unavailable")
    edges, sc, dc = _build_edges(df, inferred)
    if edges.empty:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_graph_edges")
    G = _build_nx_undirected(edges)
    if G is None or G.number_of_nodes() < 4:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "graph_too_small")
    _ensure_budget(timer)
    # Use largest connected component
    components = list(nx.connected_components(G))
    largest_cc = max(components, key=len)
    H = G.subgraph(largest_cc).copy()
    n = H.number_of_nodes()
    if n < 4:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "largest_component_too_small")
    eigenvalues = _laplacian_eigenvalues(H, k=5)
    if len(eigenvalues) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "eigenvalue_computation_failed")
    fiedler = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
    spectral_gap = float(eigenvalues[1] / max(eigenvalues[-1], 1e-9)) if len(eigenvalues) > 1 else 0.0
    findings = []
    findings.append(_make_finding(
        plugin_id, "fiedler", f"Fiedler value = {fiedler:.4f}",
        f"Algebraic connectivity (Fiedler value) = {fiedler:.4f} for graph with {n} nodes; spectral gap ratio = {spectral_gap:.4f}",
        "Low Fiedler value indicates the graph has a bottleneck (nearly disconnected components); high value indicates robust connectivity",
        {"fiedler_value": round(fiedler, 6), "spectral_gap": round(spectral_gap, 6), "n_nodes": n, "n_components": len(components), "eigenvalues": [round(float(e), 6) for e in eigenvalues[:5]]},
        recommendation="Low Fiedler value suggests a bridge/bottleneck between graph regions; add redundant paths to improve resilience" if fiedler < 1.0 else "Graph is well-connected; no critical bottleneck detected",
        severity="warn" if fiedler < 0.5 else "info", confidence=min(0.8, 0.5 + 0.1 * (1.0 / max(fiedler, 0.01))),
    ))
    extra = {"runtime_ms": _runtime_ms(timer), "fiedler": round(fiedler, 6)}
    return _finalize(plugin_id, ctx, df, sample_meta, f"Fiedler value: {fiedler:.4f}", findings, [], extra_metrics=extra)


def _handler_graph_wavelet_multiscale_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """71. Graph wavelets at multiple scales for localized anomalies."""
    _log_start(ctx, plugin_id, df, config, inferred)
    if not HAS_NETWORKX:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "networkx_unavailable")
    edges, sc, dc = _build_edges(df, inferred)
    if edges.empty:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_graph_edges")
    G = _build_nx_undirected(edges)
    if G is None or G.number_of_nodes() < 5:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "graph_too_small")
    # Use largest connected component
    largest_cc = max(nx.connected_components(G), key=len)
    H = G.subgraph(largest_cc).copy()
    n = H.number_of_nodes()
    if n < 5:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "component_too_small")
    _ensure_budget(timer)
    # Graph wavelet: use heat kernel at different scales
    L = nx.laplacian_matrix(H).astype(float)
    dense_L = L.toarray() if hasattr(L, "toarray") else np.array(L)
    try:
        eigvals, eigvecs = np.linalg.eigh(dense_L)
    except Exception:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "eigendecomposition_failed")
    # Signal: node degree
    nodes = list(H.nodes())
    signal = np.array([float(H.degree(nd)) for nd in nodes], dtype=float)
    signal = (signal - np.mean(signal)) / max(np.std(signal), 1e-9)
    scales = [0.5, 1.0, 2.0, 5.0, 10.0]
    anomalies_by_scale = {}
    for scale in scales:
        # Wavelet coefficients via heat kernel: g(s*lambda) = exp(-s*lambda)
        kernel = np.exp(-scale * np.maximum(eigvals, 0))
        coeffs = eigvecs @ np.diag(kernel) @ eigvecs.T @ signal
        coeff_std = float(np.std(coeffs))
        threshold = 2.5 * max(coeff_std, 1e-9)
        anomalous = [(nodes[i], round(float(coeffs[i]), 4)) for i in range(n) if abs(coeffs[i]) > threshold]
        anomalies_by_scale[f"scale_{scale}"] = {"n_anomalies": len(anomalous), "top": anomalous[:5]}
        _ensure_budget(timer)
    total_anomalies = sum(v["n_anomalies"] for v in anomalies_by_scale.values())
    findings = []
    if total_anomalies > 0:
        findings.append(_make_finding(
            plugin_id, "wavelet_anomaly", "Graph wavelet anomalies at multiple scales",
            f"{total_anomalies} localized anomalies across {len(scales)} scales",
            "Graph wavelet coefficients identify nodes that are anomalous relative to their neighborhood at different resolution scales",
            {"scales_analyzed": len(scales), "total_anomalies": total_anomalies, "by_scale": anomalies_by_scale},
            recommendation="Investigate nodes flagged at multiple scales as they represent robust localized anomalies in the workflow graph",
            severity="warn" if total_anomalies > 5 else "info", confidence=min(0.75, 0.4 + 0.05 * total_anomalies),
        ))
    extra = {"runtime_ms": _runtime_ms(timer), "total_anomalies": total_anomalies}
    return _finalize(plugin_id, ctx, df, sample_meta, f"Graph wavelets: {total_anomalies} anomalies", findings, [], extra_metrics=extra)


def _handler_cheeger_bottleneck_v1(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """72. Cheeger constant (isoperimetric number) for bottleneck detection."""
    _log_start(ctx, plugin_id, df, config, inferred)
    if not HAS_NETWORKX:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "networkx_unavailable")
    edges, sc, dc = _build_edges(df, inferred)
    if edges.empty:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_graph_edges")
    G = _build_nx_undirected(edges)
    if G is None or G.number_of_nodes() < 4:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "graph_too_small")
    largest_cc = max(nx.connected_components(G), key=len)
    H = G.subgraph(largest_cc).copy()
    n = H.number_of_nodes()
    if n < 4:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "component_too_small")
    _ensure_budget(timer)
    # Approximate Cheeger constant via Fiedler vector partitioning (Cheeger inequality)
    eigenvalues = _laplacian_eigenvalues(H, k=3)
    if len(eigenvalues) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "eigenvalue_failed")
    fiedler_val = float(eigenvalues[1])
    # Compute Fiedler vector for partition
    dense_L = nx.laplacian_matrix(H).astype(float)
    dense_L = dense_L.toarray() if hasattr(dense_L, "toarray") else np.array(dense_L)
    try:
        eigvals_full, eigvecs_full = np.linalg.eigh(dense_L)
    except Exception:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "eigendecomposition_failed")
    fiedler_vec = eigvecs_full[:, 1]
    nodes = list(H.nodes())
    # Sweep cut: try different thresholds on Fiedler vector
    sorted_idx = np.argsort(fiedler_vec)
    best_cheeger = float("inf")
    best_cut_size = 0
    vol_total = sum(dict(H.degree()).values())
    for cut_pos in range(1, n):
        S = set(nodes[sorted_idx[i]] for i in range(cut_pos))
        vol_S = sum(H.degree(nd) for nd in S)
        if vol_S == 0 or vol_S >= vol_total:
            continue
        boundary = sum(1 for u in S for v in H.neighbors(u) if v not in S)
        cheeger = boundary / min(vol_S, vol_total - vol_S)
        if cheeger < best_cheeger:
            best_cheeger = cheeger
            best_cut_size = cut_pos
        _ensure_budget(timer)
    if not math.isfinite(best_cheeger):
        best_cheeger = fiedler_val / 2.0
    # Cheeger inequality: fiedler/2 <= h(G) <= sqrt(2 * fiedler)
    cheeger_lb = fiedler_val / 2.0
    cheeger_ub = math.sqrt(2.0 * max(fiedler_val, 0))
    findings = []
    findings.append(_make_finding(
        plugin_id, "cheeger", f"Cheeger constant ~ {best_cheeger:.4f}",
        f"Approximate Cheeger constant = {best_cheeger:.4f} (Fiedler bounds: [{cheeger_lb:.4f}, {cheeger_ub:.4f}]); best cut at position {best_cut_size}/{n}",
        "A low Cheeger constant means there is a narrow bottleneck splitting the workflow graph into loosely connected halves",
        {"cheeger_approx": round(best_cheeger, 6), "fiedler_value": round(fiedler_val, 6), "cheeger_lower_bound": round(cheeger_lb, 6), "cheeger_upper_bound": round(cheeger_ub, 6), "n_nodes": n, "best_cut_position": best_cut_size},
        recommendation="Low Cheeger constant indicates a process bottleneck; add cross-links between the two halves to improve flow" if best_cheeger < 0.5 else "Graph connectivity is adequate; no severe bottleneck detected",
        severity="warn" if best_cheeger < 0.3 else "info", confidence=min(0.8, 0.5 + 0.3 * (1.0 / max(best_cheeger, 0.01))),
    ))
    extra = {"runtime_ms": _runtime_ms(timer), "cheeger": round(best_cheeger, 6)}
    return _finalize(plugin_id, ctx, df, sample_meta, f"Cheeger constant: {best_cheeger:.4f}", findings, [], extra_metrics=extra)


# ---------------------------------------------------------------------------
# HANDLERS registry
# ---------------------------------------------------------------------------

HANDLERS: dict[str, Callable[..., PluginResult]] = {
    "analysis_persistent_homology_regimes_v1": _handler_persistent_homology_regimes_v1,
    "analysis_mapper_subpopulations_v1": _handler_mapper_subpopulations_v1,
    "analysis_persistence_landscape_features_v1": _handler_persistence_landscape_features_v1,
    "analysis_vietoris_rips_higher_order_v1": _handler_vietoris_rips_higher_order_v1,
    "analysis_euler_characteristic_curve_v1": _handler_euler_characteristic_curve_v1,
    "analysis_sheaf_consistency_v1": _handler_sheaf_consistency_v1,
    "analysis_functorial_migration_v1": _handler_functorial_migration_v1,
    "analysis_operad_composition_v1": _handler_operad_composition_v1,
    "analysis_grothendieck_coverage_v1": _handler_grothendieck_coverage_v1,
    "analysis_lyapunov_exponent_chaos_v1": _handler_lyapunov_exponent_chaos_v1,
    "analysis_fractal_dimension_attractor_v1": _handler_fractal_dimension_attractor_v1,
    "analysis_takens_embedding_v1": _handler_takens_embedding_v1,
    "analysis_job_shop_scheduling_bound_v1": _handler_job_shop_scheduling_bound_v1,
    "analysis_bin_packing_utilization_v1": _handler_bin_packing_utilization_v1,
    "analysis_csp_binding_rules_v1": _handler_csp_binding_rules_v1,
    "analysis_min_cost_flow_throughput_v1": _handler_min_cost_flow_throughput_v1,
    "analysis_kolmogorov_complexity_ncd_v1": _handler_kolmogorov_complexity_ncd_v1,
    "analysis_renyi_entropy_spectrum_v1": _handler_renyi_entropy_spectrum_v1,
    "analysis_pid_synergy_redundancy_v1": _handler_pid_synergy_redundancy_v1,
    "analysis_excess_entropy_memory_v1": _handler_excess_entropy_memory_v1,
    "analysis_channel_capacity_bound_v1": _handler_channel_capacity_bound_v1,
    "analysis_spectral_graph_fiedler_v1": _handler_spectral_graph_fiedler_v1,
    "analysis_graph_wavelet_multiscale_v1": _handler_graph_wavelet_multiscale_v1,
    "analysis_cheeger_bottleneck_v1": _handler_cheeger_bottleneck_v1,
}
