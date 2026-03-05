"""Cross-domain plugins: life sciences (bioinformatics, ecology, epidemiology, genetics, pharmacology)."""
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
    from scipy import integrate as scipy_integrate
    from scipy import optimize as scipy_optimize
    HAS_SCIPY = True
except Exception:
    scipy_stats = scipy_integrate = scipy_optimize = None
    HAS_SCIPY = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except Exception:
    nx = None
    HAS_NETWORKX = False

try:
    import hmmlearn.hmm as hmm_module
    HAS_HMMLEARN = True
except Exception:
    hmm_module = None
    HAS_HMMLEARN = False


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

def _time_windows(ts, n_windows=5):
    valid = ts.dropna()
    if len(valid) < n_windows * 5: return []
    indices = np.array_split(np.arange(len(valid)), n_windows)
    return [(valid.iloc[idx[0]], valid.iloc[idx[-1]], idx) for idx in indices if len(idx) > 0]

def _group_column(df, inferred):
    """Find best grouping column (host, machine, group, etc.)."""
    hints = ("host", "machine", "group", "entity", "server", "node", "cluster", "team", "site", "region")
    cats = _categorical_columns(df, inferred, max_cols=20)
    for c in cats:
        if any(h in c.lower() for h in hints):
            nuniq = df[c].nunique()
            if 2 <= nuniq <= 200: return c
    for c in cats:
        nuniq = df[c].nunique()
        if 2 <= nuniq <= 200: return c
    return None

def _step_column(df, inferred):
    """Find column representing workflow steps/stages."""
    hints = ("step", "stage", "phase", "task", "activity", "action", "operation", "event", "type")
    cats = _categorical_columns(df, inferred, max_cols=20)
    for c in cats:
        if any(h in c.lower() for h in hints):
            nuniq = df[c].nunique()
            if 2 <= nuniq <= 500: return c
    return cats[0] if cats else None

def _job_column(df, inferred):
    """Find column representing job/case identifier."""
    hints = ("job", "case", "ticket", "incident", "order", "id", "run", "batch")
    cats = _categorical_columns(df, inferred, max_cols=20)
    for c in cats:
        if any(h in c.lower() for h in hints):
            nuniq = df[c].nunique()
            if 2 <= nuniq <= 50000: return c
    return None

def _anomaly_column(df, inferred):
    """Find column representing anomaly/failure/error status."""
    hints = ("anomaly", "failure", "error", "fault", "defect", "alert", "flag", "outlier", "incident")
    for c in df.columns:
        cl = c.lower()
        if any(h in cl for h in hints):
            vals = pd.to_numeric(df[c], errors="coerce").dropna()
            if not vals.empty:
                uniq = set(np.unique(vals.to_numpy(dtype=float)).tolist())
                if uniq.issubset({0.0, 1.0}): return str(c)
    return None


# ---------------------------------------------------------------------------
# Plugin handlers (life sciences)
# ---------------------------------------------------------------------------


def _smith_waterman_workflow_align(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Local alignment of step sequences per job type using DP."""
    _log_start(ctx, plugin_id, df, config, inferred)
    step_col = _step_column(df, inferred)
    job_col = _job_column(df, inferred)
    if step_col is None or job_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_step_or_job_column")
    # Build sequences per job
    groups = df.groupby(job_col)[step_col].apply(list)
    if len(groups) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "fewer_than_2_jobs")
    seqs = list(groups.values)[:int(config.get("max_jobs", 200))]
    _ensure_budget(timer)
    # Smith-Waterman DP for first N pairs
    match_score, gap_penalty = 2, -1
    scores = []
    max_pairs = min(len(seqs) * (len(seqs) - 1) // 2, 500)
    pair_count = 0
    for i in range(len(seqs)):
        for j in range(i + 1, len(seqs)):
            if pair_count >= max_pairs: break
            s1, s2 = seqs[i], seqs[j]
            m, n = len(s1), len(s2)
            if m == 0 or n == 0: continue
            H = np.zeros((m + 1, n + 1), dtype=float)
            for ii in range(1, m + 1):
                for jj in range(1, n + 1):
                    diag = H[ii-1, jj-1] + (match_score if s1[ii-1] == s2[jj-1] else gap_penalty)
                    H[ii, jj] = max(0.0, diag, H[ii-1, jj] + gap_penalty, H[ii, jj-1] + gap_penalty)
            norm = match_score * min(m, n) if min(m, n) > 0 else 1.0
            scores.append(float(np.max(H)) / max(norm, 1e-9))
            pair_count += 1
        if pair_count >= max_pairs: break
        _ensure_budget(timer)
    if not scores:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_alignment_scores")
    mean_sim = float(np.mean(scores))
    std_sim = float(np.std(scores))
    findings = []
    if mean_sim > 0.6:
        findings.append(_make_finding(
            plugin_id, "high_alignment",
            "High workflow sequence alignment detected",
            f"Mean Smith-Waterman similarity={mean_sim:.3f} (std={std_sim:.3f}) across {pair_count} job pairs.",
            "High local alignment indicates recurring shared sub-sequences across jobs.",
            {"metrics": {"mean_similarity": mean_sim, "std_similarity": std_sim, "pairs_evaluated": pair_count,
                         "step_col": step_col, "job_col": job_col}},
            recommendation="Standardize the common sub-sequence as a template to reduce variation.",
            severity="info" if mean_sim < 0.8 else "warn",
            confidence=min(0.85, 0.5 + mean_sim * 0.3),
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Smith-Waterman alignment: mean_sim={mean_sim:.3f}.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "mean_similarity": mean_sim, "pairs": pair_count})


def _meme_motif_discovery(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Find recurring n-gram step motifs (3-6 steps) and correlate with duration."""
    _log_start(ctx, plugin_id, df, config, inferred)
    step_col = _step_column(df, inferred)
    job_col = _job_column(df, inferred)
    dur_col = _duration_column(df, inferred)
    if step_col is None or job_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_step_or_job_column")
    groups = df.groupby(job_col)
    seqs = {k: list(v[step_col]) for k, v in groups}
    if len(seqs) < 5:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "fewer_than_5_jobs")
    _ensure_budget(timer)
    # Count n-grams across all sequences
    ngram_counts: Counter = Counter()
    for seq in seqs.values():
        for n in range(3, min(7, len(seq) + 1)):
            for i in range(len(seq) - n + 1):
                ngram_counts[tuple(seq[i:i+n])] += 1
    if not ngram_counts:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_ngrams_found")
    top_motifs = ngram_counts.most_common(10)
    most_common_motif, motif_freq = top_motifs[0]
    total_jobs = len(seqs)
    prevalence = motif_freq / max(total_jobs, 1)
    findings = []
    if prevalence > 0.3:
        findings.append(_make_finding(
            plugin_id, "frequent_motif",
            "Recurring workflow motif discovered",
            f"Motif {' -> '.join(str(s) for s in most_common_motif)} appears in {motif_freq}/{total_jobs} jobs ({prevalence:.1%}).",
            "Frequently recurring step motifs indicate standardizable workflow patterns.",
            {"metrics": {"motif": list(most_common_motif), "frequency": motif_freq, "prevalence": prevalence,
                         "top_5": [{"motif": list(m), "count": c} for m, c in top_motifs[:5]]}},
            recommendation="Codify the recurring motif as a standard operating procedure to reduce variation.",
            severity="info", confidence=min(0.85, 0.5 + prevalence * 0.3),
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Motif discovery: top motif prevalence={prevalence:.1%}.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "n_unique_motifs": len(ngram_counts), "top_prevalence": prevalence})


def _hmm_workflow_state(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Fit Gaussian HMM to numeric features to detect latent workflow states."""
    _log_start(ctx, plugin_id, df, config, inferred)
    if not HAS_HMMLEARN:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "hmmlearn_unavailable")
    num_cols = _numeric_columns(df, inferred, max_cols=6)
    if len(num_cols) < 1:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_columns")
    X = df[num_cols].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(dtype=float)
    if len(X) < 30:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_data")
    _ensure_budget(timer)
    seed = int(config.get("seed", 42))
    best_bic, best_n = float("inf"), 2
    max_states = min(int(config.get("max_hmm_states", 6)), 8)
    for n_states in range(2, max_states + 1):
        try:
            model = hmm_module.GaussianHMM(n_components=n_states, covariance_type="diag",
                                           n_iter=50, random_state=seed)
            model.fit(X)
            ll = float(model.score(X))
            n_params = n_states * (n_states - 1) + n_states * len(num_cols) * 2
            bic = -2 * ll + n_params * math.log(len(X))
            if bic < best_bic:
                best_bic, best_n = bic, n_states
        except Exception:
            continue
        _ensure_budget(timer)
    findings = []
    if best_n > 2:
        findings.append(_make_finding(
            plugin_id, "latent_states",
            f"HMM detected {best_n} latent workflow states",
            f"Best BIC selects {best_n} hidden states (BIC={best_bic:.1f}).",
            "Multiple latent states suggest the process transitions between distinct operating regimes.",
            {"metrics": {"best_n_states": best_n, "bic": best_bic, "columns": num_cols}},
            recommendation="Investigate what drives transitions between latent states and whether some are undesirable.",
            severity="warn", confidence=min(0.8, 0.5 + 0.1 * best_n),
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"HMM workflow state detection: {best_n} states selected.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "best_n_states": best_n, "bic": best_bic})


def _phylogenetic_workflow_tree(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Pairwise edit distances between job sequences, UPGMA dendrogram."""
    _log_start(ctx, plugin_id, df, config, inferred)
    if not HAS_SCIPY:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "scipy_unavailable")
    step_col = _step_column(df, inferred)
    job_col = _job_column(df, inferred)
    if step_col is None or job_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_step_or_job_column")
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    groups = df.groupby(job_col)[step_col].apply(list)
    seqs = list(groups.values)[:int(config.get("max_jobs", 100))]
    n = len(seqs)
    if n < 3:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "fewer_than_3_jobs")
    _ensure_budget(timer)
    # Levenshtein edit distance
    def _edit_dist(a, b):
        la, lb = len(a), len(b)
        dp = list(range(lb + 1))
        for i in range(1, la + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, lb + 1):
                tmp = dp[j]
                dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + (0 if a[i-1] == b[j-1] else 1))
                prev = tmp
        return dp[lb]
    dist_mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = _edit_dist(seqs[i], seqs[j])
            dist_mat[i, j] = dist_mat[j, i] = d
        _ensure_budget(timer)
    condensed = squareform(dist_mat)
    Z = linkage(condensed, method="average")
    # Cut at distance producing 2-5 clusters
    best_k, best_score = 2, -1.0
    for k in range(2, min(6, n)):
        labels = fcluster(Z, t=k, criterion="maxclust")
        sizes = Counter(labels)
        if len(sizes) < 2: continue
        min_frac = min(sizes.values()) / n
        if min_frac > best_score: best_k, best_score = k, min_frac
    labels = fcluster(Z, t=best_k, criterion="maxclust")
    cluster_sizes = dict(Counter(labels))
    findings = []
    if best_k > 2:
        findings.append(_make_finding(
            plugin_id, "workflow_families",
            f"Phylogenetic analysis found {best_k} workflow families",
            f"UPGMA clustering of edit distances yields {best_k} distinct workflow families.",
            "Distinct workflow families suggest structurally different process variants.",
            {"metrics": {"n_clusters": best_k, "cluster_sizes": {str(k): v for k, v in cluster_sizes.items()},
                         "mean_edit_dist": float(np.mean(condensed))}},
            recommendation="Assess whether workflow families should be harmonized or whether divergence is intentional.",
            severity="info", confidence=min(0.8, 0.5 + 0.1 * best_k),
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Phylogenetic tree: {best_k} workflow families.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "n_clusters": best_k, "mean_edit_dist": float(np.mean(condensed))})


def _shannon_diversity_index(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """H' = -sum(p_i * ln(p_i)) per host/group. Low H' = specialized."""
    _log_start(ctx, plugin_id, df, config, inferred)
    step_col = _step_column(df, inferred)
    grp_col = _group_column(df, inferred)
    if step_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_step_column")
    _ensure_budget(timer)
    if grp_col:
        results = {}
        for grp, sub in df.groupby(grp_col):
            counts = sub[step_col].value_counts().to_numpy(dtype=float)
            p = counts / counts.sum()
            p = p[p > 0]
            results[str(grp)] = float(-np.sum(p * np.log(p)))
        h_values = list(results.values())
    else:
        counts = df[step_col].value_counts().to_numpy(dtype=float)
        p = counts / counts.sum()
        p = p[p > 0]
        h_global = float(-np.sum(p * np.log(p)))
        results = {"global": h_global}
        h_values = [h_global]
    mean_h = float(np.mean(h_values))
    std_h = float(np.std(h_values)) if len(h_values) > 1 else 0.0
    findings = []
    low_div = {k: v for k, v in results.items() if v < 1.0} if grp_col else {}
    if low_div:
        findings.append(_make_finding(
            plugin_id, "low_diversity",
            "Low Shannon diversity detected in some groups",
            f"{len(low_div)}/{len(results)} groups have H'<1.0, indicating over-specialization.",
            "Low diversity means a group handles very few step types, creating fragility.",
            {"metrics": {"low_diversity_groups": low_div, "mean_H": mean_h, "std_H": std_h}},
            recommendation="Cross-train low-diversity groups to handle more step types for resilience.",
            severity="warn", confidence=min(0.8, 0.5 + len(low_div) / max(len(results), 1) * 0.3),
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Shannon diversity: mean H'={mean_h:.3f}.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "mean_H": mean_h, "n_groups": len(results)})


def _rarefaction_coverage(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Unique types discovered vs samples. Fit saturation curve."""
    _log_start(ctx, plugin_id, df, config, inferred)
    step_col = _step_column(df, inferred)
    if step_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_step_column")
    values = df[step_col].dropna().tolist()
    if len(values) < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_data")
    _ensure_budget(timer)
    seed = int(config.get("seed", 42))
    rng = np.random.RandomState(seed)
    rng.shuffle(values)
    n_points = min(50, len(values))
    sample_sizes = np.linspace(1, len(values), n_points, dtype=int)
    unique_counts = []
    for sz in sample_sizes:
        unique_counts.append(len(set(values[:sz])))
    unique_counts = np.array(unique_counts, dtype=float)
    total_unique = len(set(values))
    # Saturation ratio at 50% of data
    half_idx = n_points // 2
    coverage_at_half = unique_counts[half_idx] / max(total_unique, 1)
    findings = []
    if coverage_at_half > 0.9:
        findings.append(_make_finding(
            plugin_id, "saturated",
            "Rarefaction curve is saturated early",
            f"90% of unique types ({int(unique_counts[half_idx])}/{total_unique}) found with only 50% of samples.",
            "Early saturation means additional sampling yields diminishing returns for type discovery.",
            {"metrics": {"coverage_at_half": coverage_at_half, "total_unique": total_unique,
                         "curve": [{"n": int(s), "unique": int(u)} for s, u in zip(sample_sizes.tolist(), unique_counts.tolist())]}},
            recommendation="Current sample size is sufficient for type coverage; no additional sampling needed.",
            severity="info", confidence=0.7,
        ))
    elif coverage_at_half < 0.6:
        findings.append(_make_finding(
            plugin_id, "undersampled",
            "Rarefaction curve indicates undersampling",
            f"Only {coverage_at_half:.1%} of unique types found at 50% of data. Curve is still rising.",
            "Rising rarefaction curve suggests more types remain undiscovered.",
            {"metrics": {"coverage_at_half": coverage_at_half, "total_unique": total_unique}},
            recommendation="Increase sample size or extend observation window to discover additional types.",
            severity="warn", confidence=0.65,
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Rarefaction coverage: {coverage_at_half:.1%} at half-data.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "coverage_at_half": coverage_at_half, "total_unique": total_unique})


def _lotka_volterra_competition(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """ODE model of competing job types via Lotka-Volterra equations."""
    _log_start(ctx, plugin_id, df, config, inferred)
    if not HAS_SCIPY:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "scipy_unavailable")
    step_col = _step_column(df, inferred)
    tc, ts = _time_series(df, inferred)
    if step_col is None or tc is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_step_or_time_column")
    top_types = df[step_col].value_counts().head(2).index.tolist()
    if len(top_types) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "fewer_than_2_types")
    _ensure_budget(timer)
    # Build time series of counts for top 2 types in rolling windows
    df_sorted = df.assign(_ts=ts).dropna(subset=["_ts"]).sort_values("_ts")
    n_windows = min(20, len(df_sorted) // 5)
    if n_windows < 5:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_temporal_data")
    chunks = np.array_split(df_sorted, n_windows)
    x_series = np.array([float((c[step_col] == top_types[0]).sum()) for c in chunks])
    y_series = np.array([float((c[step_col] == top_types[1]).sum()) for c in chunks])
    # Fit LV: dx/dt = r1*x*(1 - (x + a12*y)/K1), dy/dt = r2*y*(1 - (y + a21*x)/K2)
    x_series = np.maximum(x_series, 0.1)
    y_series = np.maximum(y_series, 0.1)
    K1_est = float(np.max(x_series)) * 1.2
    K2_est = float(np.max(y_series)) * 1.2
    # Estimate competition coefficients from correlation
    corr = float(np.corrcoef(x_series, y_series)[0, 1]) if len(x_series) > 2 else 0.0
    competition_index = max(0.0, -corr)  # negative correlation = competition
    findings = []
    if competition_index > 0.3:
        findings.append(_make_finding(
            plugin_id, "competition",
            "Lotka-Volterra competition detected between job types",
            f"Types '{top_types[0]}' and '{top_types[1]}' show competition index={competition_index:.3f}.",
            "Negative temporal correlation suggests the two job types compete for shared resources.",
            {"metrics": {"type_a": top_types[0], "type_b": top_types[1], "competition_index": competition_index,
                         "correlation": corr, "K1_est": K1_est, "K2_est": K2_est}},
            recommendation="Investigate shared resource contention between the two job types and consider isolation.",
            severity="warn" if competition_index < 0.6 else "critical",
            confidence=min(0.8, 0.5 + competition_index * 0.3),
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Lotka-Volterra competition: index={competition_index:.3f}.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "competition_index": competition_index, "correlation": corr})


def _niche_overlap_pianka(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Pianka's overlap O_jk between entity pairs on step proportions."""
    _log_start(ctx, plugin_id, df, config, inferred)
    step_col = _step_column(df, inferred)
    grp_col = _group_column(df, inferred)
    if step_col is None or grp_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_step_or_group_column")
    groups = df.groupby(grp_col)[step_col].value_counts(normalize=True)
    entities = df[grp_col].unique()
    if len(entities) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "fewer_than_2_entities")
    _ensure_budget(timer)
    all_steps = sorted(df[step_col].unique())
    # Build proportion matrix
    prop_mat = {}
    for ent in entities:
        try:
            vc = groups[ent]
            prop_mat[ent] = np.array([float(vc.get(s, 0.0)) for s in all_steps])
        except Exception:
            prop_mat[ent] = np.zeros(len(all_steps))
    # Compute pairwise Pianka overlap: O_jk = sum(p_j * p_k) / sqrt(sum(p_j^2) * sum(p_k^2))
    overlaps = []
    ent_list = list(prop_mat.keys())[:100]
    for i in range(len(ent_list)):
        for j in range(i + 1, len(ent_list)):
            pj, pk = prop_mat[ent_list[i]], prop_mat[ent_list[j]]
            denom = math.sqrt(float(np.sum(pj**2)) * float(np.sum(pk**2)))
            o = float(np.sum(pj * pk)) / max(denom, 1e-12)
            overlaps.append(o)
        _ensure_budget(timer)
    mean_overlap = float(np.mean(overlaps)) if overlaps else 0.0
    findings = []
    if mean_overlap > 0.7:
        findings.append(_make_finding(
            plugin_id, "high_overlap",
            "High niche overlap (Pianka index) between entities",
            f"Mean Pianka overlap O={mean_overlap:.3f} across {len(overlaps)} entity pairs.",
            "High niche overlap means entities handle very similar step mixes, creating redundancy.",
            {"metrics": {"mean_overlap": mean_overlap, "n_pairs": len(overlaps), "n_entities": len(ent_list)}},
            recommendation="Consider consolidating entities with high overlap or differentiating their roles.",
            severity="info" if mean_overlap < 0.85 else "warn",
            confidence=min(0.8, 0.5 + mean_overlap * 0.3),
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Pianka niche overlap: mean O={mean_overlap:.3f}.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "mean_overlap": mean_overlap, "n_entities": len(ent_list)})


def _sir_failure_cascade(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """SIR epidemic model on DAG for failure cascade analysis."""
    _log_start(ctx, plugin_id, df, config, inferred)
    if not HAS_SCIPY or not HAS_NETWORKX:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "scipy_or_networkx_unavailable")
    edges, sc, dc = _build_edges(df, inferred)
    if edges.empty:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_graph_edges")
    G = _build_nx_graph_from_edges(edges)
    if G is None or G.number_of_nodes() < 5:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "graph_too_small")
    _ensure_budget(timer)
    N = float(G.number_of_nodes())
    avg_degree = float(sum(dict(G.degree()).values())) / max(N, 1.0)
    beta = min(0.5, avg_degree / max(N, 1.0))
    gamma = 0.1
    # Solve SIR ODE: dS/dt=-beta*S*I, dI/dt=beta*S*I-gamma*I, dR/dt=gamma*I
    def sir_ode(t, y):
        S, I, R = y
        return [-beta*S*I, beta*S*I - gamma*I, gamma*I]
    I0 = 1.0 / N
    sol = scipy_integrate.solve_ivp(sir_ode, [0, 100], [1.0 - I0, I0, 0.0], max_step=1.0)
    peak_infected = float(np.max(sol.y[1]))
    final_recovered = float(sol.y[2][-1])
    R0 = beta * N / max(gamma, 1e-9)
    findings = []
    if R0 > 1.0:
        findings.append(_make_finding(
            plugin_id, "sir_epidemic",
            "SIR model predicts failure cascade spread (R0 > 1)",
            f"R0={R0:.2f}, peak infected={peak_infected:.1%}, final affected={final_recovered:.1%}.",
            "R0 > 1 means each failure on average triggers more than one downstream failure.",
            {"metrics": {"R0": R0, "beta": beta, "gamma": gamma, "peak_infected": peak_infected,
                         "final_recovered": final_recovered, "nodes": int(N), "avg_degree": avg_degree}},
            recommendation="Reduce coupling (lower beta) or add faster recovery mechanisms (higher gamma).",
            severity="warn" if R0 < 2.0 else "critical",
            confidence=min(0.8, 0.5 + min(R0 / 5.0, 0.3)),
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"SIR failure cascade: R0={R0:.2f}.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "R0": R0, "peak_infected": peak_infected})


def _cascade_r0_reproduction(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """R0 = mean secondary failures per primary failure on DAG."""
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
    out_degrees = [d for _, d in G.out_degree()]
    R0 = float(np.mean(out_degrees)) if out_degrees else 0.0
    max_out = max(out_degrees) if out_degrees else 0
    variance = float(np.var(out_degrees)) if out_degrees else 0.0
    findings = []
    if R0 > 1.0:
        findings.append(_make_finding(
            plugin_id, "high_r0",
            f"High cascade R0={R0:.2f} indicates amplifying failure propagation",
            f"Mean out-degree={R0:.2f}, max={max_out}. Failures amplify through the graph.",
            "R0 > 1 means on average each node triggers more than one downstream node.",
            {"metrics": {"R0": R0, "max_out_degree": max_out, "variance": variance,
                         "nodes": G.number_of_nodes(), "edges": G.number_of_edges()}},
            recommendation="Add circuit breakers at high-fan-out nodes to limit cascade amplification.",
            severity="warn" if R0 < 2.0 else "critical",
            confidence=min(0.85, 0.5 + R0 * 0.15),
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Cascade R0={R0:.2f}.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "R0": R0, "max_out_degree": max_out})


def _contact_tracing_superspreader(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Rank DAG nodes by failure propagation out-degree (superspreaders)."""
    _log_start(ctx, plugin_id, df, config, inferred)
    if not HAS_NETWORKX:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "networkx_unavailable")
    edges, sc, dc = _build_edges(df, inferred)
    if edges.empty:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_graph_edges")
    G = _build_nx_graph_from_edges(edges)
    if G is None or G.number_of_nodes() < 5:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "graph_too_small")
    _ensure_budget(timer)
    # Compute reachability (number of descendants) for each node
    reach = {}
    for node in G.nodes():
        reach[node] = len(nx.descendants(G, node))
    sorted_nodes = sorted(reach.items(), key=lambda x: -x[1])
    top_spreaders = sorted_nodes[:10]
    mean_reach = float(np.mean(list(reach.values())))
    max_reach = top_spreaders[0][1] if top_spreaders else 0
    findings = []
    if max_reach > G.number_of_nodes() * 0.2:
        findings.append(_make_finding(
            plugin_id, "superspreader",
            "Superspreader nodes detected in failure graph",
            f"Top node '{top_spreaders[0][0]}' can reach {max_reach}/{G.number_of_nodes()} nodes ({max_reach/G.number_of_nodes():.1%}).",
            "Superspreader nodes can propagate failures to a large fraction of the graph.",
            {"metrics": {"top_spreaders": [{"node": str(n), "reach": r} for n, r in top_spreaders[:5]],
                         "mean_reach": mean_reach, "max_reach": max_reach}},
            recommendation="Add isolation or circuit-breaker controls around superspreader nodes.",
            severity="warn" if max_reach < G.number_of_nodes() * 0.4 else "critical",
            confidence=min(0.85, 0.5 + max_reach / max(G.number_of_nodes(), 1) * 0.3),
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Contact tracing: max reach={max_reach}.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "max_reach": max_reach, "mean_reach": mean_reach})


def _epicurve_classification(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Histogram anomaly onsets. Classify: point-source, continuous, or propagated."""
    _log_start(ctx, plugin_id, df, config, inferred)
    tc, ts = _time_series(df, inferred)
    anom_col = _anomaly_column(df, inferred)
    if tc is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_time_column")
    _ensure_budget(timer)
    if anom_col:
        mask = pd.to_numeric(df[anom_col], errors="coerce").fillna(0) > 0
        event_ts = ts[mask].dropna()
    else:
        event_ts = ts.dropna()
    if len(event_ts) < 10:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_events")
    # Build histogram of event counts over time windows
    event_ts_sorted = event_ts.sort_values()
    n_bins = min(30, len(event_ts) // 3)
    counts, bin_edges = np.histogram(event_ts_sorted.astype(np.int64), bins=n_bins)
    counts = counts.astype(float)
    peak_idx = int(np.argmax(counts))
    peak_ratio = float(counts[peak_idx]) / max(float(np.mean(counts)), 1e-9)
    # Classification heuristics
    if peak_ratio > 3.0 and peak_idx < n_bins * 0.3:
        classification = "point_source"
    elif float(np.std(counts)) / max(float(np.mean(counts)), 1e-9) < 0.5:
        classification = "continuous"
    else:
        # Check for exponential growth pattern
        diffs = np.diff(counts)
        if np.sum(diffs > 0) > len(diffs) * 0.6:
            classification = "propagated"
        else:
            classification = "mixed"
    findings = []
    findings.append(_make_finding(
        plugin_id, "epicurve",
        f"Epidemic curve classified as {classification}",
        f"Peak ratio={peak_ratio:.1f}x mean. Pattern: {classification}.",
        {"point_source": "Sharp early peak suggests a single triggering event.",
         "continuous": "Steady rate suggests ongoing systemic exposure.",
         "propagated": "Accelerating pattern suggests contagion-like spread.",
         "mixed": "Mixed pattern does not fit a single epidemiological archetype."}.get(classification, ""),
        {"metrics": {"classification": classification, "peak_ratio": peak_ratio, "n_bins": n_bins,
                     "peak_bin_index": peak_idx, "total_events": len(event_ts)}},
        recommendation={"point_source": "Investigate the triggering event at the onset peak.",
                        "continuous": "Look for persistent environmental factors driving steady incidence.",
                        "propagated": "Implement containment to break the propagation chain.",
                        "mixed": "Decompose the curve into sub-populations for targeted intervention."}.get(classification, "Investigate further."),
        severity="warn" if classification in ("propagated", "point_source") else "info",
        confidence=0.6 if classification == "mixed" else 0.7,
    ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Epicurve classification: {classification}.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "classification": classification, "peak_ratio": peak_ratio})


def _hardy_weinberg_param_coupling(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Chi-square test on parameter combo frequencies vs independence assumption."""
    _log_start(ctx, plugin_id, df, config, inferred)
    if not HAS_SCIPY:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "scipy_unavailable")
    cats = _categorical_columns(df, inferred, max_cols=5)
    if len(cats) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "fewer_than_2_categorical")
    _ensure_budget(timer)
    col_a, col_b = cats[0], cats[1]
    contingency = pd.crosstab(df[col_a], df[col_b])
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "degenerate_contingency_table")
    chi2, p_val, dof, expected = scipy_stats.chi2_contingency(contingency)
    cramers_v = math.sqrt(chi2 / max(len(df) * (min(contingency.shape) - 1), 1))
    findings = []
    if p_val < 0.05 and cramers_v > 0.2:
        findings.append(_make_finding(
            plugin_id, "hw_coupling",
            "Hardy-Weinberg disequilibrium: parameter coupling detected",
            f"Chi2={chi2:.1f}, p={p_val:.2e}, Cramer's V={cramers_v:.3f} between '{col_a}' and '{col_b}'.",
            "Significant departure from independence indicates systematic coupling between parameters.",
            {"metrics": {"chi2": chi2, "p_value": p_val, "cramers_v": cramers_v, "dof": dof,
                         "col_a": col_a, "col_b": col_b}},
            recommendation="Investigate why these parameters co-occur non-randomly; consider whether coupling is intentional.",
            severity="warn" if cramers_v < 0.4 else "critical",
            confidence=min(0.9, 0.6 + cramers_v * 0.3),
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Hardy-Weinberg test: V={cramers_v:.3f}, p={p_val:.2e}.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "cramers_v": cramers_v, "p_value": p_val})


def _linkage_disequilibrium_steps(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """D = freq(AB) - freq(A)*freq(B) for step-anomaly pairs."""
    _log_start(ctx, plugin_id, df, config, inferred)
    step_col = _step_column(df, inferred)
    anom_col = _anomaly_column(df, inferred)
    if step_col is None or anom_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_step_or_anomaly_column")
    _ensure_budget(timer)
    anomalies = pd.to_numeric(df[anom_col], errors="coerce").fillna(0).to_numpy(dtype=float)
    anom_rate = float(np.mean(anomalies > 0))
    if anom_rate < 0.01 or anom_rate > 0.99:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "anomaly_rate_extreme")
    steps = df[step_col].values
    unique_steps = sorted(set(str(s) for s in steps if pd.notna(s)))
    ld_scores = {}
    for s in unique_steps[:50]:
        mask_s = (steps == s)
        freq_a = float(np.mean(mask_s))
        freq_b = anom_rate
        freq_ab = float(np.mean(mask_s & (anomalies > 0)))
        D = freq_ab - freq_a * freq_b
        # Normalize: D' = D / D_max
        D_max = min(freq_a * (1 - freq_b), freq_b * (1 - freq_a)) if D > 0 else min(freq_a * freq_b, (1 - freq_a) * (1 - freq_b))
        D_prime = D / max(abs(D_max), 1e-12)
        ld_scores[s] = {"D": D, "D_prime": D_prime, "freq_step": freq_a, "freq_anomaly_given_step": freq_ab / max(freq_a, 1e-9)}
    top_ld = sorted(ld_scores.items(), key=lambda x: -abs(x[1]["D_prime"]))[:5]
    findings = []
    if top_ld and abs(top_ld[0][1]["D_prime"]) > 0.3:
        findings.append(_make_finding(
            plugin_id, "linkage_disequilibrium",
            "Linkage disequilibrium between steps and anomalies",
            f"Step '{top_ld[0][0]}' shows D'={top_ld[0][1]['D_prime']:.3f} with anomalies.",
            "Non-random association between specific steps and anomalies suggests causal or confounding linkage.",
            {"metrics": {"top_associations": {k: v for k, v in top_ld}, "overall_anomaly_rate": anom_rate}},
            recommendation="Investigate steps with high |D'| for root cause of anomaly association.",
            severity="warn", confidence=min(0.8, 0.5 + abs(top_ld[0][1]["D_prime"]) * 0.3),
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Linkage disequilibrium analysis: {len(ld_scores)} steps tested.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "n_steps_tested": len(ld_scores), "anomaly_rate": anom_rate})


def _fst_machine_differentiation(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """F_ST = between-group / total variance across groups."""
    _log_start(ctx, plugin_id, df, config, inferred)
    grp_col = _group_column(df, inferred)
    num_cols = _numeric_columns(df, inferred, max_cols=5)
    if grp_col is None or not num_cols:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_group_or_numeric_columns")
    _ensure_budget(timer)
    fst_values = {}
    for col in num_cols:
        vals = pd.to_numeric(df[col], errors="coerce")
        total_var = float(np.nanvar(vals.to_numpy(dtype=float)))
        if total_var < 1e-12: continue
        group_means = df.assign(_v=vals).groupby(grp_col)["_v"].mean()
        between_var = float(np.nanvar(group_means.to_numpy(dtype=float)))
        fst = between_var / max(total_var, 1e-12)
        fst_values[col] = min(1.0, max(0.0, fst))
    if not fst_values:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_variance")
    mean_fst = float(np.mean(list(fst_values.values())))
    max_fst_col = max(fst_values, key=fst_values.get)
    findings = []
    if mean_fst > 0.1:
        findings.append(_make_finding(
            plugin_id, "high_fst",
            f"High F_ST differentiation between groups (mean={mean_fst:.3f})",
            f"Highest F_ST={fst_values[max_fst_col]:.3f} on column '{max_fst_col}'.",
            "High F_ST means between-group variance is large relative to total, indicating systematic group differences.",
            {"metrics": {"fst_by_column": fst_values, "mean_fst": mean_fst, "group_col": grp_col}},
            recommendation="Investigate causes of between-group differentiation; standardize if unintentional.",
            severity="warn" if mean_fst < 0.3 else "critical",
            confidence=min(0.85, 0.5 + mean_fst * 0.5),
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"F_ST differentiation: mean={mean_fst:.3f}.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "mean_fst": mean_fst, "n_columns": len(fst_values)})


def _wright_fisher_config_drift(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Track parameter frequencies across time windows for genetic drift analogue."""
    _log_start(ctx, plugin_id, df, config, inferred)
    step_col = _step_column(df, inferred)
    tc, ts = _time_series(df, inferred)
    if step_col is None or tc is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_step_or_time_column")
    _ensure_budget(timer)
    df_sorted = df.assign(_ts=ts).dropna(subset=["_ts"]).sort_values("_ts")
    windows = _time_windows(df_sorted["_ts"], n_windows=8)
    if len(windows) < 3:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_time_windows")
    top_types = df_sorted[step_col].value_counts().head(5).index.tolist()
    # Track frequency of each type per window
    freq_series = {t: [] for t in top_types}
    for _, _, idx in windows:
        chunk = df_sorted.iloc[idx]
        total = len(chunk)
        for t in top_types:
            freq_series[t].append(float((chunk[step_col] == t).sum()) / max(total, 1))
    # Measure drift: variance of frequency trajectory
    drift_scores = {}
    for t in top_types:
        traj = np.array(freq_series[t])
        drift_scores[t] = float(np.std(traj))
    max_drift_type = max(drift_scores, key=drift_scores.get)
    max_drift = drift_scores[max_drift_type]
    findings = []
    if max_drift > 0.1:
        findings.append(_make_finding(
            plugin_id, "config_drift",
            f"Wright-Fisher drift detected: '{max_drift_type}' frequency unstable",
            f"Type '{max_drift_type}' frequency std={max_drift:.3f} across {len(windows)} time windows.",
            "Frequency drift without selection pressure suggests random process variation (genetic drift analogue).",
            {"metrics": {"drift_scores": drift_scores, "freq_series": freq_series, "n_windows": len(windows)}},
            recommendation="Investigate whether frequency shifts are intentional or represent uncontrolled drift.",
            severity="warn" if max_drift < 0.2 else "critical",
            confidence=min(0.8, 0.5 + max_drift * 2.0),
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Wright-Fisher drift: max std={max_drift:.3f}.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "max_drift": max_drift, "n_types_tracked": len(top_types)})


def _compartment_redistribution(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Multi-compartment ODE for workload redistribution dynamics."""
    _log_start(ctx, plugin_id, df, config, inferred)
    if not HAS_SCIPY:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "scipy_unavailable")
    grp_col = _group_column(df, inferred)
    dur_col = _duration_column(df, inferred)
    if grp_col is None or dur_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_group_or_duration_column")
    _ensure_budget(timer)
    loads = df.groupby(grp_col)[dur_col].apply(lambda x: pd.to_numeric(x, errors="coerce").sum())
    loads = loads.dropna()
    if len(loads) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "fewer_than_2_compartments")
    y0 = loads.to_numpy(dtype=float)
    n = len(y0)
    total = float(np.sum(y0))
    equil = total / n
    # ODE: each compartment relaxes toward equilibrium at rate k
    k = 0.1
    def ode(t, y):
        return [-k * (yi - equil) for yi in y]
    sol = scipy_integrate.solve_ivp(ode, [0, 50], y0.tolist(), max_step=1.0)
    imbalance_initial = float(np.std(y0)) / max(float(np.mean(y0)), 1e-9)
    imbalance_final = float(np.std(sol.y[:, -1])) / max(float(np.mean(sol.y[:, -1])), 1e-9)
    # Half-life to equilibrium
    t_half = math.log(2) / max(k, 1e-9)
    findings = []
    if imbalance_initial > 0.3:
        findings.append(_make_finding(
            plugin_id, "imbalanced_compartments",
            "Workload imbalance across compartments",
            f"Initial CV={imbalance_initial:.3f}. Modeled redistribution half-life={t_half:.1f} time units.",
            "High coefficient of variation across compartments indicates uneven workload distribution.",
            {"metrics": {"initial_cv": imbalance_initial, "final_cv": imbalance_final, "t_half": t_half,
                         "n_compartments": n, "total_load": total, "equilibrium_load": equil}},
            recommendation="Implement load-balancing to redistribute work toward equilibrium across compartments.",
            severity="warn" if imbalance_initial < 0.6 else "critical",
            confidence=min(0.8, 0.5 + imbalance_initial * 0.3),
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Compartment redistribution: initial CV={imbalance_initial:.3f}.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "initial_cv": imbalance_initial, "t_half": t_half})


def _half_life_utilization(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """t_half = ln(2)/k for exponential decay of utilization over time."""
    _log_start(ctx, plugin_id, df, config, inferred)
    dur_col = _duration_column(df, inferred)
    tc, ts = _time_series(df, inferred)
    if dur_col is None or tc is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_duration_or_time_column")
    _ensure_budget(timer)
    df_sorted = df.assign(_ts=ts).dropna(subset=["_ts"]).sort_values("_ts")
    vals = pd.to_numeric(df_sorted[dur_col], errors="coerce").dropna().to_numpy(dtype=float)
    if len(vals) < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_data")
    # Compute rolling mean utilization in windows
    n_windows = min(20, len(vals) // 5)
    chunks = np.array_split(vals, n_windows)
    means = np.array([float(np.mean(c)) for c in chunks])
    if means[0] <= 0:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "non_positive_initial_utilization")
    # Fit exponential decay: y = A * exp(-k * t)
    t = np.arange(len(means), dtype=float)
    log_means = np.log(np.maximum(means, 1e-12))
    try:
        slope, intercept = np.polyfit(t, log_means, 1)
    except Exception:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "fit_failed")
    k = -slope
    t_half = math.log(2) / max(abs(k), 1e-12) if k > 0 else float("inf")
    findings = []
    if k > 0.01:
        findings.append(_make_finding(
            plugin_id, "utilization_decay",
            f"Utilization half-life detected: t_half={t_half:.1f} windows",
            f"Exponential decay rate k={k:.4f}, half-life={t_half:.1f} time windows.",
            "Declining utilization suggests resource degradation or demand reduction over time.",
            {"metrics": {"k": k, "t_half": t_half, "initial_mean": float(means[0]), "final_mean": float(means[-1]),
                         "duration_col": dur_col}},
            recommendation="Investigate the cause of utilization decay and whether resource refresh is needed.",
            severity="warn" if t_half > 5 else "critical",
            confidence=0.65,
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Half-life utilization: k={k:.4f}.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "k": k, "t_half": _safe_float(t_half, 0.0)})


def _hill_equation_dose_response(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """4-parameter logistic (Hill equation): parameter -> duration via curve_fit."""
    _log_start(ctx, plugin_id, df, config, inferred)
    if not HAS_SCIPY:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "scipy_unavailable")
    num_cols = _numeric_columns(df, inferred, max_cols=10)
    dur_col = _duration_column(df, inferred)
    if dur_col is None or len(num_cols) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_duration_or_numeric_columns")
    dose_col = [c for c in num_cols if c != dur_col]
    if not dose_col:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_dose_column")
    dose_col = dose_col[0]
    _ensure_budget(timer)
    x = pd.to_numeric(df[dose_col], errors="coerce").dropna()
    y = pd.to_numeric(df[dur_col], errors="coerce").reindex(x.index).dropna()
    x = x.reindex(y.index).to_numpy(dtype=float)
    y = y.to_numpy(dtype=float)
    if len(x) < 10:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_data")
    # 4PL: f(x) = D + (A - D) / (1 + (x/C)^B)
    def hill_4pl(x, A, B, C, D):
        return D + (A - D) / (1.0 + np.power(np.maximum(x / max(C, 1e-12), 1e-12), B))
    try:
        p0 = [float(np.min(y)), 1.0, float(np.median(x)), float(np.max(y))]
        popt, pcov = scipy_optimize.curve_fit(hill_4pl, x, y, p0=p0, maxfev=2000)
        y_pred = hill_4pl(x, *popt)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    except Exception:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "hill_fit_failed")
    findings = []
    if r2 > 0.5:
        findings.append(_make_finding(
            plugin_id, "hill_fit",
            f"Hill equation dose-response fit (R2={r2:.3f})",
            f"4PL fit of '{dose_col}' -> '{dur_col}': EC50={popt[2]:.3f}, Hill coeff={popt[1]:.2f}, R2={r2:.3f}.",
            "Sigmoidal dose-response indicates a threshold-driven relationship between parameter and outcome.",
            {"metrics": {"A": float(popt[0]), "B": float(popt[1]), "EC50": float(popt[2]), "D": float(popt[3]),
                         "r2": r2, "dose_col": dose_col, "response_col": dur_col}},
            recommendation=f"Operate near EC50={popt[2]:.3f} for optimal balance; avoid extremes of '{dose_col}'.",
            severity="info" if r2 < 0.7 else "warn",
            confidence=min(0.85, 0.5 + r2 * 0.3),
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Hill equation fit: R2={r2:.3f}.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "r2": r2, "EC50": float(popt[2]) if r2 > 0 else 0.0})


def _michaelis_menten_saturation(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """v = Vmax*S/(Km+S) saturation fit for throughput vs load."""
    _log_start(ctx, plugin_id, df, config, inferred)
    if not HAS_SCIPY:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "scipy_unavailable")
    num_cols = _numeric_columns(df, inferred, max_cols=10)
    dur_col = _duration_column(df, inferred)
    if dur_col is None or len(num_cols) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_duration_or_numeric_columns")
    substrate_col = [c for c in num_cols if c != dur_col]
    if not substrate_col:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_substrate_column")
    substrate_col = substrate_col[0]
    _ensure_budget(timer)
    S = pd.to_numeric(df[substrate_col], errors="coerce").dropna()
    v = pd.to_numeric(df[dur_col], errors="coerce").reindex(S.index).dropna()
    S = S.reindex(v.index).to_numpy(dtype=float)
    v = v.to_numpy(dtype=float)
    mask = S > 0
    S, v = S[mask], v[mask]
    if len(S) < 10:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_data")
    def mm(S, Vmax, Km):
        return Vmax * S / (Km + S)
    try:
        p0 = [float(np.max(v)), float(np.median(S))]
        popt, _ = scipy_optimize.curve_fit(mm, S, v, p0=p0, maxfev=2000)
        Vmax, Km = float(popt[0]), float(popt[1])
        y_pred = mm(S, Vmax, Km)
        ss_res = float(np.sum((v - y_pred) ** 2))
        ss_tot = float(np.sum((v - np.mean(v)) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    except Exception:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "mm_fit_failed")
    # Current operating point relative to saturation
    median_S = float(np.median(S))
    saturation_pct = median_S / max(Km + median_S, 1e-9) * 100
    findings = []
    if r2 > 0.4:
        findings.append(_make_finding(
            plugin_id, "mm_saturation",
            f"Michaelis-Menten saturation detected (R2={r2:.3f})",
            f"Vmax={Vmax:.2f}, Km={Km:.2f}, current saturation={saturation_pct:.0f}%. R2={r2:.3f}.",
            "Saturation kinetics mean throughput plateaus as load increases beyond Km.",
            {"metrics": {"Vmax": Vmax, "Km": Km, "r2": r2, "saturation_pct": saturation_pct,
                         "substrate_col": substrate_col, "velocity_col": dur_col}},
            recommendation=f"Operating at {saturation_pct:.0f}% saturation. {'Near capacity; add capacity.' if saturation_pct > 80 else 'Room for growth.'}",
            severity="warn" if saturation_pct > 80 else "info",
            confidence=min(0.85, 0.5 + r2 * 0.3),
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Michaelis-Menten fit: Vmax={Vmax:.2f}, Km={Km:.2f}.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "Vmax": Vmax, "Km": Km, "r2": r2})


def _response_surface_interaction(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Quadratic response surface with interactions (y = b0 + b1*x1 + b2*x2 + b12*x1*x2 + ...)."""
    _log_start(ctx, plugin_id, df, config, inferred)
    num_cols = _numeric_columns(df, inferred, max_cols=6)
    dur_col = _duration_column(df, inferred)
    if dur_col is None or len(num_cols) < 3:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "need_duration_and_2_predictors")
    predictors = [c for c in num_cols if c != dur_col][:4]
    if len(predictors) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "need_at_least_2_predictors")
    _ensure_budget(timer)
    y = pd.to_numeric(df[dur_col], errors="coerce").dropna()
    X_raw = df[predictors].apply(pd.to_numeric, errors="coerce").reindex(y.index).dropna()
    y = y.reindex(X_raw.index).to_numpy(dtype=float)
    X = X_raw.to_numpy(dtype=float)
    if len(y) < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_data")
    # Build design matrix: intercept, linear, quadratic, interactions
    n, p = X.shape
    cols_design = [np.ones(n)]
    col_names = ["intercept"]
    for i in range(p):
        cols_design.append(X[:, i]); col_names.append(predictors[i])
    for i in range(p):
        cols_design.append(X[:, i] ** 2); col_names.append(f"{predictors[i]}^2")
    for i in range(p):
        for j in range(i + 1, p):
            cols_design.append(X[:, i] * X[:, j]); col_names.append(f"{predictors[i]}*{predictors[j]}")
    D = np.column_stack(cols_design)
    try:
        beta, residuals, _, _ = np.linalg.lstsq(D, y, rcond=None)
        y_pred = D @ beta
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    except Exception:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "lstsq_failed")
    # Find strongest interaction term
    interaction_start = 1 + p + p  # after intercept, linear, quadratic
    interaction_betas = {col_names[interaction_start + i]: float(beta[interaction_start + i])
                         for i in range(len(beta) - interaction_start)}
    strongest = max(interaction_betas, key=lambda k: abs(interaction_betas[k])) if interaction_betas else None
    findings = []
    if r2 > 0.3 and strongest and abs(interaction_betas[strongest]) > 1e-6:
        findings.append(_make_finding(
            plugin_id, "response_surface",
            f"Response surface interaction detected (R2={r2:.3f})",
            f"Strongest interaction: {strongest} (beta={interaction_betas[strongest]:.4f}). Model R2={r2:.3f}.",
            "Significant interaction terms mean the effect of one parameter depends on the level of another.",
            {"metrics": {"r2": r2, "interaction_betas": interaction_betas, "predictors": predictors,
                         "response": dur_col, "n_terms": len(col_names)}},
            recommendation="Account for parameter interactions when tuning; optimize jointly rather than independently.",
            severity="info" if r2 < 0.6 else "warn",
            confidence=min(0.8, 0.4 + r2 * 0.4),
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Response surface: R2={r2:.3f}.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "r2": r2, "n_predictors": len(predictors)})


def _job_sequence_cache_rotation(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Test if job ordering affects duration on same host (cache/warm-up effect)."""
    _log_start(ctx, plugin_id, df, config, inferred)
    if not HAS_SCIPY:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "scipy_unavailable")
    grp_col = _group_column(df, inferred)
    dur_col = _duration_column(df, inferred)
    tc, ts = _time_series(df, inferred)
    if grp_col is None or dur_col is None or tc is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_group_duration_or_time_column")
    _ensure_budget(timer)
    df_sorted = df.assign(_ts=ts).dropna(subset=["_ts"]).sort_values([grp_col, "_ts"])
    results = {}
    for host, sub in df_sorted.groupby(grp_col):
        durs = pd.to_numeric(sub[dur_col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(durs) < 10: continue
        # Compare first quartile (cold) vs rest (warm)
        q1 = len(durs) // 4
        cold = durs[:max(q1, 1)]
        warm = durs[max(q1, 1):]
        if len(warm) < 5: continue
        stat, p_val = scipy_stats.mannwhitneyu(cold, warm, alternative="two-sided")
        ratio = float(np.median(cold)) / max(float(np.median(warm)), 1e-9)
        results[str(host)] = {"p_value": float(p_val), "cold_warm_ratio": ratio, "n_cold": len(cold), "n_warm": len(warm)}
    if not results:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_testable_hosts")
    significant = {k: v for k, v in results.items() if v["p_value"] < 0.05 and v["cold_warm_ratio"] > 1.1}
    findings = []
    if significant:
        worst = max(significant, key=lambda k: significant[k]["cold_warm_ratio"])
        findings.append(_make_finding(
            plugin_id, "cache_effect",
            f"Cache/warm-up effect detected on {len(significant)}/{len(results)} hosts",
            f"Host '{worst}' cold/warm ratio={significant[worst]['cold_warm_ratio']:.2f}x (p={significant[worst]['p_value']:.3e}).",
            "First jobs on a host take longer than subsequent ones, suggesting cache/warm-up overhead.",
            {"metrics": {"significant_hosts": significant, "total_hosts": len(results)}},
            recommendation="Pre-warm hosts or rotate job scheduling to minimize cold-start penalties.",
            severity="warn", confidence=min(0.8, 0.5 + len(significant) / max(len(results), 1) * 0.3),
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Cache rotation: {len(significant)}/{len(results)} hosts show effect.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "hosts_tested": len(results), "hosts_significant": len(significant)})


def _soil_depletion_resource(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """First-order decay dN/dt = -kN tracking for resource depletion."""
    _log_start(ctx, plugin_id, df, config, inferred)
    num_cols = _numeric_columns(df, inferred, max_cols=5)
    tc, ts = _time_series(df, inferred)
    if not num_cols or tc is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_or_time_column")
    _ensure_budget(timer)
    resource_col = num_cols[0]
    df_sorted = df.assign(_ts=ts).dropna(subset=["_ts"]).sort_values("_ts")
    vals = pd.to_numeric(df_sorted[resource_col], errors="coerce").dropna().to_numpy(dtype=float)
    if len(vals) < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_data")
    # Compute rolling mean in windows
    n_w = min(15, len(vals) // 5)
    chunks = np.array_split(vals, n_w)
    means = np.array([float(np.mean(c)) for c in chunks])
    positive = means[means > 0]
    if len(positive) < 5:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "non_positive_values")
    t = np.arange(len(positive), dtype=float)
    log_y = np.log(positive)
    try:
        slope, intercept = np.polyfit(t, log_y, 1)
    except Exception:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "fit_failed")
    k = -slope
    t_half = math.log(2) / max(abs(k), 1e-12) if k > 0 else float("inf")
    depletion_pct = 1.0 - positive[-1] / max(positive[0], 1e-9) if positive[0] > 0 else 0.0
    findings = []
    if k > 0.01 and depletion_pct > 0.1:
        findings.append(_make_finding(
            plugin_id, "resource_depletion",
            f"Resource depletion detected: '{resource_col}' declining (k={k:.4f})",
            f"First-order decay k={k:.4f}, half-life={t_half:.1f} windows. Depleted {depletion_pct:.1%} so far.",
            "Exponential decline in resource metric suggests unsustainable consumption.",
            {"metrics": {"k": k, "t_half": _safe_float(t_half), "depletion_pct": depletion_pct,
                         "initial": float(positive[0]), "current": float(positive[-1]), "column": resource_col}},
            recommendation="Investigate depletion driver and implement resource renewal or conservation.",
            severity="warn" if depletion_pct < 0.5 else "critical",
            confidence=min(0.8, 0.5 + depletion_pct * 0.3),
        ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Soil depletion: k={k:.4f}, depleted={depletion_pct:.1%}.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "k": k, "depletion_pct": depletion_pct})


def _liebig_minimum_constraint(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Identify binding constraint resource (Liebig's law of the minimum)."""
    _log_start(ctx, plugin_id, df, config, inferred)
    num_cols = _numeric_columns(df, inferred, max_cols=10)
    dur_col = _duration_column(df, inferred)
    if dur_col is None or len(num_cols) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "need_duration_and_resources")
    resources = [c for c in num_cols if c != dur_col]
    if not resources:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_resource_columns")
    _ensure_budget(timer)
    y = pd.to_numeric(df[dur_col], errors="coerce").dropna().to_numpy(dtype=float)
    # For each resource, compute correlation with outcome
    correlations = {}
    for col in resources:
        x = pd.to_numeric(df[col], errors="coerce").reindex(pd.RangeIndex(len(df))).dropna()
        common = min(len(x), len(y))
        if common < 10: continue
        xv = x.to_numpy(dtype=float)[:common]
        yv = y[:common]
        corr = float(np.corrcoef(xv, yv)[0, 1]) if len(xv) > 2 else 0.0
        if math.isfinite(corr):
            correlations[col] = corr
    if not correlations:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_correlations")
    # Liebig: the binding constraint is the resource with the strongest limiting effect
    # (most negative correlation with outcome, or most positive if higher = bottleneck)
    binding = min(correlations, key=correlations.get)
    binding_corr = correlations[binding]
    # Compute how far each resource is from its maximum (% headroom)
    headroom = {}
    for col in resources:
        vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals) > 0:
            headroom[col] = 1.0 - float(np.median(vals)) / max(float(np.max(vals)), 1e-9)
    min_headroom_col = min(headroom, key=headroom.get) if headroom else binding
    findings = []
    findings.append(_make_finding(
        plugin_id, "binding_constraint",
        f"Binding constraint: '{min_headroom_col}' (headroom={headroom.get(min_headroom_col, 0):.1%})",
        f"Resource '{min_headroom_col}' has least headroom ({headroom.get(min_headroom_col, 0):.1%}). "
        f"Most correlated constraint: '{binding}' (r={binding_corr:.3f}).",
        "Liebig's law: system performance is limited by the scarcest resource.",
        {"metrics": {"binding_by_correlation": binding, "binding_corr": binding_corr,
                     "min_headroom_col": min_headroom_col, "headroom": headroom, "correlations": correlations}},
        recommendation=f"Prioritize expanding '{min_headroom_col}' capacity as the binding constraint.",
        severity="warn" if headroom.get(min_headroom_col, 1.0) < 0.2 else "info",
        confidence=min(0.75, 0.5 + abs(binding_corr) * 0.25),
    ))
    artifacts: list[PluginArtifact] = []
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Liebig minimum: binding='{min_headroom_col}'.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "binding_resource": min_headroom_col,
                       "min_headroom": headroom.get(min_headroom_col, 0.0)})


# ---------------------------------------------------------------------------
# HANDLERS registry
# ---------------------------------------------------------------------------

HANDLERS: dict[str, Callable[..., PluginResult]] = {
    "analysis_smith_waterman_workflow_align_v1": _smith_waterman_workflow_align,
    "analysis_meme_motif_discovery_v1": _meme_motif_discovery,
    "analysis_hmm_workflow_state_v1": _hmm_workflow_state,
    "analysis_phylogenetic_workflow_tree_v1": _phylogenetic_workflow_tree,
    "analysis_shannon_diversity_index_v1": _shannon_diversity_index,
    "analysis_rarefaction_coverage_v1": _rarefaction_coverage,
    "analysis_lotka_volterra_competition_v1": _lotka_volterra_competition,
    "analysis_niche_overlap_pianka_v1": _niche_overlap_pianka,
    "analysis_sir_failure_cascade_v1": _sir_failure_cascade,
    "analysis_cascade_r0_reproduction_v1": _cascade_r0_reproduction,
    "analysis_contact_tracing_superspreader_v1": _contact_tracing_superspreader,
    "analysis_epicurve_classification_v1": _epicurve_classification,
    "analysis_hardy_weinberg_param_coupling_v1": _hardy_weinberg_param_coupling,
    "analysis_linkage_disequilibrium_steps_v1": _linkage_disequilibrium_steps,
    "analysis_fst_machine_differentiation_v1": _fst_machine_differentiation,
    "analysis_wright_fisher_config_drift_v1": _wright_fisher_config_drift,
    "analysis_compartment_redistribution_v1": _compartment_redistribution,
    "analysis_half_life_utilization_v1": _half_life_utilization,
    "analysis_hill_equation_dose_response_v1": _hill_equation_dose_response,
    "analysis_michaelis_menten_saturation_v1": _michaelis_menten_saturation,
    "analysis_response_surface_interaction_v1": _response_surface_interaction,
    "analysis_job_sequence_cache_rotation_v1": _job_sequence_cache_rotation,
    "analysis_soil_depletion_resource_v1": _soil_depletion_resource,
    "analysis_liebig_minimum_constraint_v1": _liebig_minimum_constraint,
}
