"""Cross-domain plugins: earth & space sciences (plugins 96-121, skipping 95)."""
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
    from scipy import optimize as scipy_optimize
    from scipy import linalg as scipy_linalg
    HAS_SCIPY = True
except Exception:
    scipy_stats = scipy_signal = scipy_optimize = scipy_linalg = None
    HAS_SCIPY = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except Exception:
    nx = None
    HAS_NETWORKX = False

try:
    import astropy.timeseries as astropy_ts
    HAS_ASTROPY = True
except Exception:
    astropy_ts = None
    HAS_ASTROPY = False

try:
    import utide
    HAS_UTIDE = True
except Exception:
    utide = None
    HAS_UTIDE = False

try:
    import esda
    HAS_ESDA = True
except Exception:
    esda = None
    HAS_ESDA = False

try:
    import pykrige
    HAS_PYKRIGE = True
except Exception:
    pykrige = None
    HAS_PYKRIGE = False

try:
    import chainladder as cl
    HAS_CHAINLADDER = True
except Exception:
    cl = None
    HAS_CHAINLADDER = False

try:
    import pyextremes
    HAS_PYEXTREMES = True
except Exception:
    pyextremes = None
    HAS_PYEXTREMES = False

try:
    import lifelines
    HAS_LIFELINES = True
except Exception:
    lifelines = None
    HAS_LIFELINES = False


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

def _time_windows(ts, n_windows=5):
    valid = ts.dropna()
    if len(valid) < n_windows * 5: return []
    indices = np.array_split(np.arange(len(valid)), n_windows)
    return [(valid.iloc[idx[0]], valid.iloc[idx[-1]], idx) for idx in indices if len(idx) > 0]


# ---------------------------------------------------------------------------
# Plugin handlers (96-121)
# ---------------------------------------------------------------------------


def _bls_periodic_outage(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Box Least Squares for periodic flat-bottomed dips in metric time series."""
    _log_start(ctx, plugin_id, df, config, inferred)
    time_col, ts = _time_series(df, inferred)
    if ts is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_time_column")
    dur_col = _duration_column(df, inferred)
    if dur_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_metric_column")
    frame = pd.DataFrame({"t": ts, "y": pd.to_numeric(df[dur_col], errors="coerce")}).dropna().sort_values("t")
    if len(frame) < 30:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    t_num = (frame["t"] - frame["t"].iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    y = frame["y"].to_numpy(dtype=float)
    y_med = float(np.nanmedian(y))
    y_norm = y - y_med
    # Manual BLS: scan trial periods
    duration_range = t_num[-1] - t_num[0]
    if duration_range <= 0:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "zero_time_range")
    best_power, best_period, best_depth = 0.0, 0.0, 0.0
    trial_periods = np.linspace(duration_range / 20, duration_range / 2, 50)
    for period in trial_periods:
        _ensure_budget(timer)
        phase = (t_num % period) / period
        for frac in [0.05, 0.1, 0.15, 0.2]:
            in_transit = phase < frac
            if in_transit.sum() < 3 or (~in_transit).sum() < 3: continue
            depth = float(np.nanmean(y_norm[~in_transit]) - np.nanmean(y_norm[in_transit]))
            power = depth ** 2 * in_transit.sum()
            if power > best_power:
                best_power, best_period, best_depth = power, float(period), depth
    findings = []
    if best_depth > 0 and best_power > 0:
        findings.append(_make_finding(
            plugin_id, "periodic_dip",
            f"Periodic outage detected with period ~{best_period:.1f}s",
            f"BLS found a flat-bottomed dip of depth {best_depth:.3f} recurring every ~{best_period:.1f}s.",
            "Periodic dips suggest scheduled maintenance, cron jobs, or recurring resource contention.",
            {"metrics": {"period_s": best_period, "depth": best_depth, "power": best_power, "metric": dur_col}},
            recommendation=f"Investigate recurring dip in '{dur_col}' at ~{best_period:.0f}s intervals.",
            severity="warn" if best_depth > y_med * 0.1 else "info",
            confidence=min(0.85, 0.4 + best_power / (best_power + 100)),
        ))
    return _finalize(plugin_id, ctx, df, sample_meta,
        "BLS periodic outage scan complete.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "best_period": best_period, "best_depth": best_depth})


def _pulsar_timing_residual(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Fit polynomial to timestamps, analyze residuals for hidden periodicity."""
    _log_start(ctx, plugin_id, df, config, inferred)
    time_col, ts = _time_series(df, inferred)
    if ts is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_time_column")
    valid = ts.dropna().sort_values()
    if len(valid) < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    intervals = np.diff(valid.astype(np.int64) / 1e9)
    if len(intervals) < 15:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_intervals")
    x = np.arange(len(intervals), dtype=float)
    poly = np.polyfit(x, intervals, deg=min(3, len(intervals) - 1))
    trend = np.polyval(poly, x)
    residuals = intervals - trend
    std_res = float(np.nanstd(residuals))
    mean_int = float(np.nanmean(intervals))
    jitter_ratio = std_res / max(mean_int, 1e-9)
    # Check for periodicity in residuals via autocorrelation
    if len(residuals) > 10:
        acf = np.correlate(residuals - residuals.mean(), residuals - residuals.mean(), mode="full")
        acf = acf[len(acf)//2:] / (acf[len(acf)//2] + 1e-15)
        peaks = [i for i in range(2, min(len(acf), len(residuals)//2)) if acf[i] > 0.3]
        best_lag = peaks[0] if peaks else 0
    else:
        best_lag = 0
    findings = []
    if jitter_ratio > 0.1:
        findings.append(_make_finding(
            plugin_id, "timing_jitter",
            f"High timing jitter (ratio={jitter_ratio:.3f})",
            f"Residual std={std_res:.4f}s vs mean interval={mean_int:.4f}s after polynomial detrending.",
            "High timing residuals indicate irregular scheduling or contention-induced delays.",
            {"metrics": {"jitter_ratio": jitter_ratio, "std_residual": std_res, "mean_interval": mean_int,
                         "periodic_lag": best_lag, "poly_degree": len(poly)-1}},
            recommendation="Investigate sources of timing jitter; consider tighter scheduling.",
            severity="warn" if jitter_ratio > 0.25 else "info",
            confidence=min(0.8, 0.45 + jitter_ratio),
        ))
    return _finalize(plugin_id, ctx, df, sample_meta,
        "Pulsar timing residual analysis complete.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "jitter_ratio": jitter_ratio, "periodic_lag": best_lag})


def _hr_diagram_classification(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Plot throughput vs latency, classify hosts into main-sequence/giant/dwarf."""
    _log_start(ctx, plugin_id, df, config, inferred)
    num_cols = _numeric_columns(df, inferred, max_cols=20)
    if len(num_cols) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_numeric_columns")
    cat_cols = _categorical_columns(df, inferred, max_cols=5)
    host_col = cat_cols[0] if cat_cols else None
    throughput_hints = ("throughput", "count", "volume", "rate", "requests", "ops")
    latency_hints = ("latency", "duration", "response", "wait", "elapsed")
    tp_col = lat_col = None
    for c in num_cols:
        cl = c.lower()
        if not tp_col and any(h in cl for h in throughput_hints): tp_col = c
        if not lat_col and any(h in cl for h in latency_hints): lat_col = c
    if not tp_col: tp_col = num_cols[0]
    if not lat_col: lat_col = num_cols[1] if len(num_cols) > 1 else num_cols[0]
    if tp_col == lat_col:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "same_column_throughput_latency")
    tp = pd.to_numeric(df[tp_col], errors="coerce").to_numpy(dtype=float)
    lat = pd.to_numeric(df[lat_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(tp) & np.isfinite(lat)
    if valid.sum() < 10:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_valid_rows")
    tp_v, lat_v = tp[valid], lat[valid]
    tp_med, lat_med = float(np.median(tp_v)), float(np.median(lat_v))
    classes = []
    for t, l in zip(tp_v, lat_v):
        if t >= tp_med and l <= lat_med: classes.append("main_sequence")
        elif t >= tp_med and l > lat_med: classes.append("giant")
        else: classes.append("dwarf")
    counts = Counter(classes)
    findings = []
    if counts.get("giant", 0) > 0:
        findings.append(_make_finding(
            plugin_id, "giant_hosts",
            f"{counts['giant']} giant-class entities (high throughput, high latency)",
            f"{counts['giant']} entities show high {tp_col} but also high {lat_col}.",
            "Giant-class entities handle large volume but are slow; potential bottleneck or overload.",
            {"metrics": {"counts": dict(counts), "throughput_col": tp_col, "latency_col": lat_col}},
            recommendation=f"Investigate giant-class entities: high {tp_col} with high {lat_col} suggests overload.",
            severity="warn" if counts["giant"] > len(classes) * 0.2 else "info",
            confidence=0.6,
        ))
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"HR diagram classification: {dict(counts)}.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), **{f"n_{k}": v for k, v in counts.items()}})


def _tidal_harmonic_cycles(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Fit harmonic constituents to time series via FFT."""
    _log_start(ctx, plugin_id, df, config, inferred)
    time_col, ts = _time_series(df, inferred)
    if ts is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_time_column")
    dur_col = _duration_column(df, inferred)
    if dur_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_metric_column")
    frame = pd.DataFrame({"t": ts, "y": pd.to_numeric(df[dur_col], errors="coerce")}).dropna().sort_values("t")
    if len(frame) < 30:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    y = frame["y"].to_numpy(dtype=float)
    y = y - np.nanmean(y)
    n = len(y)
    fft_vals = np.fft.rfft(y)
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(n)
    # Top 5 harmonics (skip DC)
    if len(power) < 3:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_spectrum")
    top_idx = np.argsort(power[1:])[-5:][::-1] + 1
    total_power = float(np.sum(power[1:]))
    harmonics = []
    for idx in top_idx:
        if idx < len(freqs):
            harmonics.append({"freq": float(freqs[idx]), "period_samples": float(1.0 / freqs[idx]) if freqs[idx] > 0 else 0,
                              "power_frac": float(power[idx] / total_power) if total_power > 0 else 0})
    dominant = harmonics[0] if harmonics else {}
    findings = []
    if dominant and dominant.get("power_frac", 0) > 0.1:
        findings.append(_make_finding(
            plugin_id, "dominant_harmonic",
            f"Dominant harmonic at period ~{dominant.get('period_samples', 0):.1f} samples",
            f"Top harmonic explains {dominant['power_frac']*100:.1f}% of variance in '{dur_col}'.",
            "Strong harmonic constituents indicate regular cyclic patterns in the data.",
            {"metrics": {"harmonics": harmonics[:3], "total_power": total_power, "metric": dur_col}},
            recommendation=f"Align capacity planning with the dominant cycle period of ~{dominant.get('period_samples',0):.0f} samples.",
            severity="warn" if dominant["power_frac"] > 0.3 else "info",
            confidence=min(0.85, 0.4 + dominant["power_frac"]),
        ))
    return _finalize(plugin_id, ctx, df, sample_meta,
        "Tidal harmonic analysis complete.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "n_harmonics": len(harmonics)})


def _wave_directional_spectra(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Directional spectra from multiple metric sources."""
    _log_start(ctx, plugin_id, df, config, inferred)
    num_cols = _variance_sorted_numeric(df, inferred, limit=8)
    if len(num_cols) < 3:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_numeric_columns")
    time_col, ts = _time_series(df, inferred)
    if ts is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_time_column")
    frame = df[num_cols].apply(pd.to_numeric, errors="coerce").copy()
    frame["_ts"] = ts
    frame = frame.sort_values("_ts").dropna(subset=["_ts"])
    if len(frame) < 30:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    # Cross-spectral analysis: compute dominant direction per frequency
    spectra = {}
    for col in num_cols:
        vals = frame[col].to_numpy(dtype=float)
        vals = vals - np.nanmean(vals)
        fft_v = np.fft.rfft(vals)
        spectra[col] = np.abs(fft_v) ** 2
    n_freqs = min(len(v) for v in spectra.values())
    total_by_source = {col: float(np.sum(s[1:n_freqs])) for col, s in spectra.items()}
    dominant_source = max(total_by_source, key=total_by_source.get)
    dom_frac = total_by_source[dominant_source] / max(sum(total_by_source.values()), 1e-15)
    findings = []
    if dom_frac > 0.3:
        findings.append(_make_finding(
            plugin_id, "directional_dominance",
            f"Directional dominance: '{dominant_source}' carries {dom_frac*100:.1f}% of spectral energy",
            f"Column '{dominant_source}' dominates the combined spectral energy across {len(num_cols)} sources.",
            "One dominant source of variability may mask signals in other dimensions.",
            {"metrics": {"dominant_source": dominant_source, "fraction": dom_frac, "power_by_source": total_by_source}},
            recommendation=f"Decompose variability in '{dominant_source}' separately before analyzing other metrics.",
            severity="warn" if dom_frac > 0.5 else "info",
            confidence=min(0.8, 0.4 + dom_frac),
        ))
    return _finalize(plugin_id, ctx, df, sample_meta,
        "Wave directional spectra analysis complete.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "n_sources": len(num_cols), "dominant_source": dominant_source})


def _thermohaline_circulation(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Workload circulation through queues. Detect overturning (flow reversal)."""
    _log_start(ctx, plugin_id, df, config, inferred)
    if not HAS_NETWORKX:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "networkx_unavailable")
    edges, sc, dc = _build_edges(df, inferred)
    if edges.empty:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_graph_edges")
    time_col, ts = _time_series(df, inferred)
    G = _build_nx_graph_from_edges(edges)
    if G is None or G.number_of_edges() < 5:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_graph")
    # Detect cycles (circulation) and bidirectional edges (overturning)
    try:
        cycles = list(nx.simple_cycles(G))[:100]
    except Exception:
        cycles = []
    bidir = sum(1 for u, v in G.edges() if G.has_edge(v, u))
    n_edges = G.number_of_edges()
    overturning_ratio = bidir / max(n_edges, 1)
    findings = []
    if overturning_ratio > 0.05:
        findings.append(_make_finding(
            plugin_id, "overturning",
            f"Flow overturning detected: {bidir} bidirectional edges ({overturning_ratio*100:.1f}%)",
            f"{bidir}/{n_edges} edges are bidirectional, indicating workload recirculation.",
            "Bidirectional flow suggests rework or oscillating task routing between queues.",
            {"metrics": {"bidir_edges": bidir, "total_edges": n_edges, "overturning_ratio": overturning_ratio,
                         "n_cycles": len(cycles)}},
            recommendation="Investigate bidirectional flows for rework loops and enforce unidirectional routing.",
            severity="warn" if overturning_ratio > 0.15 else "info",
            confidence=min(0.8, 0.4 + overturning_ratio * 2),
        ))
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Thermohaline circulation: {len(cycles)} cycles, {bidir} bidirectional edges.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "n_cycles": len(cycles), "bidir_edges": bidir})


def _ekman_transport_indirect(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Detect indirect/orthogonal effects from primary flow (Ekman transport analogy)."""
    _log_start(ctx, plugin_id, df, config, inferred)
    num_cols = _variance_sorted_numeric(df, inferred, limit=8)
    if len(num_cols) < 3:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_numeric_columns")
    time_col, ts = _time_series(df, inferred)
    if ts is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_time_column")
    frame = df[num_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(frame) < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    # Compute correlation matrix; find pairs with low direct but high lagged correlation
    corr = frame.corr().to_numpy(dtype=float)
    indirect_pairs = []
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            _ensure_budget(timer)
            direct = abs(corr[i, j])
            if direct > 0.3: continue  # skip directly correlated
            # Check lagged correlation
            a = frame[num_cols[i]].to_numpy(dtype=float)
            b = frame[num_cols[j]].to_numpy(dtype=float)
            best_lag_corr = 0.0
            for lag in [1, 2, 3, 5, 8]:
                if lag >= len(a): break
                c = float(np.corrcoef(a[lag:], b[:-lag])[0, 1])
                if math.isfinite(c) and abs(c) > abs(best_lag_corr): best_lag_corr = c
            if abs(best_lag_corr) > 0.3:
                indirect_pairs.append({"col_a": num_cols[i], "col_b": num_cols[j],
                                       "direct_corr": float(direct), "lagged_corr": float(best_lag_corr)})
    findings = []
    if indirect_pairs:
        top = max(indirect_pairs, key=lambda p: abs(p["lagged_corr"]))
        findings.append(_make_finding(
            plugin_id, "indirect_effect",
            f"Indirect effect: '{top['col_a']}' -> '{top['col_b']}' (lag corr={top['lagged_corr']:.2f})",
            f"{len(indirect_pairs)} column pair(s) show weak direct but strong lagged correlation.",
            "Ekman-like transport: primary flow in one variable drives orthogonal response in another after a lag.",
            {"metrics": {"n_indirect_pairs": len(indirect_pairs), "top_pair": top}},
            recommendation=f"Monitor '{top['col_b']}' as an indirect response to changes in '{top['col_a']}'.",
            severity="warn" if abs(top["lagged_corr"]) > 0.5 else "info",
            confidence=min(0.8, 0.4 + abs(top["lagged_corr"])),
        ))
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Ekman transport: {len(indirect_pairs)} indirect effect(s) found.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "n_indirect_pairs": len(indirect_pairs)})


def _gutenberg_richter_bvalue(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Gutenberg-Richter b-value: log10(N)=a-b*M. Low b = more large events."""
    _log_start(ctx, plugin_id, df, config, inferred)
    dur_col = _duration_column(df, inferred)
    if dur_col is None:
        num_cols = _numeric_columns(df, inferred, max_cols=5)
        dur_col = num_cols[0] if num_cols else None
    if dur_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_column")
    vals = pd.to_numeric(df[dur_col], errors="coerce").dropna().to_numpy(dtype=float)
    vals = vals[vals > 0]
    if len(vals) < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_positive_values")
    magnitudes = np.log10(vals)
    mag_bins = np.linspace(float(np.min(magnitudes)), float(np.max(magnitudes)), 20)
    counts = np.array([np.sum(magnitudes >= m) for m in mag_bins], dtype=float)
    valid_mask = counts > 0
    if valid_mask.sum() < 5:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_magnitude_range")
    log_counts = np.log10(counts[valid_mask])
    mags_fit = mag_bins[valid_mask]
    slope, intercept = np.polyfit(mags_fit, log_counts, 1)
    b_value = float(-slope)
    a_value = float(intercept)
    findings = []
    sev = "info"
    if b_value < 0.8: sev = "critical"
    elif b_value < 1.0: sev = "warn"
    findings.append(_make_finding(
        plugin_id, "b_value",
        f"Gutenberg-Richter b-value={b_value:.2f} for '{dur_col}'",
        f"b={b_value:.2f}, a={a_value:.2f}. {'Low b: disproportionately many large events.' if b_value < 1.0 else 'Normal frequency-magnitude distribution.'}",
        "The b-value describes the ratio of small to large events. b<1 means heavy-tailed risk.",
        {"metrics": {"b_value": b_value, "a_value": a_value, "column": dur_col, "n_values": len(vals)}},
        recommendation="Investigate root causes of large-magnitude events." if b_value < 1.0 else "Frequency-magnitude distribution is within normal range.",
        severity=sev, confidence=min(0.85, 0.5 + 0.1 * valid_mask.sum()),
    ))
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Gutenberg-Richter: b={b_value:.2f}.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "b_value": b_value, "a_value": a_value})


def _omori_aftershock_decay(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Omori law: N(t)=K/(t+c)^p aftershock decay after major events."""
    _log_start(ctx, plugin_id, df, config, inferred)
    time_col, ts = _time_series(df, inferred)
    if ts is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_time_column")
    dur_col = _duration_column(df, inferred)
    if dur_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_metric_column")
    frame = pd.DataFrame({"t": ts, "y": pd.to_numeric(df[dur_col], errors="coerce")}).dropna().sort_values("t")
    if len(frame) < 30:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    y = frame["y"].to_numpy(dtype=float)
    threshold = float(np.nanpercentile(y, 95))
    mainshock_idx = np.where(y >= threshold)[0]
    if len(mainshock_idx) == 0:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_mainshock")
    # Analyze aftershocks after first mainshock
    ms_idx = int(mainshock_idx[0])
    after = y[ms_idx + 1:ms_idx + min(100, len(y) - ms_idx - 1)]
    if len(after) < 10:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_aftershock_data")
    t_after = np.arange(1, len(after) + 1, dtype=float)
    # Fit log(N) = log(K) - p*log(t+c) with c=1
    log_y = np.log(after + 1e-9)
    log_t = np.log(t_after + 1.0)
    slope, intercept = np.polyfit(log_t, log_y, 1)
    p_value = float(-slope)
    K_value = float(np.exp(intercept))
    findings = []
    findings.append(_make_finding(
        plugin_id, "aftershock_decay",
        f"Aftershock decay exponent p={p_value:.2f}",
        f"Post-mainshock decay follows Omori law with p={p_value:.2f}, K={K_value:.2f}.",
        "p<1 means slow recovery (aftershocks linger), p>1 means rapid recovery.",
        {"metrics": {"p": p_value, "K": K_value, "mainshock_idx": ms_idx, "n_aftershocks": len(after)}},
        recommendation="Slow decay (p<1): add cooldown mechanisms after major incidents." if p_value < 1.0
        else "Decay rate is healthy; system recovers quickly from spikes.",
        severity="warn" if p_value < 0.8 else "info",
        confidence=min(0.75, 0.4 + abs(p_value) * 0.2),
    ))
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Omori aftershock decay: p={p_value:.2f}.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "p": p_value, "K": K_value})


def _etas_self_exciting_cascade(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """ETAS: separate background rate from triggered/cascading events."""
    _log_start(ctx, plugin_id, df, config, inferred)
    time_col, ts = _time_series(df, inferred)
    if ts is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_time_column")
    valid = ts.dropna().sort_values()
    if len(valid) < 30:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    t_sec = (valid - valid.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    total_dur = t_sec[-1] - t_sec[0]
    if total_dur <= 0:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "zero_time_range")
    n = len(t_sec)
    background_rate = n / total_dur
    # Estimate triggered fraction: inter-event times below expected Poisson interval
    iet = np.diff(t_sec)
    expected_iet = total_dur / n
    triggered_mask = iet < expected_iet * 0.3  # events arriving much faster than background
    n_triggered = int(np.sum(triggered_mask))
    triggered_frac = n_triggered / max(n - 1, 1)
    branching_ratio = triggered_frac  # approximate
    findings = []
    if branching_ratio > 0.2:
        findings.append(_make_finding(
            plugin_id, "self_exciting",
            f"Self-exciting cascade detected: branching ratio={branching_ratio:.2f}",
            f"{n_triggered}/{n-1} inter-event intervals are clustered (< 30% of expected). Background rate={background_rate:.4f}/s.",
            "High branching ratio means events trigger further events (cascade/contagion).",
            {"metrics": {"branching_ratio": branching_ratio, "n_triggered": n_triggered,
                         "background_rate": background_rate, "n_events": n}},
            recommendation="Add circuit breakers or rate limiters to prevent event cascades.",
            severity="warn" if branching_ratio > 0.4 else "info",
            confidence=min(0.8, 0.4 + branching_ratio),
        ))
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"ETAS self-exciting: branching_ratio={branching_ratio:.2f}.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "branching_ratio": branching_ratio, "background_rate": background_rate})


def _psha_probabilistic_risk(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """PSHA: exceedance probability for failure thresholds."""
    _log_start(ctx, plugin_id, df, config, inferred)
    dur_col = _duration_column(df, inferred)
    if dur_col is None:
        num_cols = _numeric_columns(df, inferred, max_cols=5)
        dur_col = num_cols[0] if num_cols else None
    if dur_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_column")
    vals = pd.to_numeric(df[dur_col], errors="coerce").dropna().to_numpy(dtype=float)
    if len(vals) < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    percentiles = [90, 95, 99, 99.5]
    thresholds = {p: float(np.nanpercentile(vals, p)) for p in percentiles}
    # Estimate annual exceedance: assume data spans one observation period
    n = len(vals)
    exceedance = {p: float((100 - p) / 100) for p in percentiles}
    # Return period in units of samples
    return_periods = {p: 1.0 / max(exceedance[p], 1e-9) for p in percentiles}
    findings = []
    p99_val = thresholds[99]
    p95_val = thresholds[95]
    ratio = p99_val / max(p95_val, 1e-9)
    if ratio > 2.0:
        findings.append(_make_finding(
            plugin_id, "heavy_tail_risk",
            f"Heavy-tail risk: P99/P95 ratio={ratio:.2f} for '{dur_col}'",
            f"P99={p99_val:.3f} vs P95={p95_val:.3f}. Extreme values grow disproportionately.",
            "High P99/P95 ratio means tail risk is elevated; rare events are disproportionately severe.",
            {"metrics": {"thresholds": thresholds, "return_periods": return_periods, "p99_p95_ratio": ratio}},
            recommendation=f"Add safeguards for extreme '{dur_col}' values beyond P95={p95_val:.2f}.",
            severity="warn" if ratio > 3.0 else "info",
            confidence=min(0.8, 0.5 + (ratio - 2.0) * 0.1),
        ))
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"PSHA risk: P99/P95={ratio:.2f}.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "p99_p95_ratio": ratio, **thresholds})


def _moran_i_autocorrelation(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Moran's I spatial autocorrelation on categorical/host groups."""
    _log_start(ctx, plugin_id, df, config, inferred)
    num_cols = _numeric_columns(df, inferred, max_cols=5)
    cat_cols = _categorical_columns(df, inferred, max_cols=5)
    if not num_cols:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_columns")
    target = num_cols[0]
    vals = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(vals)
    if valid.sum() < 10:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_valid_rows")
    n = int(valid.sum())
    y = vals[valid]
    y_bar = float(np.mean(y))
    # Build proximity weights: sequential neighbors
    W = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        W[i, i + 1] = 1.0
        W[i + 1, i] = 1.0
    W_sum = W.sum()
    if W_sum == 0:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_spatial_weights")
    dev = y - y_bar
    numerator = float(np.sum(W * np.outer(dev, dev)))
    denominator = float(np.sum(dev ** 2))
    moran_i = (n / W_sum) * (numerator / max(denominator, 1e-15))
    expected_i = -1.0 / (n - 1)
    findings = []
    if abs(moran_i) > 0.1:
        findings.append(_make_finding(
            plugin_id, "spatial_autocorrelation",
            f"Moran's I={moran_i:.3f} for '{target}' ({'clustered' if moran_i > 0 else 'dispersed'})",
            f"Moran's I={moran_i:.3f} (expected={expected_i:.3f}). {'Positive=similar neighbors cluster.' if moran_i > 0 else 'Negative=dissimilar neighbors.'}",
            "Spatial autocorrelation indicates non-random arrangement of values across entities.",
            {"metrics": {"moran_i": moran_i, "expected_i": expected_i, "column": target, "n": n}},
            recommendation="Clustered patterns suggest localized effects; investigate regional/group-specific factors." if moran_i > 0
            else "Dispersed patterns may indicate compensating effects between neighbors.",
            severity="warn" if abs(moran_i) > 0.3 else "info",
            confidence=min(0.8, 0.4 + abs(moran_i)),
        ))
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Moran's I={moran_i:.3f}.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "moran_i": moran_i})


def _kriging_missing_data(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Kriging-inspired interpolation to estimate missing data quality."""
    _log_start(ctx, plugin_id, df, config, inferred)
    num_cols = _numeric_columns(df, inferred, max_cols=10)
    if not num_cols:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_columns")
    # Find column with partial missing data
    best_col, best_frac = None, 0.0
    for col in num_cols:
        frac = float(df[col].isna().mean())
        if 0.05 < frac < 0.5 and frac > best_frac:
            best_col, best_frac = col, frac
    if best_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_partially_missing_column")
    vals = pd.to_numeric(df[best_col], errors="coerce").to_numpy(dtype=float)
    known_idx = np.where(np.isfinite(vals))[0]
    missing_idx = np.where(~np.isfinite(vals))[0]
    if len(known_idx) < 10 or len(missing_idx) < 3:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_known_or_missing")
    known_vals = vals[known_idx]
    # Simple distance-weighted interpolation (kriging approximation)
    interpolated = []
    for mi in missing_idx[:50]:  # limit for performance
        dists = np.abs(known_idx.astype(float) - float(mi))
        weights = 1.0 / (dists + 1.0) ** 2
        weights /= weights.sum()
        interpolated.append(float(np.dot(weights, known_vals)))
    interp_arr = np.array(interpolated)
    interp_std = float(np.std(interp_arr))
    data_std = float(np.std(known_vals))
    smoothness = 1.0 - min(1.0, interp_std / max(data_std, 1e-9))
    findings = []
    findings.append(_make_finding(
        plugin_id, "kriging_quality",
        f"Kriging interpolation: {best_frac*100:.1f}% missing in '{best_col}', smoothness={smoothness:.2f}",
        f"{len(missing_idx)} missing values in '{best_col}'. Interpolated values have std={interp_std:.3f} vs data std={data_std:.3f}.",
        "High smoothness means missing values are predictable from neighbors; low smoothness suggests gaps at irregular points.",
        {"metrics": {"column": best_col, "missing_frac": best_frac, "smoothness": smoothness,
                     "interp_std": interp_std, "data_std": data_std}},
        recommendation=f"Missing data in '{best_col}' is {'predictable (safe to interpolate)' if smoothness > 0.5 else 'irregular (investigate root cause)'}.",
        severity="warn" if smoothness < 0.3 else "info",
        confidence=min(0.75, 0.4 + smoothness * 0.3),
    ))
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Kriging missing data: smoothness={smoothness:.2f}.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "smoothness": smoothness, "missing_frac": best_frac})


def _ripley_k_temporal_clustering(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Ripley's K on timestamps: test clustering vs random (CSR)."""
    _log_start(ctx, plugin_id, df, config, inferred)
    time_col, ts = _time_series(df, inferred)
    if ts is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_time_column")
    valid = ts.dropna().sort_values()
    if len(valid) < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    t_sec = (valid - valid.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    T = t_sec[-1]
    if T <= 0:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "zero_time_range")
    n = len(t_sec)
    lam = n / T  # intensity
    radii = np.linspace(T * 0.01, T * 0.2, 10)
    K_obs = []
    for r in radii:
        count = sum(1 for i in range(n) for j in range(i+1, min(i+50, n)) if abs(t_sec[j] - t_sec[i]) <= r)
        K_obs.append(2.0 * count / (n * lam + 1e-15))
    K_csr = [2.0 * r for r in radii]  # expected under complete spatial randomness (1D)
    # L-function: deviation from CSR
    L_dev = [float(ko - kc) for ko, kc in zip(K_obs, K_csr)]
    max_dev = max(L_dev) if L_dev else 0.0
    clustered = max_dev > 0
    findings = []
    if abs(max_dev) > T * 0.01:
        findings.append(_make_finding(
            plugin_id, "temporal_clustering",
            f"Temporal {'clustering' if clustered else 'regularity'} detected (max L-deviation={max_dev:.2f})",
            f"Ripley's K shows {'more clustering' if clustered else 'more regularity'} than random. Max deviation={max_dev:.2f}s.",
            "Clustering means events bunch together; regularity means they are more evenly spaced than random.",
            {"metrics": {"max_L_deviation": max_dev, "clustered": clustered, "intensity": lam, "n_events": n}},
            recommendation="Clustered events may indicate cascading failures or batch arrivals." if clustered
            else "Regular spacing may indicate scheduled processes.",
            severity="warn" if abs(max_dev) > T * 0.05 else "info",
            confidence=min(0.75, 0.4 + abs(max_dev) / T),
        ))
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Ripley's K temporal: max_dev={max_dev:.2f}.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "max_L_deviation": max_dev, "clustered": clustered})


def _getis_ord_hotspot(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Getis-Ord Gi* statistic for hotspot detection in sequential data."""
    _log_start(ctx, plugin_id, df, config, inferred)
    num_cols = _numeric_columns(df, inferred, max_cols=5)
    if not num_cols:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_columns")
    target = num_cols[0]
    vals = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
    valid_mask = np.isfinite(vals)
    if valid_mask.sum() < 10:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_valid_rows")
    y = vals[valid_mask]
    n = len(y)
    x_bar = float(np.mean(y))
    s = float(np.std(y))
    if s < 1e-12:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "zero_variance")
    # Compute Gi* with bandwidth=5 neighbors
    bw = min(5, n // 3)
    gi_star = np.zeros(n)
    for i in range(n):
        lo, hi = max(0, i - bw), min(n, i + bw + 1)
        w_sum = float(hi - lo)
        local_sum = float(np.sum(y[lo:hi]))
        gi_star[i] = (local_sum - x_bar * w_sum) / (s * math.sqrt((n * w_sum - w_sum ** 2) / max(n - 1, 1)))
    hotspots = int(np.sum(gi_star > 1.96))
    coldspots = int(np.sum(gi_star < -1.96))
    findings = []
    if hotspots > 0 or coldspots > 0:
        findings.append(_make_finding(
            plugin_id, "hotcoldspots",
            f"Gi* hotspots={hotspots}, coldspots={coldspots} in '{target}'",
            f"{hotspots} significant hotspot(s) and {coldspots} coldspot(s) at alpha=0.05.",
            "Hotspots are local concentrations of high values; coldspots are concentrations of low values.",
            {"metrics": {"hotspots": hotspots, "coldspots": coldspots, "column": target, "bandwidth": bw}},
            recommendation=f"Investigate {hotspots} hotspot region(s) in '{target}' for localized anomalies.",
            severity="warn" if hotspots > n * 0.1 else "info",
            confidence=min(0.8, 0.5 + (hotspots + coldspots) / n),
        ))
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Getis-Ord Gi*: {hotspots} hotspots, {coldspots} coldspots.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "hotspots": hotspots, "coldspots": coldspots})


def _lanchester_concentration(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Lanchester's square law: concentration advantage in resource allocation."""
    _log_start(ctx, plugin_id, df, config, inferred)
    cat_cols = _categorical_columns(df, inferred, max_cols=5)
    num_cols = _numeric_columns(df, inferred, max_cols=5)
    if not cat_cols or not num_cols:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "need_categorical_and_numeric")
    group_col = cat_cols[0]
    resource_col = num_cols[0]
    groups = df.groupby(group_col)[resource_col].agg(["sum", "count"]).dropna()
    if len(groups) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_groups")
    # Lanchester square law: combat power proportional to N^2
    groups["power"] = groups["count"].astype(float) ** 2
    total_power = float(groups["power"].sum())
    groups["power_share"] = groups["power"] / max(total_power, 1e-9)
    top = groups["power_share"].idxmax()
    top_share = float(groups.loc[top, "power_share"])
    hhi = float((groups["power_share"] ** 2).sum())  # concentration index
    findings = []
    if top_share > 0.4:
        findings.append(_make_finding(
            plugin_id, "concentration_advantage",
            f"Lanchester concentration: '{top}' holds {top_share*100:.1f}% of squared-power",
            f"Group '{top}' dominates with {top_share*100:.1f}% power share (HHI={hhi:.3f}).",
            "Lanchester's square law: concentrated resources have quadratically greater effectiveness.",
            {"metrics": {"top_group": str(top), "top_share": top_share, "hhi": hhi,
                         "group_col": group_col, "resource_col": resource_col}},
            recommendation=f"Rebalance resources across groups; '{top}' has disproportionate concentration.",
            severity="warn" if top_share > 0.6 else "info",
            confidence=min(0.8, 0.4 + top_share),
        ))
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Lanchester concentration: HHI={hhi:.3f}.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "hhi": hhi, "top_group": str(top)})


def _salvo_burst_threshold(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Find burst threshold that exceeds defensive capacity (salvo model)."""
    _log_start(ctx, plugin_id, df, config, inferred)
    time_col, ts = _time_series(df, inferred)
    if ts is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_time_column")
    dur_col = _duration_column(df, inferred)
    if dur_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_metric_column")
    frame = pd.DataFrame({"t": ts, "y": pd.to_numeric(df[dur_col], errors="coerce")}).dropna().sort_values("t")
    if len(frame) < 30:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    y = frame["y"].to_numpy(dtype=float)
    # Windowed burst analysis: find max burst intensity
    window_sizes = [5, 10, 20, 50]
    max_burst, burst_window = 0.0, 0
    median_load = float(np.median(y))
    for ws in window_sizes:
        if ws >= len(y): continue
        windowed_mean = np.convolve(y, np.ones(ws)/ws, mode="valid")
        peak = float(np.max(windowed_mean))
        if peak > max_burst:
            max_burst, burst_window = peak, ws
    burst_ratio = max_burst / max(median_load, 1e-9)
    findings = []
    if burst_ratio > 2.0:
        findings.append(_make_finding(
            plugin_id, "burst_threshold",
            f"Burst threshold exceeded: peak/median={burst_ratio:.2f}x (window={burst_window})",
            f"Peak windowed load={max_burst:.3f} vs median={median_load:.3f} ({burst_ratio:.1f}x).",
            "Burst loads exceeding defensive capacity cause cascading failures (salvo model).",
            {"metrics": {"burst_ratio": burst_ratio, "max_burst": max_burst, "median_load": median_load,
                         "burst_window": burst_window, "metric": dur_col}},
            recommendation=f"Provision burst capacity for {burst_ratio:.1f}x median load in '{dur_col}'.",
            severity="warn" if burst_ratio > 3.0 else "info",
            confidence=min(0.8, 0.4 + (burst_ratio - 2.0) * 0.1),
        ))
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Salvo burst: ratio={burst_ratio:.2f}.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "burst_ratio": burst_ratio})


def _koopman_search_allocation(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Koopman optimal search: distribute monitoring effort by detection probability."""
    _log_start(ctx, plugin_id, df, config, inferred)
    cat_cols = _categorical_columns(df, inferred, max_cols=5)
    num_cols = _numeric_columns(df, inferred, max_cols=5)
    if not cat_cols or not num_cols:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "need_categorical_and_numeric")
    region_col = cat_cols[0]
    signal_col = num_cols[0]
    groups = df.groupby(region_col)[signal_col].agg(["mean", "std", "count"]).dropna()
    if len(groups) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_groups")
    # Koopman: allocate effort proportional to sqrt(prior * signal_strength)
    groups["signal_strength"] = groups["std"] / groups["mean"].abs().clip(lower=1e-9)
    groups["prior"] = groups["count"] / float(groups["count"].sum())
    groups["effort_weight"] = np.sqrt(groups["prior"] * groups["signal_strength"])
    total_w = float(groups["effort_weight"].sum())
    groups["optimal_allocation"] = groups["effort_weight"] / max(total_w, 1e-9)
    top_region = groups["optimal_allocation"].idxmax()
    top_alloc = float(groups.loc[top_region, "optimal_allocation"])
    findings = []
    if top_alloc > 1.5 / len(groups):
        findings.append(_make_finding(
            plugin_id, "search_allocation",
            f"Koopman optimal: allocate {top_alloc*100:.1f}% monitoring to '{top_region}'",
            f"Optimal monitoring allocation is non-uniform across {len(groups)} regions.",
            "Koopman search theory: allocate effort where detection probability times signal strength is highest.",
            {"metrics": {"top_region": str(top_region), "top_allocation": top_alloc,
                         "allocations": groups["optimal_allocation"].to_dict()}},
            recommendation=f"Shift monitoring effort toward '{top_region}' (optimal={top_alloc*100:.0f}%).",
            severity="info", confidence=0.6,
        ))
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Koopman search: top region='{top_region}' ({top_alloc*100:.0f}%).", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "n_regions": len(groups)})


def _ooda_loop_bottleneck(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """OODA loop bottleneck: identify slowest phase in observe-orient-decide-act cycle."""
    _log_start(ctx, plugin_id, df, config, inferred)
    num_cols = _numeric_columns(df, inferred, max_cols=20)
    if len(num_cols) < 4:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "need_4_numeric_columns")
    # Map first 4 numeric columns to OODA phases
    phases = ["observe", "orient", "decide", "act"]
    phase_cols = num_cols[:4]
    phase_medians = {}
    for phase, col in zip(phases, phase_cols):
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if vals.empty:
            return _ok_with_reason(plugin_id, ctx, df, sample_meta, f"empty_phase_{phase}")
        phase_medians[phase] = {"column": col, "median": float(np.median(vals)), "std": float(np.std(vals))}
    total = sum(p["median"] for p in phase_medians.values())
    if total <= 0:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "zero_total_cycle")
    for phase in phases:
        phase_medians[phase]["fraction"] = phase_medians[phase]["median"] / total
    bottleneck = max(phases, key=lambda p: phase_medians[p]["fraction"])
    bn_frac = phase_medians[bottleneck]["fraction"]
    findings = []
    if bn_frac > 0.35:
        findings.append(_make_finding(
            plugin_id, "ooda_bottleneck",
            f"OODA bottleneck: '{bottleneck}' phase = {bn_frac*100:.1f}% of cycle",
            f"Phase '{bottleneck}' (column '{phase_medians[bottleneck]['column']}') dominates the cycle at {bn_frac*100:.1f}%.",
            "OODA loop: the slowest phase limits overall decision-action speed.",
            {"metrics": {"bottleneck": bottleneck, "fraction": bn_frac, "phases": phase_medians}},
            recommendation=f"Accelerate the '{bottleneck}' phase to reduce overall cycle time.",
            severity="warn" if bn_frac > 0.5 else "info",
            confidence=min(0.75, 0.4 + bn_frac),
        ))
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"OODA loop: bottleneck='{bottleneck}' ({bn_frac*100:.0f}%).", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "bottleneck": bottleneck, "bottleneck_frac": bn_frac})


def _chain_ladder_ibnr(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Chain-ladder method for incurred-but-not-reported estimation."""
    _log_start(ctx, plugin_id, df, config, inferred)
    time_col, ts = _time_series(df, inferred)
    if ts is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_time_column")
    dur_col = _duration_column(df, inferred)
    if dur_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_metric_column")
    frame = pd.DataFrame({"t": ts, "y": pd.to_numeric(df[dur_col], errors="coerce")}).dropna().sort_values("t")
    if len(frame) < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    # Build development triangle from time windows
    n_periods = min(8, len(frame) // 5)
    if n_periods < 3:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_periods")
    groups = np.array_split(np.arange(len(frame)), n_periods)
    cumulative = []
    for g in groups:
        vals = frame["y"].iloc[g].to_numpy(dtype=float)
        cumulative.append(float(np.nansum(vals)))
    # Chain-ladder development factors
    dev_factors = []
    for i in range(len(cumulative) - 1):
        if cumulative[i] > 0:
            dev_factors.append(cumulative[i + 1] / cumulative[i])
        else:
            dev_factors.append(1.0)
    avg_factor = float(np.mean(dev_factors)) if dev_factors else 1.0
    ultimate = cumulative[-1] * avg_factor
    ibnr = ultimate - cumulative[-1]
    ibnr_ratio = ibnr / max(cumulative[-1], 1e-9)
    findings = []
    if ibnr_ratio > 0.05:
        findings.append(_make_finding(
            plugin_id, "ibnr_estimate",
            f"IBNR estimate: {ibnr_ratio*100:.1f}% additional unreported for '{dur_col}'",
            f"Chain-ladder projects {ibnr:.2f} additional unreported (avg dev factor={avg_factor:.3f}).",
            "IBNR represents work/events that have occurred but not yet appeared in the data.",
            {"metrics": {"ibnr": ibnr, "ibnr_ratio": ibnr_ratio, "avg_dev_factor": avg_factor,
                         "ultimate": ultimate, "current": cumulative[-1]}},
            recommendation=f"Budget for ~{ibnr_ratio*100:.0f}% additional unreported {dur_col}.",
            severity="warn" if ibnr_ratio > 0.2 else "info",
            confidence=min(0.7, 0.4 + len(dev_factors) * 0.05),
        ))
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Chain-ladder IBNR: ratio={ibnr_ratio:.3f}.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "ibnr_ratio": ibnr_ratio, "avg_dev_factor": avg_factor})


def _buhlmann_credibility(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Buhlmann credibility weighting: Z = n/(n+K)."""
    _log_start(ctx, plugin_id, df, config, inferred)
    cat_cols = _categorical_columns(df, inferred, max_cols=5)
    num_cols = _numeric_columns(df, inferred, max_cols=5)
    if not cat_cols or not num_cols:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "need_categorical_and_numeric")
    group_col = cat_cols[0]
    metric_col = num_cols[0]
    grand_mean = float(pd.to_numeric(df[metric_col], errors="coerce").mean())
    groups = df.groupby(group_col)[metric_col].agg(["mean", "var", "count"]).dropna()
    if len(groups) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_groups")
    # Buhlmann: K = E[variance within] / Var[means between]
    within_var = float(groups["var"].mean())
    between_var = float(groups["mean"].var())
    K = within_var / max(between_var, 1e-9)
    groups["Z"] = groups["count"] / (groups["count"] + K)
    groups["credibility_estimate"] = groups["Z"] * groups["mean"] + (1 - groups["Z"]) * grand_mean
    low_cred = groups[groups["Z"] < 0.5]
    findings = []
    if len(low_cred) > 0:
        findings.append(_make_finding(
            plugin_id, "low_credibility",
            f"{len(low_cred)}/{len(groups)} groups have credibility Z < 0.5",
            f"K={K:.2f}. {len(low_cred)} group(s) lack sufficient data for reliable individual estimates.",
            "Buhlmann credibility: low Z means the group estimate is pulled toward the grand mean.",
            {"metrics": {"K": K, "n_low_cred": len(low_cred), "n_groups": len(groups),
                         "grand_mean": grand_mean, "group_col": group_col, "metric_col": metric_col}},
            recommendation=f"Collect more data for low-credibility groups or use pooled estimates.",
            severity="warn" if len(low_cred) > len(groups) * 0.5 else "info",
            confidence=min(0.75, 0.4 + len(groups) * 0.03),
        ))
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Buhlmann credibility: K={K:.2f}, {len(low_cred)} low-cred groups.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "K": K, "n_low_cred": len(low_cred)})


def _lee_carter_trend_decomp(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Lee-Carter SVD decomposition: separate age/period/trend components."""
    _log_start(ctx, plugin_id, df, config, inferred)
    num_cols = _variance_sorted_numeric(df, inferred, limit=8)
    if len(num_cols) < 3:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_numeric_columns")
    time_col, ts = _time_series(df, inferred)
    if ts is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_time_column")
    frame = df[num_cols].apply(pd.to_numeric, errors="coerce").copy()
    frame["_ts"] = ts
    frame = frame.sort_values("_ts").dropna(subset=["_ts"])
    if len(frame) < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    mat = frame[num_cols].to_numpy(dtype=float)
    mat = np.nan_to_num(mat, nan=0.0)
    # Center by column mean (alpha_x in Lee-Carter)
    alpha = np.mean(mat, axis=0)
    centered = mat - alpha
    # SVD
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    except Exception:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "svd_failed")
    total_var = float(np.sum(S ** 2))
    explained_first = float(S[0] ** 2 / max(total_var, 1e-15))
    kappa = U[:, 0] * S[0]  # time index (kt)
    beta = Vt[0, :]  # column loadings (bx)
    # Trend in kappa
    x_idx = np.arange(len(kappa), dtype=float)
    trend_slope = float(np.polyfit(x_idx, kappa, 1)[0]) if len(kappa) > 2 else 0.0
    findings = []
    if explained_first > 0.3:
        findings.append(_make_finding(
            plugin_id, "trend_decomposition",
            f"Lee-Carter: first SVD component explains {explained_first*100:.1f}% of variance",
            f"Trend slope={trend_slope:.4f}. Top loading columns: {', '.join(num_cols[:3])}.",
            "Lee-Carter decomposes variation into a dominant time trend (kappa) and column-specific loadings (beta).",
            {"metrics": {"explained_first": explained_first, "trend_slope": trend_slope,
                         "top_loadings": {c: float(b) for c, b in zip(num_cols, beta)}}},
            recommendation="Monitor the dominant trend component; it drives most of the variation." if trend_slope > 0
            else "Declining trend in first component may indicate improving conditions.",
            severity="warn" if abs(trend_slope) > float(np.std(kappa)) else "info",
            confidence=min(0.8, 0.4 + explained_first),
        ))
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Lee-Carter: explained={explained_first*100:.1f}%, trend={trend_slope:.4f}.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "explained_first": explained_first, "trend_slope": trend_slope})


def _experience_rating_score(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Experience modification factor (EMF) per host/entity."""
    _log_start(ctx, plugin_id, df, config, inferred)
    cat_cols = _categorical_columns(df, inferred, max_cols=5)
    num_cols = _numeric_columns(df, inferred, max_cols=5)
    if not cat_cols or not num_cols:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "need_categorical_and_numeric")
    entity_col = cat_cols[0]
    loss_col = num_cols[0]
    groups = df.groupby(entity_col)[loss_col].agg(["mean", "count"]).dropna()
    if len(groups) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_groups")
    grand_mean = float(pd.to_numeric(df[loss_col], errors="coerce").mean())
    if grand_mean == 0:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "zero_grand_mean")
    groups["emf"] = groups["mean"] / grand_mean
    worst = groups["emf"].idxmax()
    worst_emf = float(groups.loc[worst, "emf"])
    n_high = int((groups["emf"] > 1.2).sum())
    findings = []
    if n_high > 0:
        findings.append(_make_finding(
            plugin_id, "high_emf",
            f"{n_high} entities with EMF > 1.2 (worst: '{worst}' at {worst_emf:.2f}x)",
            f"{n_high}/{len(groups)} entities have experience modification factor > 1.2.",
            "EMF > 1 means worse than average experience; EMF < 1 means better. High EMF entities need attention.",
            {"metrics": {"worst_entity": str(worst), "worst_emf": worst_emf, "n_high_emf": n_high,
                         "entity_col": entity_col, "loss_col": loss_col}},
            recommendation=f"Investigate '{worst}' (EMF={worst_emf:.2f}x) and other high-EMF entities.",
            severity="warn" if worst_emf > 1.5 else "info",
            confidence=min(0.75, 0.4 + n_high * 0.05),
        ))
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Experience rating: {n_high} high-EMF entities.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "n_high_emf": n_high, "worst_emf": worst_emf})


def _teleconnection_lag(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Cross-correlation at various lags to find teleconnection (remote linkages)."""
    _log_start(ctx, plugin_id, df, config, inferred)
    num_cols = _variance_sorted_numeric(df, inferred, limit=6)
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
    teleconnections = []
    max_lag = min(20, len(frame) // 3)
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            _ensure_budget(timer)
            a = frame[num_cols[i]].to_numpy(dtype=float)
            b = frame[num_cols[j]].to_numpy(dtype=float)
            best_corr, best_lag = 0.0, 0
            for lag in range(1, max_lag + 1):
                c = float(np.corrcoef(a[lag:], b[:-lag])[0, 1])
                if math.isfinite(c) and abs(c) > abs(best_corr):
                    best_corr, best_lag = c, lag
            if abs(best_corr) > 0.3:
                teleconnections.append({"col_a": num_cols[i], "col_b": num_cols[j],
                                        "lag": best_lag, "correlation": best_corr})
    findings = []
    if teleconnections:
        top = max(teleconnections, key=lambda t: abs(t["correlation"]))
        findings.append(_make_finding(
            plugin_id, "teleconnection",
            f"Teleconnection: '{top['col_a']}' -> '{top['col_b']}' at lag {top['lag']} (r={top['correlation']:.2f})",
            f"{len(teleconnections)} lagged cross-correlation(s) detected across {len(num_cols)} variables.",
            "Teleconnections are remote linkages where changes in one variable predict future changes in another.",
            {"metrics": {"n_teleconnections": len(teleconnections), "top": top}},
            recommendation=f"Use '{top['col_a']}' as a leading indicator for '{top['col_b']}' (lag={top['lag']}).",
            severity="warn" if abs(top["correlation"]) > 0.5 else "info",
            confidence=min(0.8, 0.4 + abs(top["correlation"])),
        ))
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Teleconnection: {len(teleconnections)} linkage(s) found.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "n_teleconnections": len(teleconnections)})


def _return_period_evt(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Extreme value theory: return periods for threshold exceedances."""
    _log_start(ctx, plugin_id, df, config, inferred)
    dur_col = _duration_column(df, inferred)
    if dur_col is None:
        num_cols = _numeric_columns(df, inferred, max_cols=5)
        dur_col = num_cols[0] if num_cols else None
    if dur_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_numeric_column")
    vals = pd.to_numeric(df[dur_col], errors="coerce").dropna().to_numpy(dtype=float)
    if len(vals) < 30:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_rows")
    # Block maxima approach: fit GEV if scipy available
    threshold = float(np.percentile(vals, 90))
    exceedances = vals[vals > threshold]
    if len(exceedances) < 5:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_exceedances")
    if HAS_SCIPY:
        try:
            shape, loc, scale = scipy_stats.genextreme.fit(exceedances)
        except Exception:
            shape, loc, scale = 0.0, float(np.mean(exceedances)), float(np.std(exceedances))
    else:
        shape, loc, scale = 0.0, float(np.mean(exceedances)), float(np.std(exceedances))
    # Return periods
    return_levels = {}
    for rp in [10, 50, 100]:
        p = 1.0 / rp
        if abs(shape) > 1e-6:
            rl = loc + (scale / shape) * ((-np.log(1 - p)) ** (-shape) - 1)
        else:
            rl = loc - scale * np.log(-np.log(1 - p))
        return_levels[f"rp_{rp}"] = float(rl)
    findings = []
    rp100 = return_levels["rp_100"]
    current_max = float(np.max(vals))
    if rp100 > current_max * 1.5:
        findings.append(_make_finding(
            plugin_id, "extreme_return_period",
            f"EVT: 100-sample return level={rp100:.2f} vs observed max={current_max:.2f}",
            f"Extreme value analysis projects return levels significantly above current observations.",
            "Return period analysis estimates the magnitude of rare events beyond the observed sample.",
            {"metrics": {"return_levels": return_levels, "shape": float(shape), "loc": float(loc),
                         "scale": float(scale), "current_max": current_max, "column": dur_col}},
            recommendation=f"Prepare for '{dur_col}' values up to {rp100:.2f} (100-sample return level).",
            severity="warn" if shape > 0 else "info",
            confidence=min(0.75, 0.4 + len(exceedances) * 0.02),
        ))
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"EVT return periods computed for '{dur_col}'.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), **return_levels, "shape": float(shape)})


def _climate_attribution_far(
    plugin_id: str, ctx, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    """Fraction Attributable Risk (FAR): attribute extreme events to a factor."""
    _log_start(ctx, plugin_id, df, config, inferred)
    cat_cols = _categorical_columns(df, inferred, max_cols=5)
    num_cols = _numeric_columns(df, inferred, max_cols=5)
    if not cat_cols or not num_cols:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "need_categorical_and_numeric")
    factor_col = cat_cols[0]
    metric_col = num_cols[0]
    vals = pd.to_numeric(df[metric_col], errors="coerce")
    threshold = float(np.nanpercentile(vals.dropna(), 90))
    groups = df.groupby(factor_col).apply(
        lambda g: float((pd.to_numeric(g[metric_col], errors="coerce") > threshold).mean()),
        include_groups=False,
    )
    if len(groups) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_groups")
    overall_rate = float((vals > threshold).mean())
    if overall_rate <= 0:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_exceedances")
    far_values = {}
    for group_name, rate in groups.items():
        far = 1.0 - (overall_rate / max(rate, 1e-9)) if rate > overall_rate else 0.0
        far_values[str(group_name)] = {"rate": float(rate), "far": float(far)}
    max_far_group = max(far_values, key=lambda g: far_values[g]["far"])
    max_far = far_values[max_far_group]["far"]
    findings = []
    if max_far > 0.1:
        findings.append(_make_finding(
            plugin_id, "attributable_risk",
            f"FAR={max_far:.2f} for '{max_far_group}' (factor: '{factor_col}')",
            f"{max_far*100:.1f}% of extreme events in '{max_far_group}' are attributable to group membership.",
            "Fraction Attributable Risk quantifies how much a factor increases the probability of extreme events.",
            {"metrics": {"max_far_group": max_far_group, "max_far": max_far,
                         "factor_col": factor_col, "metric_col": metric_col, "threshold": threshold,
                         "far_by_group": far_values}},
            recommendation=f"Investigate why '{max_far_group}' has {max_far*100:.0f}% attributable risk for extreme '{metric_col}'.",
            severity="warn" if max_far > 0.3 else "info",
            confidence=min(0.75, 0.4 + max_far),
        ))
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Climate attribution FAR: max={max_far:.2f} for '{max_far_group}'.", findings, [],
        extra_metrics={"runtime_ms": _runtime_ms(timer), "max_far": max_far, "max_far_group": max_far_group})


# ---------------------------------------------------------------------------
# HANDLERS registry
# ---------------------------------------------------------------------------

HANDLERS: dict[str, Callable[..., PluginResult]] = {
    "analysis_bls_periodic_outage_v1": _bls_periodic_outage,
    "analysis_pulsar_timing_residual_v1": _pulsar_timing_residual,
    "analysis_hr_diagram_classification_v1": _hr_diagram_classification,
    "analysis_tidal_harmonic_cycles_v1": _tidal_harmonic_cycles,
    "analysis_wave_directional_spectra_v1": _wave_directional_spectra,
    "analysis_thermohaline_circulation_v1": _thermohaline_circulation,
    "analysis_ekman_transport_indirect_v1": _ekman_transport_indirect,
    "analysis_gutenberg_richter_bvalue_v1": _gutenberg_richter_bvalue,
    "analysis_omori_aftershock_decay_v1": _omori_aftershock_decay,
    "analysis_etas_self_exciting_cascade_v1": _etas_self_exciting_cascade,
    "analysis_psha_probabilistic_risk_v1": _psha_probabilistic_risk,
    "analysis_moran_i_autocorrelation_v1": _moran_i_autocorrelation,
    "analysis_kriging_missing_data_v1": _kriging_missing_data,
    "analysis_ripley_k_temporal_clustering_v1": _ripley_k_temporal_clustering,
    "analysis_getis_ord_hotspot_v1": _getis_ord_hotspot,
    "analysis_lanchester_concentration_v1": _lanchester_concentration,
    "analysis_salvo_burst_threshold_v1": _salvo_burst_threshold,
    "analysis_koopman_search_allocation_v1": _koopman_search_allocation,
    "analysis_ooda_loop_bottleneck_v1": _ooda_loop_bottleneck,
    "analysis_chain_ladder_ibnr_v1": _chain_ladder_ibnr,
    "analysis_buhlmann_credibility_v1": _buhlmann_credibility,
    "analysis_lee_carter_trend_decomp_v1": _lee_carter_trend_decomp,
    "analysis_experience_rating_score_v1": _experience_rating_score,
    "analysis_teleconnection_lag_v1": _teleconnection_lag,
    "analysis_return_period_evt_v1": _return_period_evt,
    "analysis_climate_attribution_far_v1": _climate_attribution_far,
}
