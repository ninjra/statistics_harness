"""Cross-domain plugins: signal processing, music, network & queueing (plugins 122-148)."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from typing import Any, Callable

import numpy as np
import pandas as pd

from statistic_harness.core.stat_plugins import (
    BudgetTimer, deterministic_sample, robust_center_scale, stable_id, standardized_median_diff,
)
from statistic_harness.core.types import PluginArtifact, PluginResult

try:
    from scipy import stats as scipy_stats
    from scipy import signal as scipy_signal
    from scipy import optimize as scipy_optimize
    HAS_SCIPY = True
except Exception:
    scipy_stats = scipy_signal = scipy_optimize = None
    HAS_SCIPY = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except Exception:
    nx = None
    HAS_NETWORKX = False

try:
    import librosa
    HAS_LIBROSA = True
except Exception:
    librosa = None
    HAS_LIBROSA = False

try:
    import ssqueezepy
    HAS_SSQ = True
except Exception:
    ssqueezepy = None
    HAS_SSQ = False

try:
    import ciw
    HAS_CIW = True
except Exception:
    ciw = None
    HAS_CIW = False

try:
    import leidenalg
    HAS_LEIDEN = True
except Exception:
    leidenalg = None
    HAS_LEIDEN = False


# ---------------------------------------------------------------------------
# Module-private helpers (duplicated per addon, standard pattern)
# ---------------------------------------------------------------------------

def _safe_id(plugin_id: str, key: str) -> str:
    try:
        return stable_id((plugin_id, key))
    except Exception:
        return hashlib.sha256(f"{plugin_id}:{key}".encode("utf-8")).hexdigest()[:16]


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
    kind: str | None = None,
) -> dict[str, Any]:
    finding: dict[str, Any] = {
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
        finding["kind"] = kind
    return finding


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
    runtime_ms = int(metrics.pop("runtime_ms", 0))
    ctx.logger(f"END runtime_ms={runtime_ms} findings={len(findings)}")
    merged_debug = dict(debug or {})
    merged_debug["runtime_ms"] = runtime_ms
    return PluginResult(
        "ok",
        summary,
        metrics,
        findings,
        artifacts,
        None,
        debug=merged_debug,
    )


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
        lname = str(col).lower()
        if "time" not in lname and "date" not in lname:
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
    scored: list[tuple[float, str]] = []
    for col in cols:
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        scored.append((float(np.nanvar(vals)), col))
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
    edges = edges[
        (edges["src"] != "") & (edges["dst"] != "") &
        (edges["src"] != "nan") & (edges["dst"] != "nan")
    ]
    edges = edges[edges["src"] != edges["dst"]]
    if len(edges) > max_edges:
        edges = edges.iloc[:max_edges]
    return edges, sc, dc


def _artifact_json(
    ctx: Any,
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


def _get_ts_values(df: pd.DataFrame, inferred: dict[str, Any]) -> tuple[np.ndarray, str] | None:
    """Extract a duration/numeric time series as sorted numpy array."""
    tc, ts = _time_series(df, inferred)
    col = _duration_column(df, inferred)
    if col is None:
        return None
    vals = pd.to_numeric(df[col], errors="coerce")
    if ts is not None:
        order = ts.argsort()
        vals = vals.iloc[order]
    arr = vals.dropna().to_numpy(dtype=float)
    if len(arr) < 30:
        return None
    return arr, col


def _build_nx_graph(edges: pd.DataFrame) -> Any:
    """Build a networkx DiGraph from an edges DataFrame with src/dst columns."""
    if not HAS_NETWORKX or edges.empty:
        return None
    G = nx.DiGraph()
    for _, row in edges.iterrows():
        G.add_edge(row["src"], row["dst"])
    return G


# ---------------------------------------------------------------------------
# Plugin handlers 122-148 (skipping 133 and 144)
# ---------------------------------------------------------------------------


# 122. analysis_cepstral_decomposition_v1
def _cepstral_decomposition(
    plugin_id: str, ctx: Any, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    result = _get_ts_values(df, inferred)
    if result is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_time_series")
    signal_arr, col = result
    n = len(signal_arr)
    # Compute real cepstrum
    spectrum = np.fft.rfft(signal_arr - np.mean(signal_arr))
    log_mag = np.log(np.maximum(np.abs(spectrum), 1e-12))
    cepstrum = np.fft.irfft(log_mag, n=n)
    _ensure_budget(timer)
    # Detect echoes: peaks in cepstrum beyond quefrency 2
    ceps_abs = np.abs(cepstrum[2: n // 2])
    if len(ceps_abs) == 0:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "signal_too_short")
    threshold = float(np.mean(ceps_abs) + 2.0 * np.std(ceps_abs))
    peaks = np.where(ceps_abs > threshold)[0]
    peak_quefrencies = (peaks + 2).tolist()[:10]
    peak_values = ceps_abs[peaks].tolist()[:10]
    # Lifter: separate envelope (low quefrency) from detail (high quefrency)
    lifter_cut = max(2, n // 20)
    envelope_ceps = np.copy(cepstrum)
    envelope_ceps[lifter_cut:] = 0.0
    envelope_spec = np.exp(np.fft.rfft(envelope_ceps, n=n).real)
    envelope_energy = float(np.sum(envelope_spec ** 2))
    total_energy = float(np.sum(np.abs(spectrum) ** 2))
    envelope_ratio = _safe_float(envelope_energy / max(total_energy, 1e-12))
    _ensure_budget(timer)
    findings: list[dict[str, Any]] = []
    if len(peak_quefrencies) > 0:
        findings.append(_make_finding(
            plugin_id, "echo_detected",
            "Echo/repetition pattern detected in cepstrum",
            f"Found {len(peak_quefrencies)} cepstral peak(s) indicating periodic echoes at quefrencies {peak_quefrencies[:5]}.",
            "Cepstral peaks reveal hidden periodicity or echo effects in the process signal.",
            {"metrics": {"column": col, "peak_quefrencies": peak_quefrencies, "peak_values": [round(v, 4) for v in peak_values], "envelope_ratio": round(envelope_ratio, 4)}},
            recommendation="Investigate periodic echo sources; consider deconvolution to remove repetitive artifacts.",
            severity="warn" if len(peak_quefrencies) >= 3 else "info",
            confidence=min(0.8, 0.4 + 0.1 * len(peak_quefrencies)),
        ))
    artifacts = [_artifact_json(ctx, plugin_id, "cepstral.json", {
        "column": col, "n_points": n, "lifter_cut": lifter_cut,
        "envelope_ratio": round(envelope_ratio, 4),
        "peak_quefrencies": peak_quefrencies, "peak_values": [round(v, 4) for v in peak_values],
    }, "Cepstral decomposition summary")]
    return _finalize(plugin_id, ctx, df, sample_meta,
        "Computed cepstral decomposition with echo detection.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "n_peaks": len(peak_quefrencies), "envelope_ratio": round(envelope_ratio, 4)})


# 123. analysis_synchrosqueezing_drift_v1
def _synchrosqueezing_drift(
    plugin_id: str, ctx: Any, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    result = _get_ts_values(df, inferred)
    if result is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_time_series")
    signal_arr, col = result
    n = len(signal_arr)
    if n < 64:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "signal_too_short")
    _ensure_budget(timer)
    # Use ssqueezepy if available, else fall back to STFT-based approach
    if HAS_SSQ:
        try:
            Tx, Wx, freqs, scales = ssqueezepy.ssq_cwt(signal_arr - np.mean(signal_arr))
            power = np.abs(Tx) ** 2
        except Exception:
            power = None
    else:
        power = None
    if power is None:
        # Fallback: compute STFT manually
        win_size = min(64, n // 4)
        hop = max(1, win_size // 2)
        windows = []
        for start in range(0, n - win_size + 1, hop):
            seg = signal_arr[start: start + win_size] - np.mean(signal_arr[start: start + win_size])
            spec = np.abs(np.fft.rfft(seg * np.hanning(win_size))) ** 2
            windows.append(spec)
        if not windows:
            return _ok_with_reason(plugin_id, ctx, df, sample_meta, "stft_failed")
        power = np.array(windows).T  # freq x time
    _ensure_budget(timer)
    # Track dominant frequency over time
    n_freq, n_time = power.shape
    dom_freq_idx = np.argmax(power, axis=0)
    # Detect drift: measure variance of dominant frequency index
    freq_drift_std = float(np.std(dom_freq_idx))
    freq_drift_range = int(np.max(dom_freq_idx) - np.min(dom_freq_idx))
    # Split into halves and compare dominant frequency
    half = n_time // 2
    mean_freq_first = float(np.mean(dom_freq_idx[:half]))
    mean_freq_second = float(np.mean(dom_freq_idx[half:]))
    freq_shift = abs(mean_freq_second - mean_freq_first)
    findings: list[dict[str, Any]] = []
    if freq_shift > 1.0 or freq_drift_std > 2.0:
        findings.append(_make_finding(
            plugin_id, "frequency_drift",
            "Drifting oscillatory component detected",
            f"Dominant frequency shifted by {freq_shift:.2f} bins between first and second half of signal.",
            "Frequency drift indicates non-stationary periodic behavior, suggesting changing process dynamics.",
            {"metrics": {"column": col, "freq_shift": round(freq_shift, 3), "freq_drift_std": round(freq_drift_std, 3), "freq_drift_range": freq_drift_range, "used_ssq": HAS_SSQ}},
            recommendation="Investigate root cause of changing periodicity; consider adaptive monitoring.",
            severity="warn" if freq_shift > 3.0 else "info",
            confidence=min(0.85, 0.5 + 0.05 * freq_shift),
        ))
    artifacts = [_artifact_json(ctx, plugin_id, "synchrosqueezing.json", {
        "column": col, "n_time_bins": n_time, "n_freq_bins": n_freq,
        "freq_drift_std": round(freq_drift_std, 4), "freq_shift": round(freq_shift, 4),
        "used_ssq": HAS_SSQ,
    }, "Synchrosqueezing drift analysis")]
    return _finalize(plugin_id, ctx, df, sample_meta,
        "Computed synchrosqueezing time-frequency drift analysis.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "freq_shift": round(freq_shift, 4)})


# 124. analysis_wigner_ville_coupling_v1
def _wigner_ville_coupling(
    plugin_id: str, ctx: Any, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    cols = _variance_sorted_numeric(df, inferred, limit=4)
    if len(cols) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_numeric_columns")
    tc, ts = _time_series(df, inferred)
    x1 = pd.to_numeric(df[cols[0]], errors="coerce").dropna().to_numpy(dtype=float)
    x2 = pd.to_numeric(df[cols[1]], errors="coerce").dropna().to_numpy(dtype=float)
    min_len = min(len(x1), len(x2))
    if min_len < 32:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "signal_too_short")
    x1, x2 = x1[:min_len], x2[:min_len]
    x1 = (x1 - np.mean(x1)) / max(np.std(x1), 1e-9)
    x2 = (x2 - np.mean(x2)) / max(np.std(x2), 1e-9)
    _ensure_budget(timer)
    # Pseudo Wigner-Ville: compute cross-term energy via cross-ambiguity
    # Simplified: use cross-spectral density as proxy
    n = min_len
    fft1 = np.fft.rfft(x1)
    fft2 = np.fft.rfft(x2)
    cross_spec = fft1 * np.conj(fft2)
    cross_power = np.abs(cross_spec) ** 2
    auto1 = np.abs(fft1) ** 2
    auto2 = np.abs(fft2) ** 2
    # Coherence per frequency bin
    denom = np.maximum(auto1 * auto2, 1e-12)
    coherence = np.abs(cross_spec) ** 2 / denom
    mean_coherence = float(np.mean(coherence))
    max_coherence = float(np.max(coherence))
    max_coherence_freq_bin = int(np.argmax(coherence))
    # Cross-term energy ratio (proxy for WV interference)
    total_energy = float(np.sum(auto1) + np.sum(auto2))
    cross_energy = float(np.sum(np.abs(cross_spec)))
    cross_ratio = _safe_float(cross_energy / max(total_energy, 1e-12))
    _ensure_budget(timer)
    findings: list[dict[str, Any]] = []
    if mean_coherence > 0.3:
        findings.append(_make_finding(
            plugin_id, "wv_coupling",
            "Wigner-Ville cross-term coupling detected",
            f"Mean coherence between {cols[0]} and {cols[1]} is {mean_coherence:.3f}, indicating frequency-domain coupling.",
            "Cross-terms in the Wigner-Ville distribution reveal interactions between signal components.",
            {"metrics": {"col_a": cols[0], "col_b": cols[1], "mean_coherence": round(mean_coherence, 4), "max_coherence": round(max_coherence, 4), "max_coherence_freq_bin": max_coherence_freq_bin, "cross_ratio": round(cross_ratio, 4)}},
            recommendation="Investigate shared drivers between these metrics; consider joint monitoring.",
            severity="warn" if mean_coherence > 0.6 else "info",
            confidence=min(0.85, mean_coherence),
        ))
    artifacts = [_artifact_json(ctx, plugin_id, "wigner_ville.json", {
        "col_a": cols[0], "col_b": cols[1], "mean_coherence": round(mean_coherence, 4),
        "max_coherence": round(max_coherence, 4), "cross_ratio": round(cross_ratio, 4),
    }, "Wigner-Ville coupling analysis")]
    return _finalize(plugin_id, ctx, df, sample_meta,
        "Computed Wigner-Ville coupling analysis.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "mean_coherence": round(mean_coherence, 4)})


# 125. analysis_spectral_coherence_coupling_v1
def _spectral_coherence_coupling(
    plugin_id: str, ctx: Any, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    cols = _variance_sorted_numeric(df, inferred, limit=6)
    if len(cols) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_numeric_columns")
    findings: list[dict[str, Any]] = []
    pair_results: list[dict[str, Any]] = []
    for i in range(min(len(cols), 5)):
        for j in range(i + 1, min(len(cols), 5)):
            _ensure_budget(timer)
            a = pd.to_numeric(df[cols[i]], errors="coerce").dropna().to_numpy(dtype=float)
            b = pd.to_numeric(df[cols[j]], errors="coerce").dropna().to_numpy(dtype=float)
            min_len = min(len(a), len(b))
            if min_len < 32:
                continue
            a, b = a[:min_len], b[:min_len]
            # Magnitude-squared coherence via Welch-like segments
            seg_len = min(64, min_len // 4)
            if seg_len < 8:
                continue
            n_segs = min_len // seg_len
            if n_segs < 2:
                continue
            Pxx = np.zeros(seg_len // 2 + 1)
            Pyy = np.zeros(seg_len // 2 + 1)
            Pxy = np.zeros(seg_len // 2 + 1, dtype=complex)
            for s in range(n_segs):
                sa = a[s * seg_len: (s + 1) * seg_len]
                sb = b[s * seg_len: (s + 1) * seg_len]
                fa = np.fft.rfft(sa - np.mean(sa))
                fb = np.fft.rfft(sb - np.mean(sb))
                Pxx += np.abs(fa) ** 2
                Pyy += np.abs(fb) ** 2
                Pxy += fa * np.conj(fb)
            denom = np.maximum(Pxx * Pyy, 1e-12)
            coh = np.abs(Pxy) ** 2 / denom
            mean_coh = float(np.mean(coh))
            max_coh = float(np.max(coh))
            max_bin = int(np.argmax(coh))
            pair_results.append({
                "col_a": cols[i], "col_b": cols[j],
                "mean_coherence": round(mean_coh, 4), "max_coherence": round(max_coh, 4),
                "max_freq_bin": max_bin,
            })
            if mean_coh > 0.35:
                findings.append(_make_finding(
                    plugin_id, f"coherence_{cols[i]}_{cols[j]}",
                    f"Spectral coherence: {cols[i]} ~ {cols[j]}",
                    f"Mean magnitude-squared coherence is {mean_coh:.3f} between {cols[i]} and {cols[j]}.",
                    "High spectral coherence at specific frequencies indicates shared oscillatory drivers.",
                    {"metrics": {"col_a": cols[i], "col_b": cols[j], "mean_coherence": round(mean_coh, 4), "max_coherence": round(max_coh, 4), "max_freq_bin": max_bin}},
                    recommendation=f"Investigate common periodic drivers between {cols[i]} and {cols[j]}.",
                    severity="warn" if mean_coh > 0.6 else "info",
                    confidence=min(0.85, mean_coh + 0.1),
                ))
    artifacts = [_artifact_json(ctx, plugin_id, "spectral_coherence.json", {"pairs": pair_results}, "Spectral coherence coupling")]
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Spectral coherence computed for {len(pair_results)} pairs.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "n_pairs": len(pair_results)})


# 126. analysis_matched_filter_anomaly_v1
def _matched_filter_anomaly(
    plugin_id: str, ctx: Any, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    result = _get_ts_values(df, inferred)
    if result is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_time_series")
    signal_arr, col = result
    n = len(signal_arr)
    if n < 50:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "signal_too_short")
    signal_arr = (signal_arr - np.mean(signal_arr)) / max(np.std(signal_arr), 1e-9)
    _ensure_budget(timer)
    # Define anomaly templates: spike, step, ramp
    templates = {}
    t_len = min(10, n // 10)
    if t_len >= 3:
        spike = np.zeros(t_len)
        spike[t_len // 2] = 1.0
        templates["spike"] = spike
        step = np.concatenate([np.zeros(t_len // 2), np.ones(t_len - t_len // 2)])
        templates["step"] = step
        ramp = np.linspace(0, 1, t_len)
        templates["ramp"] = ramp
    if not templates:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "template_too_short")
    detections: list[dict[str, Any]] = []
    for tname, template in templates.items():
        _ensure_budget(timer)
        template_norm = (template - np.mean(template))
        tnorm = np.sqrt(np.sum(template_norm ** 2))
        if tnorm < 1e-9:
            continue
        template_norm = template_norm / tnorm
        # Cross-correlate
        corr = np.correlate(signal_arr, template_norm, mode="valid")
        thresh = float(np.mean(np.abs(corr)) + 3.0 * np.std(np.abs(corr)))
        hits = np.where(np.abs(corr) > thresh)[0]
        if len(hits) > 0:
            # Cluster nearby hits
            clustered: list[int] = []
            last = -t_len * 2
            for h in hits:
                if h - last >= t_len:
                    clustered.append(int(h))
                    last = h
            for pos in clustered[:5]:
                detections.append({"template": tname, "position": pos, "score": round(float(np.abs(corr[pos])), 4)})
    findings: list[dict[str, Any]] = []
    if detections:
        findings.append(_make_finding(
            plugin_id, "matched_anomaly",
            "Matched-filter anomaly patterns detected",
            f"Detected {len(detections)} anomaly pattern occurrence(s) matching known templates.",
            "Matched filtering reveals recurrent anomaly shapes that may indicate systematic issues.",
            {"metrics": {"column": col, "n_detections": len(detections), "detections": detections[:10]}},
            recommendation="Review detected anomaly positions for root-cause patterns.",
            severity="warn" if len(detections) >= 3 else "info",
            confidence=min(0.8, 0.4 + 0.05 * len(detections)),
        ))
    artifacts = [_artifact_json(ctx, plugin_id, "matched_filter.json", {"column": col, "detections": detections, "template_length": t_len}, "Matched filter anomaly detection")]
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Matched filter anomaly detection: {len(detections)} hits.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "n_detections": len(detections)})


# 127. analysis_onset_detection_rhythm_v1
def _onset_detection_rhythm(
    plugin_id: str, ctx: Any, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    result = _get_ts_values(df, inferred)
    if result is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_time_series")
    signal_arr, col = result
    n = len(signal_arr)
    if n < 30:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "signal_too_short")
    signal_arr_norm = (signal_arr - np.mean(signal_arr)) / max(np.std(signal_arr), 1e-9)
    _ensure_budget(timer)
    # Onset detection: compute onset strength (spectral flux proxy)
    if HAS_LIBROSA:
        try:
            onset_env = librosa.onset.onset_strength(y=signal_arr_norm.astype(np.float32), sr=1, hop_length=1)
            onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=1, hop_length=1, backtrack=False)
        except Exception:
            onset_env = None
            onset_frames = None
    else:
        onset_env = None
        onset_frames = None
    if onset_env is None:
        # Fallback: first-difference energy
        diff = np.abs(np.diff(signal_arr_norm))
        onset_env = diff
        thresh = float(np.mean(diff) + 2.0 * np.std(diff))
        onset_frames = np.where(diff > thresh)[0]
    onset_positions = onset_frames.tolist() if hasattr(onset_frames, 'tolist') else list(onset_frames)
    n_onsets = len(onset_positions)
    # Inter-onset intervals
    ioi: list[float] = []
    if n_onsets >= 2:
        for k in range(1, n_onsets):
            ioi.append(float(onset_positions[k] - onset_positions[k - 1]))
    ioi_mean = float(np.mean(ioi)) if ioi else 0.0
    ioi_std = float(np.std(ioi)) if ioi else 0.0
    ioi_cv = _safe_float(ioi_std / max(ioi_mean, 1e-9))
    _ensure_budget(timer)
    findings: list[dict[str, Any]] = []
    if n_onsets >= 3:
        sev = "info"
        if ioi_cv > 0.5:
            sev = "warn"
        if ioi_cv > 1.0:
            sev = "critical"
        findings.append(_make_finding(
            plugin_id, "onset_rhythm",
            "Onset rhythm pattern detected",
            f"Detected {n_onsets} onsets with mean inter-onset interval {ioi_mean:.1f} and CV {ioi_cv:.3f}.",
            "Irregular onset spacing (high CV) indicates unpredictable process timing.",
            {"metrics": {"column": col, "n_onsets": n_onsets, "ioi_mean": round(ioi_mean, 3), "ioi_std": round(ioi_std, 3), "ioi_cv": round(ioi_cv, 4), "used_librosa": HAS_LIBROSA}},
            recommendation="Regularize process cadence to reduce timing variability." if ioi_cv > 0.5 else "Onset timing is relatively stable.",
            severity=sev,
            confidence=min(0.8, 0.4 + 0.05 * n_onsets),
        ))
    artifacts = [_artifact_json(ctx, plugin_id, "onset_rhythm.json", {
        "column": col, "n_onsets": n_onsets, "onset_positions": onset_positions[:50],
        "ioi_mean": round(ioi_mean, 4), "ioi_std": round(ioi_std, 4), "ioi_cv": round(ioi_cv, 4),
    }, "Onset detection rhythm analysis")]
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Onset detection: {n_onsets} onsets detected.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "n_onsets": n_onsets, "ioi_cv": round(ioi_cv, 4)})


# 128. analysis_tempo_tracking_cadence_v1
def _tempo_tracking_cadence(
    plugin_id: str, ctx: Any, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    result = _get_ts_values(df, inferred)
    if result is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_time_series")
    signal_arr, col = result
    n = len(signal_arr)
    if n < 40:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "signal_too_short")
    signal_norm = (signal_arr - np.mean(signal_arr)) / max(np.std(signal_arr), 1e-9)
    _ensure_budget(timer)
    # Estimate tempo via autocorrelation
    max_lag = min(n // 2, 500)
    autocorr = np.correlate(signal_norm[:max_lag * 2], signal_norm[:max_lag * 2], mode="full")
    autocorr = autocorr[len(autocorr) // 2:]  # positive lags only
    autocorr = autocorr[:max_lag]
    if len(autocorr) < 4:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "autocorrelation_too_short")
    # Normalize
    autocorr = autocorr / max(autocorr[0], 1e-9)
    # Find first significant peak after lag 1
    peak_lag = 0
    peak_val = 0.0
    for lag in range(2, len(autocorr)):
        if autocorr[lag] > peak_val and autocorr[lag] > autocorr[lag - 1] and (lag + 1 >= len(autocorr) or autocorr[lag] >= autocorr[lag + 1]):
            peak_val = float(autocorr[lag])
            peak_lag = lag
            break
    # Also try librosa tempo if available
    librosa_tempo = 0.0
    if HAS_LIBROSA and n >= 64:
        try:
            tempo = librosa.beat.tempo(y=signal_norm.astype(np.float32), sr=1, hop_length=1)
            librosa_tempo = float(tempo[0]) if len(tempo) > 0 else 0.0
        except Exception:
            pass
    dominant_period = peak_lag if peak_lag > 0 else 0
    _ensure_budget(timer)
    findings: list[dict[str, Any]] = []
    if dominant_period > 0 and peak_val > 0.2:
        findings.append(_make_finding(
            plugin_id, "cadence",
            "Dominant process cadence detected",
            f"Dominant period is {dominant_period} samples (autocorrelation peak {peak_val:.3f}).",
            "A clear cadence indicates regular process cycling; deviations from this cadence signal disruption.",
            {"metrics": {"column": col, "dominant_period": dominant_period, "peak_autocorr": round(peak_val, 4), "librosa_tempo": round(librosa_tempo, 4)}},
            recommendation="Monitor adherence to the detected cadence and alert on deviations.",
            severity="info",
            confidence=min(0.85, peak_val + 0.2),
        ))
    elif dominant_period == 0:
        findings.append(_make_finding(
            plugin_id, "no_cadence",
            "No clear process cadence found",
            "No dominant periodic pattern was detected in the signal.",
            "Lack of cadence may indicate aperiodic or chaotic process behavior.",
            {"metrics": {"column": col, "dominant_period": 0, "peak_autocorr": round(peak_val, 4)}},
            recommendation="Consider whether process should have a regular cadence; if so, investigate why none exists.",
            severity="info",
            confidence=0.5,
        ))
    artifacts = [_artifact_json(ctx, plugin_id, "tempo_cadence.json", {
        "column": col, "dominant_period": dominant_period, "peak_autocorr": round(peak_val, 4),
        "librosa_tempo": round(librosa_tempo, 4),
    }, "Tempo tracking cadence analysis")]
    return _finalize(plugin_id, ctx, df, sample_meta,
        "Computed tempo/cadence analysis.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "dominant_period": dominant_period})


# 129. analysis_self_similarity_matrix_v1
def _self_similarity_matrix(
    plugin_id: str, ctx: Any, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    result = _get_ts_values(df, inferred)
    if result is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_time_series")
    signal_arr, col = result
    n = len(signal_arr)
    # Window the signal into feature vectors
    win_size = min(16, n // 8)
    if win_size < 4:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "signal_too_short")
    hop = max(1, win_size // 2)
    features: list[np.ndarray] = []
    for start in range(0, n - win_size + 1, hop):
        seg = signal_arr[start: start + win_size]
        seg_norm = (seg - np.mean(seg)) / max(np.std(seg), 1e-9)
        features.append(seg_norm)
    if len(features) < 4:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "too_few_windows")
    _ensure_budget(timer)
    feat_matrix = np.array(features)  # n_windows x win_size
    # Compute cosine similarity matrix
    norms = np.linalg.norm(feat_matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    normed = feat_matrix / norms
    sim_matrix = normed @ normed.T  # n_windows x n_windows
    _ensure_budget(timer)
    # Detect repeating sections: look for off-diagonal high-similarity stripes
    n_win = len(features)
    diag_means: list[float] = []
    for d in range(1, min(n_win, 50)):
        diag_vals = np.array([sim_matrix[i, i + d] for i in range(n_win - d)])
        diag_means.append(float(np.mean(diag_vals)))
    # Find the lag with highest mean similarity (repetition period)
    if diag_means:
        best_lag = int(np.argmax(diag_means)) + 1
        best_sim = float(diag_means[best_lag - 1])
    else:
        best_lag = 0
        best_sim = 0.0
    overall_self_sim = float(np.mean(sim_matrix))
    findings: list[dict[str, Any]] = []
    if best_sim > 0.5:
        findings.append(_make_finding(
            plugin_id, "repeating_section",
            "Repeating structural sections detected",
            f"Self-similarity peak at lag {best_lag} windows (similarity {best_sim:.3f}).",
            "Repeating sections indicate cyclical process phases that could be optimized or standardized.",
            {"metrics": {"column": col, "best_lag_windows": best_lag, "best_similarity": round(best_sim, 4), "overall_self_sim": round(overall_self_sim, 4), "n_windows": n_win, "window_size": win_size}},
            recommendation="Analyze repeating sections for standardization opportunities.",
            severity="info",
            confidence=min(0.8, best_sim),
        ))
    artifacts = [_artifact_json(ctx, plugin_id, "self_similarity.json", {
        "column": col, "n_windows": n_win, "window_size": win_size,
        "best_lag_windows": best_lag, "best_similarity": round(best_sim, 4),
        "overall_self_sim": round(overall_self_sim, 4),
        "diag_means_top10": [round(v, 4) for v in diag_means[:10]],
    }, "Self-similarity matrix analysis")]
    return _finalize(plugin_id, ctx, df, sample_meta,
        "Computed self-similarity matrix analysis.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "best_lag": best_lag, "best_sim": round(best_sim, 4)})


# 130. analysis_markov_harmonic_progression_v1
def _markov_harmonic_progression(
    plugin_id: str, ctx: Any, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    result = _get_ts_values(df, inferred)
    if result is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_time_series")
    signal_arr, col = result
    n = len(signal_arr)
    if n < 30:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "signal_too_short")
    _ensure_budget(timer)
    # Quantize signal into states (like musical notes/chords)
    n_states = min(8, max(3, int(np.sqrt(n / 10))))
    percentiles = np.linspace(0, 100, n_states + 1)
    bins = np.percentile(signal_arr, percentiles)
    bins[0] -= 1e-9
    bins[-1] += 1e-9
    states = np.digitize(signal_arr, bins[1:-1])  # 0 to n_states-1
    # Build transition matrix
    trans = np.zeros((n_states, n_states), dtype=float)
    for i in range(len(states) - 1):
        trans[states[i], states[i + 1]] += 1.0
    row_sums = trans.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1.0)
    trans_prob = trans / row_sums
    _ensure_budget(timer)
    # Identify resolution patterns (high state -> low state = "tension release")
    # and deceptive cadences (expected resolution that doesn't happen)
    tension_states = list(range(n_states * 2 // 3, n_states))
    release_states = list(range(0, n_states // 3))
    resolution_prob = 0.0
    deceptive_prob = 0.0
    for ts_state in tension_states:
        for rs in release_states:
            resolution_prob += trans_prob[ts_state, rs]
        for ds in tension_states:
            deceptive_prob += trans_prob[ts_state, ds]
    n_tension = max(len(tension_states), 1)
    resolution_prob /= n_tension
    deceptive_prob /= n_tension
    # Entropy of transition matrix (predictability)
    entropy_per_state: list[float] = []
    for s in range(n_states):
        row = trans_prob[s]
        row = row[row > 0]
        if len(row) > 0:
            entropy_per_state.append(float(-np.sum(row * np.log(row))))
    mean_entropy = float(np.mean(entropy_per_state)) if entropy_per_state else 0.0
    max_entropy = float(math.log(max(n_states, 2)))
    norm_entropy = _safe_float(mean_entropy / max(max_entropy, 1e-9))
    findings: list[dict[str, Any]] = []
    if deceptive_prob > resolution_prob and deceptive_prob > 0.3:
        findings.append(_make_finding(
            plugin_id, "deceptive_cadence",
            "Deceptive cadence pattern detected",
            f"Tension states resolve back to tension {deceptive_prob:.2%} of the time vs release {resolution_prob:.2%}.",
            "Deceptive cadences indicate process escalation loops where high-load states persist.",
            {"metrics": {"column": col, "n_states": n_states, "resolution_prob": round(resolution_prob, 4), "deceptive_prob": round(deceptive_prob, 4), "norm_entropy": round(norm_entropy, 4)}},
            recommendation="Break escalation loops by adding forced resolution mechanisms.",
            severity="warn",
            confidence=min(0.8, deceptive_prob),
        ))
    elif norm_entropy > 0.8:
        findings.append(_make_finding(
            plugin_id, "high_entropy",
            "High transition entropy in Markov progression",
            f"Normalized transition entropy is {norm_entropy:.3f}, indicating unpredictable state changes.",
            "High entropy means the process state sequence is nearly random, limiting predictability.",
            {"metrics": {"column": col, "n_states": n_states, "norm_entropy": round(norm_entropy, 4)}},
            recommendation="Investigate sources of unpredictability in process state transitions.",
            severity="info",
            confidence=0.6,
        ))
    artifacts = [_artifact_json(ctx, plugin_id, "markov_harmonic.json", {
        "column": col, "n_states": n_states, "resolution_prob": round(resolution_prob, 4),
        "deceptive_prob": round(deceptive_prob, 4), "norm_entropy": round(norm_entropy, 4),
        "transition_matrix": trans_prob.tolist(),
    }, "Markov harmonic progression analysis")]
    return _finalize(plugin_id, ctx, df, sample_meta,
        "Computed Markov harmonic progression analysis.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "norm_entropy": round(norm_entropy, 4)})


# 131. analysis_polyrhythm_contention_v1
def _polyrhythm_contention(
    plugin_id: str, ctx: Any, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    result = _get_ts_values(df, inferred)
    if result is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_time_series")
    signal_arr, col = result
    n = len(signal_arr)
    if n < 64:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "signal_too_short")
    signal_norm = (signal_arr - np.mean(signal_arr)) / max(np.std(signal_arr), 1e-9)
    _ensure_budget(timer)
    # Find multiple periodic components via FFT peaks
    spectrum = np.abs(np.fft.rfft(signal_norm))
    freqs = np.fft.rfftfreq(n)
    # Ignore DC and very high freqs
    spectrum[0] = 0.0
    if len(spectrum) > 3:
        spectrum[-1] = 0.0
    # Find top peaks
    peak_threshold = float(np.mean(spectrum) + 2.0 * np.std(spectrum))
    peak_indices = np.where(spectrum > peak_threshold)[0]
    if len(peak_indices) == 0:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_periodic_components")
    # Sort by amplitude
    sorted_peaks = sorted(peak_indices, key=lambda i: -spectrum[i])[:6]
    periods = []
    for idx in sorted_peaks:
        if freqs[idx] > 0:
            period = 1.0 / freqs[idx]
            periods.append((period, float(spectrum[idx])))
    if len(periods) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "single_period_only")
    _ensure_budget(timer)
    # Compute least common period (LCM of approximate integer periods)
    int_periods = [max(2, int(round(p[0]))) for p in periods[:4]]
    lcm_val = int_periods[0]
    for ip in int_periods[1:]:
        lcm_val = lcm_val * ip // math.gcd(lcm_val, ip)
        if lcm_val > n * 10:
            lcm_val = n * 10
            break
    # Identify interference points: where multiple rhythms collide
    collision_count = 0
    collision_positions: list[int] = []
    for t in range(n):
        hits = sum(1 for ip in int_periods if ip > 0 and t % ip == 0)
        if hits >= 2:
            collision_count += 1
            if len(collision_positions) < 20:
                collision_positions.append(t)
    collision_density = _safe_float(collision_count / max(n, 1))
    findings: list[dict[str, Any]] = []
    findings.append(_make_finding(
        plugin_id, "polyrhythm",
        "Polyrhythmic contention pattern detected",
        f"Found {len(periods)} overlapping periodicities with LCM period {lcm_val}; {collision_count} collision points ({collision_density:.1%} density).",
        "Multiple overlapping rhythms create interference at collision points, causing contention.",
        {"metrics": {"column": col, "n_rhythms": len(periods), "periods": [(round(p, 2), round(a, 4)) for p, a in periods], "lcm_period": lcm_val, "collision_count": collision_count, "collision_density": round(collision_density, 4)}},
        recommendation="Stagger periodic processes to reduce collision density at interference points.",
        severity="warn" if collision_density > 0.1 else "info",
        confidence=min(0.8, 0.4 + 0.1 * len(periods)),
    ))
    artifacts = [_artifact_json(ctx, plugin_id, "polyrhythm.json", {
        "column": col, "periods": [(round(p, 2), round(a, 4)) for p, a in periods],
        "int_periods": int_periods, "lcm_period": lcm_val,
        "collision_count": collision_count, "collision_density": round(collision_density, 4),
        "collision_positions_sample": collision_positions,
    }, "Polyrhythm contention analysis")]
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Polyrhythm contention: {len(periods)} rhythms, {collision_count} collisions.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "n_rhythms": len(periods), "collision_density": round(collision_density, 4)})


# 132. analysis_community_detection_dag_v1
def _community_detection_dag(
    plugin_id: str, ctx: Any, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    if not HAS_NETWORKX:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "networkx_unavailable")
    edges, sc, dc = _build_edges(df, inferred)
    if edges.empty or not sc:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_graph_columns")
    G = nx.Graph()
    for _, row in edges.iterrows():
        G.add_edge(row["src"], row["dst"])
    if G.number_of_nodes() < 4:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "graph_too_small")
    _ensure_budget(timer)
    # Community detection
    communities: list[set[str]] = []
    if HAS_LEIDEN:
        try:
            import igraph as ig
            ig_g = ig.Graph.TupleList(edges[["src", "dst"]].itertuples(index=False), directed=False)
            partition = leidenalg.find_partition(ig_g, leidenalg.ModularityVertexPartition)
            names = ig_g.vs["name"]
            for comm_indices in partition:
                communities.append({names[i] for i in comm_indices})
        except Exception:
            communities = []
    if not communities:
        # Fallback to networkx greedy modularity
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            communities = [set(c) for c in greedy_modularity_communities(G)]
        except Exception:
            communities = []
    if not communities:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "community_detection_failed")
    _ensure_budget(timer)
    n_communities = len(communities)
    sizes = sorted([len(c) for c in communities], reverse=True)
    # Modularity
    try:
        partition_map = {}
        for idx, comm in enumerate(communities):
            for node in comm:
                partition_map[node] = idx
        modularity = nx.community.modularity(G, communities)
    except Exception:
        modularity = 0.0
    findings: list[dict[str, Any]] = []
    if n_communities >= 2:
        findings.append(_make_finding(
            plugin_id, "communities",
            "Workflow community structure detected",
            f"Found {n_communities} communities (sizes: {sizes[:5]}) with modularity {modularity:.3f}.",
            "Community structure reveals natural workflow clusters that may benefit from independent optimization.",
            {"metrics": {"src_col": sc, "dst_col": dc, "n_communities": n_communities, "sizes": sizes[:10], "modularity": round(modularity, 4), "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(), "used_leiden": HAS_LEIDEN}},
            recommendation="Consider organizing teams/processes along detected community boundaries.",
            severity="warn" if n_communities > 5 else "info",
            confidence=min(0.85, 0.4 + modularity),
        ))
    artifacts = [_artifact_json(ctx, plugin_id, "communities.json", {
        "src_col": sc, "dst_col": dc, "n_communities": n_communities,
        "sizes": sizes, "modularity": round(modularity, 4),
    }, "Community detection analysis")]
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Community detection: {n_communities} communities found.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "n_communities": n_communities, "modularity": round(modularity, 4)})


# 134. analysis_network_controllability_v1
def _network_controllability(
    plugin_id: str, ctx: Any, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    if not HAS_NETWORKX:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "networkx_unavailable")
    edges, sc, dc = _build_edges(df, inferred)
    if edges.empty or not sc:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_graph_columns")
    G = _build_nx_graph(edges)
    if G is None or G.number_of_nodes() < 3:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "graph_too_small")
    _ensure_budget(timer)
    # Structural controllability: minimum driver nodes via maximum matching
    # For a directed graph, unmatched nodes in maximum matching are driver nodes
    n_nodes = G.number_of_nodes()
    try:
        # Use bipartite matching on the directed graph
        B = nx.DiGraph(G)
        matching = nx.bipartite.maximum_matching(G.to_undirected())
        # Driver nodes = nodes not in the matching as targets
        matched_targets = set()
        for u, v in matching.items():
            if u in G and v in G and G.has_edge(u, v):
                matched_targets.add(v)
        driver_nodes = [n for n in G.nodes() if n not in matched_targets]
    except Exception:
        # Simpler heuristic: nodes with in-degree 0 are driver nodes
        driver_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
        if not driver_nodes:
            driver_nodes = [min(G.nodes(), key=lambda x: G.in_degree(x))]
    n_drivers = len(driver_nodes)
    driver_fraction = _safe_float(n_drivers / max(n_nodes, 1))
    _ensure_budget(timer)
    # Find uncontrollable components (nodes unreachable from any driver)
    reachable: set[Any] = set()
    for d in driver_nodes[:50]:
        reachable.update(nx.descendants(G, d))
        reachable.add(d)
    unreachable = [n for n in G.nodes() if n not in reachable]
    findings: list[dict[str, Any]] = []
    findings.append(_make_finding(
        plugin_id, "controllability",
        "Network controllability analysis",
        f"Minimum {n_drivers} driver nodes needed ({driver_fraction:.1%} of network); {len(unreachable)} nodes unreachable.",
        "Structural controllability reveals how many independent inputs are needed to steer the entire workflow.",
        {"metrics": {"src_col": sc, "dst_col": dc, "n_nodes": n_nodes, "n_drivers": n_drivers, "driver_fraction": round(driver_fraction, 4), "n_unreachable": len(unreachable), "driver_nodes_sample": [str(d) for d in driver_nodes[:10]]}},
        recommendation="Focus control efforts on identified driver nodes; investigate unreachable components.",
        severity="warn" if len(unreachable) > n_nodes * 0.1 else "info",
        confidence=0.7,
    ))
    artifacts = [_artifact_json(ctx, plugin_id, "controllability.json", {
        "n_nodes": n_nodes, "n_drivers": n_drivers, "driver_fraction": round(driver_fraction, 4),
        "n_unreachable": len(unreachable), "driver_nodes_sample": [str(d) for d in driver_nodes[:20]],
    }, "Network controllability analysis")]
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Network controllability: {n_drivers} driver nodes, {len(unreachable)} unreachable.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "n_drivers": n_drivers})


# 135. analysis_temporal_network_bottleneck_v1
def _temporal_network_bottleneck(
    plugin_id: str, ctx: Any, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    if not HAS_NETWORKX:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "networkx_unavailable")
    edges, sc, dc = _build_edges(df, inferred)
    if edges.empty or not sc:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_graph_columns")
    tc, ts = _time_series(df, inferred)
    if ts is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_time_column")
    _ensure_budget(timer)
    # Split data into time windows
    ts_valid = ts.dropna()
    if len(ts_valid) < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_timestamps")
    n_windows = min(5, max(2, len(df) // 100))
    window_size = len(df) // n_windows
    bottleneck_evolution: list[dict[str, Any]] = []
    for w in range(n_windows):
        _ensure_budget(timer)
        start_idx = w * window_size
        end_idx = min((w + 1) * window_size, len(df))
        window_df = df.iloc[start_idx:end_idx]
        w_edges, _, _ = _build_edges(window_df, inferred)
        if w_edges.empty:
            continue
        wG = nx.DiGraph()
        for _, row in w_edges.iterrows():
            wG.add_edge(row["src"], row["dst"])
        if wG.number_of_nodes() < 3:
            continue
        # Find bottleneck: node with highest betweenness centrality
        try:
            bc = nx.betweenness_centrality(wG)
            top_node = max(bc, key=bc.get)
            top_bc = bc[top_node]
        except Exception:
            top_node = ""
            top_bc = 0.0
        bottleneck_evolution.append({
            "window": w, "top_node": str(top_node), "betweenness": round(top_bc, 4),
            "n_nodes": wG.number_of_nodes(), "n_edges": wG.number_of_edges(),
        })
    if not bottleneck_evolution:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_valid_windows")
    # Check if bottleneck shifts
    top_nodes = [b["top_node"] for b in bottleneck_evolution]
    unique_bottlenecks = len(set(top_nodes))
    is_shifting = unique_bottlenecks > 1
    findings: list[dict[str, Any]] = []
    findings.append(_make_finding(
        plugin_id, "temporal_bottleneck",
        "Temporal network bottleneck analysis",
        f"Tracked bottlenecks across {len(bottleneck_evolution)} time windows; {unique_bottlenecks} unique bottleneck node(s).",
        "Shifting bottlenecks indicate dynamic process constraints that require adaptive optimization.",
        {"metrics": {"src_col": sc, "dst_col": dc, "n_windows": len(bottleneck_evolution), "unique_bottlenecks": unique_bottlenecks, "is_shifting": is_shifting, "evolution": bottleneck_evolution}},
        recommendation="Address shifting bottlenecks with adaptive capacity planning." if is_shifting else "Focus optimization on the stable bottleneck node.",
        severity="warn" if is_shifting else "info",
        confidence=0.7,
    ))
    artifacts = [_artifact_json(ctx, plugin_id, "temporal_bottleneck.json", {
        "evolution": bottleneck_evolution, "unique_bottlenecks": unique_bottlenecks,
    }, "Temporal network bottleneck analysis")]
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Temporal bottleneck: {unique_bottlenecks} unique bottleneck(s) across {len(bottleneck_evolution)} windows.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "unique_bottlenecks": unique_bottlenecks})


# 136. analysis_kshell_decomposition_v1
def _kshell_decomposition(
    plugin_id: str, ctx: Any, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    if not HAS_NETWORKX:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "networkx_unavailable")
    edges, sc, dc = _build_edges(df, inferred)
    if edges.empty or not sc:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_graph_columns")
    G = nx.Graph()
    for _, row in edges.iterrows():
        G.add_edge(row["src"], row["dst"])
    if G.number_of_nodes() < 4:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "graph_too_small")
    _ensure_budget(timer)
    # K-shell decomposition
    core_numbers = nx.core_number(G)
    max_core = max(core_numbers.values()) if core_numbers else 0
    # Group by core number
    shell_sizes: dict[int, int] = defaultdict(int)
    for node, k in core_numbers.items():
        shell_sizes[k] += 1
    # Innermost core nodes
    inner_core_nodes = [n for n, k in core_numbers.items() if k == max_core]
    n_nodes = G.number_of_nodes()
    inner_fraction = _safe_float(len(inner_core_nodes) / max(n_nodes, 1))
    _ensure_budget(timer)
    findings: list[dict[str, Any]] = []
    findings.append(_make_finding(
        plugin_id, "kshell",
        "K-shell decomposition reveals core structure",
        f"Max k-shell = {max_core}; innermost core has {len(inner_core_nodes)} nodes ({inner_fraction:.1%} of network).",
        "Deeply embedded core nodes are structurally critical and hard to remove without fragmenting the workflow.",
        {"metrics": {"src_col": sc, "dst_col": dc, "max_core": max_core, "n_inner_core": len(inner_core_nodes), "inner_fraction": round(inner_fraction, 4), "shell_sizes": dict(sorted(shell_sizes.items())), "inner_core_sample": [str(n) for n in inner_core_nodes[:10]]}},
        recommendation="Prioritize reliability and monitoring for innermost core nodes.",
        severity="warn" if max_core >= 4 else "info",
        confidence=0.75,
    ))
    artifacts = [_artifact_json(ctx, plugin_id, "kshell.json", {
        "max_core": max_core, "n_inner_core": len(inner_core_nodes),
        "shell_sizes": dict(sorted(shell_sizes.items())),
        "inner_core_sample": [str(n) for n in inner_core_nodes[:20]],
    }, "K-shell decomposition analysis")]
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"K-shell decomposition: max k={max_core}, {len(inner_core_nodes)} inner core nodes.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "max_core": max_core})


# 137. analysis_kingman_vut_decomposition_v1
def _kingman_vut_decomposition(
    plugin_id: str, ctx: Any, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    dur_col = _duration_column(df, inferred)
    if dur_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_duration_column")
    vals = pd.to_numeric(df[dur_col], errors="coerce").dropna().to_numpy(dtype=float)
    vals = vals[vals > 0]
    if len(vals) < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_data")
    _ensure_budget(timer)
    # VUT formula: Wq = (V * U * T) / (1 - U) approximately
    # V = variability factor = (ca^2 + cs^2) / 2
    # U = utilization (estimate from data)
    # T = mean service time
    service_mean = float(np.mean(vals))
    service_std = float(np.std(vals))
    cs_squared = _safe_float((service_std / max(service_mean, 1e-9)) ** 2)  # service CV^2
    # Estimate arrival rate from inter-arrival times if time series available
    tc, ts = _time_series(df, inferred)
    ca_squared = 1.0  # default: Poisson arrivals
    utilization = 0.7  # default estimate
    if ts is not None:
        ts_sorted = ts.dropna().sort_values()
        if len(ts_sorted) >= 10:
            iat = ts_sorted.diff().dt.total_seconds().dropna().to_numpy(dtype=float)
            iat = iat[iat > 0]
            if len(iat) >= 5:
                iat_mean = float(np.mean(iat))
                iat_std = float(np.std(iat))
                ca_squared = _safe_float((iat_std / max(iat_mean, 1e-9)) ** 2)
                arrival_rate = 1.0 / max(iat_mean, 1e-9)
                service_rate = 1.0 / max(service_mean, 1e-9)
                utilization = min(0.99, _safe_float(arrival_rate / max(service_rate, 1e-9)))
    V = (ca_squared + cs_squared) / 2.0
    U = utilization
    T = service_mean
    # Kingman formula
    if U < 0.99:
        Wq = V * (U / (1.0 - U)) * T
    else:
        Wq = V * 99.0 * T  # cap at U=0.99
    _ensure_budget(timer)
    # Decompose delay into components
    variability_component = V * T
    utilization_component = U / max(1.0 - U, 0.01)
    findings: list[dict[str, Any]] = []
    dominant = "variability" if V > utilization_component else "utilization"
    findings.append(_make_finding(
        plugin_id, "vut",
        "Kingman VUT delay decomposition",
        f"Expected wait Wq={Wq:.2f}; V(variability)={V:.3f}, U(utilization)={U:.3f}, T(service)={T:.2f}. Dominant factor: {dominant}.",
        "The VUT formula decomposes queueing delay into variability, utilization, and service time components.",
        {"metrics": {"column": dur_col, "Wq": round(Wq, 4), "V_factor": round(V, 4), "U_utilization": round(U, 4), "T_service_mean": round(T, 4), "ca_squared": round(ca_squared, 4), "cs_squared": round(cs_squared, 4), "dominant_factor": dominant}},
        recommendation=f"Reduce {dominant} to lower queueing delays." if Wq > T else "Queue delay is modest relative to service time.",
        severity="warn" if Wq > 2.0 * T else "info",
        confidence=0.7,
    ))
    artifacts = [_artifact_json(ctx, plugin_id, "kingman_vut.json", {
        "column": dur_col, "Wq": round(Wq, 4), "V": round(V, 4), "U": round(U, 4), "T": round(T, 4),
        "ca_squared": round(ca_squared, 4), "cs_squared": round(cs_squared, 4),
    }, "Kingman VUT decomposition")]
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Kingman VUT decomposition: Wq={Wq:.2f}, dominant={dominant}.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "Wq": round(Wq, 4)})


# 138. analysis_fork_join_straggler_v1
def _fork_join_straggler(
    plugin_id: str, ctx: Any, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    # Look for parallel branches: multiple numeric duration-like columns or a group column
    dur_cols = []
    for col in _numeric_columns(df, inferred, max_cols=20):
        if any(h in col.lower() for h in ("duration", "latency", "time", "elapsed", "wait")):
            dur_cols.append(col)
    if len(dur_cols) < 2:
        # Try using a group column to identify branches
        cats = _categorical_columns(df, inferred, max_cols=5)
        dur_col = _duration_column(df, inferred)
        if dur_col and cats:
            group_col = cats[0]
            groups = df.groupby(group_col)[dur_col].apply(lambda x: pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float))
            branch_data = {str(k): v for k, v in groups.items() if len(v) >= 5}
            if len(branch_data) < 2:
                return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_branches")
        else:
            return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_parallel_branches")
    else:
        branch_data = {}
        for col in dur_cols[:8]:
            vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
            if len(vals) >= 5:
                branch_data[col] = vals
    if len(branch_data) < 2:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_branches")
    _ensure_budget(timer)
    # Fork-join: completion = max(branch durations)
    # Straggler = branch that is most often the max
    branch_names = list(branch_data.keys())
    min_len = min(len(v) for v in branch_data.values())
    min_len = min(min_len, 5000)
    matrix = np.column_stack([branch_data[b][:min_len] for b in branch_names])
    max_indices = np.argmax(matrix, axis=1)
    straggler_counts = Counter(max_indices.tolist())
    straggler_fractions = {branch_names[k]: round(v / min_len, 4) for k, v in straggler_counts.items()}
    worst_branch_idx = max(straggler_counts, key=straggler_counts.get)
    worst_branch = branch_names[worst_branch_idx]
    worst_fraction = straggler_fractions[worst_branch]
    # Impact: how much would removing the straggler help?
    actual_completion = np.max(matrix, axis=1)
    without_worst = np.delete(matrix, worst_branch_idx, axis=1)
    counterfactual = np.max(without_worst, axis=1) if without_worst.shape[1] > 0 else np.zeros(min_len)
    savings = float(np.mean(actual_completion - counterfactual))
    savings_pct = _safe_float(savings / max(float(np.mean(actual_completion)), 1e-9))
    _ensure_budget(timer)
    findings: list[dict[str, Any]] = []
    findings.append(_make_finding(
        plugin_id, "straggler",
        "Fork-join straggler branch identified",
        f"Branch '{worst_branch}' is the straggler {worst_fraction:.1%} of the time; optimizing it would save {savings_pct:.1%} of completion time.",
        "In fork-join workflows, the slowest branch determines completion time. Reducing straggler delay has outsized impact.",
        {"metrics": {"n_branches": len(branch_names), "worst_branch": worst_branch, "worst_fraction": worst_fraction, "straggler_fractions": straggler_fractions, "mean_savings": round(savings, 4), "savings_pct": round(savings_pct, 4)}},
        recommendation=f"Prioritize optimizing branch '{worst_branch}' to reduce fork-join completion time.",
        severity="warn" if worst_fraction > 0.5 else "info",
        confidence=min(0.85, 0.5 + worst_fraction * 0.3),
    ))
    artifacts = [_artifact_json(ctx, plugin_id, "fork_join.json", {
        "branches": branch_names, "straggler_fractions": straggler_fractions,
        "worst_branch": worst_branch, "mean_savings": round(savings, 4), "savings_pct": round(savings_pct, 4),
    }, "Fork-join straggler analysis")]
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Fork-join straggler: '{worst_branch}' ({worst_fraction:.1%}).", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "worst_fraction": worst_fraction})


# 139. analysis_fluid_model_backlog_v1
def _fluid_model_backlog(
    plugin_id: str, ctx: Any, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    tc, ts = _time_series(df, inferred)
    dur_col = _duration_column(df, inferred)
    if ts is None or dur_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_time_series_or_duration")
    _ensure_budget(timer)
    # Sort by time
    frame = df.copy()
    frame["_ts"] = ts
    frame = frame.sort_values("_ts").dropna(subset=["_ts"])
    durations = pd.to_numeric(frame[dur_col], errors="coerce")
    ok = durations.notna()
    frame = frame[ok]
    durations = durations[ok].to_numpy(dtype=float)
    if len(frame) < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_data")
    # Estimate arrival rate in windows
    n = len(frame)
    n_windows = min(10, max(3, n // 50))
    window_size = n // n_windows
    backlog_periods: list[dict[str, Any]] = []
    total_backlog_time = 0
    for w in range(n_windows):
        _ensure_budget(timer)
        start = w * window_size
        end = min((w + 1) * window_size, n)
        w_ts = frame["_ts"].iloc[start:end]
        w_dur = durations[start:end]
        # Arrival rate
        if len(w_ts) < 3:
            continue
        time_span = (w_ts.iloc[-1] - w_ts.iloc[0]).total_seconds()
        if time_span <= 0:
            continue
        arrival_rate = len(w_ts) / time_span
        service_rate = 1.0 / max(float(np.mean(w_dur)), 1e-9)
        net_flow = arrival_rate - service_rate
        is_backlog = net_flow > 0
        if is_backlog:
            total_backlog_time += 1
        backlog_periods.append({
            "window": w, "arrival_rate": round(arrival_rate, 6), "service_rate": round(service_rate, 6),
            "net_flow": round(net_flow, 6), "is_backlog": is_backlog,
        })
    backlog_fraction = _safe_float(total_backlog_time / max(len(backlog_periods), 1))
    findings: list[dict[str, Any]] = []
    if backlog_fraction > 0.0:
        findings.append(_make_finding(
            plugin_id, "backlog",
            "Fluid model detects backlog accumulation",
            f"Arrival rate exceeds service rate in {backlog_fraction:.0%} of time windows.",
            "Fluid model backlog periods indicate capacity shortfalls where work accumulates faster than it is processed.",
            {"metrics": {"duration_col": dur_col, "n_windows": len(backlog_periods), "backlog_fraction": round(backlog_fraction, 4), "periods": backlog_periods}},
            recommendation="Add capacity during backlog periods or smooth arrival rate.",
            severity="critical" if backlog_fraction > 0.5 else "warn" if backlog_fraction > 0.2 else "info",
            confidence=min(0.8, 0.5 + backlog_fraction * 0.3),
        ))
    artifacts = [_artifact_json(ctx, plugin_id, "fluid_backlog.json", {
        "duration_col": dur_col, "backlog_fraction": round(backlog_fraction, 4), "periods": backlog_periods,
    }, "Fluid model backlog analysis")]
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Fluid model backlog: {backlog_fraction:.0%} of windows in backlog.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "backlog_fraction": round(backlog_fraction, 4)})


# 140. analysis_diffusion_approx_heavy_traffic_v1
def _diffusion_approx_heavy_traffic(
    plugin_id: str, ctx: Any, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    dur_col = _duration_column(df, inferred)
    if dur_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_duration_column")
    vals = pd.to_numeric(df[dur_col], errors="coerce").dropna().to_numpy(dtype=float)
    vals = vals[vals > 0]
    if len(vals) < 30:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_data")
    _ensure_budget(timer)
    service_mean = float(np.mean(vals))
    service_var = float(np.var(vals))
    cs_squared = _safe_float(service_var / max(service_mean ** 2, 1e-12))
    # Estimate utilization
    tc, ts = _time_series(df, inferred)
    rho = 0.85  # default high utilization assumption
    ca_squared = 1.0
    if ts is not None:
        ts_sorted = ts.dropna().sort_values()
        if len(ts_sorted) >= 10:
            iat = ts_sorted.diff().dt.total_seconds().dropna().to_numpy(dtype=float)
            iat = iat[iat > 0]
            if len(iat) >= 5:
                iat_mean = float(np.mean(iat))
                iat_var = float(np.var(iat))
                ca_squared = _safe_float(iat_var / max(iat_mean ** 2, 1e-12))
                rho = min(0.99, _safe_float(service_mean / max(iat_mean, 1e-9)))
    _ensure_budget(timer)
    # Heavy-traffic diffusion: queue modeled as reflected Brownian motion
    # Drift: mu = (1 - rho) * service_rate
    # Variance: sigma^2 = lambda * (ca^2 + cs^2 * rho^2)
    # Stationary distribution ~ Exponential(2*mu / sigma^2)
    service_rate = 1.0 / max(service_mean, 1e-9)
    drift = (1.0 - rho) * service_rate
    diffusion_var = service_rate * rho * (ca_squared + cs_squared * rho ** 2)
    if drift > 0 and diffusion_var > 0:
        # Exponential rate parameter
        exp_rate = 2.0 * drift / max(diffusion_var, 1e-9)
        mean_queue = 1.0 / max(exp_rate, 1e-9)
        p95_queue = -math.log(0.05) / max(exp_rate, 1e-9)
    else:
        mean_queue = float("inf")
        p95_queue = float("inf")
        exp_rate = 0.0
    mean_queue = min(mean_queue, 1e6)
    p95_queue = min(p95_queue, 1e6)
    findings: list[dict[str, Any]] = []
    findings.append(_make_finding(
        plugin_id, "heavy_traffic",
        "Heavy-traffic diffusion approximation",
        f"At utilization {rho:.2%}, diffusion model predicts mean queue {mean_queue:.2f} and P95 queue {p95_queue:.2f}.",
        "The heavy-traffic approximation models queue behavior near capacity as reflected Brownian motion.",
        {"metrics": {"column": dur_col, "rho": round(rho, 4), "ca_squared": round(ca_squared, 4), "cs_squared": round(cs_squared, 4), "mean_queue": round(mean_queue, 4), "p95_queue": round(p95_queue, 4), "exp_rate": round(exp_rate, 6)}},
        recommendation="Reduce utilization below 85% to prevent heavy-traffic queue explosion." if rho > 0.85 else "Current utilization is within safe bounds.",
        severity="critical" if rho > 0.95 else "warn" if rho > 0.85 else "info",
        confidence=0.65,
    ))
    artifacts = [_artifact_json(ctx, plugin_id, "diffusion_heavy_traffic.json", {
        "column": dur_col, "rho": round(rho, 4), "mean_queue": round(mean_queue, 4),
        "p95_queue": round(p95_queue, 4),
    }, "Heavy-traffic diffusion approximation")]
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Heavy-traffic diffusion: rho={rho:.2%}, mean_queue={mean_queue:.2f}.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "rho": round(rho, 4)})


# 141. analysis_phase_type_multimodal_v1
def _phase_type_multimodal(
    plugin_id: str, ctx: Any, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    dur_col = _duration_column(df, inferred)
    if dur_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_duration_column")
    vals = pd.to_numeric(df[dur_col], errors="coerce").dropna().to_numpy(dtype=float)
    vals = vals[vals > 0]
    if len(vals) < 30:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_data")
    _ensure_budget(timer)
    # Detect multimodality via kernel density estimation
    # Use histogram-based approach for robustness
    n_bins = min(50, max(10, int(np.sqrt(len(vals)))))
    hist, bin_edges = np.histogram(vals, bins=n_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    # Smooth histogram
    if len(hist) >= 5:
        kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        smoothed = np.convolve(hist, kernel, mode="same")
    else:
        smoothed = hist.astype(float)
    # Find peaks in smoothed histogram
    peaks: list[int] = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
            peaks.append(i)
    n_modes = max(1, len(peaks))
    _ensure_budget(timer)
    # Fit mixture of exponentials (phase-type proxy)
    mode_info: list[dict[str, Any]] = []
    if n_modes >= 2 and peaks:
        # Split data at valleys between peaks
        for p_idx, peak_bin in enumerate(peaks[:4]):
            center = float(bin_centers[peak_bin])
            density = float(smoothed[peak_bin])
            mode_info.append({"mode": p_idx, "center": round(center, 4), "density": round(density, 6)})
    else:
        mode_info.append({"mode": 0, "center": round(float(np.mean(vals)), 4), "density": 1.0})
    # Dip test proxy: compare actual distribution to unimodal
    overall_mean = float(np.mean(vals))
    overall_std = float(np.std(vals))
    cv = _safe_float(overall_std / max(overall_mean, 1e-9))
    # High CV with multiple peaks strongly suggests multimodality
    is_multimodal = n_modes >= 2
    findings: list[dict[str, Any]] = []
    if is_multimodal:
        findings.append(_make_finding(
            plugin_id, "multimodal",
            "Multimodal service time distribution detected",
            f"Service times show {n_modes} modes, suggesting {n_modes} distinct sub-processes.",
            "Multimodal service times indicate items follow different processing paths with different time profiles.",
            {"metrics": {"column": dur_col, "n_modes": n_modes, "cv": round(cv, 4), "modes": mode_info, "mean": round(overall_mean, 4), "std": round(overall_std, 4)}},
            recommendation="Segment work items by mode and optimize each sub-process independently.",
            severity="warn" if n_modes >= 3 else "info",
            confidence=min(0.8, 0.4 + 0.15 * n_modes),
        ))
    else:
        findings.append(_make_finding(
            plugin_id, "unimodal",
            "Service time distribution is unimodal",
            f"Service times show a single mode (CV={cv:.3f}).",
            "A unimodal distribution suggests a single dominant processing path.",
            {"metrics": {"column": dur_col, "n_modes": 1, "cv": round(cv, 4), "mean": round(overall_mean, 4)}},
            recommendation="Service time variability is the primary optimization lever." if cv > 1.0 else "Distribution is well-behaved.",
            severity="info",
            confidence=0.7,
        ))
    artifacts = [_artifact_json(ctx, plugin_id, "phase_type.json", {
        "column": dur_col, "n_modes": n_modes, "cv": round(cv, 4), "modes": mode_info,
    }, "Phase-type multimodal analysis")]
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Phase-type analysis: {n_modes} mode(s) detected.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "n_modes": n_modes})


# 142. analysis_erlang_bc_blocking_v1
def _erlang_bc_blocking(
    plugin_id: str, ctx: Any, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    dur_col = _duration_column(df, inferred)
    if dur_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_duration_column")
    vals = pd.to_numeric(df[dur_col], errors="coerce").dropna().to_numpy(dtype=float)
    vals = vals[vals > 0]
    if len(vals) < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_data")
    _ensure_budget(timer)
    service_mean = float(np.mean(vals))
    # Estimate arrival rate
    tc, ts = _time_series(df, inferred)
    arrival_rate = len(vals) / max(len(vals) * service_mean, 1e-9)  # default
    if ts is not None:
        ts_sorted = ts.dropna().sort_values()
        if len(ts_sorted) >= 5:
            total_time = (ts_sorted.iloc[-1] - ts_sorted.iloc[0]).total_seconds()
            if total_time > 0:
                arrival_rate = len(ts_sorted) / total_time
    traffic_intensity = arrival_rate * service_mean  # Erlangs

    # Erlang B formula: P_block = (A^N / N!) / sum(A^k / k! for k=0..N)
    def _erlang_b(A: float, N: int) -> float:
        if N <= 0 or A <= 0:
            return 1.0
        inv_b = 1.0
        for k in range(1, N + 1):
            inv_b = 1.0 + inv_b * k / A
        return 1.0 / inv_b

    # Erlang C formula: P_wait = (A^N / N!) * N/(N-A) / (sum(A^k/k!, k=0..N-1) + A^N/N! * N/(N-A))
    def _erlang_c(A: float, N: int) -> float:
        if N <= 0 or A <= 0 or A >= N:
            return 1.0
        pb = _erlang_b(A, N)
        return pb * N / (N - A * (1.0 - pb))

    # Test different server counts
    results: list[dict[str, Any]] = []
    for n_servers in range(max(1, int(traffic_intensity)), int(traffic_intensity) + 10):
        _ensure_budget(timer)
        pb = _erlang_b(traffic_intensity, n_servers)
        pc = _erlang_c(traffic_intensity, n_servers)
        results.append({
            "servers": n_servers, "erlang_b_blocking": round(pb, 6),
            "erlang_c_wait_prob": round(pc, 6),
        })
        if pb < 0.01 and pc < 0.05:
            break
    # Find minimum servers for acceptable blocking
    min_servers_1pct = int(traffic_intensity) + 1
    for r in results:
        if r["erlang_b_blocking"] < 0.01:
            min_servers_1pct = r["servers"]
            break
    current_blocking = results[0]["erlang_b_blocking"] if results else 1.0
    findings: list[dict[str, Any]] = []
    findings.append(_make_finding(
        plugin_id, "erlang",
        "Erlang B/C blocking probability analysis",
        f"Traffic intensity={traffic_intensity:.2f} Erlangs; need {min_servers_1pct} servers for <1% blocking.",
        "Erlang formulas model the probability of blocking (B) or waiting (C) given traffic load and capacity.",
        {"metrics": {"column": dur_col, "traffic_intensity": round(traffic_intensity, 4), "min_servers_1pct": min_servers_1pct, "results": results[:5]}},
        recommendation=f"Ensure at least {min_servers_1pct} servers/resources to maintain <1% blocking probability.",
        severity="warn" if current_blocking > 0.05 else "info",
        confidence=0.7,
    ))
    artifacts = [_artifact_json(ctx, plugin_id, "erlang_bc.json", {
        "traffic_intensity": round(traffic_intensity, 4), "min_servers_1pct": min_servers_1pct,
        "results": results,
    }, "Erlang B/C blocking analysis")]
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Erlang B/C: {traffic_intensity:.2f} Erlangs, need {min_servers_1pct} servers.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "traffic_intensity": round(traffic_intensity, 4)})


# 143. analysis_cdma_interference_sir_v1
def _cdma_interference_sir(
    plugin_id: str, ctx: Any, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    dur_col = _duration_column(df, inferred)
    if dur_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_duration_column")
    vals = pd.to_numeric(df[dur_col], errors="coerce").dropna().to_numpy(dtype=float)
    vals = vals[vals > 0]
    if len(vals) < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_data")
    _ensure_budget(timer)
    # Model concurrent jobs as CDMA interference
    # Estimate concurrency from timestamp overlap
    tc, ts = _time_series(df, inferred)
    if ts is not None:
        ts_sorted = ts.dropna().sort_values()
        if len(ts_sorted) >= 10:
            iat = ts_sorted.diff().dt.total_seconds().dropna().to_numpy(dtype=float)
            iat = iat[iat > 0]
            if len(iat) >= 5:
                mean_iat = float(np.mean(iat))
                mean_service = float(np.mean(vals))
                # Little's law estimate of concurrency
                concurrency = _safe_float(mean_service / max(mean_iat, 1e-9))
            else:
                concurrency = 5.0
        else:
            concurrency = 5.0
    else:
        concurrency = 5.0
    N = max(2, int(round(concurrency)))
    # SIR = signal / ((N-1) * interference)
    # Model: each job's "signal" is its ideal service time; interference adds delay
    ideal_service = float(np.percentile(vals, 10))  # fastest = least interference
    actual_mean = float(np.mean(vals))
    interference_per_job = _safe_float((actual_mean - ideal_service) / max(N - 1, 1))
    sir = _safe_float(ideal_service / max((N - 1) * interference_per_job, 1e-9))
    sir_db = 10.0 * math.log10(max(sir, 1e-9)) if sir > 0 else -99.0
    degradation_factor = _safe_float(actual_mean / max(ideal_service, 1e-9))
    _ensure_budget(timer)
    findings: list[dict[str, Any]] = []
    findings.append(_make_finding(
        plugin_id, "sir",
        "CDMA interference model for concurrent job contention",
        f"Estimated concurrency N={N}, SIR={sir_db:.1f} dB, degradation factor={degradation_factor:.2f}x.",
        "Modeling jobs as CDMA signals reveals how much concurrent contention degrades individual performance.",
        {"metrics": {"column": dur_col, "concurrency_N": N, "sir": round(sir, 4), "sir_db": round(sir_db, 2), "ideal_service": round(ideal_service, 4), "actual_mean": round(actual_mean, 4), "interference_per_job": round(interference_per_job, 4), "degradation_factor": round(degradation_factor, 4)}},
        recommendation=f"Reduce concurrency from {N} to lower interference; consider batching or scheduling.",
        severity="warn" if degradation_factor > 2.0 else "info",
        confidence=0.6,
    ))
    artifacts = [_artifact_json(ctx, plugin_id, "cdma_sir.json", {
        "column": dur_col, "concurrency_N": N, "sir_db": round(sir_db, 2),
        "degradation_factor": round(degradation_factor, 4),
    }, "CDMA interference SIR analysis")]
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"CDMA interference: N={N}, SIR={sir_db:.1f}dB.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "sir_db": round(sir_db, 2)})


# 145. analysis_heft_list_scheduling_v1
def _heft_list_scheduling(
    plugin_id: str, ctx: Any, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    if not HAS_NETWORKX:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "networkx_unavailable")
    edges, sc, dc = _build_edges(df, inferred)
    if edges.empty or not sc:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_graph_columns")
    dur_col = _duration_column(df, inferred)
    G = _build_nx_graph(edges)
    if G is None or G.number_of_nodes() < 3:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "graph_too_small")
    _ensure_budget(timer)
    # Assign weights (durations) to nodes
    node_weights: dict[Any, float] = {}
    if dur_col is not None:
        # Map average duration per node
        for node in G.nodes():
            mask = (df[sc].astype(str) == str(node)) | (df[dc].astype(str) == str(node))
            subset = pd.to_numeric(df.loc[mask, dur_col], errors="coerce").dropna()
            node_weights[node] = float(np.mean(subset)) if len(subset) > 0 else 1.0
    else:
        for node in G.nodes():
            node_weights[node] = 1.0
    # HEFT: compute upward rank for each node
    topo_order = list(nx.topological_sort(G)) if nx.is_directed_acyclic_graph(G) else list(G.nodes())
    upward_rank: dict[Any, float] = {}
    for node in reversed(topo_order):
        w = node_weights.get(node, 1.0)
        successors = list(G.successors(node))
        if not successors:
            upward_rank[node] = w
        else:
            upward_rank[node] = w + max(upward_rank.get(s, 0.0) for s in successors)
    _ensure_budget(timer)
    # Sort by upward rank (HEFT priority)
    priority = sorted(upward_rank.items(), key=lambda x: -x[1])
    # Compute critical path length
    cp_length = max(upward_rank.values()) if upward_rank else 0.0
    total_work = sum(node_weights.values())
    # Theoretical speedup with infinite processors
    parallelism = _safe_float(total_work / max(cp_length, 1e-9))
    # Compute actual makespan (sequential)
    sequential_makespan = total_work
    speedup = _safe_float(sequential_makespan / max(cp_length, 1e-9))
    findings: list[dict[str, Any]] = []
    findings.append(_make_finding(
        plugin_id, "heft",
        "HEFT scheduling analysis",
        f"Critical path={cp_length:.2f}, total work={total_work:.2f}, max parallelism={parallelism:.2f}x.",
        "HEFT scheduling reveals the optimal task ordering and the theoretical speedup from parallelization.",
        {"metrics": {"src_col": sc, "dst_col": dc, "critical_path": round(cp_length, 4), "total_work": round(total_work, 4), "parallelism": round(parallelism, 4), "n_nodes": G.number_of_nodes(), "top_priority": [(str(n), round(r, 4)) for n, r in priority[:5]]}},
        recommendation=f"Parallelize to achieve up to {parallelism:.1f}x speedup; prioritize critical-path tasks.",
        severity="warn" if parallelism > 3.0 else "info",
        confidence=0.7,
    ))
    artifacts = [_artifact_json(ctx, plugin_id, "heft.json", {
        "critical_path": round(cp_length, 4), "total_work": round(total_work, 4),
        "parallelism": round(parallelism, 4), "top_priority": [(str(n), round(r, 4)) for n, r in priority[:20]],
    }, "HEFT list scheduling analysis")]
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"HEFT scheduling: parallelism={parallelism:.2f}x.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "parallelism": round(parallelism, 4)})


# 146. analysis_dataflow_dead_output_v1
def _dataflow_dead_output(
    plugin_id: str, ctx: Any, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    if not HAS_NETWORKX:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "networkx_unavailable")
    edges, sc, dc = _build_edges(df, inferred)
    if edges.empty or not sc:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_graph_columns")
    G = _build_nx_graph(edges)
    if G is None or G.number_of_nodes() < 3:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "graph_too_small")
    _ensure_budget(timer)
    # Dead outputs: nodes that produce output (have outgoing edges from predecessors)
    # but whose output is never consumed (out-degree = 0 AND not a final sink)
    # More precisely: nodes with in-degree > 0 but out-degree = 0 that are not
    # labeled as terminal/final
    sinks = [n for n in G.nodes() if G.out_degree(n) == 0]
    sources = [n for n in G.nodes() if G.in_degree(n) == 0]
    # Dead outputs = intermediate nodes that have in-degree > 0 and out-degree 0
    # but are not in the "expected" sinks set
    # Without domain knowledge, all sinks could be legitimate or dead
    # Heuristic: sinks that have only one predecessor and that predecessor has other successors
    dead_candidates: list[dict[str, Any]] = []
    for sink in sinks:
        preds = list(G.predecessors(sink))
        if not preds:
            continue  # source-sink, not dead
        for p in preds:
            p_successors = list(G.successors(p))
            if len(p_successors) > 1:
                # This sink's parent has other children, so this sink might be dead
                dead_candidates.append({
                    "node": str(sink), "parent": str(p),
                    "parent_out_degree": len(p_successors),
                })
    # Also find intermediate nodes with low out-degree relative to in-degree (potential waste)
    n_nodes = G.number_of_nodes()
    dead_fraction = _safe_float(len(dead_candidates) / max(n_nodes, 1))
    _ensure_budget(timer)
    findings: list[dict[str, Any]] = []
    if dead_candidates:
        findings.append(_make_finding(
            plugin_id, "dead_output",
            "Potential dead outputs in workflow dataflow",
            f"Found {len(dead_candidates)} potential dead output node(s) ({dead_fraction:.1%} of graph).",
            "Dead outputs represent work products that are computed but never consumed downstream, indicating waste.",
            {"metrics": {"src_col": sc, "dst_col": dc, "n_dead_candidates": len(dead_candidates), "dead_fraction": round(dead_fraction, 4), "candidates": dead_candidates[:10], "n_sinks": len(sinks), "n_sources": len(sources)}},
            recommendation="Verify whether identified dead outputs are necessary; eliminate if they are pure waste.",
            severity="warn" if dead_fraction > 0.1 else "info",
            confidence=0.6,
        ))
    artifacts = [_artifact_json(ctx, plugin_id, "dead_output.json", {
        "n_dead_candidates": len(dead_candidates), "dead_fraction": round(dead_fraction, 4),
        "candidates": dead_candidates[:30],
    }, "Dataflow dead output analysis")]
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Dataflow analysis: {len(dead_candidates)} potential dead outputs.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "n_dead": len(dead_candidates)})


# 147. analysis_amdahl_law_serial_fraction_v1
def _amdahl_law_serial_fraction(
    plugin_id: str, ctx: Any, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    dur_col = _duration_column(df, inferred)
    if dur_col is None:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_duration_column")
    vals = pd.to_numeric(df[dur_col], errors="coerce").dropna().to_numpy(dtype=float)
    vals = vals[vals > 0]
    if len(vals) < 20:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "insufficient_data")
    _ensure_budget(timer)
    # Estimate serial fraction from scaling data
    # If we have a concurrency/parallelism column, use it directly
    par_col = None
    for col in _numeric_columns(df, inferred, max_cols=20):
        if any(h in col.lower() for h in ("parallel", "threads", "workers", "concurrency", "cores", "procs")):
            par_col = col
            break
    if par_col is not None:
        par_vals = pd.to_numeric(df[par_col], errors="coerce").dropna()
        dur_vals = pd.to_numeric(df[dur_col], errors="coerce").dropna()
        common = par_vals.index.intersection(dur_vals.index)
        if len(common) >= 5:
            N_arr = par_vals.loc[common].to_numpy(dtype=float)
            T_arr = dur_vals.loc[common].to_numpy(dtype=float)
            # Fit Amdahl's law: T(N) = T1 * (s + (1-s)/N)
            T1 = float(T_arr[np.argmin(N_arr)])  # time at lowest parallelism
            if T1 > 0:
                speedups = T1 / np.maximum(T_arr, 1e-9)
                # Estimate s via least squares: speedup = 1/(s + (1-s)/N)
                # => 1/speedup = s + (1-s)/N
                inv_speedup = 1.0 / np.maximum(speedups, 1e-9)
                inv_N = 1.0 / np.maximum(N_arr, 1.0)
                # Linear regression: inv_speedup = s + (1-s) * inv_N
                # => inv_speedup = s*(1 - inv_N) + inv_N
                A = np.column_stack([1.0 - inv_N, np.ones(len(inv_N))])
                try:
                    result_fit = np.linalg.lstsq(A, inv_speedup, rcond=None)
                    s = float(np.clip(result_fit[0][0], 0.0, 1.0))
                except Exception:
                    s = 0.5
            else:
                s = 0.5
        else:
            s = 0.5
    else:
        # Estimate from variance structure: serial fraction proxy from coefficient of variation
        cv = _safe_float(float(np.std(vals)) / max(float(np.mean(vals)), 1e-9))
        # Higher CV suggests more parallelizable work (variable portion)
        s = max(0.01, min(0.99, 1.0 - cv / max(cv + 1.0, 1e-9)))
    _ensure_budget(timer)
    # Amdahl predictions
    predictions: list[dict[str, Any]] = []
    for N in [1, 2, 4, 8, 16, 32, 64]:
        speedup_pred = 1.0 / (s + (1.0 - s) / N)
        efficiency = speedup_pred / N
        predictions.append({"N": N, "speedup": round(speedup_pred, 3), "efficiency": round(efficiency, 4)})
    max_speedup = 1.0 / max(s, 1e-9)
    findings: list[dict[str, Any]] = []
    findings.append(_make_finding(
        plugin_id, "amdahl",
        "Amdahl's law serial fraction estimate",
        f"Serial fraction s={s:.3f}; max theoretical speedup={max_speedup:.1f}x.",
        "Amdahl's law bounds the achievable speedup from parallelization given the serial fraction of work.",
        {"metrics": {"column": dur_col, "serial_fraction": round(s, 4), "max_speedup": round(max_speedup, 3), "has_parallelism_col": par_col is not None, "predictions": predictions}},
        recommendation=f"To improve beyond {max_speedup:.1f}x, reduce the serial fraction (currently {s:.1%}).",
        severity="warn" if s > 0.5 else "info",
        confidence=0.65 if par_col is None else 0.8,
    ))
    artifacts = [_artifact_json(ctx, plugin_id, "amdahl.json", {
        "serial_fraction": round(s, 4), "max_speedup": round(max_speedup, 3), "predictions": predictions,
    }, "Amdahl's law analysis")]
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Amdahl's law: s={s:.3f}, max speedup={max_speedup:.1f}x.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "serial_fraction": round(s, 4)})


# 148. analysis_graph_coloring_resources_v1
def _graph_coloring_resources(
    plugin_id: str, ctx: Any, df: pd.DataFrame, config: dict[str, Any],
    inferred: dict[str, Any], timer: BudgetTimer, sample_meta: dict[str, Any],
) -> PluginResult:
    _log_start(ctx, plugin_id, df, config, inferred)
    if not HAS_NETWORKX:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "networkx_unavailable")
    edges, sc, dc = _build_edges(df, inferred)
    if edges.empty or not sc:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "no_graph_columns")
    # Build undirected conflict graph
    G = nx.Graph()
    for _, row in edges.iterrows():
        G.add_edge(row["src"], row["dst"])
    if G.number_of_nodes() < 3:
        return _ok_with_reason(plugin_id, ctx, df, sample_meta, "graph_too_small")
    _ensure_budget(timer)
    # Graph coloring: greedy coloring
    coloring = nx.coloring.greedy_color(G, strategy="largest_first")
    n_colors = max(coloring.values()) + 1 if coloring else 1
    # Color distribution
    color_counts: dict[int, int] = defaultdict(int)
    for node, color in coloring.items():
        color_counts[color] += 1
    color_dist = dict(sorted(color_counts.items()))
    # Chromatic number lower bound: clique number
    try:
        clique_number = nx.graph_clique_number(G)
    except Exception:
        clique_number = 1
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = _safe_float(2.0 * n_edges / max(n_nodes * (n_nodes - 1), 1))
    _ensure_budget(timer)
    findings: list[dict[str, Any]] = []
    findings.append(_make_finding(
        plugin_id, "coloring",
        "Graph coloring resource analysis",
        f"Minimum {n_colors} resource slots needed (greedy coloring); clique lower bound={clique_number}.",
        "Graph coloring determines the minimum resources needed for conflict-free parallel scheduling.",
        {"metrics": {"src_col": sc, "dst_col": dc, "n_colors": n_colors, "clique_number": clique_number, "color_distribution": color_dist, "n_nodes": n_nodes, "n_edges": n_edges, "density": round(density, 4)}},
        recommendation=f"Provision at least {n_colors} independent resource slots for conflict-free scheduling.",
        severity="warn" if n_colors > 8 else "info",
        confidence=0.75,
    ))
    artifacts = [_artifact_json(ctx, plugin_id, "graph_coloring.json", {
        "n_colors": n_colors, "clique_number": clique_number,
        "color_distribution": color_dist, "density": round(density, 4),
    }, "Graph coloring resource analysis")]
    return _finalize(plugin_id, ctx, df, sample_meta,
        f"Graph coloring: {n_colors} colors needed.", findings, artifacts,
        extra_metrics={"runtime_ms": _runtime_ms(timer), "n_colors": n_colors})


# ---------------------------------------------------------------------------
# HANDLERS registry (25 entries, skipping #133 and #144)
# ---------------------------------------------------------------------------

HANDLERS: dict[str, Callable[..., PluginResult]] = {
    "analysis_cepstral_decomposition_v1": _cepstral_decomposition,
    "analysis_synchrosqueezing_drift_v1": _synchrosqueezing_drift,
    "analysis_wigner_ville_coupling_v1": _wigner_ville_coupling,
    "analysis_spectral_coherence_coupling_v1": _spectral_coherence_coupling,
    "analysis_matched_filter_anomaly_v1": _matched_filter_anomaly,
    "analysis_onset_detection_rhythm_v1": _onset_detection_rhythm,
    "analysis_tempo_tracking_cadence_v1": _tempo_tracking_cadence,
    "analysis_self_similarity_matrix_v1": _self_similarity_matrix,
    "analysis_markov_harmonic_progression_v1": _markov_harmonic_progression,
    "analysis_polyrhythm_contention_v1": _polyrhythm_contention,
    "analysis_community_detection_dag_v1": _community_detection_dag,
    "analysis_network_controllability_v1": _network_controllability,
    "analysis_temporal_network_bottleneck_v1": _temporal_network_bottleneck,
    "analysis_kshell_decomposition_v1": _kshell_decomposition,
    "analysis_kingman_vut_decomposition_v1": _kingman_vut_decomposition,
    "analysis_fork_join_straggler_v1": _fork_join_straggler,
    "analysis_fluid_model_backlog_v1": _fluid_model_backlog,
    "analysis_diffusion_approx_heavy_traffic_v1": _diffusion_approx_heavy_traffic,
    "analysis_phase_type_multimodal_v1": _phase_type_multimodal,
    "analysis_erlang_bc_blocking_v1": _erlang_bc_blocking,
    "analysis_cdma_interference_sir_v1": _cdma_interference_sir,
    "analysis_heft_list_scheduling_v1": _heft_list_scheduling,
    "analysis_dataflow_dead_output_v1": _dataflow_dead_output,
    "analysis_amdahl_law_serial_fraction_v1": _amdahl_law_serial_fraction,
    "analysis_graph_coloring_resources_v1": _graph_coloring_resources,
}
