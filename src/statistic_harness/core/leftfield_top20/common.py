from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import KFold
    from sklearn.neighbors import NearestNeighbors

    HAS_SKLEARN = True
except Exception:  # pragma: no cover - optional dependency guard
    RandomForestClassifier = None
    LogisticRegression = None
    roc_auc_score = None
    KFold = None
    NearestNeighbors = None
    HAS_SKLEARN = False


@dataclass(frozen=True)
class PreparedData:
    frame: pd.DataFrame
    matrix: np.ndarray
    numeric_cols: list[str]
    time_col: str | None
    process_col: str | None


def build_config(ctx) -> dict[str, Any]:
    return dict(getattr(ctx, "settings", None) or {})


def artifact(ctx, plugin_id: str, name: str, payload: Any, kind: str = "json") -> PluginArtifact:
    out_dir = ctx.artifacts_dir(plugin_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    write_json(path, payload)
    return PluginArtifact(path=str(path.relative_to(ctx.run_dir)), type=kind, description=name)


def finding(
    plugin_id: str,
    title: str,
    recommendation: str,
    score: float | None = None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "plugin_id": plugin_id,
        "kind": "leftfield_signal",
        "title": title,
        "recommendation": recommendation,
    }
    if isinstance(score, (int, float)):
        payload["score"] = float(score)
    if isinstance(evidence, dict) and evidence:
        payload["evidence"] = evidence
    return payload


def degraded(plugin_id: str, summary: str, metrics: dict[str, Any] | None = None) -> PluginResult:
    return PluginResult("degraded", summary, metrics or {}, [], [], None)


def rng(ctx, config: dict[str, Any]) -> np.random.Generator:
    seed = int(config.get("seed") or getattr(ctx, "run_seed", 0) or 0)
    return np.random.default_rng(seed)


def pick_time_col(df: pd.DataFrame) -> str | None:
    preferred = [
        "start_time",
        "end_time",
        "timestamp",
        "event_time",
        "created_at",
        "time",
        "date",
    ]
    cols = {str(c).strip().lower(): c for c in df.columns}
    for key in preferred:
        if key in cols:
            return str(cols[key])
    for col in df.columns:
        name = str(col).strip().lower()
        if "time" in name or "date" in name:
            return str(col)
    return None


def pick_process_col(df: pd.DataFrame) -> str | None:
    preferred = ["process_norm", "process", "activity", "task", "step"]
    cols = {str(c).strip().lower(): c for c in df.columns}
    for key in preferred:
        if key in cols:
            return str(cols[key])
    return None


def prepare_data(ctx, config: dict[str, Any]) -> PreparedData | PluginResult:
    df = ctx.dataset_loader()
    if df is None or df.empty:
        return degraded("leftfield", "Empty dataset", {"rows_seen": 0})

    # Leftfield methods include O(n^2)/O(n^3) kernels/eigensolvers; keep
    # deterministic subsampling bounded for full baseline runs.
    max_rows = int(config.get("max_rows") or 2000)
    max_features = int(config.get("max_features") or 12)
    local_rng = rng(ctx, config)
    if len(df) > max_rows:
        idx = np.sort(local_rng.choice(len(df), size=max_rows, replace=False))
        df = df.iloc[idx].reset_index(drop=True)

    time_col = pick_time_col(df)
    if time_col is not None:
        parsed = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        if parsed.notna().sum() >= max(20, int(len(df) * 0.2)):
            df = df.assign(__time__=parsed).sort_values("__time__").drop(columns=["__time__"]).reset_index(drop=True)
        else:
            time_col = None

    numeric = df.select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        converted = {}
        for col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if float(s.notna().mean()) >= 0.6:
                converted[str(col)] = s
        if converted:
            numeric = pd.DataFrame(converted)

    if numeric.empty:
        return degraded("leftfield", "No usable numeric columns", {"rows_seen": int(len(df))})

    valid_cols = [c for c in numeric.columns if numeric[c].notna().sum() >= 5]
    numeric = numeric[valid_cols]
    if numeric.empty:
        return degraded("leftfield", "Numeric columns were all sparse", {"rows_seen": int(len(df))})

    if numeric.shape[1] > max_features:
        variances = numeric.var(axis=0, skipna=True).sort_values(ascending=False)
        numeric = numeric[variances.index[:max_features]]

    filled = numeric.copy()
    for col in filled.columns:
        med = float(filled[col].median(skipna=True)) if filled[col].notna().any() else 0.0
        filled[col] = filled[col].fillna(med)

    mat = filled.to_numpy(dtype=float)
    process_col = pick_process_col(df)
    return PreparedData(
        frame=df,
        matrix=mat,
        numeric_cols=[str(c) for c in filled.columns],
        time_col=time_col,
        process_col=process_col,
    )


def split_early_late(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = int(x.shape[0])
    split = max(1, n // 2)
    return x[:split], x[split:]


def rbf_kernel(x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    y = x if y is None else y
    x2 = np.sum(x * x, axis=1, keepdims=True)
    y2 = np.sum(y * y, axis=1, keepdims=True).T
    d2 = np.maximum(0.0, x2 + y2 - 2.0 * (x @ y.T))
    med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
    gamma = 1.0 / max(1e-9, med)
    return np.exp(-gamma * d2)


def center_kernel(k: np.ndarray) -> np.ndarray:
    n = k.shape[0]
    h = np.eye(n) - np.ones((n, n)) / max(1, n)
    return h @ k @ h


def hsic_score(x: np.ndarray, y: np.ndarray) -> float:
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    kx = center_kernel(rbf_kernel(x))
    ky = center_kernel(rbf_kernel(y))
    n = int(x.shape[0])
    return float(np.trace(kx @ ky) / max(1.0, (n - 1) ** 2))


def safe_svd(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    return u, s, vt


def ssa_reconstruct(signal: np.ndarray, window: int, components: int) -> tuple[np.ndarray, np.ndarray]:
    n = int(signal.size)
    w = int(max(4, min(window, n - 1)))
    k = n - w + 1
    hankel = np.vstack([signal[i : i + k] for i in range(w)])
    u, s, vt = np.linalg.svd(hankel, full_matrices=False)
    r = max(1, min(components, len(s)))
    rec = (u[:, :r] * s[:r]) @ vt[:r, :]
    out = np.zeros(n, dtype=float)
    counts = np.zeros(n, dtype=float)
    for i in range(w):
        out[i : i + k] += rec[i, :]
        counts[i : i + k] += 1.0
    out /= np.maximum(1.0, counts)
    return out, s
