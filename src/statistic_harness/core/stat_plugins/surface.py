from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math

import numpy as np
import pandas as pd

try:  # optional
    from sklearn.decomposition import PCA

    HAS_SKLEARN = True
except Exception:  # pragma: no cover - optional
    PCA = None
    HAS_SKLEARN = False


@dataclass(frozen=True)
class SurfaceGrid:
    z: np.ndarray  # (H,W) float64 with NaNs for missing
    mask: np.ndarray  # (H,W) bool, True where z is finite
    x_edges: np.ndarray  # (W+1,)
    y_edges: np.ndarray  # (H+1,)
    z_column: str
    x_column: str
    y_column: str


def infer_xy_columns(df: pd.DataFrame, numeric_cols: list[str]) -> tuple[str | None, str | None]:
    tokens_x = ("x", "lon", "lng", "long", "longitude", "coord_x")
    tokens_y = ("y", "lat", "latitude", "coord_y")
    x = None
    y = None
    for col in numeric_cols:
        name = str(col).lower()
        if x is None and any(tok == name or tok in name for tok in tokens_x):
            x = col
        if y is None and any(tok == name or tok in name for tok in tokens_y):
            y = col
    if x and y and x != y:
        return x, y
    # Fall back to first two numeric columns if present.
    if len(numeric_cols) >= 2:
        return numeric_cols[0], numeric_cols[1]
    return None, None


def _pca_embed_2d(X: np.ndarray) -> np.ndarray:
    if not (HAS_SKLEARN and PCA is not None):
        # Deterministic fallback: take first two columns (or pad with zeros).
        if X.shape[1] >= 2:
            return X[:, :2]
        if X.shape[1] == 1:
            return np.concatenate([X, np.zeros((X.shape[0], 1))], axis=1)
        return np.zeros((X.shape[0], 2))
    pca = PCA(n_components=2, random_state=0)
    return pca.fit_transform(X)


def build_surface(
    df: pd.DataFrame,
    *,
    z_column: str,
    x_column: str | None = None,
    y_column: str | None = None,
    numeric_columns: list[str] | None = None,
    grid_size: int = 32,
    max_points: int = 2000,
    fill: str = "neighbors",
) -> SurfaceGrid | None:
    """Build a deterministic 2D gridded surface from a tabular dataset.

    If explicit x/y are not provided (or missing), a 2D PCA embedding over numeric columns
    is used as a fallback coordinate system.
    """

    if df is None or df.empty or z_column not in df.columns:
        return None

    grid_size = int(max(8, min(256, grid_size)))
    max_points = int(max(50, max_points))

    frame = df.head(max_points).copy()
    z = pd.to_numeric(frame[z_column], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(z)
    if keep.sum() < 50:
        return None

    if x_column and y_column and x_column in frame.columns and y_column in frame.columns:
        x = pd.to_numeric(frame.loc[keep, x_column], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(frame.loc[keep, y_column], errors="coerce").to_numpy(dtype=float)
        z = z[keep]
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        x = x[valid]
        y = y[valid]
        z = z[valid]
        if x.size < 50:
            return None
        x_col = x_column
        y_col = y_column
    else:
        # PCA embedding fallback.
        numeric_columns = list(numeric_columns or [])
        cols = [c for c in numeric_columns if c in frame.columns and c != z_column]
        if len(cols) < 1:
            return None
        X = frame.loc[keep, cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        med = np.nanmedian(X, axis=0)
        X = np.where(np.isfinite(X), X, med)
        coords = _pca_embed_2d(X)
        x = coords[:, 0].astype(float, copy=False)
        y = coords[:, 1].astype(float, copy=False)
        z = z[keep].astype(float, copy=False)
        x_col = x_column or "pca_x"
        y_col = y_column or "pca_y"

    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    if not (math.isfinite(x_min) and math.isfinite(x_max) and math.isfinite(y_min) and math.isfinite(y_max)):
        return None
    if x_max == x_min:
        x_max = x_min + 1.0
    if y_max == y_min:
        y_max = y_min + 1.0

    x_edges = np.linspace(x_min, x_max, grid_size + 1)
    y_edges = np.linspace(y_min, y_max, grid_size + 1)

    xi = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, grid_size - 1)
    yi = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, grid_size - 1)

    # Aggregate cell medians (deterministic).
    cells: dict[tuple[int, int], list[float]] = {}
    for i in range(int(z.shape[0])):
        key = (int(yi[i]), int(xi[i]))
        cells.setdefault(key, []).append(float(z[i]))

    grid = np.full((grid_size, grid_size), np.nan, dtype=float)
    for (r, c), vals in cells.items():
        grid[r, c] = float(np.nanmedian(np.asarray(vals, dtype=float)))

    if fill == "neighbors":
        grid = _fill_neighbors(grid)
    mask = np.isfinite(grid)
    if mask.sum() < 10:
        return None

    return SurfaceGrid(
        z=grid,
        mask=mask,
        x_edges=x_edges,
        y_edges=y_edges,
        z_column=str(z_column),
        x_column=str(x_col),
        y_column=str(y_col),
    )


def _fill_neighbors(grid: np.ndarray, passes: int = 3) -> np.ndarray:
    filled = grid.copy()
    H, W = filled.shape
    for _ in range(int(max(1, passes))):
        updated = filled.copy()
        for r in range(H):
            for c in range(W):
                if math.isfinite(updated[r, c]):
                    continue
                neighbors: list[float] = []
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        rr = r + dr
                        cc = c + dc
                        if 0 <= rr < H and 0 <= cc < W and math.isfinite(updated[rr, cc]):
                            neighbors.append(float(updated[rr, cc]))
                if neighbors:
                    filled[r, c] = float(np.nanmean(np.asarray(neighbors, dtype=float)))
    return filled


def top_hotspots(surface: SurfaceGrid, *, top_n: int = 5) -> list[dict[str, Any]]:
    """Return top-N grid cells by z value (indices + approximate coordinates)."""

    z = surface.z
    mask = np.isfinite(z)
    if not mask.any():
        return []
    flat = z[mask]
    if flat.size == 0:
        return []
    order = np.argsort(flat)[::-1]
    # Map back to (r,c)
    coords = np.argwhere(mask)
    out: list[dict[str, Any]] = []
    for idx in order[: int(max(1, top_n))]:
        r, c = coords[int(idx)]
        x0 = float(surface.x_edges[int(c)])
        x1 = float(surface.x_edges[int(c) + 1])
        y0 = float(surface.y_edges[int(r)])
        y1 = float(surface.y_edges[int(r) + 1])
        out.append(
            {
                "row": int(r),
                "col": int(c),
                "z": float(z[int(r), int(c)]),
                "x_center": (x0 + x1) / 2.0,
                "y_center": (y0 + y1) / 2.0,
            }
        )
    return out


__all__ = [
    "SurfaceGrid",
    "build_surface",
    "infer_xy_columns",
    "top_hotspots",
]

