from __future__ import annotations

from typing import Iterable

import numpy as np


def robust_center_scale(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    scale = mad * 1.4826
    if scale <= 0:
        q1, q3 = np.percentile(arr, [25, 75])
        scale = float((q3 - q1) / 1.349) if q3 > q1 else 0.0
    if scale <= 0:
        scale = float(np.std(arr))
    if scale <= 0:
        scale = 1.0
    return median, scale


def robust_zscores(values: Iterable[float]) -> np.ndarray:
    median, scale = robust_center_scale(values)
    arr = np.asarray(list(values), dtype=float)
    return (arr - median) / scale


def standardized_median_diff(left: Iterable[float], right: Iterable[float]) -> float:
    left_arr = np.asarray(list(left), dtype=float)
    right_arr = np.asarray(list(right), dtype=float)
    left_arr = left_arr[~np.isnan(left_arr)]
    right_arr = right_arr[~np.isnan(right_arr)]
    if left_arr.size == 0 or right_arr.size == 0:
        return 0.0
    combined = np.concatenate([left_arr, right_arr])
    _, scale = robust_center_scale(combined)
    if scale == 0:
        return 0.0
    return float(np.median(left_arr) - np.median(right_arr)) / scale


def cliffs_delta(left: Iterable[float], right: Iterable[float], max_pairs: int = 2000000) -> float:
    left_arr = np.asarray(list(left), dtype=float)
    right_arr = np.asarray(list(right), dtype=float)
    left_arr = left_arr[~np.isnan(left_arr)]
    right_arr = right_arr[~np.isnan(right_arr)]
    n_left = left_arr.size
    n_right = right_arr.size
    if n_left == 0 or n_right == 0:
        return 0.0
    pair_count = n_left * n_right
    if pair_count > max_pairs:
        left_idx = np.linspace(0, n_left - 1, int(np.sqrt(max_pairs)), dtype=int)
        right_idx = np.linspace(0, n_right - 1, int(np.sqrt(max_pairs)), dtype=int)
        left_arr = left_arr[left_idx]
        right_arr = right_arr[right_idx]
        n_left = left_arr.size
        n_right = right_arr.size
        pair_count = n_left * n_right
    diff = left_arr[:, None] - right_arr[None, :]
    more = float(np.sum(diff > 0))
    less = float(np.sum(diff < 0))
    return (more - less) / float(pair_count)


def cramers_v(table: np.ndarray) -> float:
    observed = np.asarray(table, dtype=float)
    if observed.size == 0:
        return 0.0
    total = observed.sum()
    if total <= 0:
        return 0.0
    row_sums = observed.sum(axis=1, keepdims=True)
    col_sums = observed.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / total
    valid = expected > 0
    chi2 = float(((observed - expected) ** 2 / np.where(valid, expected, 1.0)).sum())
    r, c = observed.shape
    denom = total * (min(r - 1, c - 1) or 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(chi2 / denom))
