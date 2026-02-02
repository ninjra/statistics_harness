from __future__ import annotations

from typing import Iterable


def benjamini_hochberg(p_values: Iterable[float]) -> list[float]:
    values = [float(p) for p in p_values]
    n = len(values)
    if n == 0:
        return []
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    q_values = [0.0] * n
    prev = 1.0
    for rank, (idx, pval) in enumerate(reversed(indexed), start=1):
        adj = min(prev, (n / float(n - rank + 1)) * pval)
        prev = adj
        q_values[idx] = max(0.0, min(1.0, adj))
    return q_values


def confidence_from_p(p_value: float) -> float:
    try:
        pval = float(p_value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, 1.0 - pval))


def effect_size_ratio(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return float(a) / float(b)
