from __future__ import annotations

from typing import Iterable


def bh_fdr(p_values: Iterable[float], alpha: float = 0.05) -> tuple[list[float], list[bool]]:
    p_list = [float(p) if p is not None else 1.0 for p in p_values]
    n = len(p_list)
    if n == 0:
        return [], []
    indexed = sorted(enumerate(p_list), key=lambda item: item[1])
    q_values = [1.0] * n
    prev_q = 1.0
    for rank, (idx, pval) in enumerate(reversed(indexed), start=1):
        i = n - rank + 1
        q = min(prev_q, pval * n / i)
        q_values[idx] = q
        prev_q = q
    decisions = [q <= alpha for q in q_values]
    return q_values, decisions
