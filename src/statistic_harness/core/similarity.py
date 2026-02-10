from __future__ import annotations

import math
from typing import Iterable

from simhash import Simhash  # type: ignore


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter) / float(union) if union else 0.0


def cosine_sparse_counts(a: dict[str, int], b: dict[str, int]) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for k, va in a.items():
        na += float(va) * float(va)
        vb = b.get(k)
        if vb is not None:
            dot += float(va) * float(vb)
    for vb in b.values():
        nb += float(vb) * float(vb)
    denom = math.sqrt(na) * math.sqrt(nb)
    return float(dot) / float(denom) if denom else 0.0


def simhash_fingerprint(tokens: Iterable[str], *, bits: int = 64) -> int:
    toks = [t for t in tokens if str(t).strip()]
    if not toks:
        return 0
    return int(Simhash(toks, f=int(bits)).value)


def hamming_distance(a: int, b: int) -> int:
    return int((int(a) ^ int(b)).bit_count())

