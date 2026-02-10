from __future__ import annotations

from statistic_harness.core.similarity import (
    cosine_sparse_counts,
    hamming_distance,
    jaccard,
    simhash_fingerprint,
)


def test_jaccard_edge_cases() -> None:
    assert jaccard(set(), set()) == 1.0
    assert jaccard({"a"}, set()) == 0.0
    assert jaccard({"a", "b"}, {"b", "c"}) == 1.0 / 3.0


def test_cosine_sparse_counts() -> None:
    a = {"x": 1, "y": 2}
    b = {"x": 1, "y": 2}
    c = {"x": 1}
    assert abs(cosine_sparse_counts(a, b) - 1.0) < 1e-12
    assert 0.0 < cosine_sparse_counts(a, c) < 1.0


def test_simhash_is_deterministic_for_same_tokens() -> None:
    tokens = ["a", "b", "c", "c"]
    fp1 = simhash_fingerprint(tokens, bits=64)
    fp2 = simhash_fingerprint(tokens, bits=64)
    assert fp1 == fp2
    assert hamming_distance(fp1, fp2) == 0
