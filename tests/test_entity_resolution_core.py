from __future__ import annotations

from statistic_harness.core.entity_resolution import (
    build_token_inverted_index,
    match_entity,
    normalize_org_name,
    normalize_org_name_aggressive,
    tokenize,
)


def test_normalize_org_name_strips_suffix_and_punctuation() -> None:
    assert normalize_org_name("Acme, Inc.") == "ACME"
    assert normalize_org_name("Global Widgets LLC") == "GLOBAL WIDGETS"
    assert normalize_org_name("Foo & Bar Co.") == "FOO AND BAR"


def test_aggressive_normalization_sorts_unique_tokens() -> None:
    value = normalize_org_name_aggressive("Beta Alpha Alpha, Inc.")
    assert value == "ALPHA BETA"


def test_tokenize_respects_min_token_len() -> None:
    assert tokenize("A BB CCC", min_token_len=2) == ["BB", "CCC"]


def test_build_token_inverted_index_and_match_entity() -> None:
    rows = [
        {"key": "v1", "name": "ACME SYSTEMS"},
        {"key": "v2", "name": "GLOBAL WIDGETS"},
    ]
    index = build_token_inverted_index(rows, "name", key_field="key", min_token_len=3)
    assert "ACME" in index
    result = match_entity(
        "Acme Systems Inc.",
        {"v1": "ACME SYSTEMS", "v2": "GLOBAL WIDGETS"},
        token_index=index,
        fuzzy_threshold=80,
        min_token_len=3,
        token_overlap_min_ratio=0.5,
        min_overlap_tokens=1,
    )
    assert result is not None
    assert result.candidate_key == "v1"
    assert result.match_type in {"exact", "fuzzy", "token_overlap"}


def test_match_entity_tiebreak_is_deterministic() -> None:
    candidates = {
        "b_key": "ACME GROUP",
        "a_key": "ACME GROUP",
    }
    result = match_entity(
        "Acme Group",
        candidates,
        token_index=None,
        fuzzy_threshold=90,
        min_token_len=2,
        token_overlap_min_ratio=0.5,
        min_overlap_tokens=1,
    )
    assert result is not None
    assert result.candidate_key == "a_key"

