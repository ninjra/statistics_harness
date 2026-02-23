from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any


_LEGAL_SUFFIXES = {
    "INC",
    "INCORPORATED",
    "LLC",
    "L L C",
    "LTD",
    "LIMITED",
    "CO",
    "COMPANY",
    "CORP",
    "CORPORATION",
    "LP",
    "L P",
    "LLP",
    "L L P",
    "PLC",
    "PC",
}
_PUNCT_RE = re.compile(r"[^A-Z0-9 ]+")
_SPACE_RE = re.compile(r"\s+")


def _strip_legal_suffixes(text: str) -> str:
    tokens = text.split()
    while tokens:
        tail = tokens[-1]
        if tail in _LEGAL_SUFFIXES:
            tokens.pop()
            continue
        if len(tokens) >= 2:
            tail2 = f"{tokens[-2]} {tokens[-1]}"
            if tail2 in _LEGAL_SUFFIXES:
                tokens = tokens[:-2]
                continue
        break
    return " ".join(tokens)


def normalize_org_name(name: str | None) -> str:
    if name is None:
        return ""
    text = str(name).upper().strip()
    if not text:
        return ""
    text = text.replace("&", " AND ")
    text = text.replace('"', " ").replace("'", " ")
    text = _PUNCT_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    text = _strip_legal_suffixes(text)
    return _SPACE_RE.sub(" ", text).strip()


def tokenize(norm: str, *, min_token_len: int = 1) -> list[str]:
    out: list[str] = []
    for token in (norm or "").split():
        t = token.strip()
        if t and len(t) >= int(min_token_len):
            out.append(t)
    return out


def normalize_org_name_aggressive(name: str | None) -> str:
    norm = normalize_org_name(name)
    if not norm:
        return ""
    uniq = sorted(set(tokenize(norm, min_token_len=1)))
    return " ".join(uniq)


def build_token_inverted_index(
    rows_iter: Any,
    name_field: str,
    *,
    key_field: str = "key",
    min_token_len: int = 4,
) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for row in rows_iter:
        if not isinstance(row, dict):
            continue
        key_raw = row.get(key_field)
        if key_raw is None:
            continue
        key = str(key_raw)
        norm = normalize_org_name(row.get(name_field))
        for token in tokenize(norm, min_token_len=max(1, int(min_token_len))):
            out.setdefault(token, set()).add(key)
    return out


def token_sort_ratio(a: str, b: str) -> float:
    sa = " ".join(sorted(tokenize(normalize_org_name(a), min_token_len=1)))
    sb = " ".join(sorted(tokenize(normalize_org_name(b), min_token_len=1)))
    if not sa and not sb:
        return 100.0
    try:
        from rapidfuzz import fuzz  # type: ignore

        return float(fuzz.ratio(sa, sb))
    except Exception:
        return 100.0 * SequenceMatcher(None, sa, sb).ratio()


@dataclass(frozen=True)
class MatchResult:
    candidate_key: str
    candidate_norm: str
    match_type: str
    confidence_tier: str
    score: float
    fuzzy_score: float
    overlap_ratio: float
    overlap_tokens: list[str]


def _confidence_for(match_type: str, fuzzy: float, overlap: float) -> str:
    if match_type == "exact":
        return "high"
    if match_type == "fuzzy":
        if fuzzy >= 95.0:
            return "high"
        if fuzzy >= 87.0:
            return "medium"
        return "low"
    if match_type == "token_overlap":
        if overlap >= 0.8:
            return "medium"
        return "low"
    return "low"


def match_entity(
    query: str | None,
    candidate_norms: dict[str, str],
    *,
    token_index: dict[str, set[str]] | None = None,
    fuzzy_threshold: int = 82,
    min_token_len: int = 4,
    token_overlap_min_ratio: float = 0.6,
    min_overlap_tokens: int = 2,
) -> MatchResult | None:
    query_norm = normalize_org_name(query)
    if not query_norm or not candidate_norms:
        return None

    exact_keys = sorted(k for k, v in candidate_norms.items() if v == query_norm)
    if exact_keys:
        key = exact_keys[0]
        return MatchResult(
            candidate_key=key,
            candidate_norm=candidate_norms[key],
            match_type="exact",
            confidence_tier="high",
            score=100.0,
            fuzzy_score=100.0,
            overlap_ratio=1.0,
            overlap_tokens=sorted(set(tokenize(query_norm, min_token_len=1))),
        )

    q_tokens = tokenize(query_norm, min_token_len=max(1, int(min_token_len)))
    candidate_keys: set[str] = set()
    if token_index and q_tokens:
        for token in q_tokens:
            candidate_keys.update(token_index.get(token, set()))
    if not candidate_keys:
        if len(candidate_norms) <= 5000:
            candidate_keys = set(candidate_norms.keys())
        else:
            return None

    ranked: list[MatchResult] = []
    q_set = set(q_tokens)
    for key in candidate_keys:
        norm = candidate_norms.get(key)
        if not norm:
            continue
        c_tokens = tokenize(norm, min_token_len=max(1, int(min_token_len)))
        c_set = set(c_tokens)
        overlap_tokens = sorted(q_set.intersection(c_set))
        denom = max(len(q_set), len(c_set), 1)
        overlap_ratio = float(len(overlap_tokens)) / float(denom)
        fuzzy = token_sort_ratio(query_norm, norm)
        if fuzzy >= float(fuzzy_threshold):
            match_type = "fuzzy"
            score = float(fuzzy)
        elif overlap_ratio >= float(token_overlap_min_ratio) and len(overlap_tokens) >= int(min_overlap_tokens):
            match_type = "token_overlap"
            score = overlap_ratio * 100.0
        else:
            continue
        ranked.append(
            MatchResult(
                candidate_key=str(key),
                candidate_norm=norm,
                match_type=match_type,
                confidence_tier=_confidence_for(match_type, fuzzy, overlap_ratio),
                score=score,
                fuzzy_score=fuzzy,
                overlap_ratio=overlap_ratio,
                overlap_tokens=overlap_tokens,
            )
        )
    if not ranked:
        return None
    ranked.sort(
        key=lambda m: (
            -float(m.score),
            str(m.candidate_norm),
            str(m.candidate_key),
        )
    )
    return ranked[0]

