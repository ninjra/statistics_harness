from __future__ import annotations

import re
from typing import Any, Iterable


_SPLIT_RE = re.compile(r"[;|,\\n]+")


def normalize_kv_pairs(kv_pairs: Iterable[tuple[Any, Any]]) -> tuple[str, list[tuple[str, str]]]:
    """Normalize arbitrary (key,value) pairs into a deterministic canonical form.

    This is the same logical format used by `profile_basic` when populating
    `parameter_entities` / `parameter_kv`:
    - keys are lowercased + stripped
    - values are stripped
    - duplicates are removed
    - canonical string is sorted key/value pairs joined by ';'
    """

    cleaned: list[tuple[str, str]] = []
    for key, value in kv_pairs:
        k = str(key).strip().lower()
        v = str(value).strip()
        if k:
            cleaned.append((k, v))
    if not cleaned:
        return "", []
    cleaned = sorted(set(cleaned))
    canonical = ";".join(f"{k}={v}" for k, v in cleaned)
    return canonical, cleaned


def parse_parameter_text(text: Any) -> tuple[str, list[tuple[str, str]]] | None:
    """Best-effort parse of a parameter cell into canonical text + kv pairs.

    Prefer using the normalized parameter tables (`parameter_entities`, `parameter_kv`,
    `row_parameter_link`) when available. This parser is only for degraded/fallback paths.
    """

    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    tokens = _SPLIT_RE.split(raw)
    kv_pairs: list[tuple[str, str]] = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if "=" in token:
            k, v = token.split("=", 1)
        elif ":" in token:
            k, v = token.split(":", 1)
        else:
            continue
        kv_pairs.append((k, v))
    if kv_pairs:
        canonical, cleaned = normalize_kv_pairs(kv_pairs)
        if canonical:
            return canonical, cleaned
    canonical, cleaned = normalize_kv_pairs([("raw", raw)])
    return (canonical, cleaned) if canonical else None


def tokenize_kv_pairs(
    kv_pairs: Iterable[tuple[str, str]],
    *,
    ignore_keys_regex: str | None = None,
    include_values: bool = True,
) -> set[str]:
    """Convert normalized kv pairs into a token set for similarity methods."""

    ignore_re = None
    if ignore_keys_regex:
        try:
            ignore_re = re.compile(ignore_keys_regex, flags=re.IGNORECASE)
        except re.error:
            ignore_re = None
    tokens: set[str] = set()
    for k, v in kv_pairs:
        k2 = str(k).strip().lower()
        if not k2:
            continue
        if ignore_re and ignore_re.search(k2):
            continue
        if include_values and v is not None and str(v).strip() != "":
            tokens.add(f"{k2}={str(v).strip()}")
        else:
            tokens.add(k2)
    return tokens

