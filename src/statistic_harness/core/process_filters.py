from __future__ import annotations

from fnmatch import fnmatch
from typing import Iterable


def normalize_process(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def process_matches_token(process_id: object, token: object) -> bool:
    proc = normalize_process(process_id)
    term = normalize_process(token)
    if not proc or not term:
        return False
    if any(ch in term for ch in "*?["):
        return fnmatch(proc, term)
    return term in proc


def process_is_excluded(process_id: object, tokens: Iterable[object]) -> bool:
    return any(process_matches_token(process_id, token) for token in tokens)
