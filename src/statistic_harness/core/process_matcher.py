from __future__ import annotations

import os
import re
from fnmatch import fnmatchcase
from typing import Callable, Iterable


# Conservative defaults; can be overridden by env/known-issues settings.
_DEFAULT_EXCLUDE_PROCESS_PATTERNS = [
    "LOS*",
    "POSTWKFL",
    "POSTWJFL",
    "BKRV*",
    "JEPOST*",
]


def default_exclude_process_patterns() -> list[str]:
    return list(_DEFAULT_EXCLUDE_PROCESS_PATTERNS)


def parse_exclude_patterns_env(
    env_var: str = "STAT_HARNESS_EXCLUDE_PROCESSES",
) -> list[str]:
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return []
    out: list[str] = []
    for entry in re.split(r"[;,\s]+", raw):
        entry = entry.strip()
        if entry:
            out.append(entry)
    return out


def merge_patterns(*sources: Iterable[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for src in sources:
        for p in src:
            if not isinstance(p, str):
                continue
            s = p.strip()
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(s)
    return merged


def compile_patterns(patterns: Iterable[str]) -> Callable[[str], bool]:
    """Compile process exclude patterns into a predicate.

    Supported inputs:
    - Exact: POSTWKFL
    - Glob: LOS*
    - SQL-like: JEPOST% (and _ as single character)
    - Optional regex: re:^JEPOST.*
    """

    exact: set[str] = set()
    globs: list[str] = []
    regexes: list[re.Pattern[str]] = []

    for raw in patterns:
        if not isinstance(raw, str):
            continue
        p = raw.strip()
        if not p:
            continue

        if p.lower().startswith("re:"):
            # Regex is powerful and can be abused, but process names are short and we
            # keep pattern size bounded. Keep it enabled (doc asks for it), and fail
            # closed by ignoring invalid regexes.
            body = p[3:].strip()
            if not body or len(body) > 200:
                continue
            try:
                regexes.append(re.compile(body, flags=re.IGNORECASE))
            except re.error:
                continue
            continue

        # Convert SQL LIKE patterns to glob semantics.
        if "%" in p or "_" in p:
            p = p.replace("%", "*").replace("_", "?")

        lowered = p.lower()
        if any(ch in p for ch in ("*", "?")):
            # Use lowercased globs for case-insensitive matching.
            globs.append(lowered)
        else:
            exact.add(lowered)

    def _matches(value: str) -> bool:
        if not isinstance(value, str):
            return False
        s = value.strip()
        if not s:
            return False
        lowered = s.lower()
        if lowered in exact:
            return True
        for rgx in regexes:
            if rgx.search(s) is not None:
                return True
        for g in globs:
            if fnmatchcase(lowered, g):
                return True
        return False

    return _matches

