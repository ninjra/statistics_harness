from __future__ import annotations

from typing import Any

_PROCESS_MATCH_KEYS: tuple[str, ...] = (
    "process",
    "process_norm",
    "process_name",
    "process_id",
    "activity",
    "process_matches",
)


def _match_key_aliases(key: str) -> tuple[str, ...]:
    token = str(key or "").strip().lower()
    if token in _PROCESS_MATCH_KEYS:
        return _PROCESS_MATCH_KEYS
    return (key,)


def _normalize_match_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip().lower()
    return value


def _collect_alias_values(item: dict[str, Any], key: str) -> list[Any]:
    out: list[Any] = []
    for alias in _match_key_aliases(key):
        if alias in item:
            out.append(item.get(alias))
    return out


def _matches_where_value(actual: Any, expected: Any) -> bool:
    if isinstance(actual, (list, tuple, set)):
        actual_norm = {_normalize_match_value(v) for v in actual}
        if isinstance(expected, (list, tuple, set)):
            expected_norm = {_normalize_match_value(v) for v in expected}
            return expected_norm.issubset(actual_norm)
        return _normalize_match_value(expected) in actual_norm
    if isinstance(expected, (list, tuple, set)):
        expected_norm = {_normalize_match_value(v) for v in expected}
        return _normalize_match_value(actual) in expected_norm
    return _normalize_match_value(actual) == _normalize_match_value(expected)


def _matches_contains_value(actual: Any, expected: Any) -> bool:
    if isinstance(actual, str):
        actual_text = actual.strip().lower()
        if isinstance(expected, (list, tuple, set)):
            return all(str(token).strip().lower() in actual_text for token in expected)
        return str(expected).strip().lower() in actual_text
    if isinstance(actual, (list, tuple, set)):
        actual_norm = {_normalize_match_value(v) for v in actual}
        if isinstance(expected, (list, tuple, set)):
            expected_norm = {_normalize_match_value(v) for v in expected}
            return expected_norm.issubset(actual_norm)
        return _normalize_match_value(expected) in actual_norm
    if isinstance(expected, (list, tuple, set)):
        expected_norm = {_normalize_match_value(v) for v in expected}
        return _normalize_match_value(actual) in expected_norm
    return _normalize_match_value(actual) == _normalize_match_value(expected)


def _matches_expected_impl(
    item: dict[str, Any],
    where: dict[str, Any] | None,
    contains: dict[str, Any] | None,
) -> bool:
    if where:
        for key, expected in where.items():
            values = _collect_alias_values(item, key)
            if not values or not any(_matches_where_value(actual, expected) for actual in values):
                return False
    if contains:
        for key, expected in contains.items():
            values = _collect_alias_values(item, key)
            if not values or not any(_matches_contains_value(actual, expected) for actual in values):
                return False
    return True


def _matches_expected(
    item: dict[str, Any],
    where: dict[str, Any] | None,
    contains: dict[str, Any] | None,
) -> bool:
    return _matches_expected_impl(item, where, contains)
