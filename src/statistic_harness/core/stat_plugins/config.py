from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_COMMON_CONFIG: dict[str, Any] = {
    "seed": 1337,
    # For large datasets (~2M rows), row sampling is disallowed by default. Plugins
    # must scan via batching/streaming or accept the full in-memory load.
    "time_budget_ms": None,
    "max_rows": None,
    "allow_row_sampling": False,
    "max_cols": 80,
    "max_groups": 30,
    "max_findings": 30,
    "time_column": "auto",
    "group_by": "auto",
    "value_columns": "auto",
    "focus": {
        "mode": "full_scan",
        "windows": [],
        "include_full_scan": True,
    },
    "privacy": {
        "enable_redaction": True,
        "redact_patterns": ["email", "ip", "uuid", "credit_card", "phone"],
        "max_exemplars": 3,
        "allow_exemplar_snippets": False,
    },
    "verbosity": "normal",
}


def merge_config(config: dict[str, Any] | None) -> dict[str, Any]:
    if config is None:
        return deepcopy(DEFAULT_COMMON_CONFIG)
    merged = deepcopy(DEFAULT_COMMON_CONFIG)
    _deep_merge(merged, config)
    return merged


def _deep_merge(target: dict[str, Any], incoming: dict[str, Any]) -> None:
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
