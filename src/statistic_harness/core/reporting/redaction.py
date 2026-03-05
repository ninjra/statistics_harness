"""Redaction: scrub hostnames, user IDs, and sensitive data before output.

Applied before writing business_summary.md and slide_kit CSVs.
"""
from __future__ import annotations

import re
from typing import Any


# Patterns that match typical hostnames and server identifiers
_HOSTNAME_PATTERNS = [
    re.compile(r"\b(?:[a-zA-Z][\w-]*\.)+[a-zA-Z]{2,}\b"),  # FQDN
    re.compile(r"\b(?:srv|server|host|node|vm|app|web|db|ws)[-_]?\d+\b", re.IGNORECASE),
]

# Patterns that match user identifiers
_USER_ID_PATTERNS = [
    re.compile(r"\b[A-Z]{2,5}\d{4,8}\b"),  # corporate IDs like AB12345
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),  # email addresses
]

# Forbidden column names that must not appear in slide_kit outputs
FORBIDDEN_COLUMNS = frozenset({
    "LOCAL_MACHINE_ID",
    "local_machine_id",
    "hostname",
    "server_name",
    "user_id",
    "user_email",
    "operator_id",
    "ip_address",
    "email",
})


def redact_hostnames(text: str, *, counter: dict[str, str] | None = None) -> str:
    """Replace hostname-like tokens with pseudonymized host_NN labels."""
    if counter is None:
        counter = {}
    for pattern in _HOSTNAME_PATTERNS:
        for match in pattern.finditer(text):
            token = match.group()
            if token not in counter:
                counter[token] = f"host_{len(counter) + 1:02d}"
            text = text.replace(token, counter[token])
    return text


def redact_user_ids(text: str, *, counter: dict[str, str] | None = None) -> str:
    """Replace user-ID-like tokens with pseudonymized user_NN labels."""
    if counter is None:
        counter = {}
    for pattern in _USER_ID_PATTERNS:
        for match in pattern.finditer(text):
            token = match.group()
            if token not in counter:
                counter[token] = f"user_{len(counter) + 1:02d}"
            text = text.replace(token, counter[token])
    return text


def pseudonymize(text: str) -> str:
    """Apply all redaction passes to text."""
    host_map: dict[str, str] = {}
    user_map: dict[str, str] = {}
    text = redact_hostnames(text, counter=host_map)
    text = redact_user_ids(text, counter=user_map)
    return text


def check_forbidden_columns(headers: list[str]) -> list[str]:
    """Return list of forbidden column names found in headers."""
    return sorted(set(headers) & FORBIDDEN_COLUMNS)


def redact_dict_values(
    data: dict[str, Any],
    keys_to_redact: set[str] | None = None,
) -> dict[str, Any]:
    """Redact values in a dict for specified keys."""
    if keys_to_redact is None:
        keys_to_redact = {"hostname", "local_machine_id", "user_id", "email"}
    out: dict[str, Any] = {}
    for key, value in data.items():
        if key.lower() in keys_to_redact and isinstance(value, str):
            out[key] = pseudonymize(value)
        elif isinstance(value, dict):
            out[key] = redact_dict_values(value, keys_to_redact)
        else:
            out[key] = value
    return out
