from __future__ import annotations

import re
from typing import Callable, Iterable

import pandas as pd

DEFAULT_PATTERNS: dict[str, str] = {
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "ip": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    "uuid": r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b",
    "credit_card": r"\b(?:\d[ -]*?){13,16}\b",
    "phone": r"\b\+?\d{1,3}[ -]?(?:\(\d{2,4}\)|\d{2,4})[ -]?\d{3,4}[ -]?\d{4}\b",
}


def build_redactor(privacy_config: dict | None) -> Callable[[str], str]:
    if not privacy_config or not privacy_config.get("enable_redaction", False):
        return lambda value: value
    patterns = privacy_config.get("redact_patterns") or list(DEFAULT_PATTERNS.keys())
    compiled = []
    for name in patterns:
        pattern = DEFAULT_PATTERNS.get(name, name)
        compiled.append(re.compile(pattern))

    def _redact(value: str) -> str:
        result = value
        for regex in compiled:
            result = regex.sub("[REDACTED]", result)
        return result

    return _redact


def redact_text(value: str | None, privacy_config: dict | None = None) -> str:
    if value is None:
        return ""
    redactor = build_redactor(privacy_config)
    return redactor(str(value))


def redact_series(values: Iterable[object], privacy_config: dict | None = None) -> list[str]:
    redactor = build_redactor(privacy_config)
    return [redactor(str(value)) if value is not None else "" for value in values]


def redact_frame(df: pd.DataFrame, columns: Iterable[str], privacy_config: dict | None = None) -> pd.DataFrame:
    if not privacy_config or not privacy_config.get("enable_redaction", False):
        return df
    redactor = build_redactor(privacy_config)
    output = df.copy()
    for col in columns:
        if col in output.columns:
            output[col] = output[col].astype(str).map(redactor)
    return output
