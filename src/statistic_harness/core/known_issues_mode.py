from __future__ import annotations

import os


_KNOWN_ISSUES_MODE_ENV = "STAT_HARNESS_KNOWN_ISSUES_MODE"
_DISABLED_VALUES = {"0", "false", "no", "off", "disabled", "none"}


def known_issues_enabled(raw: str | None = None) -> bool:
    value = raw if raw is not None else os.environ.get(_KNOWN_ISSUES_MODE_ENV, "")
    text = str(value or "").strip().lower()
    if not text:
        return True
    return text not in _DISABLED_VALUES


def known_issues_mode_label(raw: str | None = None) -> str:
    return "on" if known_issues_enabled(raw) else "off"

