from __future__ import annotations

"""Canonical accounting-window models and helpers.

This module is the shared home for accounting-month window contracts.
Implementation wiring is completed in follow-up changes.
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class AccountingWindow:
    accounting_month: str
    accounting_month_start_ts: datetime | None
    accounting_month_end_ts: datetime | None
    close_static_start_ts: datetime | None
    close_static_end_ts: datetime | None
    close_dynamic_start_ts: datetime | None
    close_dynamic_end_ts: datetime | None
    source_plugin: str | None
    confidence: float | None
    fallback_reason: str | None

