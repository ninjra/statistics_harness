from __future__ import annotations

from typing import Iterable


_QUORUM_SIGNATURE_FIELDS = {
    "PROCESS_QUEUE_ID",
    "PROCESS_ID",
    "STATUS_CD",
    "LOCAL_MACHINE_ID",
    "QUEUE_DT",
    "START_DT",
    "END_DT",
}


def normalize_field_name(name: str | None) -> str:
    value = str(name or "").strip().upper()
    if not value:
        return ""
    return value


def infer_erp_type_from_field_names(field_names: Iterable[str | None]) -> str:
    normalized = {normalize_field_name(name) for name in field_names}
    normalized.discard("")
    if not normalized:
        return "unknown"

    quorum_hits = len(_QUORUM_SIGNATURE_FIELDS.intersection(normalized))
    # Require strong evidence to avoid false positives.
    if quorum_hits >= 6:
        return "quorum"
    return "unknown"

