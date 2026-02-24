from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from .dataset_io import DatasetAccessor


@dataclass(frozen=True)
class DatasetSpec:
    role: str
    dataset_version_id: str


def parse_dataset_specs(settings: dict[str, Any]) -> dict[str, DatasetSpec]:
    out: dict[str, DatasetSpec] = {}
    raw = settings.get("datasets")
    if not isinstance(raw, dict):
        return out
    for role, payload in raw.items():
        if not isinstance(role, str) or not role.strip():
            continue
        dataset_version_id = None
        if isinstance(payload, dict):
            value = payload.get("dataset_version_id")
            if isinstance(value, str) and value.strip():
                dataset_version_id = value.strip()
        elif isinstance(payload, str) and payload.strip():
            dataset_version_id = payload.strip()
        if dataset_version_id:
            out[role.strip()] = DatasetSpec(
                role=role.strip(), dataset_version_id=dataset_version_id
            )
    return out


def ensure_roles(specs: dict[str, DatasetSpec], required_roles: list[str]) -> list[str]:
    missing: list[str] = []
    for role in required_roles:
        if role not in specs:
            missing.append(role)
    return missing


def fetch_columns(storage: Any, dataset_version_id: str) -> list[dict[str, Any]]:
    return list(storage.fetch_dataset_columns(dataset_version_id) or [])


def resolve_column(
    storage: Any,
    dataset_version_id: str,
    *,
    explicit: str | None = None,
    candidates: list[str] | None = None,
) -> str | None:
    cols = fetch_columns(storage, dataset_version_id)
    if not cols:
        return None
    names = [str(c.get("original_name") or "") for c in cols]
    lower = {n.lower(): n for n in names if n}
    if isinstance(explicit, str) and explicit.strip():
        e = explicit.strip()
        if e in names:
            return e
        mapped = lower.get(e.lower())
        if mapped:
            return mapped
        return None
    for cand in candidates or []:
        key = str(cand).strip().lower()
        if not key:
            continue
        if key in lower:
            return lower[key]
    return None


def iter_rows(
    storage: Any,
    dataset_version_id: str,
    *,
    columns: list[str] | None = None,
    batch_size: int = 100_000,
    sql: Any | None = None,
):
    accessor = DatasetAccessor(storage, dataset_version_id, sql=sql)
    for batch in accessor.iter_batches(columns=columns, batch_size=max(1, int(batch_size))):
        if batch.empty:
            continue
        for idx, row in batch.iterrows():
            payload = row.to_dict()
            payload["row_index"] = int(idx)
            yield payload


def matches_when(row: dict[str, Any], when: dict[str, Any] | None) -> bool:
    if not when:
        return True
    for key, value in when.items():
        if not isinstance(key, str):
            continue
        if key.endswith("_in") and isinstance(value, (list, tuple, set)):
            field = key[: -len("_in")]
            actual = str(row.get(field, "")).strip()
            allowed = {str(v).strip() for v in value}
            if actual not in allowed:
                return False
            continue
        expected = value
        actual = row.get(key)
        if isinstance(expected, (list, tuple, set)):
            if str(actual) not in {str(v) for v in expected}:
                return False
            continue
        if str(actual) != str(expected):
            return False
    return True


def parse_dt(value: Any) -> datetime | None:
    try:
        dt = pd.to_datetime(value, errors="coerce")
    except Exception:
        return None
    if dt is None or pd.isna(dt):
        return None
    return dt.to_pydatetime()


def parse_amount(value: Any) -> float | None:
    if value is None:
        return None
    raw = str(value).strip().replace(",", "")
    if not raw:
        return None
    try:
        return float(raw)
    except Exception:
        return None

