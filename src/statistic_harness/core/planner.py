from __future__ import annotations

from typing import Any

import pandas as pd

from .column_inference import infer_timestamp_series
from .utils import quote_identifier

from .plugin_manager import PluginSpec
from .storage import Storage


def _is_numeric_dtype(dtype: str | None) -> bool:
    if not dtype:
        return False
    lowered = dtype.lower()
    return "int" in lowered or "float" in lowered or "double" in lowered


def _infer_dataset_features(
    storage: Storage, dataset_version_id: str
) -> dict[str, Any]:
    dataset_template = storage.fetch_dataset_template(dataset_version_id)
    if dataset_template and dataset_template.get("status") == "ready":
        fields = storage.fetch_template_fields(int(dataset_template["template_id"]))
        numeric_cols = [field for field in fields if _is_numeric_dtype(field.get("dtype"))]
        roles = {field["name"]: (field.get("role") or "") for field in fields}
        names = [field["name"].lower() for field in fields]
    else:
        columns = storage.fetch_dataset_columns(dataset_version_id)
        numeric_cols = [col for col in columns if _is_numeric_dtype(col.get("dtype"))]
        roles = {col["original_name"]: (col.get("role") or "") for col in columns}
        names = [col["original_name"].lower() for col in columns]

    role_values = [str(role).lower() for role in roles.values() if role]
    has_timestamp = any(
        role == "timestamp" or "time" in role or "date" in role for role in role_values
    )
    if not has_timestamp:
        has_timestamp = any("time" in name or "date" in name for name in names)

    if not has_timestamp:
        columns = storage.fetch_dataset_columns(dataset_version_id)
        if columns:
            safe_cols = [col.get("safe_name") for col in columns if col.get("safe_name")]
            table_row = storage.get_dataset_version_context(dataset_version_id)
            if safe_cols and table_row and table_row.get("table_name"):
                quoted = ", ".join(quote_identifier(col) for col in safe_cols)
                sql = (
                    f"SELECT {quoted} FROM {quote_identifier(table_row['table_name'])} "
                    "ORDER BY row_index LIMIT ?"
                )
                with storage.connection() as conn:
                    df = pd.read_sql_query(sql, conn, params=(200,))
                df = df.rename(
                    columns={
                        col["safe_name"]: col["original_name"]
                        for col in columns
                        if col.get("safe_name") in df.columns
                    }
                )
                for col in df.columns:
                    info = infer_timestamp_series(df[col], name_hint=col, sample_size=200)
                    if info.valid:
                        has_timestamp = True
                        break

    has_id = any("id" in role for role in role_values)
    if not has_id:
        has_id = any(name.endswith("id") or " id" in name for name in names)

    has_activity = any("activity" in name or "event" in name or "step" in name for name in names)
    has_eventlog = has_id and has_activity

    host_tokens = ("host", "server", "node", "instance", "machine")
    has_host = any(
        any(token in role for token in host_tokens) for role in role_values
    )
    if not has_host:
        has_host = any(any(token in name for token in host_tokens) for name in names)

    return {
        "numeric_count": len(numeric_cols),
        "has_numeric": len(numeric_cols) > 0,
        "has_multi_numeric": len(numeric_cols) > 1,
        "has_timestamp": has_timestamp,
        "has_eventlog": has_eventlog,
        "has_host": has_host,
    }


def _capabilities_satisfied(capabilities: list[str], features: dict[str, Any]) -> bool:
    for cap in capabilities:
        if cap == "needs_numeric" and not features["has_numeric"]:
            return False
        if cap == "needs_multi_numeric" and not features["has_multi_numeric"]:
            return False
        if cap == "needs_timestamp" and not features["has_timestamp"]:
            return False
        if cap == "needs_eventlog" and not features["has_eventlog"]:
            return False
    return True


def select_plugins(
    specs: list[PluginSpec], storage: Storage, dataset_version_id: str
) -> list[str]:
    features = _infer_dataset_features(storage, dataset_version_id)
    dataset_template = storage.fetch_dataset_template(dataset_version_id)
    template_ready = bool(dataset_template and dataset_template.get("status") == "ready")
    selected = []
    spec_ids = {spec.plugin_id for spec in specs}
    for spec in specs:
        if spec.type != "analysis":
            continue
        if not _capabilities_satisfied(spec.capabilities, features):
            continue
        selected.append(spec.plugin_id)
    if (
        "analysis_close_cycle_capacity_model" in spec_ids
        and "analysis_close_cycle_capacity_model" not in selected
        and features.get("has_timestamp")
        and features.get("has_host")
    ):
        selected.append("analysis_close_cycle_capacity_model")
    if not template_ready:
        for spec in specs:
            if spec.plugin_id == "transform_normalize_mixed":
                selected.append(spec.plugin_id)
                break
    return sorted(set(selected))
