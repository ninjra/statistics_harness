from __future__ import annotations

import os
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

    # "Eventlog" datasets commonly use process/activity synonyms (process_name, task, action, job).
    activity_tokens = ("activity", "event", "step", "process", "task", "action", "job")
    has_activity = any(any(tok in name for tok in activity_tokens) for name in names)
    has_eventlog = has_id and has_activity

    host_tokens = ("host", "server", "node", "instance", "machine")
    has_host = any(
        any(token in role for token in host_tokens) for role in role_values
    )
    if not has_host:
        has_host = any(any(token in name for token in host_tokens) for name in names)

    # New (Topo/TDA add-on) capabilities: prefer fast metadata inference, then
    # fall back to a small sample when necessary.
    coord_x_tokens = ("x", "lon", "lng", "long", "longitude", "coord_x")
    coord_y_tokens = ("y", "lat", "latitude", "coord_y")
    has_coords = any(any(tok in name for tok in coord_x_tokens) for name in names) and any(
        any(tok in name for tok in coord_y_tokens) for name in names
    )

    text_tokens = ("message", "log", "exception", "error", "stack", "trace", "comment", "note", "details")
    has_text = any("text" in role or "string" in role for role in role_values) or any(
        any(tok in name for tok in text_tokens) for name in names
    )

    epoch_tokens = ("epoch", "unix", "nanos", "millis", "ms", "sec", "seconds", "timestamp")
    has_epoch = any(any(tok in name for tok in epoch_tokens) for name in names) and any(
        _is_numeric_dtype(col.get("dtype")) for col in (numeric_cols or [])
        if isinstance(col, dict)
    )

    has_groupable = any("category" in role or "categorical" in role for role in role_values)
    has_point_id = has_id

    # Sample-based refinement: only when we have a dataset table.
    try:
        columns = storage.fetch_dataset_columns(dataset_version_id)
        table_row = storage.get_dataset_version_context(dataset_version_id)
        if columns and table_row and table_row.get("table_name"):
            safe_cols = [col.get("safe_name") for col in columns if col.get("safe_name")]
            orig_by_safe = {
                col.get("safe_name"): col.get("original_name") for col in columns if col.get("safe_name")
            }
            quoted = ", ".join(quote_identifier(col) for col in safe_cols)
            sql = (
                f"SELECT {quoted} FROM {quote_identifier(table_row['table_name'])} "
                "ORDER BY row_index LIMIT ?"
            )
            with storage.connection() as conn:
                df = pd.read_sql_query(sql, conn, params=(200,))
            df = df.rename(columns={k: v for k, v in orig_by_safe.items() if k in df.columns})

            # has_text: detect free-text by median length on object columns.
            if not has_text:
                for col in df.columns:
                    series = df[col].dropna()
                    if series.empty:
                        continue
                    if pd.api.types.is_numeric_dtype(series):
                        continue
                    lengths = series.astype(str).map(len)
                    if float(lengths.median()) >= 30.0 and series.nunique() / max(len(series), 1) >= 0.5:
                        has_text = True
                        break

            # has_groupable: detect bounded-cardinality categoricals.
            if not has_groupable:
                for col in df.columns:
                    series = df[col].dropna()
                    if series.empty:
                        continue
                    if pd.api.types.is_numeric_dtype(series):
                        continue
                    nunique = int(series.nunique())
                    if 2 <= nunique <= 50:
                        has_groupable = True
                        break

            # has_point_id: detect repeatable identifiers (not near-unique, not constant).
            if not has_point_id:
                for col in df.columns:
                    series = df[col].dropna()
                    if series.empty:
                        continue
                    nunique = int(series.nunique())
                    if nunique < 2:
                        continue
                    ratio = float(nunique) / float(max(len(series), 1))
                    if 0.05 <= ratio <= 0.95 and ("id" in str(col).lower() or str(col).lower().endswith("id")):
                        has_point_id = True
                        break

            # has_coords: detect plausible x/y numeric columns.
            if not has_coords:
                numeric = []
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        numeric.append(str(col).lower())
                has_coords = any(any(tok == c or tok in c for tok in coord_x_tokens) for c in numeric) and any(
                    any(tok == c or tok in c for tok in coord_y_tokens) for c in numeric
                )

            # has_epoch: detect numeric epoch-like columns (large magnitude).
            if not has_epoch:
                for col in df.columns:
                    name = str(col).lower()
                    if not any(tok in name for tok in epoch_tokens):
                        continue
                    series = pd.to_numeric(df[col], errors="coerce").dropna()
                    if series.empty:
                        continue
                    med = float(series.median())
                    if med >= 1e9:
                        has_epoch = True
                        break
    except Exception:
        pass

    return {
        "numeric_count": len(numeric_cols),
        "has_numeric": len(numeric_cols) > 0,
        "has_multi_numeric": len(numeric_cols) > 1,
        "has_timestamp": has_timestamp,
        "has_eventlog": has_eventlog,
        "has_host": has_host,
        "has_coords": has_coords,
        "has_text": has_text,
        "has_groupable": has_groupable,
        "has_epoch": has_epoch,
        "has_point_id": has_point_id,
    }


def _capabilities_satisfied(capabilities: list[str], features: dict[str, Any]) -> bool:
    for cap in capabilities:
        # Legacy "needs_*" capability tags.
        if cap == "needs_numeric" and not features["has_numeric"]:
            return False
        if cap == "needs_multi_numeric" and not features["has_multi_numeric"]:
            return False
        if cap == "needs_timestamp" and not features["has_timestamp"]:
            return False
        if cap == "needs_eventlog" and not features["has_eventlog"]:
            return False
        if cap == "needs_host" and not features["has_host"]:
            return False
        # Topo/TDA add-on capability tags (plan uses has_*, allow needs_* as well).
        if cap in {"has_coords", "needs_coords"} and not features.get("has_coords"):
            return False
        if cap in {"has_text", "needs_text"} and not features.get("has_text"):
            return False
        if cap in {"has_groupable", "needs_groupable"} and not features.get("has_groupable"):
            return False
        if cap in {"has_epoch", "needs_epoch"} and not features.get("has_epoch"):
            return False
        if cap in {"has_point_id", "needs_point_id"} and not features.get("has_point_id"):
            return False
    return True


def select_plugins(
    specs: list[PluginSpec], storage: Storage, dataset_version_id: str
) -> list[str]:
    features = _infer_dataset_features(storage, dataset_version_id)
    dataset_template = storage.fetch_dataset_template(dataset_version_id)
    template_ready = bool(dataset_template and dataset_template.get("status") == "ready")
    include_untagged = os.environ.get("STAT_HARNESS_PLANNER_INCLUDE_UNTAGGED", "").lower() in {
        "1",
        "true",
        "yes",
    }
    enable_topo_tda = os.environ.get("STAT_HARNESS_ENABLE_TOPO_TDA", "").lower() in {
        "1",
        "true",
        "yes",
    }
    # Keep auto runs lightweight; plugins without recognized capability tags are opt-in
    # (either via allowlist or STAT_HARNESS_PLANNER_INCLUDE_UNTAGGED).
    essential = {
        "analysis_queue_delay_decomposition",
        "analysis_busy_period_segmentation_v2",
        "analysis_close_cycle_contention",
        "analysis_close_cycle_duration_shift",
        "analysis_close_cycle_capacity_model",
        "analysis_close_cycle_capacity_impact",
        "analysis_close_cycle_capacity_model",
        "analysis_capacity_scaling",
        "analysis_waterfall_summary_v2",
        "analysis_traceability_manifest_v2",
        "analysis_recommendation_dedupe_v2",
        "analysis_issue_cards_v2",
    }
    recognized_caps = {
        "needs_numeric",
        "needs_multi_numeric",
        "needs_timestamp",
        "needs_eventlog",
        "needs_host",
        "has_coords",
        "needs_coords",
        "has_text",
        "needs_text",
        "has_groupable",
        "needs_groupable",
        "has_epoch",
        "needs_epoch",
        "has_point_id",
        "needs_point_id",
        "topo_tda_addon",
    }

    selected: list[str] = []
    spec_ids = {spec.plugin_id for spec in specs}
    for spec in specs:
        if spec.type != "analysis":
            continue
        caps = list(spec.capabilities or [])
        if "topo_tda_addon" in caps and not enable_topo_tda:
            continue
        if spec.plugin_id not in essential:
            if not caps:
                if not include_untagged:
                    continue
            elif not any(cap in recognized_caps for cap in caps):
                if not include_untagged:
                    continue
        if not _capabilities_satisfied(caps, features):
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
