from __future__ import annotations

import hashlib
from typing import Any

from .storage import Storage
from .utils import json_dumps, now_iso, quote_identifier


def mapping_hash(mapping: dict[str, Any]) -> str:
    return hashlib.sha256(json_dumps(mapping).encode("utf-8")).hexdigest()


def apply_template(
    storage: Storage,
    dataset_version_id: str,
    template_id: int,
    mapping: dict[str, Any],
) -> int:
    template = storage.fetch_template(template_id)
    if not template:
        raise ValueError("Template not found")
    template_fields = storage.fetch_template_fields(template_id)
    dataset = storage.get_dataset_version(dataset_version_id)
    if not dataset:
        raise ValueError("Dataset version not found")

    mapping_h = mapping_hash(mapping)
    storage.upsert_dataset_template(
        dataset_version_id,
        template_id,
        json_dumps(mapping),
        mapping_h,
        "pending",
        now_iso(),
        now_iso(),
    )

    row_count = 0
    try:
        with storage.connection() as conn:
            columns = storage.fetch_dataset_columns(dataset_version_id, conn)
            raw_table = dataset["table_name"]
            raw_col_map = {
                col["original_name"]: col["safe_name"] for col in columns
            }
            template_table = template["table_name"]

            conn.execute(
                f"DELETE FROM {quote_identifier(template_table)} WHERE dataset_version_id = ?",
                (dataset_version_id,),
            )

            select_cols = []
            json_parts = []
            for field in template_fields:
                field_name = field["name"]
                raw_name = mapping.get(field_name)
                if raw_name and raw_name in raw_col_map:
                    safe_raw = quote_identifier(raw_col_map[raw_name])
                    select_cols.append(safe_raw)
                    json_parts.append("?")
                    json_parts.append(safe_raw)
                else:
                    select_cols.append("NULL")
                    json_parts.append("?")
                    json_parts.append("NULL")
            json_expr = "NULL"
            if json_parts:
                json_expr = "json_object(" + ", ".join(json_parts) + ")"

            template_safe_cols = [quote_identifier(field["safe_name"]) for field in template_fields]
            insert_cols = (
                "dataset_version_id, row_index, row_json"
                + (", " + ", ".join(template_safe_cols) if template_safe_cols else "")
            )
            select_list = (
                "?, row_index, " + json_expr
                + (", " + ", ".join(select_cols) if select_cols else "")
            )
            sql = (
                f"INSERT INTO {quote_identifier(template_table)} ({insert_cols}) "
                f"SELECT {select_list} FROM {quote_identifier(raw_table)} ORDER BY row_index"
            )
            params = [dataset_version_id]
            for field in template_fields:
                params.append(field["name"])
            conn.execute(sql, params)
            cur = conn.execute(
                f"SELECT COUNT(*) FROM {quote_identifier(template_table)} WHERE dataset_version_id = ?",
                (dataset_version_id,),
            )
            row_count = int(cur.fetchone()[0])

        storage.upsert_dataset_template(
            dataset_version_id,
            template_id,
            json_dumps(mapping),
            mapping_h,
            "ready",
            now_iso(),
            now_iso(),
        )
        storage.record_template_conversion(
            dataset_version_id,
            template_id,
            "completed",
            now_iso(),
            now_iso(),
            mapping_h,
            row_count=row_count,
        )
        return row_count
    except Exception as exc:  # pragma: no cover - error flow
        storage.upsert_dataset_template(
            dataset_version_id,
            template_id,
            json_dumps(mapping),
            mapping_h,
            "error",
            now_iso(),
            now_iso(),
        )
        storage.record_template_conversion(
            dataset_version_id,
            template_id,
            "error",
            now_iso(),
            now_iso(),
            mapping_h,
            row_count=row_count,
            error={"message": str(exc)},
        )
        raise
