from __future__ import annotations

import json
import re
from typing import Any

from statistic_harness.core.template import mapping_hash
from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import json_dumps, now_iso, quote_identifier, write_json


_NUMERIC_RE = re.compile(r"^[+-]?(\d+(\.\d+)?|\.\d+)$")
_WS_RE = re.compile(r"\s+")


def _normalize_name(raw: str, fallback: str) -> str:
    cleaned = str(raw).strip()
    return cleaned if cleaned else fallback


def _is_numeric_like(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return True
    text = str(value).strip()
    if not text:
        return False
    candidate = text.replace(",", "").replace("_", "")
    return bool(_NUMERIC_RE.match(candidate))


def _normalize_value(
    value: Any,
    *,
    allow_numeric: bool,
    lowercase: bool,
    strip: bool,
    collapse_whitespace: bool,
    empty_as_null: bool = True,
) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    text = str(value)
    if strip:
        text = text.strip()
    if collapse_whitespace:
        text = _WS_RE.sub(" ", text)
    if lowercase:
        text = text.lower()
    if empty_as_null and text == "":
        return None
    if allow_numeric:
        candidate = text.replace(",", "").replace("_", "")
        if _NUMERIC_RE.match(candidate):
            try:
                if "." in candidate:
                    return float(candidate)
                return int(candidate)
            except ValueError:
                return text
    return text


class Plugin:
    def run(self, ctx) -> PluginResult:
        dataset_version_id = ctx.dataset_version_id
        if not dataset_version_id:
            return PluginResult("error", "Missing dataset version", {}, [], [], None)

        template_name = str(ctx.settings.get("template_name") or "").strip()
        lowercase = bool(ctx.settings.get("lowercase", True))
        strip = bool(ctx.settings.get("strip", True))
        collapse_whitespace = bool(ctx.settings.get("collapse_whitespace", True))
        numeric_coercion = bool(ctx.settings.get("numeric_coercion", True))
        numeric_threshold = float(ctx.settings.get("numeric_threshold", 0.98))
        exclude_patterns = ctx.settings.get("exclude_name_patterns")
        if not exclude_patterns:
            exclude_patterns = ["id", "uuid", "guid", "key"]
        exclude_patterns = [
            str(pat).strip().lower()
            for pat in exclude_patterns
            if str(pat).strip()
        ]
        chunk_size = int(ctx.settings.get("chunk_size", 1000))
        sample_rows = int(ctx.settings.get("sample_rows", 500))

        dataset = ctx.storage.get_dataset_version(dataset_version_id)
        if not dataset:
            return PluginResult("error", "Dataset not found", {}, [], [], None)
        raw_format_id = dataset.get("raw_format_id")

        columns = ctx.storage.fetch_dataset_columns(dataset_version_id)
        if not columns:
            return PluginResult("skipped", "No columns found", {}, [], [], None)

        name_counts: dict[str, int] = {}
        mapping: dict[str, str] = {}
        column_safe_names: list[str] = []
        original_names: list[str] = []

        for idx, col in enumerate(columns, start=1):
            original = _normalize_name(col["original_name"], f"column_{idx}")
            base = original
            if base in name_counts:
                name_counts[base] += 1
                field_name = f"{base}_{name_counts[base]}"
            else:
                name_counts[base] = 1
                field_name = base
            mapping[field_name] = original
            original_names.append(original)
            column_safe_names.append(col["safe_name"])

        safe_by_original = {
            orig: safe for orig, safe in zip(original_names, column_safe_names)
        }

        coercion_allowed: dict[str, bool] = {}
        if numeric_coercion:
            sample_limit = max(sample_rows, 0)
            if sample_limit > 0:
                with ctx.storage.connection() as conn:
                    raw_table = dataset["table_name"]
                    quoted_cols = ", ".join(
                        quote_identifier(col) for col in column_safe_names
                    )
                    sql = (
                        f"SELECT {quoted_cols} FROM {quote_identifier(raw_table)} "
                        "ORDER BY row_id LIMIT ?"
                    )
                    cur = conn.execute(sql, (sample_limit,))
                    sample = cur.fetchall()
                counts = {col: {"num": 0, "total": 0} for col in column_safe_names}
                for row in sample:
                    for col in column_safe_names:
                        value = row[col]
                        if value is None:
                            continue
                        counts[col]["total"] += 1
                        if _is_numeric_like(value):
                            counts[col]["num"] += 1
                for col in column_safe_names:
                    total = counts[col]["total"]
                    ratio = counts[col]["num"] / total if total else 0.0
                    coercion_allowed[col] = ratio >= numeric_threshold
            else:
                coercion_allowed = {col: True for col in column_safe_names}
        else:
            coercion_allowed = {col: False for col in column_safe_names}

        for col, name in zip(column_safe_names, original_names):
            lowered = name.lower()
            if any(token in lowered for token in exclude_patterns):
                coercion_allowed[col] = False

        field_defs: list[dict[str, Any]] = []
        for field_name, original in zip(mapping.keys(), original_names):
            safe_name = safe_by_original.get(original, "")
            sqlite_type = "REAL" if coercion_allowed.get(safe_name) else "TEXT"
            field_defs.append(
                {
                    "name": field_name,
                    "dtype": next(
                        (col.get("dtype") for col in columns if col["original_name"] == original),
                        None,
                    ),
                    "sqlite_type": sqlite_type,
                }
            )

        if not template_name:
            if raw_format_id:
                template_name = f"normalized_rawformat_{raw_format_id}"
            else:
                template_name = f"normalized_{dataset_version_id}"

        templates = ctx.storage.list_templates()
        template = next(
            (item for item in templates if item.get("name") == template_name), None
        )
        if template:
            template_id = int(template["template_id"])
            template_fields = ctx.storage.fetch_template_fields(template_id)
            if len(template_fields) != len(field_defs):
                return PluginResult(
                    "error",
                    "Template fields mismatch for existing template",
                    {},
                    [],
                    [],
                    None,
                )
        else:
            template_id = ctx.storage.create_template(
                template_name,
                field_defs,
                "Normalized mixed-type view",
                "v1",
                now_iso(),
            )
            template = ctx.storage.fetch_template(template_id)
            template_fields = ctx.storage.fetch_template_fields(template_id)

        template_fields = template_fields if template else []
        if not template_fields:
            return PluginResult("error", "Template fields missing", {}, [], [], None)

        table_name = template["table_name"] if template else None
        if not table_name:
            return PluginResult("error", "Template table missing", {}, [], [], None)

        normalized_settings = {
            "lowercase": lowercase,
            "strip": strip,
            "collapse_whitespace": collapse_whitespace,
            "numeric_coercion": numeric_coercion,
            "numeric_threshold": numeric_threshold,
            "exclude_name_patterns": exclude_patterns,
        }
        mapping_payload = {"mapping": mapping, "normalization": normalized_settings}
        mapping_json = json_dumps(mapping_payload)
        mapping_h = mapping_hash(mapping_payload)

        ctx.storage.upsert_dataset_template(
            dataset_version_id,
            template_id,
            mapping_json,
            mapping_h,
            "pending",
            now_iso(),
            now_iso(),
        )

        row_count = 0
        coerced_columns = [
            name for name, safe in safe_by_original.items() if coercion_allowed.get(safe)
        ]

        try:
            with ctx.storage.connection() as conn:
                conn.execute(
                    f"DELETE FROM {quote_identifier(table_name)} WHERE dataset_version_id = ?",
                    (dataset_version_id,),
                )
                raw_table = dataset["table_name"]
                quoted_cols = ", ".join(
                    quote_identifier(col) for col in column_safe_names
                )
                select_sql = (
                    f"SELECT row_id, row_index, {quoted_cols} "
                    f"FROM {quote_identifier(raw_table)} WHERE row_id > ? "
                    "ORDER BY row_id LIMIT ?"
                )
                last_row_id = 0
                template_safe_cols = [field["safe_name"] for field in template_fields]
                field_names = [field["name"] for field in template_fields]

                while True:
                    cur = conn.execute(select_sql, (last_row_id, chunk_size))
                    batch_rows = cur.fetchall()
                    if not batch_rows:
                        break
                    batch = []
                    for row in batch_rows:
                        last_row_id = int(row["row_id"])
                        raw_by_name = {
                            orig: row[safe]
                            for orig, safe in zip(original_names, column_safe_names)
                        }
                        values = []
                        row_data = {}
                        for field_name in field_names:
                            raw_name = mapping.get(field_name)
                            value = raw_by_name.get(raw_name)
                            safe_name = safe_by_original.get(raw_name, "")
                            allow_numeric = coercion_allowed.get(safe_name, False)
                            normalized = _normalize_value(
                                value,
                                allow_numeric=allow_numeric,
                                lowercase=lowercase,
                                strip=strip,
                                collapse_whitespace=collapse_whitespace,
                            )
                            values.append(normalized)
                            row_data[field_name] = normalized
                        row_json = json.dumps(row_data, ensure_ascii=False)
                        batch.append(
                            (
                                dataset_version_id,
                                row["row_index"],
                                row_json,
                                *values,
                            )
                        )
                        row_count += 1
                    ctx.storage.insert_template_rows(
                        table_name, template_safe_cols, batch, conn
                    )

            ctx.storage.upsert_dataset_template(
                dataset_version_id,
                template_id,
                mapping_json,
                mapping_h,
                "ready",
                now_iso(),
                now_iso(),
            )
            ctx.storage.record_template_conversion(
                dataset_version_id,
                template_id,
                "completed",
                now_iso(),
                now_iso(),
                mapping_h,
                row_count=row_count,
            )
        except Exception as exc:  # pragma: no cover - error flow
            ctx.storage.upsert_dataset_template(
                dataset_version_id,
                template_id,
                mapping_json,
                mapping_h,
                "error",
                now_iso(),
                now_iso(),
            )
            ctx.storage.record_template_conversion(
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

        artifacts_dir = ctx.artifacts_dir("transform_normalize_mixed")
        map_path = artifacts_dir / "mapping.json"
        write_json(map_path, mapping_payload)
        artifacts = [
            PluginArtifact(
                path=str(map_path.relative_to(ctx.run_dir)),
                type="json",
                description="Normalization mapping",
            )
        ]

        return PluginResult(
            status="ok",
            summary="Normalized template generated",
            metrics={
                "row_count": int(row_count),
                "column_count": int(len(field_defs)),
                "template_id": int(template_id),
                "coerced_columns": coerced_columns,
            },
            findings=[],
            artifacts=artifacts,
            error=None,
        )
