from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import json_dumps, now_iso, write_json


class Plugin:
    def run(self, ctx) -> PluginResult:
        input_file = ctx.settings.get("input_file")
        if not input_file:
            raise ValueError("Input file not found")
        path = Path(input_file)
        if not path.exists():
            raise ValueError("Input file not found")

        encoding = ctx.settings.get("encoding", "utf-8")
        delimiter = ctx.settings.get("delimiter")
        sheet_name = ctx.settings.get("sheet_name")
        chunk_size = int(ctx.settings.get("chunk_size", 1000))

        dataset_version_id = ctx.dataset_version_id
        if not dataset_version_id:
            raise ValueError("Dataset version not provided")
        existing = ctx.storage.get_dataset_version(dataset_version_id)
        if existing and int(existing.get("row_count") or 0) > 0:
            columns = ctx.storage.fetch_dataset_columns(dataset_version_id)
            schema = {
                "columns": [
                    {
                        "name": col["original_name"],
                        "dtype": col.get("dtype"),
                    }
                    for col in columns
                ]
            }
            schema_path = ctx.run_dir / "dataset" / "schema.json"
            write_json(schema_path, schema)
            return PluginResult(
                status="ok",
                summary=f"Dataset already ingested ({existing['row_count']} rows)",
                metrics={
                    "rows": int(existing.get("row_count") or 0),
                    "cols": int(existing.get("column_count") or 0),
                },
                findings=[],
                artifacts=[
                    PluginArtifact(
                        path=str(schema_path.relative_to(ctx.run_dir)),
                        type="json",
                        description="Schema",
                    )
                ],
                error=None,
            )

        def dtype_to_sqlite(dtype: Any) -> str:
            if pd.api.types.is_integer_dtype(dtype):
                return "INTEGER"
            if pd.api.types.is_float_dtype(dtype):
                return "REAL"
            if pd.api.types.is_bool_dtype(dtype):
                return "INTEGER"
            return "TEXT"

        def normalize_value(value: Any) -> Any:
            if pd.isna(value):
                return None
            if isinstance(value, pd.Timestamp):
                return value.isoformat()
            return value

        def build_columns(columns: list[str], dtypes: list[Any]) -> list[dict[str, Any]]:
            safe_names = [f"c{idx+1}" for idx in range(len(columns))]
            out = []
            for idx, (orig, safe, dtype) in enumerate(zip(columns, safe_names, dtypes), start=1):
                out.append(
                    {
                        "column_id": idx,
                        "safe_name": safe,
                        "original_name": str(orig),
                        "dtype": str(dtype),
                        "sqlite_type": dtype_to_sqlite(dtype),
                    }
                )
            return out

        def iter_csv_chunks() -> Iterable[pd.DataFrame]:
            return pd.read_csv(
                path, delimiter=delimiter, encoding=encoding, chunksize=chunk_size
            )

        def iter_json_chunks() -> Iterable[pd.DataFrame]:
            try:
                for chunk in pd.read_json(path, lines=True, chunksize=chunk_size):
                    yield chunk
                return
            except ValueError:
                df = pd.read_json(path)
                yield df

        def iter_excel_rows() -> Iterable[list[Any]]:
            import mimetypes

            # Prevent mimetypes from probing system files blocked by the sandbox.
            mimetypes.knownfiles = []
            from openpyxl import load_workbook

            wb = load_workbook(path, read_only=True, data_only=True)
            ws = wb[sheet_name] if sheet_name else wb.active
            rows = ws.iter_rows(values_only=True)
            try:
                headers = next(rows)
            except StopIteration:
                return
            headers = [
                (str(h) if h is not None else f"column_{idx+1}")
                for idx, h in enumerate(headers)
            ]
            yield headers
            for row in rows:
                yield list(row)

        columns_meta: list[dict[str, Any]] = []
        row_index = 0
        table_name = f"dataset_{dataset_version_id}"

        with ctx.storage.connection() as conn:
            if path.suffix.lower() in {".xlsx"}:
                row_iter = iter_excel_rows()
                headers = next(row_iter, None)
                if headers is None:
                    raise ValueError("No rows found")
                sample = []
                for _ in range(chunk_size):
                    row = next(row_iter, None)
                    if row is None:
                        break
                    sample.append(row)
                sample_df = pd.DataFrame(sample, columns=headers)
                columns_meta = build_columns(
                    list(sample_df.columns), list(sample_df.dtypes)
                )
                ctx.storage.create_dataset_table(table_name, columns_meta, conn)
                ctx.storage.add_append_only_triggers(table_name, conn)
                ctx.storage.replace_dataset_columns(dataset_version_id, columns_meta, conn)

                def emit_rows(rows: list[list[Any]]) -> None:
                    nonlocal row_index
                    safe_columns = [col["safe_name"] for col in columns_meta]
                    batch = []
                    for row in rows:
                        values = [normalize_value(v) for v in row]
                        row_dict = dict(zip(headers, values))
                        row_json = json.dumps(row_dict, ensure_ascii=False)
                        batch.append((row_index, row_json, *values))
                        row_index += 1
                    ctx.storage.insert_dataset_rows(table_name, safe_columns, batch, conn)

                emit_rows(sample)
                buffer = []
                for row in row_iter:
                    buffer.append(row)
                    if len(buffer) >= chunk_size:
                        emit_rows(buffer)
                        buffer = []
                if buffer:
                    emit_rows(buffer)
                ctx.storage.update_dataset_version_stats(
                    dataset_version_id,
                    row_index,
                    len(columns_meta),
                    conn,
                )
            elif path.suffix.lower() == ".json":
                for chunk in iter_json_chunks():
                    if not columns_meta:
                        columns_meta = build_columns(
                            list(chunk.columns), list(chunk.dtypes)
                        )
                        ctx.storage.create_dataset_table(table_name, columns_meta, conn)
                        ctx.storage.add_append_only_triggers(table_name, conn)
                        ctx.storage.replace_dataset_columns(
                            dataset_version_id, columns_meta, conn
                        )
                    safe_columns = [col["safe_name"] for col in columns_meta]
                    batch = []
                    for row in chunk.itertuples(index=False, name=None):
                        values = [normalize_value(v) for v in row]
                        row_dict = dict(zip(chunk.columns, values))
                        row_json = json.dumps(row_dict, ensure_ascii=False)
                        batch.append((row_index, row_json, *values))
                        row_index += 1
                    ctx.storage.insert_dataset_rows(table_name, safe_columns, batch, conn)
                ctx.storage.update_dataset_version_stats(
                    dataset_version_id, row_index, len(columns_meta), conn
                )
            else:
                for chunk in iter_csv_chunks():
                    if not columns_meta:
                        columns_meta = build_columns(
                            list(chunk.columns), list(chunk.dtypes)
                        )
                        ctx.storage.create_dataset_table(table_name, columns_meta, conn)
                        ctx.storage.add_append_only_triggers(table_name, conn)
                        ctx.storage.replace_dataset_columns(
                            dataset_version_id, columns_meta, conn
                        )
                    safe_columns = [col["safe_name"] for col in columns_meta]
                    batch = []
                    for row in chunk.itertuples(index=False, name=None):
                        values = [normalize_value(v) for v in row]
                        row_dict = dict(zip(chunk.columns, values))
                        row_json = json.dumps(row_dict, ensure_ascii=False)
                        batch.append((row_index, row_json, *values))
                        row_index += 1
                    ctx.storage.insert_dataset_rows(table_name, safe_columns, batch, conn)
                ctx.storage.update_dataset_version_stats(
                    dataset_version_id, row_index, len(columns_meta), conn
                )

        schema = {
            "columns": [
                {
                    "name": col["original_name"],
                    "dtype": col.get("dtype"),
                }
                for col in columns_meta
            ]
        }
        schema_path = ctx.run_dir / "dataset" / "schema.json"
        write_json(schema_path, schema)

        if ctx.dataset_version_id:
            fingerprint_payload = [
                {
                    "name": col["original_name"].lower().strip(),
                    "dtype": col.get("dtype"),
                }
                for col in columns_meta
            ]
            fingerprint_payload = sorted(
                fingerprint_payload, key=lambda item: item["name"]
            )
            fingerprint = hashlib.sha256(
                json_dumps(fingerprint_payload).encode("utf-8")
            ).hexdigest()
            format_id = ctx.storage.ensure_raw_format(
                fingerprint, name=None, created_at=now_iso()
            )
            ctx.storage.set_dataset_raw_format(ctx.dataset_version_id, format_id)

        return PluginResult(
            status="ok",
            summary=f"Ingested {row_index} rows",
            metrics={"rows": int(row_index), "cols": int(len(columns_meta))},
            findings=[],
            artifacts=[
                PluginArtifact(
                    path=str(schema_path.relative_to(ctx.run_dir)),
                    type="json",
                    description="Schema",
                )
            ],
            error=None,
        )
