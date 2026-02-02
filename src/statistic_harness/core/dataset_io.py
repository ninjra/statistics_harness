from __future__ import annotations

import json
from typing import Any

import pandas as pd

from .storage import Storage
from .utils import quote_identifier


class DatasetAccessor:
    def __init__(self, storage: Storage, dataset_version_id: str) -> None:
        self.storage = storage
        self.dataset_version_id = dataset_version_id
        self._df: pd.DataFrame | None = None

    def _load_df(self) -> pd.DataFrame:
        with self.storage.connection() as conn:
            version = self.storage.get_dataset_version(self.dataset_version_id, conn)
            if not version:
                raise ValueError("Dataset version not found")
            columns = self.storage.fetch_dataset_columns(
                self.dataset_version_id, conn
            )
            safe_cols = [col["safe_name"] for col in columns]
            original_cols = [col["original_name"] for col in columns]
            if safe_cols:
                quoted_cols = ", ".join(quote_identifier(col) for col in safe_cols)
                sql = (
                    f"SELECT row_index, {quoted_cols} FROM "
                    f"{quote_identifier(version['table_name'])} ORDER BY row_index"
                )
                df = pd.read_sql_query(sql, conn)
            else:
                df = pd.DataFrame()
        df = df.rename(columns=dict(zip(safe_cols, original_cols)))
        if "row_index" in df.columns:
            df = df.set_index("row_index")
            df.index.name = None
        return df

    def load(self) -> pd.DataFrame:
        if self._df is None:
            self._df = self._load_df()
        return self._df.copy()

    def info(self) -> dict[str, Any]:
        with self.storage.connection() as conn:
            version = self.storage.get_dataset_version(self.dataset_version_id, conn)
            if not version:
                raise ValueError("Dataset version not found")
            columns = self.storage.fetch_dataset_columns(
                self.dataset_version_id, conn
            )
        inferred = {col["original_name"]: col.get("dtype") for col in columns}
        return {
            "rows": int(version.get("row_count") or 0),
            "cols": int(version.get("column_count") or 0),
            "inferred_types": inferred,
        }


class TemplateAccessor:
    def __init__(
        self,
        storage: Storage,
        dataset_version_id: str,
        template_id: int,
        table_name: str,
        scope: str = "dataset",
        filters: dict[str, Any] | None = None,
    ) -> None:
        self.storage = storage
        self.dataset_version_id = dataset_version_id
        self.template_id = template_id
        self.table_name = table_name
        self.scope = scope
        self.filters = filters or {}
        self._df: pd.DataFrame | None = None

    def _filtered_dataset_versions(self, conn) -> list[str] | None:
        filters = self.filters or {}
        project_ids = filters.get("project_ids") or []
        dataset_ids = filters.get("dataset_ids") or []
        dataset_version_ids = filters.get("dataset_version_ids") or []
        raw_format_ids = filters.get("raw_format_ids") or []
        created_after = filters.get("created_after")
        created_before = filters.get("created_before")
        conditions = []
        params: list[Any] = []
        join = ""

        if project_ids:
            join = " JOIN datasets d ON d.dataset_id = dv.dataset_id"
            placeholders = ", ".join(["?"] * len(project_ids))
            conditions.append(f"d.project_id IN ({placeholders})")
            params.extend(project_ids)
        if dataset_ids:
            placeholders = ", ".join(["?"] * len(dataset_ids))
            conditions.append(f"dv.dataset_id IN ({placeholders})")
            params.extend(dataset_ids)
        if dataset_version_ids:
            placeholders = ", ".join(["?"] * len(dataset_version_ids))
            conditions.append(f"dv.dataset_version_id IN ({placeholders})")
            params.extend(dataset_version_ids)
        if raw_format_ids:
            placeholders = ", ".join(["?"] * len(raw_format_ids))
            conditions.append(f"dv.raw_format_id IN ({placeholders})")
            params.extend(raw_format_ids)
        if created_after:
            conditions.append("dv.created_at >= ?")
            params.append(created_after)
        if created_before:
            conditions.append("dv.created_at <= ?")
            params.append(created_before)

        if not conditions:
            return None
        query = (
            "SELECT dv.dataset_version_id FROM dataset_versions dv"
            + join
            + " WHERE "
            + " AND ".join(conditions)
            + " ORDER BY dv.created_at, dv.dataset_version_id"
        )
        cur = conn.execute(query, params)
        return [row[0] for row in cur.fetchall()]

    def _load_df(self) -> pd.DataFrame:
        fields = self.storage.fetch_template_fields(self.template_id)
        safe_cols = [field["safe_name"] for field in fields]
        names = [field["name"] for field in fields]
        with self.storage.connection() as conn:
            if safe_cols:
                quoted_cols = ", ".join(quote_identifier(col) for col in safe_cols)
                if self.scope == "all":
                    ids = self._filtered_dataset_versions(conn)
                    if ids is None:
                        sql = (
                            f"SELECT dataset_version_id, row_index, {quoted_cols} FROM "
                            f"{quote_identifier(self.table_name)} ORDER BY dataset_version_id, row_index"
                        )
                        df = pd.read_sql_query(sql, conn)
                    elif ids:
                        placeholders = ", ".join(["?"] * len(ids))
                        sql = (
                            f"SELECT dataset_version_id, row_index, {quoted_cols} FROM "
                            f"{quote_identifier(self.table_name)} WHERE dataset_version_id IN ({placeholders}) "
                            f"ORDER BY dataset_version_id, row_index"
                        )
                        df = pd.read_sql_query(sql, conn, params=ids)
                    else:
                        df = pd.DataFrame(
                            columns=["dataset_version_id", "row_index", *safe_cols]
                        )
                else:
                    sql = (
                        f"SELECT row_index, {quoted_cols} FROM "
                        f"{quote_identifier(self.table_name)} WHERE dataset_version_id = ? "
                        f"ORDER BY row_index"
                    )
                    df = pd.read_sql_query(sql, conn, params=(self.dataset_version_id,))
            else:
                df = pd.DataFrame()
        df = df.rename(columns=dict(zip(safe_cols, names)))
        if "row_index" in df.columns:
            if self.scope == "all":
                df = df.reset_index(drop=True)
            else:
                df = df.set_index("row_index")
                df.index.name = None
        return df

    def load(self) -> pd.DataFrame:
        if self._df is None:
            self._df = self._load_df()
        return self._df.copy()

    def info(self) -> dict[str, Any]:
        fields = self.storage.fetch_template_fields(self.template_id)
        with self.storage.connection() as conn:
            if self.scope == "all":
                ids = self._filtered_dataset_versions(conn)
                if ids is None:
                    cur = conn.execute(
                        f"SELECT COUNT(*) FROM {quote_identifier(self.table_name)}"
                    )
                elif ids:
                    placeholders = ", ".join(["?"] * len(ids))
                    cur = conn.execute(
                        f"SELECT COUNT(*) FROM {quote_identifier(self.table_name)} WHERE dataset_version_id IN ({placeholders})",
                        ids,
                    )
                else:
                    rows = 0
                    inferred = {field["name"]: field.get("dtype") for field in fields}
                    return {"rows": rows, "cols": len(fields), "inferred_types": inferred}
            else:
                cur = conn.execute(
                    f"SELECT COUNT(*) FROM {quote_identifier(self.table_name)} WHERE dataset_version_id = ?",
                    (self.dataset_version_id,),
                )
            rows = int(cur.fetchone()[0])
        inferred = {field["name"]: field.get("dtype") for field in fields}
        return {"rows": rows, "cols": len(fields), "inferred_types": inferred}


def resolve_dataset_accessor(
    storage: Storage, dataset_version_id: str
) -> tuple[Any, dict[str, Any] | None]:
    dataset_template = storage.fetch_dataset_template(dataset_version_id)
    if dataset_template and dataset_template.get("status") == "ready":
        mapping = {}
        try:
            mapping = json.loads(dataset_template.get("mapping_json") or "{}")
        except json.JSONDecodeError:
            mapping = {}
        scope = "all" if mapping.get("scope") == "all" else "dataset"
        filters = mapping.get("filters") if scope == "all" else None
        if isinstance(filters, dict):
            filters = {k: v for k, v in filters.items() if v not in (None, [], "")}
            if not filters:
                filters = None
        else:
            filters = None
        accessor = TemplateAccessor(
            storage,
            dataset_version_id,
            int(dataset_template["template_id"]),
            dataset_template["table_name"],
            scope=scope,
            filters=filters,
        )
        return accessor, dataset_template
    return DatasetAccessor(storage, dataset_version_id), None
