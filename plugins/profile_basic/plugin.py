from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from statistic_harness.core.column_inference import infer_timestamp_series
from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import DEFAULT_TENANT_ID, write_json


class Plugin:
    def run(self, ctx) -> PluginResult:
        df = ctx.dataset_loader()
        columns = []
        numeric = df.select_dtypes(include="number")
        role_by_name: dict[str, str] = {}

        def infer_role(col_name: str, series: pd.Series) -> str | None:
            lname = col_name.lower()
            if "param" in lname or "parameter" in lname or "params" in lname:
                return "parameter"
            if "meta" in lname or "config" in lname:
                return "parameter"
            ts_info = infer_timestamp_series(series, name_hint=col_name, sample_size=500)
            if ts_info.valid and (
                "time" in lname
                or "date" in lname
                or "timestamp" in lname
                or ts_info.score >= 2.5
            ):
                return "timestamp"
            if lname.endswith("id") or " id" in lname:
                return "id"
            if pd.api.types.is_numeric_dtype(series):
                return "numeric"
            return None

        for col in df.columns:
            series = df[col]
            role = infer_role(col, series)
            if role:
                role_by_name[col] = role
            entry = {
                "name": col,
                "dtype": str(series.dtype),
                "missing_pct": float(series.isna().mean()),
                "unique": int(series.nunique()),
            }
            if pd.api.types.is_numeric_dtype(series):
                entry.update(
                    {
                        "min": float(series.min()),
                        "max": float(series.max()),
                        "mean": float(series.mean()),
                        "std": float(series.std(ddof=0)),
                    }
                )
            columns.append(entry)

        pii_patterns = {
            "email": re.compile(
                r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
            ),
            "ssn": re.compile(r"^\d{3}-?\d{2}-?\d{4}$"),
            "phone": re.compile(r"^\+?\d[\d\s().-]{6,}\d$"),
            "address": re.compile(
                r"\b\d{1,5}\s+\w+(?:\s+\w+){0,4}\s+"
                r"(st|street|ave|avenue|rd|road|blvd|lane|ln|dr|drive|ct|court)\b",
                re.IGNORECASE,
            ),
        }
        pii_tags_by_name: dict[str, list[str]] = {}
        sample_size = int(ctx.settings.get("pii_sample_size", 200))
        threshold = float(ctx.settings.get("pii_match_threshold", 0.6))
        for col in df.columns:
            series = df[col]
            if series.empty:
                continue
            values = series.dropna().astype(str)
            if values.empty:
                continue
            sample = values.head(sample_size)
            tags: list[str] = []
            for tag, pattern in pii_patterns.items():
                matches = sample.str.match(pattern, na=False)
                if matches.any():
                    ratio = float(matches.mean())
                    if ratio >= threshold:
                        tags.append(tag)
            if tags:
                pii_tags_by_name[col] = tags

        artifacts = []
        artifacts_dir = ctx.artifacts_dir("profile_basic")
        columns_path = artifacts_dir / "columns.json"
        write_json(columns_path, columns)
        artifacts.append(
            PluginArtifact(
                path=str(columns_path.relative_to(ctx.run_dir)),
                type="json",
                description="Column stats",
            )
        )

        max_cols = int(ctx.settings.get("max_corr_cols", 10))
        if numeric.shape[1] > 1 and numeric.shape[1] <= max_cols:
            corr = numeric.corr()
            corr_path = artifacts_dir / "correlation.csv"
            corr.to_csv(corr_path)
            artifacts.append(
                PluginArtifact(
                    path=str(corr_path.relative_to(ctx.run_dir)),
                    type="csv",
                    description="Correlation",
                )
            )

        if ctx.dataset_version_id:
            ctx.storage.update_dataset_column_roles(
                ctx.dataset_version_id, role_by_name
            )
            if pii_tags_by_name:
                ctx.storage.update_dataset_column_pii_tags(
                    ctx.dataset_version_id, pii_tags_by_name
                )
                tenant_id = ctx.tenant_id or DEFAULT_TENANT_ID
                for col, tags in pii_tags_by_name.items():
                    values = (
                        df[col]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )
                    values = sorted({str(v) for v in values})
                    for tag in tags:
                        ctx.storage.upsert_pii_entities(tenant_id, tag, values)

        def normalize_kv_pairs(kv_pairs: list[tuple[str, str]]) -> tuple[str, list[tuple[str, str]]]:
            cleaned = []
            for key, value in kv_pairs:
                key = str(key).strip().lower()
                value = str(value).strip()
                if key:
                    cleaned.append((key, value))
            if not cleaned:
                return "", []
            cleaned = sorted(set(cleaned))
            canonical = ";".join(f"{k}={v}" for k, v in cleaned)
            return canonical, cleaned

        def parse_parameter_value(value: Any) -> tuple[str, list[tuple[str, str]]] | None:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return None
            raw = str(value).strip()
            if not raw:
                return None
            if raw.startswith("{") and raw.endswith("}"):
                try:
                    data = json.loads(raw)
                    if isinstance(data, dict):
                        kv_pairs = [(str(k), str(v)) for k, v in data.items()]
                        canonical, cleaned = normalize_kv_pairs(kv_pairs)
                        if canonical:
                            return canonical, cleaned
                except json.JSONDecodeError:
                    pass
            tokens = re.split(r"[;|,\\n]+", raw)
            kv_pairs = []
            for token in tokens:
                token = token.strip()
                if not token:
                    continue
                if "=" in token:
                    key, value = token.split("=", 1)
                elif ":" in token:
                    key, value = token.split(":", 1)
                else:
                    continue
                kv_pairs.append((key, value))
            if kv_pairs:
                canonical, cleaned = normalize_kv_pairs(kv_pairs)
                if canonical:
                    return canonical, cleaned
            canonical, cleaned = normalize_kv_pairs([("raw", raw)])
            if canonical:
                return canonical, cleaned
            return None

        parameter_columns = [
            col
            for col in df.columns
            if role_by_name.get(col) == "parameter"
            or (
                df[col].dtype == object
                and df[col]
                .dropna()
                .astype(str)
                .head(20)
                .str.contains(r"[=:]")
                .any()
            )
        ]

        if parameter_columns and ctx.dataset_version_id:
            cache: dict[str, int] = {}
            links: list[tuple[int, int]] = []
            edges: list[tuple[int, int, str, dict[str, Any] | None, float | None]] = []
            with ctx.storage.connection() as conn:
                dataset_entity_id = ctx.storage.ensure_entity(
                    "dataset_version", ctx.dataset_version_id, conn
                )
                for col in parameter_columns:
                    series = df[col]
                    for row_index, value in series.items():
                        parsed = parse_parameter_value(value)
                        if not parsed:
                            continue
                        canonical, kv_pairs = parsed
                        if canonical not in cache:
                            entity_id = ctx.storage.get_or_create_parameter_entity(
                                canonical, conn
                            )
                            ctx.storage.insert_parameter_kv(
                                entity_id, kv_pairs, conn
                            )
                            cache[canonical] = entity_id
                            param_entity_id = ctx.storage.ensure_entity(
                                "parameter", canonical, conn
                            )
                            edges.append(
                                (
                                    dataset_entity_id,
                                    param_entity_id,
                                    "uses_parameter",
                                    {"column": col},
                                    None,
                                )
                            )
                        links.append((int(row_index), cache[canonical]))
                        if len(links) >= 1000:
                            ctx.storage.insert_row_parameter_links(
                                ctx.dataset_version_id, links, conn
                            )
                            links = []
                if links:
                    ctx.storage.insert_row_parameter_links(
                        ctx.dataset_version_id, links, conn
                    )
                if edges:
                    ctx.storage.add_edges(edges, conn)

        return PluginResult(
            status="ok",
            summary="Profiled dataset",
            metrics={"columns": len(columns)},
            findings=[],
            artifacts=artifacts,
            error=None,
        )
