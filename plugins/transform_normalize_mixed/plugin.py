from __future__ import annotations

import json
import re
import time
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
        # Larger default chunk improves normalization throughput for multi-million row datasets.
        chunk_size = int(ctx.settings.get("chunk_size", 10_000))
        # Deprecated: this plugin must not row-sample for type decisions. Kept for back-compat config parsing.
        _sample_rows = int(ctx.settings.get("sample_rows", 500))

        dataset = ctx.storage.get_dataset_version(dataset_version_id)
        if not dataset:
            return PluginResult("error", "Dataset not found", {}, [], [], None)
        raw_format_id = dataset.get("raw_format_id")

        columns = ctx.storage.fetch_dataset_columns(dataset_version_id)
        if not columns:
            return PluginResult("skipped", "No columns found", {}, [], [], None)

        name_counts: dict[str, int] = {}
        col_specs: list[dict[str, Any]] = []
        # Initial mapping keyed by dataset-derived field names (often column headers).
        # For pre-existing templates with semantic field names, this will be reconciled later.
        mapping: dict[str, Any] = {}
        column_safe_names: list[str] = []

        for idx, col in enumerate(columns, start=1):
            original = _normalize_name(col.get("original_name"), f"column_{idx}")
            safe = str(col.get("safe_name") or "").strip()
            if not safe:
                return PluginResult("error", "Dataset column missing safe_name", {}, [], [], None)
            base = original
            if base in name_counts:
                name_counts[base] += 1
                field_name = f"{base}_{name_counts[base]}"
            else:
                name_counts[base] = 1
                field_name = base

            col_specs.append(
                {
                    "column_id": int(col.get("column_id") if col.get("column_id") is not None else (idx - 1)),
                    "field_name": field_name,
                    "original_name": original,
                    "safe_name": safe,
                    "dtype": col.get("dtype"),
                }
            )
            mapping[field_name] = {"original_name": original, "safe_name": safe}
            column_safe_names.append(safe)

        coercion_allowed: dict[str, bool] = {col: False for col in column_safe_names}
        if numeric_coercion:
            raw_table = str(dataset["table_name"])
            row_count = int(dataset.get("row_count") or 0)

            # Read declared raw-table column types once (cheap) so we can accept already-numeric cols.
            declared_types: dict[str, str] = {}
            with ctx.storage.connection() as conn:
                try:
                    info = conn.execute(
                        f"PRAGMA table_info({quote_identifier(raw_table)})"
                    ).fetchall()
                    for r in info:
                        name = str(r["name"])
                        ctype = str(r["type"] or "")
                        declared_types[name] = ctype.upper()
                except Exception:
                    declared_types = {}

            # Prefer full-dataset numeric-like ratios computed during ingest (no re-scan).
            stats_by_safe: dict[str, dict[str, Any]] = {}
            for col in columns:
                safe = str(col.get("safe_name") or "")
                st = col.get("stats")
                if safe and isinstance(st, dict):
                    stats_by_safe[safe] = st

            # Decide coercion using (in order):
            # 1) declared type is already numeric => allow
            # 2) stored ingest stats numeric_like_ratio => allow if >= threshold
            # 3) fallback: full DB scan for that column (no sampling)
            pending: list[str] = []
            for safe in column_safe_names:
                ctype = declared_types.get(safe, "")
                if "INT" in ctype or "REAL" in ctype or "FLOA" in ctype or "DOUB" in ctype:
                    coercion_allowed[safe] = True
                    continue
                st = stats_by_safe.get(safe) or {}
                ratio = st.get("numeric_like_ratio")
                if isinstance(ratio, (int, float)):
                    coercion_allowed[safe] = float(ratio) >= float(numeric_threshold)
                else:
                    pending.append(safe)

            # Full-scan fallback for missing stats (kept for older datasets/tests).
            if pending and row_count > 0:
                with ctx.storage.connection() as conn:
                    last_row_id = 0
                    while True:
                        cols_sql = ", ".join(quote_identifier(c) for c in pending)
                        sql = (
                            f"SELECT row_id, {cols_sql} FROM {quote_identifier(raw_table)} "
                            f"WHERE row_id > ? ORDER BY row_id LIMIT ?"
                        )
                        rows = conn.execute(sql, (last_row_id, int(chunk_size))).fetchall()
                        if not rows:
                            break
                        for r in rows:
                            last_row_id = int(r["row_id"])
                            for c in pending:
                                v = r[c]
                                if v is None:
                                    continue
                                # Compute exact numeric-like ratio. This is row-complete (no sampling),
                                # but can be expensive on very wide datasets.
                                key_total = f"__tmp_total__{c}"
                                key_num = f"__tmp_num__{c}"
                                stats_by_safe.setdefault(c, {})
                                stats_by_safe[c][key_total] = int(stats_by_safe[c].get(key_total) or 0) + 1
                                if _is_numeric_like(v):
                                    stats_by_safe[c][key_num] = int(stats_by_safe[c].get(key_num) or 0) + 1
                        if last_row_id <= 0:
                            break
                for c in pending:
                    st = stats_by_safe.get(c) or {}
                    total = int(st.get(f"__tmp_total__{c}") or 0)
                    num = int(st.get(f"__tmp_num__{c}") or 0)
                    ratio = (float(num) / float(total)) if total else 0.0
                    coercion_allowed[c] = ratio >= float(numeric_threshold)
                # Persist any newly computed full-dataset ratios back into dataset column stats.
                # This avoids repeating this expensive scan in future runs.
                try:
                    stats_by_safe_name: dict[str, dict[str, Any]] = {}
                    for col in columns:
                        safe = str(col.get("safe_name") or "")
                        if not safe:
                            continue
                        st = stats_by_safe.get(safe) or {}
                        total = int(st.get(f"__tmp_total__{safe}") or 0)
                        num = int(st.get(f"__tmp_num__{safe}") or 0)
                        if total > 0:
                            stats_by_safe_name[safe] = {
                                **(col.get("stats") or {}),
                                "numeric_like_ratio": float(num) / float(total),
                            }
                    if stats_by_safe_name:
                        ctx.storage.update_dataset_column_stats(dataset_version_id, stats_by_safe_name)
                except Exception:
                    pass

        for spec in col_specs:
            lowered = str(spec["original_name"]).lower()
            if any(token in lowered for token in exclude_patterns):
                coercion_allowed[str(spec["safe_name"])] = False

        field_defs: list[dict[str, Any]] = []
        for spec in col_specs:
            safe_name = str(spec["safe_name"])
            sqlite_type = "REAL" if coercion_allowed.get(safe_name) else "TEXT"
            field_defs.append(
                {
                    "name": str(spec["field_name"]),
                    "dtype": spec.get("dtype"),
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

        # Reconcile mapping keys with template field names.
        # This matters when a template already exists with semantic field names (e.g., QUEUE_DT),
        # but the dataset columns are generic (e.g., c1..cN). In that case, we map by column order
        # so we don't silently populate the normalized table with NULLs.
        template_field_names = [str(field.get("name") or "") for field in template_fields]
        if template_field_names and not set(template_field_names).issubset(set(mapping.keys())):
            ordered = sorted(col_specs, key=lambda s: int(s.get("column_id") or 0))
            if len(ordered) != len(template_field_names):
                return PluginResult(
                    "error",
                    "Template/dataset column count mismatch; cannot infer mapping",
                    {
                        "template_name": template_name,
                        "template_id": int(template_id),
                        "template_fields": int(len(template_field_names)),
                        "dataset_columns": int(len(ordered)),
                    },
                    [],
                    [],
                    None,
                )
            mapping = {}
            for i, tname in enumerate(template_field_names):
                src = ordered[i]
                mapping[tname] = {
                    "original_name": str(src.get("original_name") or ""),
                    "safe_name": str(src.get("safe_name") or ""),
                    "inferred_by": "column_order",
                    "source_column_id": int(src.get("column_id") or 0),
                }

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
            str(spec["field_name"])
            for spec in col_specs
            if coercion_allowed.get(str(spec["safe_name"]))
        ]

        try:
            with ctx.storage.connection() as conn:
                conn.execute(
                    f"DELETE FROM {quote_identifier(table_name)} WHERE dataset_version_id = ?",
                    (dataset_version_id,),
                )
                raw_table = dataset["table_name"]
                # Ensure key indexes and planner stats exist before the bulk scan.
                try:
                    ctx.storage.ensure_dataset_row_index_index(str(raw_table), conn)
                except Exception:
                    pass
                try:
                    ctx.storage.analyze_table(str(raw_table), conn)
                except Exception:
                    pass

                # Cache declared raw-table sqlite types (cheap, and used for stats + role hints).
                declared_types: dict[str, str] = {}
                try:
                    info = conn.execute(
                        f"PRAGMA table_info({quote_identifier(str(raw_table))})"
                    ).fetchall()
                    for r in info:
                        declared_types[str(r["name"])] = str(r["type"] or "").upper()
                except Exception:
                    declared_types = {}

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

                # Full-dataset abstract representation (computed during normalization to avoid
                # repeated scans across many plugins):
                # - null counts for every column
                # - numeric min/max/mean
                # - text min/max length + numeric_like_ratio
                stats_by_safe: dict[str, dict[str, Any]] = {}
                for col in columns:
                    safe = str(col.get("safe_name") or "")
                    orig = str(col.get("original_name") or "")
                    if not safe:
                        continue
                    # Prefer the normalized/template typing decision for downstream plugin efficiency.
                    sqlite_type = "REAL" if coercion_allowed.get(safe, False) else (declared_types.get(safe, "TEXT") or "TEXT")
                    sqlite_type = str(sqlite_type).upper()
                    st: dict[str, Any] = {
                        "sqlite_type": sqlite_type,
                        "original_name": orig,
                        "n": 0,
                        "nulls": 0,
                    }
                    if sqlite_type in {"INTEGER", "REAL"}:
                        st.update({"min": None, "max": None, "sum": 0.0})
                    else:
                        st.update(
                            {
                                "min_len": None,
                                "max_len": None,
                                "numeric_like_total": 0,
                                "numeric_like_num": 0,
                            }
                        )
                    stats_by_safe[safe] = st

                def _update_stats_value(safe: str, value: Any) -> None:
                    st = stats_by_safe.get(safe)
                    if not st:
                        return
                    st["n"] = int(st.get("n") or 0) + 1
                    if value is None:
                        st["nulls"] = int(st.get("nulls") or 0) + 1
                        return
                    sqlite_type = str(st.get("sqlite_type") or "TEXT").upper()
                    if sqlite_type in {"INTEGER", "REAL"}:
                        if isinstance(value, bool):
                            num = float(int(value))
                        elif isinstance(value, (int, float)) and not isinstance(value, bool):
                            num = float(value)
                        else:
                            # Keep stats conservative if coercion produced a non-numeric.
                            return
                        prev_min = st.get("min")
                        prev_max = st.get("max")
                        st["min"] = num if prev_min is None else float(min(float(prev_min), num))
                        st["max"] = num if prev_max is None else float(max(float(prev_max), num))
                        st["sum"] = float(st.get("sum") or 0.0) + num
                    else:
                        text = str(value)
                        ln = len(text)
                        prev_min = st.get("min_len")
                        prev_max = st.get("max_len")
                        st["min_len"] = ln if prev_min is None else int(min(int(prev_min), ln))
                        st["max_len"] = ln if prev_max is None else int(max(int(prev_max), ln))
                        st["numeric_like_total"] = int(st.get("numeric_like_total") or 0) + 1
                        if _is_numeric_like(text):
                            st["numeric_like_num"] = int(st.get("numeric_like_num") or 0) + 1

                started = time.perf_counter()
                last_log = started
                while True:
                    cur = conn.execute(select_sql, (last_row_id, chunk_size))
                    batch_rows = cur.fetchall()
                    if not batch_rows:
                        break
                    # Log at chunk start so operators see progress even when the chunk is large.
                    try:
                        ctx.logger(
                            f"chunk_start rows={len(batch_rows)} last_row_id={last_row_id}"
                        )
                    except Exception:
                        pass
                    batch = []
                    for row in batch_rows:
                        last_row_id = int(row["row_id"])
                        values = []
                        row_data = {}
                        for field_name in field_names:
                            src = mapping.get(field_name) or {}
                            if not isinstance(src, dict):
                                src = {}
                            src_safe = str(src.get("safe_name") or "")
                            if src_safe:
                                try:
                                    value = row[src_safe]
                                except Exception:
                                    value = None
                            else:
                                value = None
                            allow_numeric = coercion_allowed.get(src_safe, False)
                            normalized = _normalize_value(
                                value,
                                allow_numeric=allow_numeric,
                                lowercase=lowercase,
                                strip=strip,
                                collapse_whitespace=collapse_whitespace,
                            )
                            values.append(normalized)
                            row_data[field_name] = normalized
                            if src_safe:
                                _update_stats_value(src_safe, normalized)
                        row_json = json.dumps(row_data, ensure_ascii=False, separators=(",", ":"))
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
                    now = time.perf_counter()
                    if (now - last_log) >= 5.0:
                        elapsed = max(0.001, now - started)
                        rate = float(row_count) / float(elapsed)
                        ctx.logger(
                            f"normalized_rows={row_count} elapsed_s={elapsed:.1f} "
                            f"rate_rows_s={rate:.1f} last_row_id={last_row_id}"
                        )
                        last_log = now

                finalized: dict[str, dict[str, Any]] = {}
                # Persist full-dataset stats + role hints (best-effort).
                try:
                    for safe, st in stats_by_safe.items():
                        out = dict(st)
                        sqlite_type = str(out.get("sqlite_type") or "TEXT").upper()
                        if sqlite_type in {"INTEGER", "REAL"}:
                            n = int(out.get("n") or 0)
                            nulls = int(out.get("nulls") or 0)
                            used = max(0, n - nulls)
                            if used > 0:
                                out["mean"] = float(out.get("sum") or 0.0) / float(used)
                            out.pop("sum", None)
                        else:
                            total = int(out.get("numeric_like_total") or 0)
                            num = int(out.get("numeric_like_num") or 0)
                            if total > 0:
                                out["numeric_like_ratio"] = float(num) / float(total)
                            out.pop("numeric_like_total", None)
                            out.pop("numeric_like_num", None)
                        finalized[safe] = out

                    if finalized:
                        ctx.storage.update_dataset_column_stats(dataset_version_id, finalized, conn)
                except Exception:
                    pass

                # Best-effort role hints + indexes to speed downstream plugin SQL filtering/grouping.
                try:
                    role_by_safe: dict[str, str] = {}

                    def _infer_role(col_name: str, sqlite_type: str) -> str | None:
                        lname = col_name.lower()
                        if any(tok in lname for tok in ("param", "parameter", "params", "meta", "config")):
                            return "parameter"
                        if any(tok in lname for tok in ("timestamp", "time", "date")):
                            return "timestamp"
                        if lname.endswith("id") or "_id" in lname or " id" in lname:
                            return "id"
                        if any(tok in lname for tok in ("event", "action", "activity")):
                            return "event"
                        if "variant" in lname:
                            return "variant"
                        if "status" in lname:
                            return "status"
                        if sqlite_type in {"INTEGER", "REAL"}:
                            return "numeric"
                        return None

                    index_candidates: list[tuple[str, str]] = []
                    for c in columns:
                        orig = str(c.get("original_name") or "")
                        safe = str(c.get("safe_name") or "")
                        if not safe:
                            continue
                        st = (finalized.get(safe) or {})
                        sqlite_type = str(st.get("sqlite_type") or declared_types.get(safe, "TEXT") or "TEXT").upper()
                        role = _infer_role(orig, sqlite_type)
                        if role:
                            role_by_safe[safe] = role
                        max_len = st.get("max_len")
                        if role in {"timestamp", "id", "event", "variant", "status"}:
                            if sqlite_type in {"INTEGER", "REAL"}:
                                index_candidates.append((orig, safe))
                            else:
                                try:
                                    if max_len is None or int(max_len) <= 256:
                                        index_candidates.append((orig, safe))
                                except (TypeError, ValueError):
                                    index_candidates.append((orig, safe))

                    if role_by_safe:
                        try:
                            ctx.storage.update_dataset_column_roles(dataset_version_id, role_by_safe, conn)
                        except Exception:
                            pass

                    # Cap index count to keep normalization predictable.
                    index_candidates = index_candidates[:12]
                    for _, safe in index_candidates:
                        try:
                            ctx.storage.ensure_dataset_column_index(str(raw_table), safe, conn)
                        except Exception:
                            pass

                    # Add exact categorical summaries for a small set of key-like columns.
                    # This is full-dataset and DB-backed, but limited to a few columns.
                    safe_table_q = quote_identifier(str(raw_table))

                    def _sensitive_name(col_name: str) -> bool:
                        lname = col_name.lower()
                        return any(tok in lname for tok in ("email", "ssn", "phone", "address"))

                    for orig, safe in index_candidates[:8]:
                        role = role_by_safe.get(safe)
                        if role not in {"event", "variant", "status"}:
                            continue
                        st = finalized.get(safe) or {}
                        sqlite_type = str(st.get("sqlite_type") or "TEXT").upper()
                        if sqlite_type != "TEXT":
                            continue
                        try:
                            max_len = st.get("max_len")
                            if max_len is not None and int(max_len) > 256:
                                continue
                        except (TypeError, ValueError):
                            pass

                        qcol = quote_identifier(safe)
                        try:
                            row = conn.execute(
                                f"SELECT COUNT(DISTINCT {qcol}) AS d FROM {safe_table_q}"
                            ).fetchone()
                            if row is not None:
                                st["distinct_count"] = int(row["d"] or 0)
                        except Exception:
                            pass
                        if not _sensitive_name(orig):
                            try:
                                rows = conn.execute(
                                    f"""
                                    SELECT {qcol} AS v, COUNT(*) AS c
                                    FROM {safe_table_q}
                                    GROUP BY {qcol}
                                    ORDER BY c DESC
                                    LIMIT 20
                                    """
                                ).fetchall()
                                top = []
                                for r in rows or []:
                                    v = r["v"]
                                    if v is None:
                                        sval = "(null)"
                                    else:
                                        sval = str(v)
                                        if len(sval) > 200:
                                            sval = sval[:200] + "..."
                                    top.append({"value": sval, "count": int(r["c"] or 0)})
                                if top:
                                    st["top_values"] = top
                            except Exception:
                                pass
                        finalized[safe] = st

                    if finalized:
                        ctx.storage.update_dataset_column_stats(dataset_version_id, finalized, conn)
                except Exception:
                    pass

                # Improve planner stats after bulk load.
                try:
                    ctx.storage.analyze_table(table_name, conn)
                except Exception:
                    pass

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
