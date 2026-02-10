from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_FORBIDDEN_TOKENS = (
    "pragma",
    "attach",
    "detach",
    "vacuum",
)


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sql_first_keyword(sql: str) -> str:
    s = sql.strip().lstrip("(").strip()
    m = re.match(r"^([A-Za-z_]+)", s)
    return m.group(1).lower() if m else ""


def _validate_sql(sql: str, *, mode: str) -> None:
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("SQL must be a non-empty string")
    raw = sql.strip()
    lowered = raw.lower()

    # Fail-closed on multi-statement. Semicolons are the simplest robust signal.
    # (sqlite accepts semicolons inside strings, but that complexity is not worth it here.)
    if ";" in raw:
        raise ValueError("SQL must be a single statement (no ';' allowed)")

    for tok in _FORBIDDEN_TOKENS:
        if re.search(rf"\\b{re.escape(tok)}\\b", lowered):
            raise ValueError(f"Forbidden SQL token: {tok}")

    kw = _sql_first_keyword(raw)
    if mode == "ro":
        if kw not in {"select", "with"}:
            raise ValueError(f"Read-only SQL must start with SELECT/WITH (got {kw or 'unknown'})")
        return

    if mode == "scratch":
        # Scratch DB allows DDL/DML, but we still keep it bounded to single-statement
        # and forbid file-oriented features (ATTACH/PRAGMA/VACUUM).
        if kw not in {
            "select",
            "with",
            "create",
            "drop",
            "insert",
            "update",
            "delete",
        }:
            raise ValueError(f"Scratch SQL unsupported statement (got {kw or 'unknown'})")
        return

    if mode == "plugin":
        # Plugin-owned tables are permitted. We still ban file/system escape hatches
        # and require single-statement SQL.
        if kw not in {
            "select",
            "with",
            "create",
            "drop",
            "insert",
            "update",
            "delete",
        }:
            raise ValueError(f"Plugin SQL unsupported statement (got {kw or 'unknown'})")
        return

    raise ValueError(f"Unknown SQL assist mode: {mode}")


def _rows_to_jsonable(rows: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        if hasattr(r, "keys"):
            out.append({str(k): r[k] for k in r.keys()})
        elif isinstance(r, dict):
            out.append({str(k): r[k] for k in sorted(r.keys())})
        else:
            out.append({"row": r})
    return out


@dataclass(frozen=True)
class SqlQueryResult:
    row_count: int
    sample_rows: list[dict[str, Any]]
    result_sha256: str


class SqlAssist:
    """Deterministic SQL execution helper with provenance capture.

    `mode`:
    - "ro": read-only against the dataset DB
    - "scratch": read/write against a plugin-owned scratch DB (still guarded)
    """

    def __init__(
        self,
        *,
        storage: Any,
        run_dir: Path,
        plugin_id: str,
        schema_hash: str | None,
        mode: str,
        allowed_prefix: str | None = None,
    ) -> None:
        self._storage = storage
        self._run_dir = Path(run_dir)
        self._plugin_id = str(plugin_id)
        self._schema_hash = str(schema_hash) if schema_hash else ""
        self._mode = str(mode)
        self._allowed_prefix = str(allowed_prefix) if allowed_prefix else f"plg__{self._plugin_id}__"

    def _validate_plugin_object_prefix(self, sql: str) -> None:
        if self._mode != "plugin":
            return
        kw = _sql_first_keyword(sql)
        prefix = self._allowed_prefix.lower()
        lowered = sql.strip().lower()

        def _extract_name(pat: str) -> str | None:
            m = re.search(pat, lowered)
            return m.group(1) if m else None

        # Very conservative name extraction; if we can't prove it is prefixed, deny.
        if kw == "create":
            name = _extract_name(r"\bcreate\s+(?:table|view|index)\s+([a-z0-9_]+)")
            if not name or not name.startswith(prefix):
                raise ValueError(f"Plugin SQL must create prefixed objects: {self._allowed_prefix}*")
        if kw == "drop":
            name = _extract_name(r"\bdrop\s+(?:table|view|index)\s+([a-z0-9_]+)")
            if not name or not name.startswith(prefix):
                raise ValueError(f"Plugin SQL must drop prefixed objects: {self._allowed_prefix}*")
        if kw == "insert":
            name = _extract_name(r"\binsert\s+into\s+([a-z0-9_]+)")
            if not name or not name.startswith(prefix):
                raise ValueError(f"Plugin SQL must insert into prefixed tables: {self._allowed_prefix}*")
        if kw == "update":
            name = _extract_name(r"\bupdate\s+([a-z0-9_]+)")
            if not name or not name.startswith(prefix):
                raise ValueError(f"Plugin SQL must update prefixed tables: {self._allowed_prefix}*")
        if kw == "delete":
            name = _extract_name(r"\bdelete\s+from\s+([a-z0-9_]+)")
            if not name or not name.startswith(prefix):
                raise ValueError(f"Plugin SQL must delete from prefixed tables: {self._allowed_prefix}*")

    def _sql_dir(self) -> Path:
        path = self._run_dir / "artifacts" / self._plugin_id / "sql"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def query_rows(
        self,
        sql: str,
        params: Any | None = None,
        *,
        query_id: str | None = None,
        max_rows: int = 50_000,
        sample_rows: int = 500,
        explain: bool = False,
    ) -> SqlQueryResult:
        _validate_sql(sql, mode=self._mode)
        qid = (query_id or "").strip() or f"q_{_sha256_text(sql)[:12]}"
        base = self._sql_dir() / qid

        # Always persist the SQL itself.
        sql_path = base.with_suffix(".sql")
        sql_path.write_text(sql.rstrip() + "\n", encoding="utf-8")

        plan: list[dict[str, Any]] = []
        rows_json: list[dict[str, Any]] = []
        row_count = 0

        with self._storage.connection() as conn:
            if explain:
                try:
                    cur = conn.execute("EXPLAIN QUERY PLAN " + sql, params or ())
                    plan_rows = cur.fetchall()
                    plan = _rows_to_jsonable(plan_rows)
                except Exception:
                    plan = []

            cur = conn.execute(sql, params or ())
            fetched = cur.fetchmany(int(max_rows) + 1)
            if len(fetched) > int(max_rows):
                raise ValueError(
                    f"SQL returned >{max_rows} rows; add aggregation/filters or increase max_rows explicitly"
                )
            row_count = len(fetched)
            rows_json = _rows_to_jsonable(fetched[: int(sample_rows)])

        result_hash = _sha256_text(_stable_json(rows_json))
        manifest = {
            "query_id": qid,
            "mode": self._mode,
            "schema_hash": self._schema_hash or None,
            "sql_path": str(sql_path.relative_to(self._run_dir)),
            "sql_sha256": _sha256_text(sql),
            "params": params,
            "row_count": int(row_count),
            "sample_row_count": int(len(rows_json)),
            "sample_sha256": result_hash,
            "explain_query_plan": plan,
        }

        manifest_path = base.with_suffix(".manifest.json")
        manifest_path.write_text(_stable_json(manifest), encoding="utf-8")
        sample_path = base.with_suffix(".sample.json")
        sample_path.write_text(_stable_json(rows_json), encoding="utf-8")

        return SqlQueryResult(
            row_count=int(row_count),
            sample_rows=rows_json,
            result_sha256=result_hash,
        )

    def validate_ro_sql(
        self,
        sql: str,
        params: Any | None = None,
        *,
        query_id: str | None = None,
    ) -> None:
        """Validate a read-only SQL statement without executing it.

        This runs `EXPLAIN QUERY PLAN` to ensure the statement parses and is
        compatible with the current SQLite connection. It does not run the query.
        """

        if self._mode not in {"ro", "plugin", "scratch"}:
            raise ValueError("validate_ro_sql: unsupported sql assist mode")
        _validate_sql(sql, mode="ro")
        qid = (query_id or "").strip() or f"validate_{_sha256_text(sql)[:12]}"
        base = self._sql_dir() / qid
        sql_path = base.with_suffix(".sql")
        sql_path.write_text(sql.rstrip() + "\n", encoding="utf-8")

        plan: list[dict[str, Any]] = []
        with self._storage.connection() as conn:
            cur = conn.execute("EXPLAIN QUERY PLAN " + sql, params or ())
            plan_rows = cur.fetchall()
            plan = _rows_to_jsonable(plan_rows)

        manifest = {
            "query_id": qid,
            "mode": "validate_ro",
            "schema_hash": self._schema_hash or None,
            "sql_path": str(sql_path.relative_to(self._run_dir)),
            "sql_sha256": _sha256_text(sql),
            "params": params,
            "explain_query_plan": plan,
        }
        manifest_path = base.with_suffix(".manifest.json")
        manifest_path.write_text(_stable_json(manifest), encoding="utf-8")

    def exec_scratch(
        self,
        sql: str,
        params: Any | None = None,
        *,
        query_id: str | None = None,
    ) -> None:
        if self._mode != "scratch":
            raise ValueError("exec_scratch is only allowed for scratch mode")
        _validate_sql(sql, mode="scratch")
        qid = (query_id or "").strip() or f"exec_{_sha256_text(sql)[:12]}"
        base = self._sql_dir() / qid
        sql_path = base.with_suffix(".sql")
        sql_path.write_text(sql.rstrip() + "\n", encoding="utf-8")
        with self._storage.connection() as conn:
            conn.execute(sql, params or ())
        manifest = {
            "query_id": qid,
            "mode": "scratch",
            "sql_path": str(sql_path.relative_to(self._run_dir)),
            "sql_sha256": _sha256_text(sql),
            "params": params,
        }
        manifest_path = base.with_suffix(".manifest.json")
        manifest_path.write_text(_stable_json(manifest), encoding="utf-8")

    def exec_plugin(
        self,
        sql: str,
        params: Any | None = None,
        *,
        query_id: str | None = None,
    ) -> None:
        if self._mode != "plugin":
            raise ValueError("exec_plugin is only allowed for plugin mode")
        _validate_sql(sql, mode="plugin")
        self._validate_plugin_object_prefix(sql)
        qid = (query_id or "").strip() or f"exec_{_sha256_text(sql)[:12]}"
        base = self._sql_dir() / qid
        sql_path = base.with_suffix(".sql")
        sql_path.write_text(sql.rstrip() + "\n", encoding="utf-8")
        with self._storage.connection() as conn:
            conn.execute(sql, params or ())
        manifest = {
            "query_id": qid,
            "mode": "plugin",
            "schema_hash": self._schema_hash or None,
            "sql_path": str(sql_path.relative_to(self._run_dir)),
            "sql_sha256": _sha256_text(sql),
            "params": params,
        }
        manifest_path = base.with_suffix(".manifest.json")
        manifest_path.write_text(_stable_json(manifest), encoding="utf-8")
