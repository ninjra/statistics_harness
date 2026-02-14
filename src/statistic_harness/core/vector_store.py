"""Local sqlite-vec vector store backend."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from .utils import DEFAULT_TENANT_ID, ensure_dir, json_dumps, now_iso, quote_identifier, scope_key, vector_store_enabled


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def hash_embedding(text: str, dimensions: int = 128) -> list[float]:
    tokens = _TOKEN_RE.findall(text.lower())
    vector = [0.0] * dimensions
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[idx] += sign
    norm = math.sqrt(sum(val * val for val in vector))
    if norm > 0:
        vector = [val / norm for val in vector]
    return vector


class VectorStore:
    def __init__(self, db_path: Path, tenant_id: str | None = None) -> None:
        if not vector_store_enabled():
            raise RuntimeError(
                "Vector store disabled (set STAT_HARNESS_ENABLE_VECTOR_STORE=1)"
            )
        ensure_dir(db_path.parent)
        self.db_path = db_path
        self.tenant_id = tenant_id or DEFAULT_TENANT_ID
        self._vec_available = False
        with self.connection() as conn:
            self._ensure_meta(conn)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        self._vec_available = self._ensure_extension(conn)
        return conn

    @contextmanager
    def connection(self) -> Iterable[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _ensure_extension(self, conn: sqlite3.Connection) -> bool:
        try:
            conn.execute("SELECT vec_version()").fetchone()
            return True
        except sqlite3.OperationalError:
            pass

        path = os.environ.get("STAT_HARNESS_SQLITE_VEC_PATH", "").strip()
        if not path:
            return False
        try:
            conn.enable_load_extension(True)
            conn.load_extension(path)
        except (AttributeError, sqlite3.OperationalError):
            return False
        finally:
            try:
                conn.enable_load_extension(False)
            except Exception:
                pass
        try:
            conn.execute("SELECT vec_version()").fetchone()
        except sqlite3.OperationalError:
            return False
        return True

    def _ensure_meta(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vector_collections (
                collection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT NOT NULL,
                name TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                table_name TEXT NOT NULL,
                created_at TEXT,
                UNIQUE (tenant_id, name, dimensions)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_vector_collections_tenant ON vector_collections(tenant_id)"
        )

    def _collection_table_name(self, name: str, dimensions: int) -> str:
        digest = scope_key("vector_collection", f"{self.tenant_id}:{name}:{dimensions}")
        return f"vec_{digest}"

    def _ensure_collection(self, conn: sqlite3.Connection, name: str, dimensions: int) -> str:
        row = conn.execute(
            """
            SELECT table_name
            FROM vector_collections
            WHERE tenant_id = ? AND name = ? AND dimensions = ?
            """,
            (self.tenant_id, name, int(dimensions)),
        ).fetchone()
        if row:
            return str(row["table_name"])
        table_name = self._collection_table_name(name, dimensions)
        conn.execute(
            """
            INSERT INTO vector_collections (tenant_id, name, dimensions, table_name, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (self.tenant_id, name, int(dimensions), table_name, now_iso()),
        )
        if self._vec_available:
            conn.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {quote_identifier(table_name)} USING vec0(
                    embedding float[{int(dimensions)}],
                    tenant_id TEXT,
                    collection TEXT,
                    item_id TEXT,
                    payload TEXT,
                    created_at TEXT
                )
                """
            )
        else:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {quote_identifier(table_name)} (
                    embedding_json TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    collection TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    payload TEXT,
                    created_at TEXT,
                    PRIMARY KEY (tenant_id, collection, item_id)
                )
                """
            )
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {quote_identifier(table_name + "_lookup_idx")}
                ON {quote_identifier(table_name)} (tenant_id, collection, created_at)
                """
            )
        return table_name

    def _has_column(self, conn: sqlite3.Connection, table_name: str, column: str) -> bool:
        cur = conn.execute(
            f"PRAGMA table_info({quote_identifier(table_name)})"
        )
        return any(row["name"] == column for row in cur.fetchall())

    def _lookup_collection(
        self, conn: sqlite3.Connection, name: str, dimensions: int
    ) -> str | None:
        row = conn.execute(
            """
            SELECT table_name
            FROM vector_collections
            WHERE tenant_id = ? AND name = ? AND dimensions = ?
            """,
            (self.tenant_id, name, int(dimensions)),
        ).fetchone()
        return str(row["table_name"]) if row else None

    def list_collections(self) -> list[dict[str, Any]]:
        with self.connection() as conn:
            cur = conn.execute(
                """
                SELECT name, dimensions, created_at
                FROM vector_collections
                WHERE tenant_id = ?
                ORDER BY created_at
                """,
                (self.tenant_id,),
            )
            return [dict(row) for row in cur.fetchall()]

    def collection_dimensions(self, name: str) -> list[int]:
        with self.connection() as conn:
            cur = conn.execute(
                """
                SELECT dimensions
                FROM vector_collections
                WHERE tenant_id = ? AND name = ?
                ORDER BY dimensions
                """,
                (self.tenant_id, name),
            )
            return [int(row["dimensions"]) for row in cur.fetchall()]

    def add(
        self,
        collection: str,
        vectors: list[list[float]],
        item_ids: list[str] | None = None,
        payloads: list[dict[str, Any] | None] | None = None,
    ) -> list[str]:
        if not vectors:
            return []
        dimensions = len(vectors[0])
        for vec in vectors:
            if len(vec) != dimensions:
                raise ValueError("Vector dimension mismatch")
        if item_ids is None:
            item_ids = [
                self._default_item_id(collection, vec, payloads[idx] if payloads else None)
                for idx, vec in enumerate(vectors)
            ]
        if len(item_ids) != len(vectors):
            raise ValueError("item_ids length mismatch")
        if payloads is None:
            payloads = [None] * len(vectors)
        if len(payloads) != len(vectors):
            raise ValueError("payloads length mismatch")

        with self.connection() as conn:
            table_name = self._ensure_collection(conn, collection, dimensions)
            has_created_at = self._has_column(conn, table_name, "created_at")
            rows = []
            for vec, item_id, payload in zip(vectors, item_ids, payloads):
                vec_json = json.dumps(vec, separators=(",", ":"))
                payload_json = json_dumps(payload) if payload is not None else ""
                if has_created_at:
                    rows.append(
                        (
                            vec_json,
                            self.tenant_id,
                            collection,
                            item_id,
                            payload_json,
                            now_iso(),
                        )
                    )
                else:
                    rows.append(
                        (vec_json, self.tenant_id, collection, item_id, payload_json)
                    )
            if self._vec_available:
                if has_created_at:
                    columns = "(embedding, tenant_id, collection, item_id, payload, created_at)"
                    placeholders = ", ".join(["?"] * 6)
                else:
                    columns = "(embedding, tenant_id, collection, item_id, payload)"
                    placeholders = ", ".join(["?"] * 5)
                conn.executemany(
                    f"""
                    INSERT INTO {quote_identifier(table_name)}
                    {columns}
                    VALUES ({placeholders})
                    """,
                    rows,
                )
            else:
                if has_created_at:
                    columns = "(embedding_json, tenant_id, collection, item_id, payload, created_at)"
                    placeholders = ", ".join(["?"] * 6)
                else:
                    columns = "(embedding_json, tenant_id, collection, item_id, payload)"
                    placeholders = ", ".join(["?"] * 5)
                conn.executemany(
                    f"""
                    INSERT OR REPLACE INTO {quote_identifier(table_name)}
                    {columns}
                    VALUES ({placeholders})
                    """,
                    rows,
                )
        return item_ids

    def query(
        self,
        collection: str,
        vector: list[float],
        k: int = 10,
        as_of: str | None = None,
    ) -> list[dict[str, Any]]:
        if k <= 0:
            return []
        dimensions = len(vector)
        with self.connection() as conn:
            table_name = self._lookup_collection(conn, collection, dimensions)
            if not table_name:
                return []
            has_created_at = self._has_column(conn, table_name, "created_at")
            results: list[dict[str, Any]] = []
            if self._vec_available:
                vec_json = json.dumps(vector, separators=(",", ":"))
                params: list[Any] = [vec_json, int(k)]
                where = "WHERE embedding MATCH ? AND k = ?"
                if as_of and has_created_at:
                    where += " AND (created_at IS NULL OR created_at <= ?)"
                    params.append(as_of)
                where += " AND tenant_id = ? AND collection = ?"
                params.extend([self.tenant_id, collection])
                cur = conn.execute(
                    f"""
                    SELECT item_id, payload, distance
                    FROM {quote_identifier(table_name)}
                    {where}
                    ORDER BY distance ASC
                    """,
                    params,
                )
                rows = cur.fetchall()
                for row in rows:
                    payload = row["payload"]
                    if payload is None or payload == "":
                        payload = None
                    elif payload:
                        try:
                            payload = json.loads(payload)
                        except json.JSONDecodeError:
                            pass
                    results.append(
                        {
                            "item_id": row["item_id"],
                            "distance": row["distance"],
                            "payload": payload,
                        }
                    )
            else:
                params = [self.tenant_id, collection]
                where = "WHERE tenant_id = ? AND collection = ?"
                if as_of and has_created_at:
                    where += " AND (created_at IS NULL OR created_at <= ?)"
                    params.append(as_of)
                cur = conn.execute(
                    f"""
                    SELECT item_id, payload, embedding_json
                    FROM {quote_identifier(table_name)}
                    {where}
                    """,
                    params,
                )
                rows = cur.fetchall()
                for row in rows:
                    embedding_raw = row["embedding_json"]
                    if not isinstance(embedding_raw, str) or not embedding_raw:
                        continue
                    try:
                        embedding = json.loads(embedding_raw)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(embedding, list) or len(embedding) != dimensions:
                        continue
                    try:
                        distance = self._l2_distance(vector, embedding)
                    except Exception:
                        continue
                    payload = row["payload"]
                    if payload is None or payload == "":
                        payload = None
                    elif payload:
                        try:
                            payload = json.loads(payload)
                        except json.JSONDecodeError:
                            pass
                    results.append(
                        {
                            "item_id": row["item_id"],
                            "distance": distance,
                            "payload": payload,
                        }
                    )
                results = results[:]
            results.sort(key=lambda item: (item["distance"], item["item_id"]))
            return results[: int(k)]

    def _l2_distance(self, left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            raise ValueError("Vector dimension mismatch")
        total = 0.0
        for a, b in zip(left, right):
            af = float(a)
            bf = float(b)
            delta = af - bf
            total += delta * delta
        return float(math.sqrt(total))

    def delete(self, collection: str, item_ids: list[str], dimensions: int) -> int:
        if not item_ids:
            return 0
        with self.connection() as conn:
            table_name = self._lookup_collection(conn, collection, dimensions)
            if not table_name:
                return 0
            placeholders = ", ".join(["?"] * len(item_ids))
            cur = conn.execute(
                f"""
                DELETE FROM {quote_identifier(table_name)}
                WHERE tenant_id = ? AND collection = ? AND item_id IN ({placeholders})
                """,
                [self.tenant_id, collection, *item_ids],
            )
            return cur.rowcount if cur.rowcount is not None else 0

    def _default_item_id(
        self, collection: str, vector: list[float], payload: dict[str, Any] | None
    ) -> str:
        vec_json = json.dumps(vector, separators=(",", ":"))
        payload_json = json_dumps(payload) if payload is not None else ""
        digest = hashlib.sha256(
            f"{collection}:{vec_json}:{payload_json}".encode("utf-8")
        ).hexdigest()
        return digest
