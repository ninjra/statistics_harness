from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SchemaSnapshot:
    schema_hash: str
    snapshot: dict[str, Any]


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def snapshot_schema(storage) -> SchemaSnapshot:
    """Return a deterministic snapshot of the current sqlite schema.

    This is used both for citeability and for LLM prompt context. It must be stable
    across runs given the same DB schema state.
    """

    with storage.connection() as conn:
        tables = [
            str(r["name"])
            for r in conn.execute(
                "select name from sqlite_master where type='table' and name not like 'sqlite_%' order by name"
            ).fetchall()
        ]
        indexes = [
            str(r["name"])
            for r in conn.execute(
                "select name from sqlite_master where type='index' and name not like 'sqlite_%' order by name"
            ).fetchall()
        ]

        table_blocks: list[dict[str, Any]] = []
        for t in tables:
            cols = conn.execute(f"PRAGMA table_info({json.dumps(t)})").fetchall()
            col_payload = [
                {
                    "cid": int(c["cid"]),
                    "name": str(c["name"]),
                    "type": str(c["type"] or ""),
                    "notnull": int(c["notnull"] or 0),
                    "dflt_value": c["dflt_value"],
                    "pk": int(c["pk"] or 0),
                }
                for c in cols
            ]
            idxs = conn.execute(f"PRAGMA index_list({json.dumps(t)})").fetchall()
            idx_payload: list[dict[str, Any]] = []
            for i in idxs:
                iname = str(i["name"])
                info = conn.execute(f"PRAGMA index_info({json.dumps(iname)})").fetchall()
                idx_payload.append(
                    {
                        "name": iname,
                        "unique": int(i["unique"] or 0),
                        "origin": str(i["origin"] or ""),
                        "partial": int(i["partial"] or 0),
                        "columns": [str(x["name"]) for x in info],
                    }
                )
            idx_payload = sorted(idx_payload, key=lambda r: r["name"])

            table_blocks.append(
                {
                    "name": t,
                    "columns": col_payload,
                    "indexes": idx_payload,
                }
            )

    payload = {"tables": table_blocks, "indexes": indexes}
    text = _stable_json(payload)
    return SchemaSnapshot(schema_hash=_sha256_text(text), snapshot=payload)


def write_schema_snapshot(path: Path, storage) -> SchemaSnapshot:
    snap = snapshot_schema(storage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_stable_json({"schema_hash": snap.schema_hash, **snap.snapshot}), encoding="utf-8")
    return snap

