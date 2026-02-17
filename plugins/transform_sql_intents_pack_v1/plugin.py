from __future__ import annotations

import json
from typing import Any

from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.sql_intents import default_sql_intents


def _quote_ident(name: str) -> str:
    return '"' + str(name or "").replace('"', '""') + '"'


def _safe_query_id(text: str, fallback: str) -> str:
    raw = "".join(ch if ch.isalnum() else "_" for ch in str(text or "").lower()).strip("_")
    if not raw:
        raw = fallback
    while "__" in raw:
        raw = raw.replace("__", "_")
    return raw


def _select_bootstrap_table(schema: dict[str, Any]) -> dict[str, Any] | None:
    tables = schema.get("tables") if isinstance(schema, dict) else None
    if not isinstance(tables, list):
        return None
    candidates: list[dict[str, Any]] = []
    for item in tables:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        cols = item.get("columns") if isinstance(item.get("columns"), list) else []
        if not name or not cols:
            continue
        candidates.append({"name": name, "columns": cols})
    if not candidates:
        return None
    preferred = [
        c for c in candidates if str(c["name"]).startswith("template_normalized_")
    ]
    pool = preferred or candidates
    pool.sort(key=lambda c: (-len(c.get("columns") or []), str(c.get("name") or "")))
    return pool[0]


def _build_bootstrap_pack(schema: dict[str, Any], intents: list[dict[str, Any]]) -> dict[str, Any]:
    table = _select_bootstrap_table(schema)
    if table is None:
        raise ValueError("Schema snapshot did not include queryable tables")
    table_name = str(table.get("name") or "")
    quoted_table = _quote_ident(table_name)
    queries: list[dict[str, Any]] = [
        {
            "id": "bootstrap_row_count",
            "purpose": f"Row count bootstrap for {table_name}",
            "mode": "ro",
            "sql": f"SELECT COUNT(*) AS row_count FROM {quoted_table}",
        },
        {
            "id": "bootstrap_head_sample",
            "purpose": f"Head sample bootstrap for {table_name}",
            "mode": "ro",
            "sql": f"SELECT * FROM {quoted_table} LIMIT 100",
        },
    ]
    for idx, intent in enumerate(intents[:6], start=1):
        if not isinstance(intent, dict):
            continue
        iid = _safe_query_id(str(intent.get("id") or ""), f"intent_{idx}")
        purpose = str(intent.get("purpose") or f"Intent bootstrap query {idx}")
        queries.append(
            {
                "id": f"{iid}_bootstrap",
                "purpose": purpose,
                "mode": "ro",
                "sql": f"SELECT * FROM {quoted_table} LIMIT 250",
            }
        )
    return {
        "schema_hash": str(schema.get("schema_hash") or ""),
        "dialect": "sqlite",
        "model": {"name": "bootstrap_sql_pack", "source": "deterministic"},
        # Keep bootstrap packs schema-valid so downstream sqlpack materialization can run deterministically.
        "decode": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 256},
        "queries": queries,
    }


class Plugin:
    def run(self, ctx) -> PluginResult:
        schema = ctx.sql_schema_snapshot if isinstance(ctx.sql_schema_snapshot, dict) else None
        if not schema:
            return PluginResult(
                "error",
                "Schema snapshot unavailable (sql assist not wired); cannot build SQL intents pack",
                metrics={},
                findings=[],
                artifacts=[],
                error=None,
            )

        intents = default_sql_intents(schema)
        bootstrap_pack = _build_bootstrap_pack(schema, intents)
        artifacts_dir = ctx.artifacts_dir("transform_sql_intents_pack_v1")
        schema_path = artifacts_dir / "schema_snapshot.json"
        intents_path = artifacts_dir / str(ctx.settings.get("out_intents_name") or "sql_intents.json")
        bootstrap_pack_path = artifacts_dir / "sql_pack_bootstrap.json"
        schema_path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        intents_path.write_text(json.dumps({"intents": intents}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        bootstrap_pack_path.write_text(
            json.dumps(bootstrap_pack, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        artifacts = [
            PluginArtifact(
                path=str(schema_path.relative_to(ctx.run_dir)),
                type="json",
                description="Schema snapshot for SQL generation",
            ),
            PluginArtifact(
                path=str(intents_path.relative_to(ctx.run_dir)),
                type="json",
                description="SQL query intents",
            ),
            PluginArtifact(
                path=str(bootstrap_pack_path.relative_to(ctx.run_dir)),
                type="json",
                description="Deterministic bootstrap SQL pack",
            ),
        ]
        return PluginResult(
            "ok",
            f"Wrote {len(intents)} SQL intent(s) and bootstrap SQL pack ({len(bootstrap_pack.get('queries') or [])} queries)",
            metrics={
                "intents": int(len(intents)),
                "bootstrap_queries": int(len(bootstrap_pack.get("queries") or [])),
            },
            findings=[
                {
                    "kind": "sql_intents_pack",
                    "intents": int(len(intents)),
                    "measurement_type": "measured",
                    "reason_code": "SQL_ASSIST_READY",
                },
                {
                    "kind": "sql_pack_bootstrap",
                    "queries": int(len(bootstrap_pack.get("queries") or [])),
                    "measurement_type": "measured",
                    "reason_code": "SQL_PACK_BOOTSTRAPPED",
                }
            ],
            artifacts=artifacts,
            error=None,
        )
