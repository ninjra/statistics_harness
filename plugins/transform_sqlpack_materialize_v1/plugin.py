from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from statistic_harness.core.sql_pack import load_sql_pack, validate_sql_pack
from statistic_harness.core.types import PluginArtifact, PluginResult


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


class Plugin:
    def run(self, ctx) -> PluginResult:
        if not bool(ctx.settings.get("enabled", False)):
            return PluginResult(
                "skipped",
                "SQL pack materialization disabled (set enabled=true for this plugin)",
                metrics={},
                findings=[],
                artifacts=[],
                error=None,
            )

        rel = str(ctx.settings.get("source_relpath") or "").strip()
        if not rel:
            return PluginResult(
                "error",
                "Missing source_relpath",
                metrics={},
                findings=[],
                artifacts=[],
                error=None,
            )
        pack_path = ctx.run_dir / rel
        if not pack_path.exists():
            return PluginResult(
                "error",
                f"Missing sql pack at {pack_path}",
                metrics={},
                findings=[],
                artifacts=[],
                error=None,
            )

        pack = load_sql_pack(pack_path)
        validate_sql_pack(pack, Path("docs/sql_pack.schema.json"))
        schema_hash = str(pack.get("schema_hash") or "").strip()

        table_prefix = str(ctx.settings.get("table_prefix") or "plg__transform_sqlpack_materialize_v1__")
        # Enforce plugin prefix policy: materializer owns derived objects.
        if not table_prefix.startswith("plg__transform_sqlpack_materialize_v1__"):
            table_prefix = "plg__transform_sqlpack_materialize_v1__"

        if ctx.sql_exec is None:
            return PluginResult(
                "error",
                "sql_exec unavailable (sql assist not wired)",
                metrics={},
                findings=[],
                artifacts=[],
                error=None,
            )

        artifacts_dir = ctx.artifacts_dir("transform_sqlpack_materialize_v1")
        created: list[str] = []
        errors: list[str] = []

        for q in (pack.get("queries") or []):
            if not isinstance(q, dict):
                continue
            qid = str(q.get("id") or "").strip()
            if not qid:
                continue
            mode = str(q.get("mode") or "ro")
            sql = str(q.get("sql") or "").strip()
            if not sql:
                continue
            if mode != "ro":
                # By default we only accept read-only queries and materialize them ourselves.
                continue

            dest = f"{table_prefix}{qid}"
            # Materialize as a derived table; drop/recreate for idempotency.
            try:
                ctx.sql_exec.exec_plugin(f"DROP TABLE IF EXISTS {dest}", query_id=f"drop_{qid}")
                ctx.sql_exec.exec_plugin(f"CREATE TABLE {dest} AS {sql}", query_id=f"create_{qid}")
                created.append(dest)
            except Exception as exc:
                errors.append(f"{qid}: {type(exc).__name__}: {exc}")

        manifest = {"schema_hash": schema_hash, "created_tables": created, "errors": errors}
        manifest_path = artifacts_dir / "materialize_manifest.json"
        manifest_path.write_text(_stable_json(manifest), encoding="utf-8")

        artifacts = [
            PluginArtifact(
                path=str(manifest_path.relative_to(ctx.run_dir)),
                type="json",
                description="Materialization manifest",
            )
        ]

        if errors:
            return PluginResult(
                "degraded",
                f"Materialized {len(created)} table(s) with {len(errors)} error(s)",
                metrics={"tables_created": len(created), "errors": len(errors)},
                findings=[
                    {
                        "kind": "sqlpack_materialize_errors",
                        "tables_created": len(created),
                        "errors": errors[:10],
                        "measurement_type": "measured",
                    }
                ],
                artifacts=artifacts,
                error=None,
            )

        return PluginResult(
            "ok",
            f"Materialized {len(created)} table(s)",
            metrics={"tables_created": len(created)},
            findings=[
                {
                    "kind": "sqlpack_materialized",
                    "tables_created": len(created),
                    "measurement_type": "measured",
                }
            ],
            artifacts=artifacts,
            error=None,
        )

