from __future__ import annotations

import json
from typing import Any

from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.sql_intents import default_sql_intents


class Plugin:
    def run(self, ctx) -> PluginResult:
        schema = ctx.sql_schema_snapshot if isinstance(ctx.sql_schema_snapshot, dict) else None
        if not schema:
            return PluginResult(
                "skipped",
                "Schema snapshot unavailable (sql assist not wired)",
                metrics={},
                findings=[],
                artifacts=[],
                error=None,
            )

        intents = default_sql_intents(schema)
        artifacts_dir = ctx.artifacts_dir("transform_sql_intents_pack_v1")
        schema_path = artifacts_dir / "schema_snapshot.json"
        intents_path = artifacts_dir / str(ctx.settings.get("out_intents_name") or "sql_intents.json")
        schema_path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        intents_path.write_text(json.dumps({"intents": intents}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

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
        ]
        return PluginResult(
            "ok",
            f"Wrote {len(intents)} SQL intent(s)",
            metrics={"intents": int(len(intents))},
            findings=[
                {
                    "kind": "sql_intents_pack",
                    "intents": int(len(intents)),
                    "measurement_type": "measured",
                }
            ],
            artifacts=artifacts,
            error=None,
        )
