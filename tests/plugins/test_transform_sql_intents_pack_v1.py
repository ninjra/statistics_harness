from __future__ import annotations

import json
from pathlib import Path

from plugins.transform_sql_intents_pack_v1.plugin import Plugin


class _Ctx:
    def __init__(self, run_dir: Path, schema: dict | None) -> None:
        self.run_dir = run_dir
        self.sql_schema_snapshot = schema
        self.settings = {}

    def artifacts_dir(self, plugin_id: str) -> Path:
        path = self.run_dir / "artifacts" / plugin_id
        path.mkdir(parents=True, exist_ok=True)
        return path


def test_transform_sql_intents_requires_schema_snapshot(tmp_path: Path) -> None:
    ctx = _Ctx(tmp_path, None)
    result = Plugin().run(ctx)
    assert result.status == "error"
    assert "schema snapshot unavailable" in result.summary.lower()


def test_transform_sql_intents_writes_bootstrap_pack(tmp_path: Path) -> None:
    schema = {
        "schema_hash": "abcd1234abcd1234",
        "tables": [
            {
                "name": "template_normalized_dataset_v1",
                "columns": [
                    {"name": "PROCESS_ID"},
                    {"name": "QUEUE_DT"},
                    {"name": "START_DT"},
                    {"name": "END_DT"},
                ],
                "indexes": [],
            }
        ],
    }
    ctx = _Ctx(tmp_path, schema)
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert any(a.path.endswith("sql_pack_bootstrap.json") for a in result.artifacts)
    bootstrap_path = tmp_path / "artifacts" / "transform_sql_intents_pack_v1" / "sql_pack_bootstrap.json"
    payload = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    assert payload["dialect"] == "sqlite"
    assert int(payload["decode"]["max_tokens"]) >= 1
    assert len(payload["queries"]) >= 1
