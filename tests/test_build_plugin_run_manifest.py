from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from jsonschema import validate


def _init_minimal_root(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    (root / "plugins" / "plugin_alpha").mkdir(parents=True, exist_ok=True)
    (root / "appdata" / "runs" / "run_alpha").mkdir(parents=True, exist_ok=True)
    (root / "plugins" / "plugin_alpha" / "plugin.yaml").write_text(
        json.dumps(
            {
                "id": "plugin_alpha",
                "type": "analysis",
                "version": "1.0.0",
                "entrypoint": "run.py:run",
            }
        ),
        encoding="utf-8",
    )
    db_path = root / "appdata" / "state.sqlite"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE runs (
            run_id TEXT PRIMARY KEY,
            status TEXT,
            dataset_version_id TEXT,
            run_seed INTEGER,
            requested_run_seed INTEGER,
            input_hash TEXT
        );
        CREATE TABLE plugin_executions (
            execution_id INTEGER PRIMARY KEY,
            run_id TEXT,
            plugin_id TEXT,
            status TEXT
        );
        CREATE TABLE plugin_results_v2 (
            result_id INTEGER PRIMARY KEY,
            run_id TEXT,
            plugin_id TEXT,
            status TEXT,
            plugin_version TEXT,
            code_hash TEXT,
            settings_hash TEXT,
            execution_fingerprint TEXT,
            dataset_hash TEXT
        );
        """
    )
    conn.execute(
        """
        INSERT INTO runs(run_id, status, dataset_version_id, run_seed, requested_run_seed, input_hash)
        VALUES('run_alpha', 'completed', 'dataset_alpha', 123, 123, 'hash_input_alpha')
        """
    )
    conn.execute(
        "INSERT INTO plugin_executions(execution_id, run_id, plugin_id, status) VALUES(1, 'run_alpha', 'plugin_alpha', 'ok')"
    )
    conn.execute(
        """
        INSERT INTO plugin_results_v2(result_id, run_id, plugin_id, status, plugin_version, code_hash, settings_hash, execution_fingerprint, dataset_hash)
        VALUES(1, 'run_alpha', 'plugin_alpha', 'ok', '1.0.1', 'code_hash', 'settings_hash', 'fingerprint', 'dataset_hash')
        """
    )
    conn.commit()
    conn.close()
    return root, db_path


def test_build_plugin_run_manifest_validates_schema(tmp_path: Path) -> None:
    root, db_path = _init_minimal_root(tmp_path)
    out_path = root / "appdata" / "runs" / "run_alpha" / "plugin_run_manifest.json"
    cmd = [
        sys.executable,
        "scripts/build_plugin_run_manifest.py",
        "--root",
        str(root),
        "--run-id",
        "run_alpha",
        "--db",
        str(db_path),
        "--out-json",
        str(out_path),
    ]
    proc = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out_path.read_text(encoding="utf-8"))

    schema_path = Path("docs/plugin_run_manifest.schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validate(instance=payload, schema=schema)

    assert payload["run_id"] == "run_alpha"
    assert payload["plugin_count"] == 1
    assert payload["plugins"][0]["plugin_id"] == "plugin_alpha"
    assert payload["plugins"][0]["result_status"] == "ok"
