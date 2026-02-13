import json
from pathlib import Path

from jsonschema import validate

from statistic_harness.core.pipeline import Pipeline


def test_run_manifest_written_and_validates(tmp_path, monkeypatch):
    appdata = tmp_path / "appdata"
    monkeypatch.setenv("STAT_HARNESS_APPDATA", str(appdata))
    pipeline = Pipeline(appdata, Path("plugins"))
    run_id = pipeline.run(Path("tests/fixtures/synth_linear.csv"), ["profile_basic"], {}, 0)

    run_dir = appdata / "runs" / run_id
    manifest_path = run_dir / "run_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    schema = json.loads((Path("docs") / "run_manifest.schema.json").read_text(encoding="utf-8"))
    validate(instance=manifest, schema=schema)

    run_row = pipeline.storage.fetch_run(run_id)
    assert run_row
    assert manifest["run_fingerprint"] == (run_row.get("run_fingerprint") or "")


def test_events_emitted_for_run(tmp_path, monkeypatch):
    appdata = tmp_path / "appdata"
    monkeypatch.setenv("STAT_HARNESS_APPDATA", str(appdata))
    pipeline = Pipeline(appdata, Path("plugins"))
    run_id = pipeline.run(Path("tests/fixtures/synth_linear.csv"), ["profile_basic"], {}, 123)

    events = pipeline.storage.list_events(run_id)
    kinds = [e.get("kind") for e in events]
    assert "run_started" in kinds
    assert "run_fingerprint" in kinds
    assert "run_completed" in kinds

