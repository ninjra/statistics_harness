import json

import pandas as pd
from plugins.analysis_traceability_manifest_v2.plugin import Plugin
from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import now_iso, write_json
from tests.conftest import make_context


def _register_plugin_result(storage, run_id, plugin_id, artifact_path):
    result = PluginResult(
        "ok",
        "test",
        {},
        [],
        [PluginArtifact(path=artifact_path, type="json", description="test")],
        None,
    )
    storage.save_plugin_result(
        run_id,
        plugin_id,
        "0.0",
        now_iso(),
        None,
        None,
        None,
        result,
    )


def test_traceability_manifest_builds_claims(run_dir):
    ctx = make_context(run_dir, df=pd.DataFrame(), settings={}, populate=False)
    storage = ctx.storage
    upload_id = "upload1"
    storage.create_upload(upload_id, "test.csv", 0, "a" * 64, now_iso())
    storage.create_run(
        ctx.run_id,
        now_iso(),
        "completed",
        upload_id,
        "test.csv",
        "",
        {},
        None,
        run_seed=42,
        project_id=ctx.project_id,
        dataset_id=ctx.dataset_id,
        dataset_version_id=ctx.dataset_version_id,
        input_hash="hash",
    )

    busy_dir = run_dir / "artifacts" / "analysis_busy_period_segmentation_v2"
    busy_dir.mkdir(parents=True, exist_ok=True)
    busy_path = busy_dir / "busy_periods.json"
    write_json(
        busy_path,
        {
            "busy_periods": [
                {
                    "busy_period_id": "bp_0001",
                    "total_over_threshold_wait_sec": 120.0,
                    "per_process_over_threshold_wait_sec": {"qemail": 120.0},
                }
            ]
        },
    )
    _register_plugin_result(
        storage,
        ctx.run_id,
        "analysis_busy_period_segmentation_v2",
        str(busy_path.relative_to(run_dir)),
    )

    waterfall_dir = run_dir / "artifacts" / "analysis_waterfall_summary_v2"
    waterfall_dir.mkdir(parents=True, exist_ok=True)
    waterfall_path = waterfall_dir / "waterfall_summary.json"
    write_json(
        waterfall_path,
        {
            "rows": [
                {"row_id": "total_bp_over_threshold_wait", "value_hours": 0.03}
            ],
            "total_over_threshold_wait_sec": 120.0,
        },
    )
    _register_plugin_result(
        storage,
        ctx.run_id,
        "analysis_waterfall_summary_v2",
        str(waterfall_path.relative_to(run_dir)),
    )

    result = Plugin().run(ctx)
    assert result.status == "ok"
    payload = json.loads(
        (run_dir / "slide_kit" / "traceability_manifest.json").read_text(encoding="utf-8")
    )
    claims = payload.get("claims") or []
    assert claims
