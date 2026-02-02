import pandas as pd

from plugins.analysis_determinism_discipline.plugin import Plugin
from statistic_harness.core.types import PluginResult
from statistic_harness.core.utils import now_iso
from tests.conftest import make_context


def test_determinism_discipline_flags_missing_measurement(run_dir):
    df = pd.DataFrame({"value": [1]})
    ctx = make_context(run_dir, df, {})

    ctx.storage.create_run(
        run_id=ctx.run_id,
        created_at=now_iso(),
        status="running",
        upload_id="local",
        input_filename="fixture.csv",
        canonical_path=str(ctx.run_dir / "dataset" / "canonical.csv"),
        settings={},
        error=None,
        run_seed=ctx.run_seed,
        project_id=ctx.project_id,
        dataset_id=ctx.dataset_id,
        dataset_version_id=ctx.dataset_version_id,
        input_hash=ctx.input_hash,
    )

    ctx.storage.save_plugin_result(
        ctx.run_id,
        "analysis_fake_missing",
        None,
        now_iso(),
        None,
        None,
        None,
        PluginResult(
            "ok",
            "missing measurement",
            {},
            [{"kind": "fake"}],
            [],
            None,
        ),
    )
    ctx.storage.save_plugin_result(
        ctx.run_id,
        "analysis_fake_modeled",
        None,
        now_iso(),
        None,
        None,
        None,
        PluginResult(
            "ok",
            "modeled missing assumption",
            {},
            [{"kind": "fake", "measurement_type": "modeled"}],
            [],
            None,
        ),
    )
    ctx.storage.save_plugin_result(
        ctx.run_id,
        "analysis_fake_good",
        None,
        now_iso(),
        None,
        None,
        None,
        PluginResult(
            "ok",
            "measured",
            {},
            [{"kind": "fake", "measurement_type": "measured"}],
            [],
            None,
        ),
    )

    result = Plugin().run(ctx)

    assert result.status == "ok"
    assert result.findings
    summary = result.findings[0]
    assert summary["kind"] == "determinism_discipline_summary"
    assert summary["missing_measurement_type"] >= 1
    assert summary["modeled_missing_assumption"] >= 1
    violations = [f for f in result.findings if f.get("kind") == "determinism_violation"]
    assert any(v.get("issue") == "missing_measurement_type" for v in violations)
    assert any(v.get("issue") == "modeled_missing_assumption" for v in violations)
