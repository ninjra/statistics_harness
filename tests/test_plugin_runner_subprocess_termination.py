from pathlib import Path
import subprocess

from statistic_harness.core.plugin_manager import PluginSpec
from statistic_harness.core.plugin_runner import run_plugin_subprocess


def test_run_plugin_subprocess_reports_signal_termination(monkeypatch, tmp_path: Path) -> None:
    spec = PluginSpec(
        plugin_id="analysis_dummy",
        name="Dummy",
        version="0.1.0",
        type="analysis",
        entrypoint="plugin.py:Plugin",
        depends_on=[],
        settings={},
        path=tmp_path,
        capabilities=[],
        config_schema=tmp_path / "config.schema.json",
        output_schema=tmp_path / "output.schema.json",
        sandbox={},
    )
    request = {
        "run_id": "run_1",
        "run_seed": 1337,
        "plugin_seed": 1337,
        "started_at": "2026-02-16T00:00:00+00:00",
    }

    def _fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=-9, stdout="", stderr="killed")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    response = run_plugin_subprocess(spec, request, tmp_path, tmp_path)
    assert response.exit_code == -9
    assert response.result.status == "error"
    assert response.result.error is not None
    assert response.result.error.type == "ProcessTerminated"
    assert "signal 9" in response.result.error.message.lower()
