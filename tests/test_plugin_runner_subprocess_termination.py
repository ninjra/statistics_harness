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

    class _FakePopen:
        def __init__(self, *args, **kwargs) -> None:
            self.returncode = -9
            self.stdout = None
            self.stderr = None

        def communicate(self, timeout=None):
            return "", "killed"

    monkeypatch.setattr(subprocess, "Popen", _FakePopen)

    response = run_plugin_subprocess(spec, request, tmp_path, tmp_path)
    assert response.exit_code == -9
    assert response.result.status == "error"
    assert response.result.error is not None
    assert response.result.error.type == "ProcessTerminated"
    assert "signal 9" in response.result.error.message.lower()


def test_run_plugin_subprocess_timeout_detaches_if_child_stuck(monkeypatch, tmp_path: Path) -> None:
    spec = PluginSpec(
        plugin_id="report_bundle",
        name="Report",
        version="0.1.0",
        type="report",
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
        "run_id": "run_2",
        "run_seed": 1337,
        "plugin_seed": 1337,
        "started_at": "2026-02-16T00:00:00+00:00",
        "budget": {"time_limit_ms": 1},
    }

    class _Pipe:
        def close(self) -> None:
            return None

    class _FakeHungPopen:
        def __init__(self, *args, **kwargs) -> None:
            self.returncode = None
            self.stdout = _Pipe()
            self.stderr = _Pipe()
            self._calls = 0

        def communicate(self, timeout=None):
            self._calls += 1
            raise subprocess.TimeoutExpired(
                cmd="fake",
                timeout=float(timeout or 0),
                output="partial-out",
                stderr="partial-err",
            )

        def kill(self) -> None:
            return None

    monkeypatch.setattr(subprocess, "Popen", _FakeHungPopen)

    response = run_plugin_subprocess(spec, request, tmp_path, tmp_path)
    assert response.result.status == "error"
    assert response.result.error is not None
    assert response.result.error.type == "TimeoutError"
    assert "detached" in response.stderr.lower()
    assert response.exit_code == -1
