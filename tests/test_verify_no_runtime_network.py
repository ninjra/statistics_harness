from __future__ import annotations

import json
from pathlib import Path

import scripts.verify_no_runtime_network as mod


class _Spec:
    def __init__(self, plugin_id: str, plugin_type: str, no_network: bool) -> None:
        self.plugin_id = plugin_id
        self.type = plugin_type
        self.sandbox = {"no_network": no_network}


def test_verify_no_runtime_network_detects_violation(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        mod.PluginManager,
        "discover",
        lambda self: [_Spec("analysis_a", "analysis", False), _Spec("report_x", "report", True)],
    )
    payload = mod.verify_no_runtime_network(
        plugins_dir=tmp_path / "plugins",
        run_id=None,
        runs_root=tmp_path / "runs",
    )
    assert payload["ok"] is False
    assert payload["violation_count"] == 1


def test_verify_no_runtime_network_run_scoped(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        mod.PluginManager,
        "discover",
        lambda self: [_Spec("analysis_a", "analysis", True), _Spec("analysis_b", "analysis", False)],
    )
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"plugins": [{"plugin_id": "analysis_a"}]}),
        encoding="utf-8",
    )
    payload = mod.verify_no_runtime_network(
        plugins_dir=tmp_path / "plugins",
        run_id="r1",
        runs_root=tmp_path / "runs",
    )
    assert payload["ok"] is True


def test_verify_no_runtime_network_repo_manifests_are_hardened() -> None:
    payload = mod.verify_no_runtime_network(
        plugins_dir=Path("plugins"),
        run_id=None,
        runs_root=Path("appdata") / "runs",
    )
    assert payload["ok"] is True
