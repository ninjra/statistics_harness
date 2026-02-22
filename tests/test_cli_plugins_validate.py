from __future__ import annotations

import json

import pytest

import statistic_harness.cli as cli
from statistic_harness.cli import cmd_plugins_validate
from statistic_harness.core.plugin_manager import PluginManager

def test_cli_plugins_validate_profile_basic():
    # Keep this fast: validate a single known-good plugin.
    cmd_plugins_validate("profile_basic")


def test_cli_plugins_validate_uses_resolve_config(monkeypatch):
    called = {"value": False}
    original = PluginManager.resolve_config

    def wrapped(self, spec, config):
        called["value"] = True
        return original(self, spec, config)

    monkeypatch.setattr(PluginManager, "resolve_config", wrapped)
    cmd_plugins_validate("profile_basic")
    assert called["value"] is True


def test_cli_plugins_validate_writes_json_caps_report(tmp_path):
    out = tmp_path / "plugins_validate.json"
    cmd_plugins_validate("profile_basic", caps=True, json_path=str(out))
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["summary"]["total"] == 1
    assert payload["summary"]["passed"] == 1
    row = payload["plugins"][0]
    assert row["plugin_id"] == "profile_basic"
    assert row["status"] == "pass"
    assert "capabilities" in row
    assert "sandbox" in row


def test_cli_plugins_validate_isolated_unhealthy_fails(monkeypatch):
    monkeypatch.setattr(cli, "_run_isolated_health", lambda spec: {"status": "unhealthy"})
    with pytest.raises(SystemExit):
        cmd_plugins_validate("profile_basic", isolated=True)


def test_cli_plugins_validate_smoke_profile_basic():
    cmd_plugins_validate("profile_basic", smoke=True)
