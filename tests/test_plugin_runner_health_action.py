from __future__ import annotations

from pathlib import Path

import pytest

from statistic_harness.core.plugin_runner import _run_request


def test_plugin_runner_health_action_profile_basic() -> None:
    request = {
        "action": "health",
        "plugin_id": "profile_basic",
        "plugin_type": "profile",
        "entrypoint": "plugin.py:Plugin",
        "run_id": "validate",
        "run_dir": str(Path(".").resolve()),
        "run_seed": 0,
        "plugin_seed": 0,
        "root_dir": str(Path(".").resolve()),
        "sandbox": {"no_network": True},
    }
    response = _run_request(request)
    assert response["result"]["status"] == "ok"
    health = response["result"]["metrics"].get("health") or {}
    assert str(health.get("status") or "").lower() in {"ok", "healthy"}


def test_plugin_runner_rejects_unknown_action() -> None:
    with pytest.raises(ValueError):
        _run_request({"action": "unsupported"})
