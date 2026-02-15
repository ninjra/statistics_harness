from __future__ import annotations

import socket

from statistic_harness.core import plugin_runner


def test_network_guard_blocks_socket_connect(monkeypatch):
    # Ensure the guard blocks regardless of env.
    monkeypatch.delenv("STAT_HARNESS_NETWORK_MODE", raising=False)
    monkeypatch.delenv("STAT_HARNESS_ALLOW_NETWORK", raising=False)

    orig_socket = socket.socket
    orig_create = socket.create_connection
    try:
        plugin_runner._install_network_guard()
        try:
            socket.create_connection(("example.com", 80), timeout=1)
        except RuntimeError as exc:
            assert "Network disabled" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("Expected network guard to raise RuntimeError")
    finally:
        # Restore for other tests.
        socket.socket = orig_socket  # type: ignore[assignment]
        socket.create_connection = orig_create  # type: ignore[assignment]


def test_network_allowed_flag_is_false_by_default(monkeypatch):
    monkeypatch.delenv("STAT_HARNESS_NETWORK_MODE", raising=False)
    monkeypatch.delenv("STAT_HARNESS_ALLOW_NETWORK", raising=False)
    assert plugin_runner._network_mode() == "off"
    assert plugin_runner._network_allowed() is False


def test_network_mode_legacy_allow_network_maps_to_on(monkeypatch):
    monkeypatch.delenv("STAT_HARNESS_NETWORK_MODE", raising=False)
    monkeypatch.setenv("STAT_HARNESS_ALLOW_NETWORK", "1")
    assert plugin_runner._network_mode() == "on"
    assert plugin_runner._network_allowed() is True


def test_network_mode_env_overrides_legacy_flag(monkeypatch):
    monkeypatch.setenv("STAT_HARNESS_NETWORK_MODE", "off")
    monkeypatch.setenv("STAT_HARNESS_ALLOW_NETWORK", "1")
    assert plugin_runner._network_mode() == "off"
    assert plugin_runner._network_allowed() is False


def test_network_guard_localhost_mode_allows_loopback_only(monkeypatch):
    monkeypatch.setenv("STAT_HARNESS_NETWORK_MODE", "localhost")
    monkeypatch.delenv("STAT_HARNESS_ALLOW_NETWORK", raising=False)

    orig_socket = socket.socket
    orig_create = socket.create_connection

    def fake_create_connection(*args, **kwargs):
        raise ConnectionRefusedError("simulated loopback failure")

    socket.create_connection = fake_create_connection  # type: ignore[assignment]
    try:
        plugin_runner._install_network_guard(mode="localhost")
        try:
            socket.create_connection(("127.0.0.1", 8000), timeout=0.01)
        except ConnectionRefusedError:
            pass
        else:  # pragma: no cover
            raise AssertionError("Expected passthrough loopback error")

        try:
            socket.create_connection(("8.8.8.8", 53), timeout=0.01)
        except RuntimeError as exc:
            assert "loopback only" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("Expected guard to block non-loopback destination")
    finally:
        socket.socket = orig_socket  # type: ignore[assignment]
        socket.create_connection = orig_create  # type: ignore[assignment]
