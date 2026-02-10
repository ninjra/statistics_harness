from __future__ import annotations

import socket

from statistic_harness.core import plugin_runner


def test_network_guard_blocks_socket_connect(monkeypatch):
    # Ensure the guard blocks regardless of env.
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
    monkeypatch.delenv("STAT_HARNESS_ALLOW_NETWORK", raising=False)
    assert plugin_runner._network_allowed() is False

