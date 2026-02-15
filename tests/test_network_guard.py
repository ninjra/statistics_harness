from __future__ import annotations

import socket

from statistic_harness.core import plugin_runner


def test_network_guard_blocks_socket_create_connection(monkeypatch) -> None:
    monkeypatch.setenv("STAT_HARNESS_NETWORK_MODE", "off")
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
            raise AssertionError("Expected network guard to block create_connection")
    finally:
        socket.socket = orig_socket  # type: ignore[assignment]
        socket.create_connection = orig_create  # type: ignore[assignment]
