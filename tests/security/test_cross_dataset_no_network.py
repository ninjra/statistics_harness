from __future__ import annotations

import socket

from plugins.transform_entity_resolution_map_v1.plugin import Plugin
from tests.plugins.openplanter_pack_test_utils import make_openplanter_context


def test_cross_dataset_plugins_do_not_attempt_network(run_dir, monkeypatch) -> None:
    calls: list[tuple[object, ...]] = []

    def _block_connect(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(tuple(args))
        raise AssertionError("network call attempted")

    monkeypatch.setattr(socket.socket, "connect", _block_connect, raising=True)
    ctx = make_openplanter_context(run_dir)
    ctx.settings = {}
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert calls == []
