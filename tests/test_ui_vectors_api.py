import asyncio
import importlib

from starlette.requests import Request


def test_vectors_api_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("STAT_HARNESS_APPDATA", str(tmp_path / "appdata"))
    monkeypatch.delenv("STAT_HARNESS_ENABLE_VECTOR_STORE", raising=False)
    from statistic_harness.ui import server as server_mod

    server_mod = importlib.reload(server_mod)

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/vectors/collections",
        "query_string": b"",
        "headers": [],
    }

    async def call_api() -> int:
        request = Request(scope)
        try:
            await server_mod.vector_collections_api(request)
        except Exception as exc:
            return getattr(exc, "status_code", 0)
        return 200

    status = asyncio.run(call_api())
    assert status == 400
