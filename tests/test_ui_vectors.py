import asyncio
import importlib

from starlette.requests import Request


def test_vectors_page_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("STAT_HARNESS_APPDATA", str(tmp_path / "appdata"))
    monkeypatch.delenv("STAT_HARNESS_ENABLE_VECTOR_STORE", raising=False)
    from statistic_harness.ui import server as server_mod

    server_mod = importlib.reload(server_mod)

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/vectors",
        "query_string": b"",
        "headers": [],
    }

    async def render_body() -> str:
        request = Request(scope)
        response = await server_mod.vectors_view(request)
        return response.template.render(response.context)

    body = asyncio.run(render_body())
    assert "sqlite-vec extension unavailable" in body


def test_vectors_page_prefill(monkeypatch, tmp_path):
    monkeypatch.setenv("STAT_HARNESS_APPDATA", str(tmp_path / "appdata"))
    monkeypatch.delenv("STAT_HARNESS_ENABLE_VECTOR_STORE", raising=False)
    from statistic_harness.ui import server as server_mod

    server_mod = importlib.reload(server_mod)

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/vectors",
        "query_string": b"collection=report_run&text=queue%20delay",
        "headers": [],
    }

    async def render_body() -> str:
        request = Request(scope)
        response = await server_mod.vectors_view(request, collection="report_run", text="queue delay")
        return response.template.render(response.context)

    body = asyncio.run(render_body())
    assert "queue delay" in body
