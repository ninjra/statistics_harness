import asyncio
import importlib

from starlette.requests import Request


def test_template_results_filters(monkeypatch, tmp_path):
    monkeypatch.setenv("STAT_HARNESS_APPDATA", str(tmp_path / "appdata"))
    from statistic_harness.ui import server as server_mod

    server_mod = importlib.reload(server_mod)
    template_id = server_mod.pipeline.storage.create_template(
        name="FilterUI",
        fields=[{"name": "value", "dtype": "float"}],
        description=None,
        version=None,
        created_at=server_mod.now_iso(),
    )

    query = (
        b"project_ids=b,a&raw_format_ids=2,1&created_after=2026-01-01T00:00:00Z"
    )
    scope = {
        "type": "http",
        "method": "GET",
        "path": f"/templates/{template_id}/results",
        "query_string": query,
        "headers": [],
    }

    async def render_body() -> str:
        request = Request(scope)
        response = await server_mod.template_results(request, template_id)
        return response.template.render(response.context)

    body = asyncio.run(render_body())
    assert "Active filters" in body
    assert "project_ids" in body
    assert "raw_format_ids" in body
    assert "created_after" in body
    assert 'value="b,a"' in body
    assert 'value="2,1"' in body
    assert 'value="2026-01-01T00:00:00Z"' in body
