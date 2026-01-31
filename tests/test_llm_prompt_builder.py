import json

from plugins.llm_prompt_builder.plugin import Plugin
import pandas as pd

from tests.conftest import make_context


def test_llm_prompt_builder(run_dir):
    report = {"run_id": "test", "created_at": "now", "status": "completed", "input": {"filename": "x", "rows": 1, "cols": 1, "inferred_types": {}}, "plugins": {}}
    (run_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
    df = pd.DataFrame({"a": [1]})
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert any(a.path.endswith("prompt.md") for a in result.artifacts)
