import json

from plugins.llm_prompt_builder.plugin import Plugin
import pandas as pd

from tests.conftest import make_context


def test_llm_prompt_builder(run_dir):
    pii_value = "user@example.com"
    report = {
        "run_id": "test",
        "created_at": "now",
        "status": "completed",
        "input": {"filename": "x", "rows": 1, "cols": 1, "inferred_types": {}},
        "lineage": {
            "run": {
                "run_id": "test",
                "created_at": "now",
                "status": "completed",
                "run_seed": 0,
            },
            "input": {
                "upload_id": "",
                "filename": "x",
                "canonical_path": "",
                "input_hash": "",
                "sha256": "",
                "size_bytes": 0,
            },
            "dataset": {"dataset_version_id": "dv"},
            "raw_format": None,
            "template": None,
            "plugins": {},
        },
        "plugins": {
            "profile_basic": {
                "status": "ok",
                "summary": "Profiled",
                "metrics": {},
                "findings": [{"kind": "pii_sample", "value": pii_value}],
                "artifacts": [],
                "budget": {
                    "row_limit": None,
                    "sampled": False,
                    "time_limit_ms": None,
                    "cpu_limit_ms": None,
                },
                "error": None,
            }
        },
    }
    (run_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
    df = pd.DataFrame({"a": [1]})
    ctx = make_context(run_dir, df, {})
    ctx.storage.upsert_pii_entities(ctx.tenant_id or "default", "email", [pii_value])
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert any(a.path.endswith("prompt.md") for a in result.artifacts)
    prompt_path = next(
        a.path for a in result.artifacts if a.path.endswith("prompt.md")
    )
    prompt_text = (run_dir / prompt_path).read_text(encoding="utf-8")
    assert pii_value not in prompt_text
    assert "pii:email:" in prompt_text
