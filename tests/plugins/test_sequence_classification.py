import pandas as pd

from plugins.analysis_sequence_classification.plugin import Plugin
from tests.conftest import make_context


def test_sequence_classification_detects_dependency_rows(run_dir):
    df = pd.DataFrame(
        {
            "process": ["alpha", "alpha", "beta", "beta"],
            "dep_process_queue_id": [None, "123", None, "999"],
        }
    )
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "ok"
    findings = [f for f in result.findings if f.get("kind") == "sequence_classification"]
    assert findings
    assert any(f.get("process") == "alpha" for f in findings)
