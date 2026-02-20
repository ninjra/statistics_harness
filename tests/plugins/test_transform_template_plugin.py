from __future__ import annotations

import pandas as pd

from plugins.transform_template.plugin import Plugin
from tests.conftest import make_context


def test_transform_template_missing_template_id_is_ok_passthrough(run_dir):
    df = pd.DataFrame({"x": [1, 2, 3]})
    ctx = make_context(run_dir, df, {})
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert "bypassed" in str(result.summary).lower()
    findings = result.findings if isinstance(result.findings, list) else []
    assert findings
    row = findings[0] if isinstance(findings[0], dict) else {}
    assert row.get("kind") == "template_mapping_passthrough"
    assert row.get("reason_code") == "TEMPLATE_ID_MISSING"
