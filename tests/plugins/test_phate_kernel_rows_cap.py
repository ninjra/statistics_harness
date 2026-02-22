from __future__ import annotations

import pandas as pd

from plugins.analysis_phate_trajectory_embedding_v1.plugin import Plugin
from conftest import make_context


def test_phate_plugin_caps_kernel_rows(tmp_path) -> None:
    rows = 2200
    df = pd.DataFrame(
        {
            "process": ["qpec" if i % 3 == 0 else "qemail" for i in range(rows)],
            "start_time": pd.date_range("2026-01-01", periods=rows, freq="min").astype(str),
            "value_a": [float(i % 17) for i in range(rows)],
            "value_b": [float((i * 3) % 19) for i in range(rows)],
            "value_c": [float((i * 5) % 23) for i in range(rows)],
        }
    )
    ctx = make_context(tmp_path, df, {"max_rows": 2200, "max_kernel_rows": 600}, run_seed=42)
    res = Plugin().run(ctx)
    assert res.status in {"ok", "degraded"}
    if res.status == "ok":
        assert int(res.metrics.get("rows_used_for_kernel", 0)) <= 600
