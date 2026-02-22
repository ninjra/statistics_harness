from __future__ import annotations

import pandas as pd

from conftest import make_context


def test_top20_hawkes_and_dtw_plugins_run_after_normalization(tmp_path) -> None:
    rows = []
    for day in range(1, 10):
        for hour in range(0, 24):
            for rep in range(3):
                rows.append(
                    {
                        "PROCESS": "QEMAIL" if hour % 2 == 0 else "QPEC",
                        "START_TIME": f"2026-01-{day:02d}T{hour:02d}:{rep:02d}:00",
                        "END_TIME": f"2026-01-{day:02d}T{hour:02d}:{(rep+1):02d}:00",
                    }
                )
    df = pd.DataFrame(rows)
    ctx = make_context(tmp_path, df, settings={})

    from plugins.profile_basic.plugin import Plugin as ProfileBasic
    from plugins.profile_eventlog.plugin import Plugin as ProfileEventlog
    from plugins.transform_normalize_mixed.plugin import Plugin as Normalize
    from plugins.analysis_burst_modeling_hawkes_v1.plugin import Plugin as Hawkes
    from plugins.analysis_daily_pattern_alignment_dtw_v1.plugin import Plugin as Dtw

    ctx.settings = {}
    assert ProfileBasic().run(ctx).status == "ok"
    ctx.settings = {}
    assert ProfileEventlog().run(ctx).status == "ok"
    ctx.settings = {"chunk_size": 1000}
    assert Normalize().run(ctx).status in {"ok", "degraded"}

    ctx.settings = {"top_k": 5}
    hawkes_res = Hawkes().run(ctx)
    assert hawkes_res.status in {"ok", "skipped", "degraded"}
    assert hawkes_res.status != "error"

    ctx.settings = {"top_k": 5}
    dtw_res = Dtw().run(ctx)
    assert dtw_res.status in {"ok", "skipped", "degraded"}
    assert dtw_res.status != "error"

