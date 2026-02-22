from __future__ import annotations

import pandas as pd

from conftest import make_context


def test_top20_simhash_plugin_runs_after_normalization(tmp_path) -> None:
    base = ";".join([f"k{i}=1" for i in range(20)])
    rows = []
    for i in range(10):
        rows.append(
            {
                "PROCESS": "RPT_POR002",
                "START_TIME": f"2026-01-20T00:{i:02d}:00",
                "END_TIME": f"2026-01-20T00:{(i + 1):02d}:00",
                "PARAMS": f"{base};payout_id={1000 + i}",
            }
        )
    df = pd.DataFrame(rows)
    ctx = make_context(tmp_path, df, settings={})

    from plugins.profile_basic.plugin import Plugin as ProfileBasic
    from plugins.profile_eventlog.plugin import Plugin as ProfileEventlog
    from plugins.transform_normalize_mixed.plugin import Plugin as Normalize
    from plugins.analysis_param_near_duplicate_simhash_v1.plugin import Plugin as Simhash

    ctx.settings = {}
    assert ProfileBasic().run(ctx).status == "ok"
    ctx.settings = {}
    assert ProfileEventlog().run(ctx).status == "ok"
    ctx.settings = {"chunk_size": 1000}
    assert Normalize().run(ctx).status in {"ok", "degraded"}

    ctx.settings = {
        "max_processes": 10,
        "max_entities_per_process": 1000,
        "max_entities_for_similarity": 200,
        "min_cluster_size": 3,
        "max_hamming_distance": 8,
        "max_pair_checks": 20000,
        "max_pair_checks_total": 50000,
        "ignore_param_keys_regex": "",
    }
    res = Simhash().run(ctx)
    assert res.status == "ok"
    assert isinstance(res.metrics, dict)
