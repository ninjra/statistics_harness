from __future__ import annotations

import pandas as pd

from tests.conftest import make_context


def test_top20_minhash_plugin_runs_after_normalization(tmp_path) -> None:
    # Build a tiny processor-log-like dataset with one process and a near-duplicate param sweep.
    base = ";".join([f"k{i}=1" for i in range(20)])
    rows = []
    for i in range(6):
        rows.append(
            {
                "PROCESS": "RPT_POR002",
                "START_TIME": f"2026-01-20T00:0{i}:00",
                "END_TIME": f"2026-01-20T00:1{i}:00",
                "PARAMS": f"{base};payout_id={1000+i}",
            }
        )
    df = pd.DataFrame(rows)

    ctx = make_context(tmp_path, df, settings={})

    from plugins.profile_basic.plugin import Plugin as ProfileBasic
    from plugins.profile_eventlog.plugin import Plugin as ProfileEventlog
    from plugins.transform_normalize_mixed.plugin import Plugin as Normalize
    from plugins.analysis_param_near_duplicate_minhash_v1.plugin import Plugin as Minhash

    ctx.settings = {}
    assert ProfileBasic().run(ctx).status == "ok"
    ctx.settings = {}
    assert ProfileEventlog().run(ctx).status == "ok"
    ctx.settings = {"chunk_size": 1000}
    assert Normalize().run(ctx).status in {"ok", "degraded"}

    ctx.settings = {
        "max_processes": 10,
        "max_entities_per_process": 1000,
        "min_cluster_size": 3,
        "lsh_threshold": 0.8,
        "num_perm": 64,
        "ignore_param_keys_regex": "",
    }
    res = Minhash().run(ctx)
    assert res.status == "ok"
    assert any(f.get("kind") == "actionable_ops_lever" for f in res.findings)

