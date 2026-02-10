from __future__ import annotations

import pandas as pd

from statistic_harness.core import planner

from tests.conftest import make_context


def test_infer_dataset_features_addon_capabilities(run_dir):
    df = pd.DataFrame(
        {
            "case_id": ["c1", "c1", "c2", "c3", "c3", "c3"] * 40,
            "activity": ["a", "b", "a", "c", "a", "b"] * 40,
            "host": ["h1", "h2", "h1", "h2", "h1", "h2"] * 40,
            "x_coord": list(range(240)),
            "y_coord": list(range(240))[::-1],
            "epoch_ms": [1_700_000_000_000 + i for i in range(240)],
            "group": ["g1", "g2", "g1", "g2", "g1", "g2"] * 40,
            "message": ["this is a fairly long free-text message " + str(i) for i in range(240)],
            "duration_seconds": [float(i % 17) for i in range(240)],
            "created_at": pd.date_range("2026-01-01", periods=240, freq="min").astype(str).tolist(),
        }
    )
    ctx = make_context(run_dir, df, settings={})
    features = planner._infer_dataset_features(ctx.storage, ctx.dataset_version_id)
    assert features["has_timestamp"] is True
    assert features["has_eventlog"] is True
    assert features["has_host"] is True
    assert features["has_coords"] is True
    assert features["has_text"] is True
    assert features["has_groupable"] is True
    assert features["has_epoch"] is True
    assert features["has_point_id"] is True
