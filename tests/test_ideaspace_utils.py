from __future__ import annotations

import re

import pandas as pd

from statistic_harness.core.stat_plugins.ideaspace import (
    apply_budget_rows,
    pick_groupable_columns,
    redact_value,
    rng_for_run,
)


def test_rng_for_run_is_deterministic():
    a = rng_for_run(123, "x").integers(0, 10, size=20).tolist()
    b = rng_for_run(123, "x").integers(0, 10, size=20).tolist()
    c = rng_for_run(123, "y").integers(0, 10, size=20).tolist()
    assert a == b
    assert a != c


def test_redact_value_masks_email_and_uuid():
    privacy = {"enable_redaction": True, "redact_patterns": ["email", "uuid"]}
    raw = "user@example.com 123e4567-e89b-12d3-a456-426614174000"
    out = redact_value(raw, privacy)
    assert "[REDACTED]" in out
    assert "user@example.com" not in out
    assert re.search(r"[0-9a-f]{8}-", out, re.I) is None


def test_pick_groupable_columns_respects_k_min_and_cardinality():
    df = pd.DataFrame(
        {
            "ok": ["a"] * 100 + ["b"] * 100,
            "too_many": [f"v{i}" for i in range(200)],
            "too_small": ["x"] * 199 + ["y"],
        }
    )
    cols = pick_groupable_columns(
        df, ["ok", "too_many", "too_small"], k_min=5, max_cardinality=50, max_columns=5
    )
    assert cols == ["ok"]


def test_apply_budget_rows_caps_rows_deterministically():
    df = pd.DataFrame({"x": list(range(500))})
    sampled_a, meta_a = apply_budget_rows(df, 100, seed=123)
    sampled_b, meta_b = apply_budget_rows(df, 100, seed=123)
    assert len(sampled_a) == 100
    assert sampled_a["x"].tolist() == sampled_b["x"].tolist()
    assert meta_a["rows_used"] == meta_b["rows_used"]
