import pandas as pd

from statistic_harness.core.stat_plugins.columns import infer_columns
from statistic_harness.core.stat_plugins.config import merge_config
from statistic_harness.core.stat_plugins.redaction import build_redactor
from statistic_harness.core.stat_plugins.sampling import deterministic_sample


def test_merge_config_defaults():
    config = merge_config(None)
    assert config["seed"] == 1337
    assert config["max_rows"] is None
    assert config["time_budget_ms"] is None
    assert config["allow_row_sampling"] is False
    assert "privacy" in config
    assert "focus" in config


def test_deterministic_sample_reproducible():
    df = pd.DataFrame({"a": range(100), "b": range(100, 200)})
    sample1, meta1 = deterministic_sample(df, max_rows=20, seed=42)
    sample2, meta2 = deterministic_sample(df, max_rows=20, seed=42)
    assert sample1.index.tolist() == sample2.index.tolist()
    assert meta1["sampled"] is True
    assert meta2["rows_used"] == 20


def test_redaction_toggle():
    text = "user@example.com"
    redactor = build_redactor({"enable_redaction": True, "redact_patterns": ["email"]})
    assert redactor(text) != text
    passthrough = build_redactor({"enable_redaction": False})
    assert passthrough(text) == text


def test_infer_columns_basic():
    df = pd.DataFrame(
        {
            "start_time": pd.date_range("2024-01-01", periods=10, freq="D"),
            "category": ["a", "b"] * 5,
            "value": range(10),
        }
    )
    config = merge_config(None)
    inferred = infer_columns(df, config)
    assert inferred["time_column"] == "start_time"
    assert "category" in inferred["group_by"]
    assert "value" in inferred["value_columns"]
