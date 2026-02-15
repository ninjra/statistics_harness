from __future__ import annotations

from statistic_harness.core.large_dataset_policy import as_budget_dict, caps_for


def test_caps_for_below_threshold_returns_none() -> None:
    caps = caps_for(
        plugin_id="analysis_example_v1",
        plugin_type="analysis",
        row_count=999_999,
        column_count=10,
    )
    assert caps is None


def test_caps_for_large_dataset_sets_batch_size() -> None:
    caps = caps_for(
        plugin_id="analysis_example_v1",
        plugin_type="analysis",
        row_count=1_000_000,
        column_count=10,
    )
    assert caps is not None
    assert caps.batch_size == 100_000


def test_caps_for_wide_dataset_reduces_batch_size() -> None:
    caps = caps_for(
        plugin_id="analysis_example_v1",
        plugin_type="analysis",
        row_count=2_000_000,
        column_count=250,
    )
    assert caps is not None
    assert caps.batch_size == 25_000


def test_budget_dict_preserves_no_sampling_contract() -> None:
    caps = caps_for(
        plugin_id="analysis_example_v1",
        plugin_type="analysis",
        row_count=2_000_000,
        column_count=20,
    )
    assert caps is not None
    budget = as_budget_dict(caps)
    assert budget["sampled"] is False
    assert budget["row_limit"] is None
