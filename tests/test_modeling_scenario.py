from __future__ import annotations

import pytest

from statistic_harness.core.modeling.scenario import ScenarioConfig


class TestScenarioConfig:
    def test_creation_minimal(self):
        cfg = ScenarioConfig(name="test")
        assert cfg.name == "test"
        assert cfg.description == ""
        assert cfg.ranking_weights is None
        assert cfg.close_cycles_per_year is None

    def test_creation_with_overrides(self):
        cfg = ScenarioConfig(
            name="full",
            description="A test scenario",
            close_cycles_per_year=24.0,
            max_obviousness=0.6,
            require_modeled_hours=True,
            discovery_top_n=10,
        )
        assert cfg.close_cycles_per_year == 24.0
        assert cfg.max_obviousness == 0.6
        assert cfg.require_modeled_hours is True
        assert cfg.discovery_top_n == 10

    def test_frozen_immutability(self):
        cfg = ScenarioConfig(name="frozen_test")
        with pytest.raises(AttributeError):
            cfg.name = "modified"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            cfg.close_cycles_per_year = 24.0  # type: ignore[misc]

    def test_with_override(self):
        original = ScenarioConfig(name="base", close_cycles_per_year=12.0)
        modified = original.with_override(close_cycles_per_year=24.0, max_obviousness=0.5)
        assert original.close_cycles_per_year == 12.0
        assert original.max_obviousness is None
        assert modified.close_cycles_per_year == 24.0
        assert modified.max_obviousness == 0.5
        assert modified.name == "base"

    def test_with_override_preserves_existing(self):
        original = ScenarioConfig(
            name="base",
            description="desc",
            require_modeled_hours=True,
        )
        modified = original.with_override(discovery_top_n=5)
        assert modified.description == "desc"
        assert modified.require_modeled_hours is True
        assert modified.discovery_top_n == 5

    def test_diff_from_defaults_empty(self):
        cfg = ScenarioConfig(name="baseline")
        diff = cfg.diff_from_defaults()
        assert diff == {}

    def test_diff_from_defaults_with_overrides(self):
        cfg = ScenarioConfig(
            name="test",
            description="ignored",
            close_cycles_per_year=24.0,
            max_obviousness=0.6,
            require_modeled_hours=True,
        )
        diff = cfg.diff_from_defaults()
        assert "name" not in diff
        assert "description" not in diff
        assert diff["close_cycles_per_year"] == 24.0
        assert diff["max_obviousness"] == 0.6
        assert diff["require_modeled_hours"] is True

    def test_frozenset_fields(self):
        cfg = ScenarioConfig(
            name="test",
            suppress_action_types=frozenset({"tune_threshold", "review"}),
            allow_action_types=frozenset({"batch_input"}),
        )
        assert "tune_threshold" in cfg.suppress_action_types
        assert "batch_input" in cfg.allow_action_types
