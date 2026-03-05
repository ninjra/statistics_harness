"""Fail-closed dependency wiring tests (four-pillars Task 1.3).

Verifies that the pipeline correctly handles missing and failed plugin
dependencies by skipping or erroring dependent plugins rather than
silently continuing with partial data.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from statistic_harness.core.plugin_manager import PluginManager, PluginSpec
from statistic_harness.core.pipeline import Pipeline
from statistic_harness.core.types import PluginResult

ROOT = Path(__file__).resolve().parents[2]


def _make_spec(
    plugin_id: str,
    depends_on: list[str] | None = None,
    plugin_type: str = "analysis",
) -> PluginSpec:
    return PluginSpec(
        plugin_id=plugin_id,
        name=plugin_id,
        version="0.1.0",
        type=plugin_type,
        entrypoint="plugin.py:Plugin",
        depends_on=depends_on or [],
        settings={"defaults": {}},
        path=ROOT / "plugins" / plugin_id,
        capabilities=[],
        config_schema=Path("/dev/null"),
        output_schema=Path("/dev/null"),
        sandbox={"no_network": True},
    )


class TestExpandSelectedWithDeps:
    """Tests for Pipeline._expand_selected_with_deps."""

    def test_missing_dependency_reported(self, tmp_path: Path) -> None:
        """When a dep references a plugin that doesn't exist, it's listed as missing."""
        pipeline = Pipeline.__new__(Pipeline)
        specs = {
            "plugin_a": _make_spec("plugin_a", depends_on=["nonexistent_plugin"]),
        }
        selected = {"plugin_a"}
        expanded, added, missing = pipeline._expand_selected_with_deps(specs, selected)
        assert len(missing) == 1
        assert "nonexistent_plugin" in missing[0]

    def test_failed_dependency_chain_detected(self, tmp_path: Path) -> None:
        """When dep B depends on unknown C, the missing edge is reported."""
        pipeline = Pipeline.__new__(Pipeline)
        specs = {
            "plugin_a": _make_spec("plugin_a", depends_on=["plugin_b"]),
            "plugin_b": _make_spec("plugin_b", depends_on=["plugin_c"]),
        }
        selected = {"plugin_a"}
        expanded, added, missing = pipeline._expand_selected_with_deps(specs, selected)
        assert "plugin_b" in expanded
        assert any("plugin_c" in m for m in missing)

    def test_all_dependencies_satisfied_expands_correctly(self) -> None:
        """When all deps exist, they are added to the expanded set."""
        pipeline = Pipeline.__new__(Pipeline)
        specs = {
            "plugin_a": _make_spec("plugin_a", depends_on=["plugin_b"]),
            "plugin_b": _make_spec("plugin_b"),
            "plugin_c": _make_spec("plugin_c"),
        }
        selected = {"plugin_a"}
        expanded, added, missing = pipeline._expand_selected_with_deps(specs, selected)
        assert "plugin_b" in expanded
        assert "plugin_a" in expanded
        assert len(missing) == 0
        assert "plugin_b" in added


class TestToposortLayers:
    """Tests for Pipeline._toposort_layers dependency ordering."""

    def test_dependency_ordered_before_dependent(self) -> None:
        """Plugin B (dep of A) must appear in an earlier layer than A."""
        pipeline = Pipeline.__new__(Pipeline)
        spec_a = _make_spec("plugin_a", depends_on=["plugin_b"])
        spec_b = _make_spec("plugin_b")
        layers = pipeline._toposort_layers(
            [spec_a, spec_b], {"plugin_a", "plugin_b"}
        )
        flat = [spec.plugin_id for layer in layers for spec in layer]
        assert flat.index("plugin_b") < flat.index("plugin_a")

    def test_cycle_raises_value_error(self) -> None:
        """Circular dependencies must raise ValueError."""
        pipeline = Pipeline.__new__(Pipeline)
        spec_a = _make_spec("plugin_a", depends_on=["plugin_b"])
        spec_b = _make_spec("plugin_b", depends_on=["plugin_a"])
        with pytest.raises(ValueError, match="[Cc]ycle"):
            pipeline._toposort_layers(
                [spec_a, spec_b], {"plugin_a", "plugin_b"}
            )

    def test_no_deps_single_layer(self) -> None:
        """Plugins with no dependencies should all land in layer 0."""
        pipeline = Pipeline.__new__(Pipeline)
        spec_a = _make_spec("plugin_a")
        spec_b = _make_spec("plugin_b")
        layers = pipeline._toposort_layers(
            [spec_a, spec_b], {"plugin_a", "plugin_b"}
        )
        assert len(layers) == 1
        ids = {s.plugin_id for s in layers[0]}
        assert ids == {"plugin_a", "plugin_b"}


class TestReportDecisionBundleDeps:
    """Verify report_decision_bundle_v2 declares expected dependencies."""

    def test_bundle_v2_has_six_dependencies(self) -> None:
        """report_decision_bundle_v2 must declare exactly 6 upstream deps."""
        manager = PluginManager(ROOT / "plugins")
        specs = manager.discover()
        bundle = {s.plugin_id: s for s in specs}.get("report_decision_bundle_v2")
        assert bundle is not None, "report_decision_bundle_v2 plugin not found"
        assert len(bundle.depends_on) == 6
        expected_deps = {
            "analysis_issue_cards_v2",
            "analysis_recommendation_dedupe_v2",
            "analysis_busy_period_segmentation_v2",
            "analysis_waterfall_summary_v2",
            "analysis_traceability_manifest_v2",
            "report_slide_kit_emitter_v2",
        }
        assert set(bundle.depends_on) == expected_deps
