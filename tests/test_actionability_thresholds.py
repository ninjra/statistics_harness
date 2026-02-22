from __future__ import annotations

from pathlib import Path

from statistic_harness.core.actionability_thresholds import (
    load_actionability_thresholds,
    meets_actionability_thresholds,
)


def test_load_actionability_thresholds_defaults() -> None:
    thresholds = load_actionability_thresholds()
    assert thresholds.delta_hours_accounting_month > 0.0
    assert thresholds.eff_pct_accounting_month > 0.0
    assert thresholds.eff_idx_accounting_month > 0.0
    assert thresholds.confidence > 0.0
    assert thresholds.fallback_status == "na"


def test_meets_actionability_thresholds_respects_yaml_config(tmp_path: Path) -> None:
    cfg = tmp_path / "thresholds.yaml"
    cfg.write_text(
        "\n".join(
            [
                "version: 1",
                "minimums:",
                "  delta_hours_accounting_month: 1.5",
                "  eff_pct_accounting_month: 20.0",
                "  eff_idx_accounting_month: 2.0",
                "  confidence: 0.9",
                "scoring:",
                "  require_all: true",
            ]
        ),
        encoding="utf-8",
    )
    thresholds = load_actionability_thresholds(cfg)
    assert not meets_actionability_thresholds(1.0, 21.0, 2.1, 0.95, thresholds=thresholds)
    assert meets_actionability_thresholds(1.6, 20.1, 2.1, 0.95, thresholds=thresholds)
