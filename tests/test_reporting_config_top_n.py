from __future__ import annotations

from pathlib import Path

import statistic_harness.core.report as report_mod


def test_discovery_top_n_reads_reporting_config(monkeypatch, tmp_path: Path) -> None:
    cfg = tmp_path / "reporting.yaml"
    cfg.write_text("recommendation_top_n: 13\n", encoding="utf-8")
    monkeypatch.delenv("STAT_HARNESS_DISCOVERY_TOP_N", raising=False)
    monkeypatch.setattr(report_mod, "_REPORTING_CONFIG_PATH", cfg)
    monkeypatch.setattr(report_mod, "_REPORTING_CONFIG_CACHE", None)
    assert report_mod._discovery_top_n() == 13

