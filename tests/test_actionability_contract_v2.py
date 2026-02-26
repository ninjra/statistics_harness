from __future__ import annotations

import pytest

from statistic_harness.core.stat_plugins.actionability_contract import (
    ActionabilityContractError,
    validate_actionability_payload,
)
from statistic_harness.core.stat_plugins.actionability_envelope import non_actionable_envelope
from statistic_harness.core.stat_plugins.actionability_metrics import (
    build_window_metric,
    build_window_triplet,
)


def test_actionability_contract_accepts_valid_payload() -> None:
    windows = build_window_triplet(
        accounting_month=build_window_metric(1.2, 5.0, 48.0),
        close_static=build_window_metric(0.8, 3.0, 22.0),
        close_dynamic=build_window_metric(0.6, 2.0, 18.0),
    )
    payload = non_actionable_envelope(
        plugin_id="analysis_example_v1",
        reason_code="NO_STATISTICAL_SIGNAL",
        recommendation="No direct action yet; monitor signal.",
        windows=windows,
        downstream_dependencies=["report_bundle"],
    )
    validate_actionability_payload(payload)


def test_actionability_contract_rejects_missing_windows() -> None:
    payload = {
        "plugin_id": "analysis_example_v1",
        "status": "non_actionable",
        "reason_code": "NO_STATISTICAL_SIGNAL",
        "recommendation": "No direct action yet; monitor signal.",
        "downstream_dependencies": [],
    }
    with pytest.raises(ActionabilityContractError):
        validate_actionability_payload(payload)

