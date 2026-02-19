from __future__ import annotations

from statistic_harness.core.actionability_explanations import derive_reason_code


def test_reason_code_capacity_impact_not_applicable() -> None:
    code = derive_reason_code(
        status="ok",
        finding_count=1,
        blank_kind_count=0,
        debug={},
        findings=[
            {
                "kind": "close_cycle_capacity_impact",
                "decision": "not_applicable",
            }
        ],
    )
    assert code == "CAPACITY_IMPACT_CONSTRAINT"


def test_reason_code_capacity_model_no_gain() -> None:
    code = derive_reason_code(
        status="ok",
        finding_count=1,
        blank_kind_count=0,
        debug={},
        findings=[
            {
                "kind": "close_cycle_capacity_model",
                "decision": "modeled",
                "baseline_value": 10.0,
                "modeled_value": 10.0,
            }
        ],
    )
    assert code == "NO_MODELED_CAPACITY_GAIN"


def test_reason_code_revenue_compression_no_pressure() -> None:
    code = derive_reason_code(
        status="ok",
        finding_count=1,
        blank_kind_count=0,
        debug={},
        findings=[
            {
                "kind": "close_cycle_revenue_compression",
                "decision": "modeled",
                "baseline_value": 2.0,
                "modeled_value": 7.0,
            }
        ],
    )
    assert code == "NO_REVENUE_COMPRESSION_PRESSURE"


def test_reason_code_na_is_prerequisite_unmet() -> None:
    code = derive_reason_code(
        status="na",
        finding_count=0,
        blank_kind_count=0,
        debug={},
        findings=[],
    )
    assert code == "PREREQUISITE_UNMET"


def test_reason_code_default_is_adapter_rule_missing() -> None:
    code = derive_reason_code(
        status="ok",
        finding_count=2,
        blank_kind_count=0,
        debug={},
        findings=[{"kind": "some_new_kind"}, {"kind": "another_kind"}],
    )
    assert code == "ADAPTER_RULE_MISSING"


def test_reason_code_plugin_observation_is_no_decision_signal() -> None:
    code = derive_reason_code(
        status="ok",
        finding_count=1,
        blank_kind_count=0,
        debug={},
        findings=[{"kind": "plugin_observation"}],
    )
    assert code == "NO_DECISION_SIGNAL"


def test_reason_code_plugin_not_applicable_is_no_decision_signal() -> None:
    code = derive_reason_code(
        status="ok",
        finding_count=1,
        blank_kind_count=0,
        debug={},
        findings=[{"kind": "plugin_not_applicable"}],
    )
    assert code == "NO_DECISION_SIGNAL"
