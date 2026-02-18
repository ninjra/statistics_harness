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
    assert code == "CAPACITY_IMPACT_NOT_APPLICABLE"


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
