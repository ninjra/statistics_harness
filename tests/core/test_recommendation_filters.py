from __future__ import annotations

from statistic_harness.core.recommendation_filters import (
    is_flow_rewire_action,
    is_specific_process_target,
    process_is_adjustable,
)


def test_specific_process_target() -> None:
    assert is_specific_process_target("rpt_por002") is True
    assert is_specific_process_target("(multiple)") is False


def test_flow_rewire_action() -> None:
    assert is_flow_rewire_action("route_process") is True
    assert is_flow_rewire_action("batch_input") is False


def test_process_is_adjustable() -> None:
    assert process_is_adjustable("rpt_por002", {"qemail"}) is True
    assert process_is_adjustable("qemail", {"qemail"}) is False

