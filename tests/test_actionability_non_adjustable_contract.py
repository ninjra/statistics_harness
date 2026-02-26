from __future__ import annotations

from statistic_harness.core.actionability_explanations import NON_ADJUSTABLE_PROCESSES


def test_qemail_is_not_non_adjustable() -> None:
    assert "qemail" not in NON_ADJUSTABLE_PROCESSES


def test_child_chain_examples_remain_non_adjustable() -> None:
    assert "jboachild" in NON_ADJUSTABLE_PROCESSES
    assert "jbcreateje" in NON_ADJUSTABLE_PROCESSES
