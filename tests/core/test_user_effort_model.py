from __future__ import annotations

from statistic_harness.core.user_effort_model import effort_score, modeled_touch_reduction


def test_modeled_touch_reduction() -> None:
    assert modeled_touch_reduction(run_count=100, top_user_share=0.5) == 50.0
    assert modeled_touch_reduction(run_count=0, top_user_share=0.5) is None


def test_effort_score_non_negative() -> None:
    assert effort_score(close_hours_saved=4.0, touches_reduced=9.0) > 0.0
    assert effort_score(close_hours_saved=-1.0, touches_reduced=-1.0) == 0.0

