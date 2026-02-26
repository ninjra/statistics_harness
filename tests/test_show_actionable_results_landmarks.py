from __future__ import annotations

from scripts.show_actionable_results import (
    _KONA_QEMAIL_TITLE,
    _KONA_QPEC_TITLE,
    _augment_known_issue_landmarks,
)


def _landmark(items: list[dict[str, object]], title: str) -> dict[str, object]:
    for item in items:
        if str(item.get("title") or "").strip() == title:
            return item
    raise AssertionError(f"missing landmark: {title}")


def test_qemail_landmark_uses_equivalent_process_when_literal_missing() -> None:
    report = {
        "plugins": {
            "analysis_ideaspace_action_planner": {
                "findings": [
                    {
                        "kind": "ideaspace_action",
                        "action_type": "tune_schedule",
                        "process_norm": "ledger_post",
                        "modeled_close_percent": 34.5,
                        "modeled_close_hours": 12.0,
                    }
                ]
            }
        }
    }
    discovery_items = [
        {
            "plugin_id": "analysis_ideaspace_action_planner",
            "action_type": "tune_schedule",
            "where": {"process_norm": "ledger_post"},
            "modeled_close_percent": 34.5,
            "modeled_close_hours": 12.0,
            "scope_class": "close_specific",
            "recommendation": "Shift workload timing to reduce close-cycle contention.",
        }
    ]

    items = _augment_known_issue_landmarks(report, [], discovery_items)
    qemail = _landmark(items, _KONA_QEMAIL_TITLE)

    assert qemail.get("status") == "confirmed"
    assert int(qemail.get("observed_count") or 0) == 1
    where = qemail.get("where")
    assert isinstance(where, dict)
    assert str(where.get("process") or "").lower() == "ledger_post"


def test_qpec_landmark_confirms_generic_add_server_when_qpec_literal_missing() -> None:
    report = {
        "plugins": {
            "analysis_capacity_scaling": {
                "findings": [
                    {
                        "kind": "capacity_scaling",
                        "action_type": "add_server",
                        "process_norm": "ledger_post",
                        "modeled_general_percent": 22.8,
                    }
                ]
            }
        }
    }
    discovery_items = [
        {
            "plugin_id": "analysis_capacity_scaling",
            "action_type": "add_server",
            "where": {"process_norm": "ledger_post"},
            "modeled_general_percent": 22.8,
            "scope_class": "general",
            "recommendation": "Add one worker/server for ledger_post.",
        }
    ]

    items = _augment_known_issue_landmarks(report, [], discovery_items)
    qpec = _landmark(items, _KONA_QPEC_TITLE)

    assert qpec.get("status") == "confirmed"
    assert int(qpec.get("observed_count") or 0) >= 1
    assert str(qpec.get("plugin_id") or "") == "analysis_capacity_scaling"
