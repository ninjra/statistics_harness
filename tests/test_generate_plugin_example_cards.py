from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_plugin_example_cards import generate_cards


def test_generate_plugin_example_cards_creates_index_and_cards(tmp_path: Path) -> None:
    matrix = {
        "run_id": "run_1",
        "classes": {
            "direct_action_generators": {
                "rationale": "Action plugins",
                "expected_output_type": "recommendation_items",
            },
            "supporting_signal_detectors": {
                "rationale": "Signal plugins",
                "expected_output_type": "findings_or_metrics",
            },
        },
        "plugins": [
            {
                "plugin_id": "p_action",
                "plugin_type": "analysis",
                "plugin_class": "direct_action_generators",
                "actionability_state": "actionable",
                "reason_code": "",
                "example": {
                    "kind": "ops",
                    "action_type": "batch_input",
                    "recommendation": "Convert to batch input",
                    "modeled_percent": 22.5,
                },
            },
            {
                "plugin_id": "p_signal",
                "plugin_type": "analysis",
                "plugin_class": "supporting_signal_detectors",
                "actionability_state": "explained_na",
                "reason_code": "NON_DECISION_PLUGIN",
                "example": {
                    "kind": "signal",
                    "plain_english_explanation": "Feeds downstream planner",
                },
            },
        ],
    }
    out_dir = tmp_path / "cards"
    index, _ = generate_cards(matrix, out_dir)
    assert int(index.get("card_count") or 0) == 2
    index_payload = json.loads((out_dir / "index.json").read_text(encoding="utf-8"))
    assert index_payload["card_count"] == 2
    assert (out_dir / "direct_action_generators.md").exists()
    assert (out_dir / "supporting_signal_detectors.md").exists()
