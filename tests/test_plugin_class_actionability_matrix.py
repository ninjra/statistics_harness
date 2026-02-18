from __future__ import annotations

from pathlib import Path

from scripts.build_plugin_class_actionability_matrix import DEFAULT_TAXONOMY, build_matrix


def test_plugin_class_actionability_matrix_maps_all_plugins() -> None:
    payload = build_matrix("", Path(DEFAULT_TAXONOMY))
    assert int(payload.get("plugin_count") or 0) > 0
    classes = payload.get("classes") if isinstance(payload.get("classes"), dict) else {}
    assert classes
    plugins = payload.get("plugins") if isinstance(payload.get("plugins"), list) else []
    assert plugins
    for row in plugins:
        assert isinstance(row, dict)
        class_id = str(row.get("plugin_class") or "")
        assert class_id in classes
        assert str(row.get("actionability_state") or "") in {"actionable", "explained_na", "missing_output"}
        if str(row.get("actionability_state") or "") == "missing_output":
            assert str(row.get("reason_code") or "") in {
                "NOT_IN_RUN_SCOPE",
                "REPORT_SNAPSHOT_OMISSION",
            }
