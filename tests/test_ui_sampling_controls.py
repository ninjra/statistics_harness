from __future__ import annotations

from pathlib import Path

from statistic_harness.ui.server import _apply_sampling_setting


def test_apply_sampling_setting_none_keeps_settings() -> None:
    settings: dict[str, object] = {"existing": True}
    _apply_sampling_setting(settings, None)
    assert settings == {"existing": True}


def test_apply_sampling_setting_sets_boolean_from_form() -> None:
    settings: dict[str, object] = {}
    _apply_sampling_setting(settings, "1")
    assert settings["allow_row_sampling"] is True
    _apply_sampling_setting(settings, "0")
    assert settings["allow_row_sampling"] is False


def test_project_template_has_sampling_control_and_form_field() -> None:
    root = Path(__file__).resolve().parents[1]
    template = (root / "src" / "statistic_harness" / "ui" / "templates" / "project.html").read_text(
        encoding="utf-8"
    )
    assert 'id="allow-row-sampling"' in template
    assert 'formData.append("allow_row_sampling"' in template
