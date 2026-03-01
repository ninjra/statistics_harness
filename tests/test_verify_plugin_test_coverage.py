from __future__ import annotations

import json
from pathlib import Path

import scripts.verify_plugin_test_coverage as mod


def _write_plugin_manifest(plugins_dir: Path, plugin_id: str) -> None:
    plugin_dir = plugins_dir / plugin_id
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.yaml").write_text("id: test\n", encoding="utf-8")


def test_verify_plugin_test_coverage_detects_uncovered(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "plugins"
    tests_root = tmp_path / "tests"
    tests_root.mkdir(parents=True, exist_ok=True)
    _write_plugin_manifest(plugins_dir, "plugin_a")
    _write_plugin_manifest(plugins_dir, "plugin_b")
    (tests_root / "test_refs.py").write_text(
        "PLUGIN_ID = 'plugin_a'\n",
        encoding="utf-8",
    )
    exemptions = tmp_path / "exemptions.json"
    exemptions.write_text("{}\n", encoding="utf-8")

    payload = mod.verify_plugin_test_coverage(
        plugins_dir=plugins_dir,
        tests_root=tests_root,
        exemptions_path=exemptions,
    )

    assert payload["ok"] is False
    assert payload["uncovered_unexempted_plugins"] == ["plugin_b"]


def test_verify_plugin_test_coverage_respects_exemptions(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "plugins"
    tests_root = tmp_path / "tests"
    tests_root.mkdir(parents=True, exist_ok=True)
    _write_plugin_manifest(plugins_dir, "plugin_a")
    _write_plugin_manifest(plugins_dir, "plugin_b")
    (tests_root / "test_refs.py").write_text(
        "PLUGIN_ID = 'plugin_a'\n",
        encoding="utf-8",
    )
    exemptions = tmp_path / "exemptions.json"
    exemptions.write_text(
        json.dumps({"exempt_plugin_ids": ["plugin_b"]}),
        encoding="utf-8",
    )

    payload = mod.verify_plugin_test_coverage(
        plugins_dir=plugins_dir,
        tests_root=tests_root,
        exemptions_path=exemptions,
    )

    assert payload["ok"] is True
    assert payload["uncovered_unexempted_plugins"] == []
    assert payload["uncovered_exempted_plugins"] == ["plugin_b"]


def test_verify_plugin_test_coverage_reports_stale_exemptions(tmp_path: Path) -> None:
    plugins_dir = tmp_path / "plugins"
    tests_root = tmp_path / "tests"
    tests_root.mkdir(parents=True, exist_ok=True)
    _write_plugin_manifest(plugins_dir, "plugin_a")
    (tests_root / "test_refs.py").write_text(
        "PLUGIN_ID = 'plugin_a'\n",
        encoding="utf-8",
    )
    exemptions = tmp_path / "exemptions.json"
    exemptions.write_text(
        json.dumps({"exempt_plugin_ids": ["plugin_stale"]}),
        encoding="utf-8",
    )

    payload = mod.verify_plugin_test_coverage(
        plugins_dir=plugins_dir,
        tests_root=tests_root,
        exemptions_path=exemptions,
    )

    assert payload["ok"] is True
    assert payload["stale_exemptions"] == ["plugin_stale"]
