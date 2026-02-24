from __future__ import annotations

from pathlib import Path

import scripts.run_release_gate as gate_mod


def test_release_gate_payload_without_run_id(monkeypatch, tmp_path: Path) -> None:
    seen_cmds: list[list[str]] = []

    def _fake_run_step(args: list[str]) -> dict[str, object]:
        seen_cmds.append(args)
        return {"cmd": args, "rc": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr(gate_mod, "_run_step", _fake_run_step)
    payload = gate_mod._release_gate_payload(
        run_id="",
        before_run_id="",
        known_issues_mode="any",
        state_db=str(tmp_path / "state.sqlite"),
        pytest_args="-q",
        top_n=20,
        out_dir=tmp_path,
    )
    assert payload["ok"] is True
    steps = payload["steps"]
    assert sorted(steps.keys()) == [
        "verify_docs_and_plugin_matrices",
        "verify_openplanter_pack_release_gate",
    ]
    assert len(seen_cmds) == 2


def test_release_gate_payload_with_run_id_tracks_bundle_failure(monkeypatch, tmp_path: Path) -> None:
    def _fake_run_step(args: list[str]) -> dict[str, object]:
        cmd_text = " ".join(args)
        rc = 1 if "build_post_run_bundle.py" in cmd_text else 0
        return {"cmd": args, "rc": rc, "stdout": "", "stderr": ""}

    monkeypatch.setattr(gate_mod, "_run_step", _fake_run_step)
    payload = gate_mod._release_gate_payload(
        run_id="run123",
        before_run_id="run122",
        known_issues_mode="on",
        state_db=str(tmp_path / "state.sqlite"),
        pytest_args="-q",
        top_n=10,
        out_dir=tmp_path,
    )
    assert payload["ok"] is False
    assert "build_post_run_bundle" in payload["steps"]
    assert int(payload["steps"]["build_post_run_bundle"]["rc"] or 0) == 1
    assert payload["files"]["post_run_bundle_json"].endswith("bundle_run123.json")
