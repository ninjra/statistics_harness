import os
import uuid

import pytest

from statistic_harness.core.utils import atomic_write_text, read_json, write_json


def test_atomic_write_text_replace_failure_leaves_original(tmp_path, monkeypatch):
    path = tmp_path / "note.txt"
    path.write_text("old", encoding="utf-8")

    class DummyUUID:
        hex = "deadbeef"

    monkeypatch.setattr(uuid, "uuid4", lambda: DummyUUID())

    def boom(src, dst):  # noqa: ARG001 - signature matches os.replace
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", boom)

    with pytest.raises(OSError):
        atomic_write_text(path, "new")

    assert path.read_text(encoding="utf-8") == "old"
    assert not (tmp_path / ".note.txt.tmp.deadbeef").exists()


def test_write_json_roundtrip(tmp_path):
    path = tmp_path / "payload.json"
    write_json(path, {"a": 1, "b": {"c": 2}})
    assert read_json(path) == {"a": 1, "b": {"c": 2}}

