from __future__ import annotations

import pytest

from statistic_harness.core.utils import resolve_env_placeholders


def test_resolve_env_placeholders_nested(monkeypatch) -> None:
    monkeypatch.setenv("SECRET_TOKEN", "s3cr3t")
    payload = {
        "a": "${ENV:SECRET_TOKEN}",
        "b": ["x", "${ENV:SECRET_TOKEN}"],
        "c": {"inner": "${ENV:SECRET_TOKEN}"},
    }
    resolved = resolve_env_placeholders(payload)
    assert resolved["a"] == "s3cr3t"
    assert resolved["b"][1] == "s3cr3t"
    assert resolved["c"]["inner"] == "s3cr3t"


def test_resolve_env_placeholders_missing_raises(monkeypatch) -> None:
    monkeypatch.delenv("MISSING_SECRET", raising=False)
    with pytest.raises(ValueError):
        resolve_env_placeholders({"a": "${ENV:MISSING_SECRET}"})

