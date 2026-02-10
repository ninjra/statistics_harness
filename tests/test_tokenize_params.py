from __future__ import annotations

from statistic_harness.core.tokenize_params import parse_parameter_text, tokenize_kv_pairs


def test_parse_parameter_text_normalizes_keys() -> None:
    parsed = parse_parameter_text("A=1; B=2")
    assert parsed is not None
    canonical, kv = parsed
    assert canonical == "a=1;b=2"
    assert ("a", "1") in kv
    assert ("b", "2") in kv


def test_tokenize_kv_pairs_ignores_keys() -> None:
    tokens = tokenize_kv_pairs([("run_id", "123"), ("x", "y")], ignore_keys_regex="run_id")
    assert "x=y" in tokens
    assert not any(t.startswith("run_id") for t in tokens)

