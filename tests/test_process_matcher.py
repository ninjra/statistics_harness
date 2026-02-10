from __future__ import annotations

import os

from statistic_harness.core.process_matcher import (
    compile_patterns,
    default_exclude_process_patterns,
    parse_exclude_patterns_env,
)


def test_compile_patterns_exact_glob_sql_like_and_regex() -> None:
    pred = compile_patterns(["POSTWKFL", "LOS*", "JEPOST%", "re:^abc[0-9]+$"])
    assert pred("postwkfl")
    assert pred("POSTWKFL")
    assert pred("losloadcld")
    assert pred("JEPOST123")
    assert pred("abC123")
    assert not pred("other")


def test_parse_exclude_patterns_env_splits() -> None:
    os.environ["STAT_HARNESS_EXCLUDE_PROCESSES"] = "A;B, C  D"
    try:
        patterns = parse_exclude_patterns_env()
    finally:
        os.environ.pop("STAT_HARNESS_EXCLUDE_PROCESSES", None)
    assert patterns == ["A", "B", "C", "D"]


def test_default_patterns_present() -> None:
    pats = default_exclude_process_patterns()
    assert "LOS*" in pats
    assert "BKRV*" in pats
