from statistic_harness.core.process_filters import (
    normalize_process,
    process_is_excluded,
    process_matches_token,
)


def test_process_filter_supports_substring_and_glob() -> None:
    assert process_is_excluded("QLOS_BATCH", ["los"])
    assert process_is_excluded("QLOS_BATCH", ["*los*"])
    assert process_is_excluded("qpec_job", ["qpec*"])
    assert not process_is_excluded("abc", ["qpec", "*los*"])


def test_normalize_and_match_edge_cases() -> None:
    assert normalize_process(None) == ""
    assert normalize_process("  Proc_A  ") == "proc_a"
    assert process_matches_token("proc_a", "") is False
    assert process_matches_token("", "proc") is False
    assert process_matches_token("proc_a", "proc_a") is True
    assert process_matches_token("proc_a", "?roc_*") is True
