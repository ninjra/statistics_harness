from statistic_harness.core.process_filters import process_is_excluded


def test_process_filter_supports_substring_and_glob() -> None:
    assert process_is_excluded("QLOS_BATCH", ["los"])
    assert process_is_excluded("QLOS_BATCH", ["*los*"])
    assert process_is_excluded("qpec_job", ["qpec*"])
    assert not process_is_excluded("abc", ["qpec", "*los*"])
