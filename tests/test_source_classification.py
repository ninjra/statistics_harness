from statistic_harness.core.utils import (
    SOURCE_CLASSIFICATION_REAL,
    SOURCE_CLASSIFICATION_SYNTHETIC,
    SOURCE_CLASSIFICATION_SYSTEM_DERIVED,
    infer_source_classification_from_filename,
    normalize_source_classification,
)


def test_normalize_source_classification_aliases() -> None:
    assert normalize_source_classification("systemderived") == SOURCE_CLASSIFICATION_SYSTEM_DERIVED
    assert normalize_source_classification("system-derived") == SOURCE_CLASSIFICATION_SYSTEM_DERIVED
    assert normalize_source_classification("synth") == SOURCE_CLASSIFICATION_SYNTHETIC
    assert normalize_source_classification("actual") == SOURCE_CLASSIFICATION_REAL


def test_infer_source_classification_from_filename() -> None:
    assert (
        infer_source_classification_from_filename("proc_log_synth_custom_issues_6mo_sheet_v2.xlsx")
        == SOURCE_CLASSIFICATION_SYNTHETIC
    )
    assert (
        infer_source_classification_from_filename("erp_systemderived_snapshot.csv")
        == SOURCE_CLASSIFICATION_SYSTEM_DERIVED
    )
    assert infer_source_classification_from_filename("baseline_real_data.csv") == SOURCE_CLASSIFICATION_REAL


def test_normalize_source_classification_defaults_to_real() -> None:
    assert normalize_source_classification(None, "plain_upload.csv") == SOURCE_CLASSIFICATION_REAL
