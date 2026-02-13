from statistic_harness.core.erp_inference import (
    infer_erp_type_from_field_names,
    normalize_field_name,
)


def test_normalize_field_name_uppercases() -> None:
    assert normalize_field_name(" process_id ") == "PROCESS_ID"


def test_infer_erp_type_quorum_signature() -> None:
    names = [
        "PROCESS_QUEUE_ID",
        "PROCESS_ID",
        "STATUS_CD",
        "LOCAL_MACHINE_ID",
        "QUEUE_DT",
        "START_DT",
        "END_DT",
    ]
    assert infer_erp_type_from_field_names(names) == "quorum"


def test_infer_erp_type_unknown_when_signature_missing() -> None:
    names = ["id", "status", "created_at", "updated_at"]
    assert infer_erp_type_from_field_names(names) == "unknown"
