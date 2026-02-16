from __future__ import annotations

from statistic_harness.core.top20_plugins import (
    TemplateInfo,
    _infer_role_from_template_field_name,
)


def test_infer_role_from_template_field_name() -> None:
    assert _infer_role_from_template_field_name("PROCESS_ID") == "process_name"
    assert _infer_role_from_template_field_name("START_DT") == "start_time"
    assert _infer_role_from_template_field_name("QUEUE_DT") == "queue_time"
    assert _infer_role_from_template_field_name("MASTER_PROCESS_QUEUE_ID") == "master_id"
    assert _infer_role_from_template_field_name("DEP_PROCESS_QUEUE_ID") == "dependency_id"
    assert _infer_role_from_template_field_name("ASSIGNED_MACHINE_ID") == "host_id"


def test_template_info_process_name_falls_back_to_process_id() -> None:
    info = TemplateInfo(
        table_name="tmpl",
        field_to_safe={},
        safe_to_role={"c26": "process_id"},
    )
    assert info.role_field("process_name") == "c26"
