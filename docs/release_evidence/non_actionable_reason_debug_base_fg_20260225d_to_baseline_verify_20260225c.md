# Actionability Burndown

- run_id: baseline_verify_20260225c
- unresolved_count: 4
- before_run_id: debug_base_fg_20260225d

## By Lane
- report_snapshot_serialization: 4 (llm_prompt_builder, llm_text2sql_local_generate_v1, report_bundle, report_plain_english_v1)

## By Reason
- REPORT_SNAPSHOT_OMISSION: 4 (llm_prompt_builder, llm_text2sql_local_generate_v1, report_bundle, report_plain_english_v1)

## Unresolved Plugins

| plugin_id | lane | reason | finding_count | next_step |
| --- | --- | --- | ---: | --- |
| llm_prompt_builder | report_snapshot_serialization | REPORT_SNAPSHOT_OMISSION | 0 | Plugin executed but is missing from report.plugins snapshot; include it in report serialization. |
| llm_text2sql_local_generate_v1 | report_snapshot_serialization | REPORT_SNAPSHOT_OMISSION | 0 | Plugin executed but is missing from report.plugins snapshot; include it in report serialization. |
| report_bundle | report_snapshot_serialization | REPORT_SNAPSHOT_OMISSION | 0 | Plugin executed but is missing from report.plugins snapshot; include it in report serialization. |
| report_plain_english_v1 | report_snapshot_serialization | REPORT_SNAPSHOT_OMISSION | 0 | Plugin executed but is missing from report.plugins snapshot; include it in report serialization. |

## Delta Vs Before

- unresolved_count_before: 0
- unresolved_count_after: 4
- unresolved_count_delta: 4

### Reason Delta
- REPORT_SNAPSHOT_OMISSION: before=0 after=4 delta=4

### Lane Delta
- report_snapshot_serialization: before=0 after=4 delta=4
