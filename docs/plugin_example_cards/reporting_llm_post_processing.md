# Plugin Class Example: reporting_llm_post_processing

- class_rationale: Presentation/report/planner/LLM orchestration plugins.
- expected_output_type: `report_or_prompt_artifacts`
- run_id: `full_loaded_3246cc7c_20260218T193803Z`

## Example

- plugin_id: `planner_basic`
- plugin_type: `planner`
- actionability_state: `explained_na`
- reason_code: `NOT_APPLICABLE`
- finding_kind: `non_actionable_explanation`
- explanation: planner_basic produced 110 finding(s), but none matched current action-routing rules. This is a planner plugin with no downstream plugin dependencies in this run.

## Traceability

- class `reporting_llm_post_processing` -> plugin `planner_basic` -> run `full_loaded_3246cc7c_20260218T193803Z`
- Source artifact: `docs/plugin_class_actionability_matrix.json`
