# Plugin Class Example: reporting_llm_post_processing

- class_rationale: Presentation/report/planner/LLM orchestration plugins.
- expected_output_type: `report_or_prompt_artifacts`
- run_id: `debug_fg_20260225T2315Z`

## Example

- plugin_id: `planner_basic`
- plugin_type: `planner`
- actionability_state: `explained_na`
- reason_code: `NON_DECISION_PLUGIN`
- finding_kind: `non_actionable_explanation`
- explanation: planner_basic is a non-decision plugin (report_or_prompt_artifacts) and is tracked as explained N/A when recommendation lanes are absent.

## Traceability

- class `reporting_llm_post_processing` -> plugin `planner_basic` -> run `debug_fg_20260225T2315Z`
- Source artifact: `docs/plugin_class_actionability_matrix.json`
