# Plugin Class Example: synthesis_verification

- class_rationale: Plugins that aggregate, verify, dedupe, or validate outputs from other plugins.
- expected_output_type: `derived_findings_or_validation`
- run_id: `full_loaded_3246cc7c_20260218T193803Z`

## Example

- plugin_id: `analysis_issue_cards_v2`
- plugin_type: `analysis`
- actionability_state: `explained_na`
- reason_code: `NOT_APPLICABLE`
- finding_kind: `non_actionable_explanation`
- explanation: analysis_issue_cards_v2 produced 1 finding(s), but none matched current action-routing rules.

## Traceability

- class `synthesis_verification` -> plugin `analysis_issue_cards_v2` -> run `full_loaded_3246cc7c_20260218T193803Z`
- Source artifact: `docs/plugin_class_actionability_matrix.json`
