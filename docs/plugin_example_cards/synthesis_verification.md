# Plugin Class Example: synthesis_verification

- class_rationale: Plugins that aggregate, verify, dedupe, or validate outputs from other plugins.
- expected_output_type: `derived_findings_or_validation`
- run_id: `baseline_verify_20260225k`

## Example

- plugin_id: `analysis_issue_cards_v2`
- plugin_type: `analysis`
- actionability_state: `actionable`
- finding_kind: `plugin_actionability_backstop`
- action_type: `batch_or_cache`
- modeled_percent: 0.00
- recommendation: Actionability backstop for analysis_issue_cards_v2: map `issue_cards_summary` signal to a direct batch or cache change on jbprepay, then validate against accounting-window deltas.

## Traceability

- class `synthesis_verification` -> plugin `analysis_issue_cards_v2` -> run `baseline_verify_20260225k`
- Source artifact: `docs/plugin_class_actionability_matrix.json`
