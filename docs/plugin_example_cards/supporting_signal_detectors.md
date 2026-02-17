# Plugin Class Example: supporting_signal_detectors

- class_rationale: Plugins that produce measurable signals feeding downstream decision plugins.
- expected_output_type: `findings_or_metrics`
- run_id: `full_loaded_3246cc7c_20260217T165636Z`

## Example

- plugin_id: `analysis_action_search_mip_batched_scheduler_v1`
- plugin_type: `analysis`
- actionability_state: `explained_na`
- reason_code: `NOT_ROUTED_TO_ACTION`
- finding_kind: `non_actionable_explanation`
- explanation: analysis_action_search_mip_batched_scheduler_v1 produced 1 finding(s), but none matched current action-routing rules.

## Traceability

- class `supporting_signal_detectors` -> plugin `analysis_action_search_mip_batched_scheduler_v1` -> run `full_loaded_3246cc7c_20260217T165636Z`
- Source artifact: `docs/plugin_class_actionability_matrix.json`
