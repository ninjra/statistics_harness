# Plugin Class Example: supporting_signal_detectors

- class_rationale: Plugins that produce measurable signals feeding downstream decision plugins.
- expected_output_type: `findings_or_metrics`
- run_id: `baseline_verify_20260225k`

## Example

- plugin_id: `analysis_action_search_mip_batched_scheduler_v1`
- plugin_type: `analysis`
- actionability_state: `actionable`
- finding_kind: `actionable_ops_lever`
- action_type: `throttle_or_dedupe`
- modeled_percent: 0.01
- recommendation: Execute the selected actions as one package; then re-run and confirm close-window spillover shrinks without moving the bottleneck.

## Traceability

- class `supporting_signal_detectors` -> plugin `analysis_action_search_mip_batched_scheduler_v1` -> run `baseline_verify_20260225k`
- Source artifact: `docs/plugin_class_actionability_matrix.json`
