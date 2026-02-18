# Plugin Class Example: supporting_signal_detectors

- class_rationale: Plugins that produce measurable signals feeding downstream decision plugins.
- expected_output_type: `findings_or_metrics`
- run_id: `full_loaded_3246cc7c_20260218T193803Z`

## Example

- plugin_id: `analysis_close_cycle_contention`
- plugin_type: `analysis`
- actionability_state: `actionable`
- finding_kind: `close_cycle_contention`
- modeled_percent: 34.48
- recommendation: Missing evidence for analysis_close_cycle_contention:close_cycle_contention (process qemail); check inputs and re-run.

## Traceability

- class `supporting_signal_detectors` -> plugin `analysis_close_cycle_contention` -> run `full_loaded_3246cc7c_20260218T193803Z`
- Source artifact: `docs/plugin_class_actionability_matrix.json`
