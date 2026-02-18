# Plugin Class Example: direct_action_generators

- class_rationale: Plugins expected to emit direct recommendations with concrete changes.
- expected_output_type: `recommendation_items`
- run_id: `full_loaded_3246cc7c_20260218T193803Z`

## Example

- plugin_id: `analysis_actionable_ops_levers_v1`
- plugin_type: `analysis`
- actionability_state: `actionable`
- finding_kind: `actionable_ops_lever`
- action_type: `batch_input`
- modeled_percent: 0.00
- recommendation: Convert specific process_id `gmicalcrun` to batch-input mode in close month 2025-08. Why: 1,177 runs handled 1,158 unique `o(18015)` values (coverage 100.0%, uniqueness 98.4%), which indicates one-call-per-value sweep behavior. Change `gmicalcrun` to accept a list of `o(18015)` values and process one close-month cohort per run (reducing job launches by ~1,176). Upper bound: 0.3h of over-threshold wait time above 60s associated with this process over the observation window (not guaranteed). Manual-loop signal: audrey_stachmus2 executed 1,904 of 2,425 close-window runs (78.5% concentration, 10 distinct users overall).

## Traceability

- class `direct_action_generators` -> plugin `analysis_actionable_ops_levers_v1` -> run `full_loaded_3246cc7c_20260218T193803Z`
- Source artifact: `docs/plugin_class_actionability_matrix.json`
