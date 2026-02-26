# Plugin Class Example: direct_action_generators

- class_rationale: Plugins expected to emit direct recommendations with concrete changes.
- expected_output_type: `recommendation_items`
- run_id: `baseline_verify_20260225k`

## Example

- plugin_id: `analysis_actionable_ops_levers_v1`
- plugin_type: `analysis`
- actionability_state: `actionable`
- finding_kind: `actionable_ops_lever`
- action_type: `throttle_or_dedupe`
- modeled_percent: 0.01
- recommendation: For jbjeminhld, bursty arrivals correlate with slower median wait/duration. Apply throttling or deduplication at the arrival point to smooth bursts. Upper bound: 0.7h (~0.0h/day) of wait time above 60s. Validate by re-running and checking the burst-correlation and queue-delay distribution.

## Traceability

- class `direct_action_generators` -> plugin `analysis_actionable_ops_levers_v1` -> run `baseline_verify_20260225k`
- Source artifact: `docs/plugin_class_actionability_matrix.json`
