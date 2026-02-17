# Plugin Class Example: direct_action_generators

- class_rationale: Plugins expected to emit direct recommendations with concrete changes.
- expected_output_type: `recommendation_items`
- run_id: `full_loaded_3246cc7c_20260217T165636Z`

## Example

- plugin_id: `analysis_actionable_ops_levers_v1`
- plugin_type: `analysis`
- actionability_state: `actionable`
- finding_kind: `actionable_ops_lever`
- action_type: `batch_or_cache`
- modeled_percent: 0.00
- recommendation: For chkregupdt, many runs repeat the same STATUS_CD. Consider caching or batching by STATUS_CD to reduce queued work. Upper bound: 0.0h (~0.0h/day) of wait time above 60s associated with this process over the window. Validate by measuring queue-delay distribution before/after batching and ensuring any SLAs still hold.

## Traceability

- class `direct_action_generators` -> plugin `analysis_actionable_ops_levers_v1` -> run `full_loaded_3246cc7c_20260217T165636Z`
- Source artifact: `docs/plugin_class_actionability_matrix.json`
