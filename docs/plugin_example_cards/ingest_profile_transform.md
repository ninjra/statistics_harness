# Plugin Class Example: ingest_profile_transform

- class_rationale: Data ingestion, profiling, and normalization plugins.
- expected_output_type: `dataset_or_schema_artifacts`
- run_id: `baseline_verify_20260225k`

## Example

- plugin_id: `profile_basic`
- plugin_type: `profile`
- actionability_state: `explained_na`
- reason_code: `NON_DECISION_PLUGIN`
- finding_kind: `non_actionable_explanation`
- explanation: profile_basic is a non-decision plugin (dataset_or_schema_artifacts) and is tracked as explained N/A when recommendation lanes are absent.

## Traceability

- class `ingest_profile_transform` -> plugin `profile_basic` -> run `baseline_verify_20260225k`
- Source artifact: `docs/plugin_class_actionability_matrix.json`
