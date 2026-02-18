# Plugin Class Example: ingest_profile_transform

- class_rationale: Data ingestion, profiling, and normalization plugins.
- expected_output_type: `dataset_or_schema_artifacts`
- run_id: `full_loaded_3246cc7c_20260218T193803Z`

## Example

- plugin_id: `profile_basic`
- plugin_type: `profile`
- actionability_state: `explained_na`
- reason_code: `NO_FINDINGS`
- finding_kind: `non_actionable_explanation`
- explanation: profile_basic completed but did not emit findings that can drive an action. This is a profile plugin; its output feeds downstream plugins: analysis_association_rules_apriori_v1, analysis_biclustering_cheng_church_v1, analysis_constrained_clustering_cop_kmeans_v1, analysis_cur_decomposition_explain_v1, analysis_density_clustering_hdbscan_v1, analysis_diffusion_maps_manifold_v1, analysis_frequent_directions_cov_sketch_v1, analysis_frequent_itemsets_fpgrowth_v1, analysis_hsic_independence_screen_v1, analysis_param_near_duplicate_minhash_v1, analysis_param_near_duplicate_simhash_v1, analysis_similarity_graph_spectral_clustering_v1.

## Traceability

- class `ingest_profile_transform` -> plugin `profile_basic` -> run `full_loaded_3246cc7c_20260218T193803Z`
- Source artifact: `docs/plugin_class_actionability_matrix.json`
