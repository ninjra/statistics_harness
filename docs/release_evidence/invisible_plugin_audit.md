# Invisible Plugin Audit

- **Total plugins with findings**: 262
- **Contributing plugins**: 44
- **Invisible (with findings)**: 218
- **Total non-actionable explanations**: 120
- **Suspect false negatives**: 11

## Reason Code Summary

| Reason Code | Count |
|---|---|
| NO_STATISTICAL_SIGNAL | 36 |
| OBSERVATION_ONLY | 31 |
| ADAPTER_RULE_MISSING | 11 |
| MISSING_PREREQUISITE | 11 |
| PLUGIN_PRECONDITION_UNMET | 10 |
| NO_ACTIONABLE_FINDING_CLASS | 6 |
| QUADRATIC_CAP_EXCEEDED | 3 |
| PLUGIN_ERROR | 2 |
| BELOW_ACTIONABILITY_THRESHOLD | 2 |
| NO_ELIGIBLE_SLICE | 1 |
| CAPACITY_IMPACT_CONSTRAINT | 1 |
| NO_REVENUE_COMPRESSION_PRESSURE | 1 |
| INSUFFICIENT_POSITIVE_SAMPLES | 1 |
| NO_HOLD_DOMINANT_PROCESSES | 1 |
| NO_FEATURES_ELIGIBLE | 1 |
| NO_ACTIONABLE_PROCESS_SERVER_LEVER | 1 |
| NO_SIGNIFICANT_EFFECT | 1 |

## Suspect False Negatives

| Plugin ID | Reason | Findings | Confidence | Process? | Priority |
|---|---|---|---|---|---|
| analysis_constrained_clustering_cop_kmeans_v1 | ADAPTER_RULE_MISSING | 3 | 0.55 | Y | 1.7 |
| analysis_bart_uplift_surrogate_v1 | ADAPTER_RULE_MISSING | 1 | 1.00 | Y | 1.0 |
| analysis_elastic_net_regularized_glm_v1 | ADAPTER_RULE_MISSING | 1 | 1.00 | Y | 1.0 |
| analysis_gaussian_process_regression_v1 | ADAPTER_RULE_MISSING | 1 | 1.00 | Y | 1.0 |
| analysis_nonnegative_matrix_factorization_v1 | ADAPTER_RULE_MISSING | 1 | 0.98 | Y | 1.0 |
| analysis_minimum_covariance_determinant_v1 | ADAPTER_RULE_MISSING | 1 | 0.93 | Y | 0.9 |
| analysis_mice_imputation_chained_equations_v1 | ADAPTER_RULE_MISSING | 1 | 0.90 | Y | 0.9 |
| analysis_umap_embedding_v1 | ADAPTER_RULE_MISSING | 1 | 0.89 | Y | 0.9 |
| analysis_ideaspace_energy_ebm_v1 | OBSERVATION_ONLY | 1 | 0.80 | N | 0.8 |
| analysis_tsne_embedding_v1 | ADAPTER_RULE_MISSING | 1 | 0.63 | Y | 0.6 |
| analysis_dependency_critical_path_v1 | ADAPTER_RULE_MISSING | 1 | 0.55 | N | 0.6 |

## ADAPTER_RULE_MISSING Detail

| Plugin ID | Mapped Kind | Finding Kinds | Count |
|---|---|---|---|
| analysis_bart_uplift_surrogate_v1 | causal | actionable_ops_lever | 1 |
| analysis_capacity_frontier_envelope_v1 | capacity_scale_model | capacity_scale_model | 1 |
| analysis_constrained_clustering_cop_kmeans_v1 | cluster | actionable_ops_lever | 3 |
| analysis_dependency_critical_path_v1 | graph | dependency_critical_path | 1 |
| analysis_elastic_net_regularized_glm_v1 | regression | actionable_ops_lever | 1 |
| analysis_gaussian_process_regression_v1 | regression | actionable_ops_lever | 1 |
| analysis_mice_imputation_chained_equations_v1 | distribution | actionable_ops_lever | 1 |
| analysis_minimum_covariance_determinant_v1 | anomaly | actionable_ops_lever | 1 |
| analysis_nonnegative_matrix_factorization_v1 | cluster | actionable_ops_lever | 1 |
| analysis_tsne_embedding_v1 | cluster | actionable_ops_lever | 1 |
| analysis_umap_embedding_v1 | cluster | actionable_ops_lever | 1 |
