# Invisible Plugin Audit

- **Total plugins with findings**: 262
- **Contributing plugins**: 37
- **Invisible (with findings)**: 225
- **Total non-actionable explanations**: 197
- **Suspect false negatives**: 66

## Reason Code Summary

| Reason Code | Count |
|---|---|
| OBSERVATION_ONLY | 109 |
| NO_STATISTICAL_SIGNAL | 36 |
| MISSING_PREREQUISITE | 11 |
| ADAPTER_RULE_MISSING | 10 |
| PLUGIN_PRECONDITION_UNMET | 10 |
| NO_ACTIONABLE_FINDING_CLASS | 5 |
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
| MISSING_REQUIRED_COLUMNS | 1 |
| NO_SIGNIFICANT_EFFECT | 1 |

## Suspect False Negatives

| Plugin ID | Reason | Findings | Confidence | Process? | Priority |
|---|---|---|---|---|---|
| analysis_control_chart_cusum | OBSERVATION_ONLY | 30 | 1.00 | N | 30.0 |
| analysis_control_chart_ewma | OBSERVATION_ONLY | 30 | 1.00 | N | 30.0 |
| analysis_control_chart_individuals | OBSERVATION_ONLY | 30 | 1.00 | N | 30.0 |
| analysis_control_chart_suite | OBSERVATION_ONLY | 30 | 1.00 | N | 30.0 |
| analysis_distribution_drift_suite | OBSERVATION_ONLY | 30 | 1.00 | N | 30.0 |
| analysis_multivariate_control_charts | OBSERVATION_ONLY | 30 | 1.00 | N | 30.0 |
| analysis_robust_pca_pcp | OBSERVATION_ONLY | 30 | 1.00 | N | 30.0 |
| analysis_robust_pca_sparse_outliers | OBSERVATION_ONLY | 30 | 1.00 | N | 30.0 |
| analysis_isolation_forest | OBSERVATION_ONLY | 30 | 1.00 | N | 30.0 |
| analysis_isolation_forest_anomaly | OBSERVATION_ONLY | 30 | 1.00 | N | 30.0 |
| analysis_stl_seasonal_decompose_v1 | OBSERVATION_ONLY | 30 | 0.95 | N | 28.5 |
| analysis_seasonal_holt_winters_forecast_residuals_v1 | OBSERVATION_ONLY | 30 | 0.90 | N | 27.0 |
| analysis_fisher_exact_enrichment_v1 | OBSERVATION_ONLY | 30 | 0.70 | N | 21.0 |
| analysis_two_sample_categorical_chi2 | OBSERVATION_ONLY | 14 | 1.00 | N | 14.0 |
| analysis_multivariate_changepoint_pelt | OBSERVATION_ONLY | 20 | 0.60 | N | 12.0 |
| analysis_drift_adwin | OBSERVATION_ONLY | 10 | 1.00 | N | 10.0 |
| analysis_effect_size_report | OBSERVATION_ONLY | 9 | 1.00 | N | 9.0 |
| analysis_quantile_regression_forest_v1 | OBSERVATION_ONLY | 10 | 0.65 | N | 6.5 |
| analysis_evt_gumbel_tail | OBSERVATION_ONLY | 5 | 1.00 | N | 5.0 |
| analysis_benfords_law_anomaly_v1 | OBSERVATION_ONLY | 5 | 0.85 | N | 4.2 |
| analysis_multiple_testing_fdr | OBSERVATION_ONLY | 2 | 1.00 | N | 2.0 |
| analysis_multicollinearity_vif_screen_v1 | OBSERVATION_ONLY | 2 | 0.90 | N | 1.8 |
| analysis_constrained_clustering_cop_kmeans_v1 | ADAPTER_RULE_MISSING | 3 | 0.55 | Y | 1.7 |
| analysis_bart_uplift_surrogate_v1 | ADAPTER_RULE_MISSING | 1 | 1.00 | Y | 1.0 |
| analysis_burst_detection_kleinberg | OBSERVATION_ONLY | 1 | 1.00 | N | 1.0 |
| analysis_elastic_net_regularized_glm_v1 | ADAPTER_RULE_MISSING | 1 | 1.00 | Y | 1.0 |
| analysis_gaussian_process_regression_v1 | ADAPTER_RULE_MISSING | 1 | 1.00 | Y | 1.0 |
| analysis_hawkes_self_exciting | OBSERVATION_ONLY | 1 | 1.00 | N | 1.0 |
| analysis_multivariate_ewma_control | OBSERVATION_ONLY | 1 | 1.00 | N | 1.0 |
| analysis_multivariate_t2_control | OBSERVATION_ONLY | 1 | 1.00 | N | 1.0 |

## ADAPTER_RULE_MISSING Detail

| Plugin ID | Mapped Kind | Finding Kinds | Count |
|---|---|---|---|
| analysis_bart_uplift_surrogate_v1 | causal | actionable_ops_lever | 1 |
| analysis_constrained_clustering_cop_kmeans_v1 | cluster | actionable_ops_lever | 3 |
| analysis_dependency_critical_path_v1 | graph | dependency_critical_path | 1 |
| analysis_elastic_net_regularized_glm_v1 | regression | actionable_ops_lever | 1 |
| analysis_gaussian_process_regression_v1 | regression | actionable_ops_lever | 1 |
| analysis_mice_imputation_chained_equations_v1 | distribution | actionable_ops_lever | 1 |
| analysis_minimum_covariance_determinant_v1 | anomaly | actionable_ops_lever | 1 |
| analysis_nonnegative_matrix_factorization_v1 | cluster | actionable_ops_lever | 1 |
| analysis_tsne_embedding_v1 | cluster | actionable_ops_lever | 1 |
| analysis_umap_embedding_v1 | cluster | actionable_ops_lever | 1 |
