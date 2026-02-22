# Plugins Validate Runbook Technique Map

| # | Technique | Plugin ID | Implemented |
|---:|---|---|:---:|
| 1 | Conformal prediction | `analysis_conformal_feature_prediction` | Y |
| 2 | False Discovery Rate (Benjamini-Hochberg) | `analysis_multiple_testing_fdr` | Y |
| 3 | BCa bootstrap confidence intervals | `analysis_bootstrap_ci_effect_sizes_v1` | Y |
| 4 | Knockoff filter | `analysis_gaussian_knockoffs` | Y |
| 5 | Quantile regression | `analysis_quantile_regression_duration` | Y |
| 6 | LASSO | `analysis_fused_lasso_trend_filtering_v1` | Y |
| 7 | Elastic Net | `analysis_elastic_net_regularized_glm_v1` | Y |
| 8 | Random Forests | `analysis_knockoff_wrapper_rf` | Y |
| 9 | Gradient Boosting Machine | `analysis_quantile_loss_boosting_v1` | Y |
| 10 | Isolation Forest | `analysis_isolation_forest_anomaly` | Y |
| 11 | Local Outlier Factor (LOF) | `analysis_local_outlier_factor` | Y |
| 12 | Minimum Covariance Determinant (MCD) | `analysis_minimum_covariance_determinant_v1` | Y |
| 13 | Huber M-estimator | `analysis_robust_regression_huber_ransac_v1` | Y |
| 14 | Gaussian Process Regression | `analysis_gaussian_process_regression_v1` | Y |
| 15 | Generalized Additive Models (GAM) | `analysis_gam_spline_regression_v1` | Y |
| 16 | Mixed-effects (random-effects) models | `analysis_mixed_effects_hierarchical_v1` | Y |
| 17 | Cox proportional hazards | `analysis_survival_time_to_event` | Y |
| 18 | Bayesian Additive Regression Trees (BART) | `analysis_bart_uplift_surrogate_v1` | Y |
| 19 | Dirichlet Process (nonparametric Bayes) | `analysis_dirichlet_multinomial_categorical_overdispersion_v1` | Y |
| 20 | Hidden Markov Models (HMM) | `analysis_hmm_latent_state_sequences` | Y |
| 21 | Kalman filter | `analysis_state_space_kalman_residuals` | Y |
| 22 | Granger causality | `analysis_granger_causality_v1` | Y |
| 23 | Transfer entropy | `analysis_transfer_entropy_directional` | Y |
| 24 | kNN mutual information estimator | `analysis_mutual_information_screen` | Y |
| 25 | Independent Component Analysis (ICA) | `analysis_ica_source_separation_v1` | Y |
| 26 | Non-negative Matrix Factorization (NMF) | `analysis_nonnegative_matrix_factorization_v1` | Y |
| 27 | t-SNE | `analysis_tsne_embedding_v1` | Y |
| 28 | UMAP | `analysis_umap_embedding_v1` | Y |
| 29 | NOTEARS | `analysis_notears_linear` | Y |
| 30 | MICE (chained equations) imputation | `analysis_mice_imputation_chained_equations_v1` | Y |
