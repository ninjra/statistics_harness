# Optimal 4-Pillars Triage

- run_id: `baseline_actionable_sync_20260220T2310Z`
- generated_at: `2026-02-21T02:07:34.233630+00:00`
- totals: plugins=200 ok=190 error=9 running=1 degraded=0

## Priority Queue

| rank | plugin_id | status | reason | duration_ms | max_rss_kb | next_step |
|---:|---|---|---|---:|---:|---|
| 1 | `analysis_control_chart_ewma` | `error` | `runtime_failure` | 38275 | 2425172 | Capture plugin-specific traceback, add guardrails, and force deterministic fallback result. |
| 2 | `analysis_control_chart_cusum` | `error` | `runtime_failure` | 35571 | 2424052 | Capture plugin-specific traceback, add guardrails, and force deterministic fallback result. |
| 3 | `analysis_control_chart_individuals` | `error` | `runtime_failure` | 35023 | 2421104 | Capture plugin-specific traceback, add guardrails, and force deterministic fallback result. |
| 4 | `analysis_multivariate_control_charts` | `error` | `runtime_failure` | 23936 | 2086008 | Capture plugin-specific traceback, add guardrails, and force deterministic fallback result. |
| 5 | `analysis_control_chart_suite` | `error` | `runtime_failure` | 27341 | 1843080 | Capture plugin-specific traceback, add guardrails, and force deterministic fallback result. |
| 6 | `analysis_distribution_drift_suite` | `error` | `runtime_failure` | 26716 | 1841884 | Capture plugin-specific traceback, add guardrails, and force deterministic fallback result. |
| 7 | `analysis_phate_trajectory_embedding_v1` | `error` | `runtime_failure` | 28721 | 1754196 | Capture plugin-specific traceback, add guardrails, and force deterministic fallback result. |
| 8 | `analysis_conformance_alignments` | `ok` | `high_duration_hotspot` | 581018 | 1638100 | Add bounded-window execution path and lightweight approximation mode for large inputs. |
| 9 | `analysis_markov_transition_shift` | `ok` | `high_duration_hotspot` | 562130 | 1640448 | Add bounded-window execution path and lightweight approximation mode for large inputs. |
| 10 | `analysis_local_outlier_factor` | `ok` | `high_memory_hotspot` | 263066 | 1640048 | Stream/batch large computations and avoid materializing wide intermediate DataFrames. |
| 11 | `analysis_close_cycle_start_backtrack_v1` | `ok` | `high_memory_hotspot` | 197935 | 1541776 | Stream/batch large computations and avoid materializing wide intermediate DataFrames. |
| 12 | `analysis_actionable_ops_levers_v1` | `ok` | `high_memory_hotspot` | 97974 | 1638304 | Stream/batch large computations and avoid materializing wide intermediate DataFrames. |
| 13 | `analysis_close_cycle_window_resolver` | `ok` | `high_memory_hotspot` | 194279 | 1539816 | Stream/batch large computations and avoid materializing wide intermediate DataFrames. |
| 14 | `analysis_dynamic_close_detection` | `ok` | `high_memory_hotspot` | 188061 | 1542364 | Stream/batch large computations and avoid materializing wide intermediate DataFrames. |
| 15 | `analysis_changepoint_energy_edivisive` | `ok` | `high_memory_hotspot` | 86058 | 1637936 | Stream/batch large computations and avoid materializing wide intermediate DataFrames. |
| 16 | `analysis_param_near_duplicate_minhash_v1` | `error` | `killed_or_time_budget_exhausted` | 600147 | 0 | Bound algorithmic complexity (pair checks, tokens, rows) and add deterministic caps. |
| 17 | `analysis_param_near_duplicate_simhash_v1` | `error` | `killed_or_time_budget_exhausted` | 600086 | 0 | Bound algorithmic complexity (pair checks, tokens, rows) and add deterministic caps. |
| 18 | `analysis_sparse_pca_interpretable_components_v1` | `ok` | `high_memory_hotspot` | 77850 | 1637632 | Stream/batch large computations and avoid materializing wide intermediate DataFrames. |
| 19 | `analysis_changepoint_method_survey_guided` | `ok` | `high_memory_hotspot` | 75761 | 1637540 | Stream/batch large computations and avoid materializing wide intermediate DataFrames. |
| 20 | `analysis_haar_wavelet_transient_detector_v1` | `ok` | `high_memory_hotspot` | 58005 | 1642276 | Stream/batch large computations and avoid materializing wide intermediate DataFrames. |
| 21 | `analysis_circular_time_of_day_drift_v1` | `ok` | `high_memory_hotspot` | 57186 | 1635824 | Stream/batch large computations and avoid materializing wide intermediate DataFrames. |
| 22 | `analysis_bayesian_online_changepoint_studentt_v1` | `ok` | `high_memory_hotspot` | 63213 | 1629572 | Stream/batch large computations and avoid materializing wide intermediate DataFrames. |
| 23 | `analysis_distance_covariance_dependence_v1` | `ok` | `high_memory_hotspot` | 55595 | 1636792 | Stream/batch large computations and avoid materializing wide intermediate DataFrames. |
| 24 | `analysis_quantile_sketch_p2_streaming_v1` | `ok` | `high_memory_hotspot` | 55761 | 1633632 | Stream/batch large computations and avoid materializing wide intermediate DataFrames. |
| 25 | `analysis_state_space_smoother_level_shift_v1` | `ok` | `high_memory_hotspot` | 48600 | 1637908 | Stream/batch large computations and avoid materializing wide intermediate DataFrames. |
| 26 | `analysis_beta_binomial_overdispersion_v1` | `ok` | `high_memory_hotspot` | 48270 | 1638028 | Stream/batch large computations and avoid materializing wide intermediate DataFrames. |
| 27 | `analysis_mutual_information_screen` | `ok` | `high_memory_hotspot` | 44365 | 1641368 | Stream/batch large computations and avoid materializing wide intermediate DataFrames. |
| 28 | `analysis_subspace_tracking_oja_v1` | `ok` | `high_memory_hotspot` | 45572 | 1640060 | Stream/batch large computations and avoid materializing wide intermediate DataFrames. |
| 29 | `analysis_ssa_decomposition_changepoint_v1` | `ok` | `high_memory_hotspot` | 67254 | 1618000 | Stream/batch large computations and avoid materializing wide intermediate DataFrames. |
| 30 | `analysis_stl_seasonal_decompose_v1` | `ok` | `high_memory_hotspot` | 42392 | 1642700 | Stream/batch large computations and avoid materializing wide intermediate DataFrames. |
| 31 | `analysis_outlier_influence_cooks_distance_v1` | `ok` | `high_memory_hotspot` | 46346 | 1638556 | Stream/batch large computations and avoid materializing wide intermediate DataFrames. |
| 32 | `analysis_spectral_radius_stability_v1` | `ok` | `high_memory_hotspot` | 47140 | 1637076 | Stream/batch large computations and avoid materializing wide intermediate DataFrames. |
| 33 | `analysis_change_score_consensus_v1` | `ok` | `high_memory_hotspot` | 47470 | 1636604 | Stream/batch large computations and avoid materializing wide intermediate DataFrames. |
| 34 | `analysis_bootstrap_ci_effect_sizes_v1` | `ok` | `high_memory_hotspot` | 43466 | 1638880 | Stream/batch large computations and avoid materializing wide intermediate DataFrames. |
| 35 | `analysis_pca_auto` | `ok` | `high_memory_hotspot` | 39850 | 1642128 | Stream/batch large computations and avoid materializing wide intermediate DataFrames. |
| 36 | `analysis_cca_crossblock_association_v1` | `ok` | `high_memory_hotspot` | 45707 | 1636212 | Stream/batch large computations and avoid materializing wide intermediate DataFrames. |
| 37 | `analysis_random_matrix_marchenko_pastur_denoise_v1` | `ok` | `high_memory_hotspot` | 43527 | 1637828 | Stream/batch large computations and avoid materializing wide intermediate DataFrames. |
| 38 | `analysis_chi_square_association` | `ok` | `high_memory_hotspot` | 42236 | 1638756 | Stream/batch large computations and avoid materializing wide intermediate DataFrames. |
| 39 | `analysis_log_template_drain` | `ok` | `high_memory_hotspot` | 37179 | 1643732 | Stream/batch large computations and avoid materializing wide intermediate DataFrames. |
| 40 | `analysis_quantile_loss_boosting_v1` | `ok` | `high_memory_hotspot` | 41576 | 1639020 | Stream/batch large computations and avoid materializing wide intermediate DataFrames. |

## Workflow

1. Fix top queue items first (failure > runtime > memory).
2. Run targeted tests for touched plugins.
3. Run one full gauntlet (`--plugin-set full --force`).
4. Compare before/after with `scripts/compare_run_outputs.py` and `scripts/compare_plugin_actionability_runs.py`.

