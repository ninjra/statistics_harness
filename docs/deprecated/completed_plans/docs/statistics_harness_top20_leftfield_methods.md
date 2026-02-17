# Statistics Harness — Top 20 “Left-Field” Methods (Not in repo / not in Next30 packs)

> **NO EVIDENCE (recency):** Web browsing is unavailable in this session, so “latest” cannot be validated.
> The items below are **modern/left-field** techniques that can produce unique insights, with **reference links as pointers**.
> Links were **not accessed** in-session.

## How to use this list
- Each item proposes a **new plugin ID** (suggested naming).
- Pillar scores are **INFERENCE** (0–3) and ranked by **leximin** (maximize the worst pillar first).
- “Deps” notes whether you can implement with existing repo deps (`numpy/pandas/sklearn/networkx`) or likely need additions.

## Ranked table (leximin over P/A/S/C)
Legend: P=Performant, A=Accurate, S=Secure, C=Citable (0–3). “Worst” shows the minimum pillar(s).

|Rank|Proposed plugin_id|Method (unique insight)|Key outputs (metrics/artifacts)|Deps|P|A|S|C|Worst|
|---:|---|---|---|---|---:|---:|---:|---:|---|
|1|analysis_ssa_decomposition_changepoint_v1|**Singular Spectrum Analysis (SSA)** for regime decomposition + changepoints|components, w-corr, cp candidates|numpy|3|2|3|3|2:A|
|2|analysis_cur_decomposition_explain_v1|**CUR / column subset selection** for interpretable low-rank structure|selected rows/cols, recon error|numpy/sklearn|3|2|3|3|2:A|
|3|analysis_hsic_independence_screen_v1|**HSIC** (kernel independence) for nonlinear dependence discovery|top dependent pairs, hsic scores|numpy (Nyström optional)|2|3|3|3|2:P|
|4|analysis_icp_invariant_causal_prediction_v1|**Invariant Causal Prediction (ICP)** across environments (time windows/groups)|invariant parent sets, stability scores|numpy/sklearn|2|3|3|3|2:P|
|5|analysis_lingam_causal_discovery_v1|**LiNGAM** (ICA-based causal discovery)|adjacency, edge confidences|sklearn (FastICA)|2|2|3|3|2:P,A|
|6|analysis_frequent_directions_cov_sketch_v1|**Frequent Directions** streaming sketch of covariance (scale to huge rows)|sketch size, approx error|numpy|3|2|3|2|2:A,C|
|7|analysis_dmd_koopman_modes_v1|**Dynamic Mode Decomposition (DMD)** / Koopman modes for multivariate dynamics|modes, eigenvalues, drift score|numpy|2|2|3|2|2:P,A,C|
|8|analysis_diffusion_maps_manifold_v1|**Diffusion Maps / Laplacian eigenmaps** to surface manifold structure & drift|embedding, eigen-gap, cluster stability|numpy/sklearn|2|2|3|2|2:P,A,C|
|9|analysis_sinkhorn_ot_drift_v1|**Sinkhorn divergence** OT drift (scalable OT)|sinkhorn cost, convergence|numpy (optional POT)|2|2|3|2|2:P,A,C|
|10|analysis_knn_graph_two_sample_test_v1|**kNN / Friedman–Rafsky** multivariate two-sample test (graph-based)|FR statistic, p-value (perm capped)|numpy/sklearn|1|3|3|2|1:P|
|11|analysis_ksd_stein_discrepancy_anomaly_v1|**Kernelized Stein Discrepancy** for “does data fit baseline?” anomaly|ksd score, threshold|numpy|1|2|3|2|1:P|
|12|analysis_pc_algorithm_causal_graph_v1|**PC algorithm** causal discovery via conditional independence tests|skeleton, CPDAG edges|numpy/sklearn|1|2|3|3|1:P|
|13|analysis_ges_score_based_causal_v1|**GES** score-based causal discovery (BIC-like)|DAG/CPDAG, score deltas|numpy|1|2|3|3|1:P|
|14|analysis_phate_trajectory_embedding_v1|**PHATE** for trajectory/transition structure in high-dim data|embedding, diffusion potential|new dep likely|1|2|3|2|1:P|
|15|analysis_node2vec_graph_embedding_drift_v1|**Node2Vec** embeddings to detect shifting roles/hotspots in graphs|embedding drift, cluster changes|new dep likely|1|2|3|2|1:P|
|16|analysis_tensor_cp_parafac_decomp_v1|**Tensor decomposition (CP/PARAFAC)** on (time×process×metric) cubes|rank, factors, recon error|numpy (or tensorly)|2|2|3|2|2:P,A,C|
|17|analysis_symbolic_regression_gp_v1|**Symbolic regression** for interpretable formula discovery|best expressions, fit error|new dep likely|1|2|2|3|1:P,2:S|
|18|analysis_normalizing_flow_density_v1|**Normalizing flows** density modeling for anomaly detection|log-likelihood, outliers|torch likely|1|2|1|1|1:S,C|
|19|analysis_tabpfn_foundation_tabular_v1|**TabPFN** (foundation tabular predictor) for few-shot modeling|predictive perf, uncertainty|torch + weights|1|2|1|2|1:S|
|20|analysis_neural_additive_model_nam_v1|**Neural Additive Models (NAM)** for interpretable nonlinear effects|shape functions, importances|torch likely|1|2|1|2|1:S|

## Reference pointers (NO EVIDENCE: not accessed)
- SSA: https://en.wikipedia.org/wiki/Singular_spectrum_analysis  
- CUR decomposition: https://en.wikipedia.org/wiki/CUR_matrix_approximation  
- HSIC: https://jmlr.org/papers/v13/gretton12a.html  
- ICP: https://arxiv.org/abs/1503.01372  
- LiNGAM: https://jmlr.org/papers/v7/shimizu06a.html  
- Frequent Directions: https://arxiv.org/abs/1501.01711  
- DMD: https://epubs.siam.org/doi/10.1137/1.9781611974508  
- Diffusion Maps: https://www.pnas.org/doi/10.1073/pnas.0500334102  
- Sinkhorn divergence: https://arxiv.org/abs/1306.0895  (Sinkhorn; divergence variants exist)  
- Friedman–Rafsky: https://www.jstor.org/stable/2286737  
- KSD: https://arxiv.org/abs/1602.03253  
- PC algorithm: https://en.wikipedia.org/wiki/PC_algorithm  
- GES: https://www.jmlr.org/papers/v3/chickering02b.html  
- PHATE: https://www.nature.com/articles/s41587-019-0336-3  
- node2vec: https://arxiv.org/abs/1607.00653  
- CP/PARAFAC: https://en.wikipedia.org/wiki/CP_decomposition  
- Symbolic regression (GP): https://en.wikipedia.org/wiki/Symbolic_regression  
- Normalizing flows (overview): https://arxiv.org/abs/1912.02762  
- TabPFN: https://arxiv.org/abs/2207.01848  
- NAM: https://arxiv.org/abs/2004.13912  

## Notes for integration into this repo
- Implement “safe-first” items (ranks 1–16) as **stat-plugin handlers** (single addon module + thin wrappers), similar to existing packs.
- Items 18–20 likely require new heavy dependencies and careful sandboxing; score reflects that.

