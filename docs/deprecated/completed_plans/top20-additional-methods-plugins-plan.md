# Plan: Top 20 Additional Methods Plugins

**Generated**: 2026-02-10  
**Estimated Complexity**: High

## Overview
Implement the 20 new analysis plugins specified in `docs/top20_additional_methods_plugins.md`, plus the shared helper utilities and the recommended output integration so results surface as **non-generic, engineering-actionable** recommendations (batch/multi-input, caching/dedupe, orchestration macros, decoupling boundaries, modeled action bundles).

Design constraints (4 pillars):
- **Performant**: SQL-first + streaming where possible; avoid full in-memory row-level feature matrices on multi-million-row datasets; cap work deterministically and emit “degraded” with explicit gating reasons when caps bind.
- **Accurate**: deterministic tokenization + cohorting; avoid “black box” outputs without evidence; validate on synthetic fixtures and the real 500k dataset.
- **Secure**: no network; no eval/shelling out; careful artifact writing under `run_dir` only.
- **Citable**: each plugin emits artifacts and evidence pointers (row ids, cohort definitions, and/or reproducible SQL) and has references captured via `default_references_for_plugin`.

## Prerequisites
- Target repo: `/mnt/d/projects/statistics_harness/statistics_harness`
- Tests must pass: `.venv/bin/python -m pytest -q`
- New plugins must follow repo plugin contract (`plugin.yaml`, `plugin.py`, `config.schema.json`, `output.schema.json`).
- Confirm availability/decision for heavy optional deps (HDBSCAN/Leiden/MIP solver). See “Questions” at end.

## Sprint 0: Matrix First (Avoid Duplicate/Overlapping Functionality)
**Goal**: Build a source-of-truth matrix for the 20 plugins (inputs, deps, outputs, overlap with existing plugins) before writing code.

**Demo/Validation**:
- A generated matrix file enumerates all 20 plugins and their execution order constraints.
- Matrix is used to drive scaffolding (no duplicated wrappers/schemas).

### Task 0.1: Create “Top20 Plugins” Implementation Matrix
- **Location**: `docs/`
- **Description**:
  - Add a new generated doc pair:
    - `docs/top20_additional_methods_plugins_matrix.json`
    - `docs/top20_additional_methods_plugins_matrix.md`
  - Fields (minimum):
    - `plugin_id`, `method_family`, `capabilities`, `depends_on`
    - `data_sources` (normalized table, parameter tables, plugin-results dependencies)
    - `runtime_mode` (`sql_first`, `batch_stream`, `in_memory_small`)
    - `output_kinds` and `primary_artifacts`
    - `overlap_with_existing_plugins` (e.g., existing `analysis_sequential_patterns_prefixspan`)
  - Add a generator script (deterministic ordering):
    - `scripts/top20_methods_matrix.py`
- **Complexity**: 5/10
- **Dependencies**: none
- **Acceptance Criteria**:
  - Exactly 20 entries match the IDs in `docs/top20_additional_methods_plugins.md`.
  - Each entry explicitly states caps and gating behavior.
- **Validation**:
  - Unit test: `tests/test_top20_methods_matrix.py` (asserts stable IDs + required fields).

### Task 0.2: Decide “New IDs vs Upgrades” for Overlapping Functionality
- **Location**: `docs/top20_additional_methods_plugins_matrix.md`
- **Description**:
  - For each overlapping method, record whether we:
    - implement the new `_v1` plugin ID as specified, or
    - upgrade an existing plugin and add an alias in `src/statistic_harness/core/stat_plugins/registry.py::ALIAS_MAP`.
- **Complexity**: 2/10
- **Dependencies**: Task 0.1
- **Acceptance Criteria**:
  - No ambiguity about which plugin ID is authoritative for each method.
- **Validation**:
  - CI/test asserts alias map consistency if used.

## Sprint 1: Shared Core Utilities (Tokenization, Similarity, Graph Builders)
**Goal**: Implement shared utilities once, then reuse across all 20 plugins.

**Demo/Validation**:
- Unit tests demonstrate deterministic tokenization and stable similarity results for the same seed/input.

### Task 1.1: Tokenization + Canonicalization Helpers
- **Location**: `src/statistic_harness/core/`
- **Description**:
  - Add `src/statistic_harness/core/tokenize_params.py`:
    - Normalize JSON-ish param strings and key/value pairs.
    - Drop volatile keys via config regex/list.
    - Emit:
      - `tokens: set[str]` for similarity methods
      - `kv: dict[str, str]` for itemset/rule mining
  - Prefer normalization layer parameter tables when present:
    - `row_parameter_link` + `parameter_kv` + `parameter_entities`
  - Provide a stable fallback parser for raw param columns (when parameter tables absent).
- **Complexity**: 7/10
- **Dependencies**: none
- **Acceptance Criteria**:
  - Deterministic output: identical input produces identical token sets across runs.
  - Works with parameter tables and without them (degraded path).
- **Validation**:
  - Unit tests: `tests/test_tokenize_params.py`.

### Task 1.2: Similarity Primitives (Jaccard/Cosine/MinHash/SimHash/LSH)
- **Location**: `src/statistic_harness/core/similarity.py`
- **Description**:
  - Implement Jaccard + cosine primitives (pure Python + numpy).
  - Implement SimHash (pure Python) with deterministic tokenizer.
  - For MinHash/LSH:
    - Prefer `datasketch` if installed; otherwise a degraded pure-Python MinHash with smaller caps.
  - Provide “caps” config shared across plugins:
    - `max_rows_considered`, `max_processes_considered`, `max_pairs_reported`.
- **Complexity**: 8/10
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Always returns results within budget/caps and with explicit gating metadata when truncated.
- **Validation**:
  - Unit tests: `tests/test_similarity.py`.

### Task 1.3: Deterministic Dependency & Similarity Graph Builders
- **Location**: `src/statistic_harness/core/graph_builders.py`
- **Description**:
  - Build graphs with deterministic edge weights from:
    - process transitions (if trace/case ID exists)
    - dependency columns (if present)
    - process co-occurrence in close-month cohorts
  - Emit graph snapshots as artifacts (`nodes.json`, `edges.json`) for citation.
- **Complexity**: 7/10
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Stable node ordering, stable weight computation, stable output hashes.
- **Validation**:
  - Unit tests: `tests/test_graph_builders.py`.

## Sprint 2: Scaffolding (20 Plugin Folders + Schemas + Wrappers)
**Goal**: Create all `plugins/<id>/` directories with correct metadata, schemas, and wrappers so the harness discovers them.

**Demo/Validation**:
- `scripts/plugins_functionality_matrix.py` includes the new plugins and their dependencies.

### Task 2.1: Add a Scaffolder Script for New Plugins
- **Location**: `scripts/scaffold_plugins_from_doc.py` (new)
- **Description**:
  - Parse `docs/top20_additional_methods_plugins.md` and scaffold:
    - `plugins/<id>/plugin.yaml`
    - `plugins/<id>/plugin.py` wrapper (calls `run_plugin(<id>, ctx)`)
    - `plugins/<id>/config.schema.json` (minimal + `additionalProperties: true`)
    - `plugins/<id>/output.schema.json` (copy canonical “Plugin Result” schema)
- **Complexity**: 6/10
- **Dependencies**: Sprint 0
- **Acceptance Criteria**:
  - All 20 plugin dirs exist and match the doc IDs exactly.
- **Validation**:
  - Test: `tests/test_top20_scaffold_integrity.py` (asserts files exist + yaml loads).

### Task 2.2: Register Handlers in a Dedicated Module (Keep Registry Maintainable)
- **Location**:
  - `src/statistic_harness/core/stat_plugins/top20_addon.py` (new)
  - `src/statistic_harness/core/stat_plugins/registry.py`
- **Description**:
  - Implement `HANDLERS` dict for the 20 plugin IDs in `top20_addon.py`.
  - Import and merge `HANDLERS` into registry.
- **Complexity**: 5/10
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - `run_plugin(<new_id>)` resolves and executes handler.
- **Validation**:
  - Smoke tests per plugin (see Sprint 8).

## Sprint 3: Parameter Near-Duplicate + Itemsets + Rules (Plugins 1–4)
**Goal**: Implement non-generic batching/caching/dedupe signals from parameter similarity and frequent bundles.

**Demo/Validation**:
- On synthetic “payout-like” data, emits `batch_refactor_candidate` and `preset_job_candidate`.

### Task 3.1: `analysis_param_near_duplicate_minhash_v1`
- **Location**:
  - `plugins/analysis_param_near_duplicate_minhash_v1/`
  - `src/statistic_harness/core/stat_plugins/top20_addon.py`
- **Description**:
  - Tokenize params via parameter tables when available; fallback parser otherwise.
  - Compute near-duplicate clusters per process with LSH bucketing.
  - Emit:
    - `kind=batch_refactor_candidate`
    - artifact `minhash_clusters.json` with cluster exemplars and row indices.
- **Complexity**: 8/10
- **Dependencies**: Sprint 1, Sprint 2
- **Validation**:
  - `tests/plugins/test_param_near_duplicate_minhash_v1.py`.

### Task 3.2: `analysis_param_near_duplicate_simhash_v1`
- **Description**:
  - Build token stream from params/log text; compute SimHash; cluster by Hamming radius.
  - Emit `near_duplicate_cluster` + `batch_refactor_candidate` when same process family.
- **Complexity**: 7/10
- **Dependencies**: Sprint 1, Sprint 2
- **Validation**:
  - `tests/plugins/test_param_near_duplicate_simhash_v1.py`.

### Task 3.3: `analysis_frequent_itemsets_fpgrowth_v1` and `analysis_association_rules_apriori_v1`
- **Description**:
  - Use kv pairs to mine frequent itemsets and association rules.
  - Prefer deterministic implementations with hard caps (max itemset size, max rules).
  - Emit `frequent_param_itemset`, `preset_job_candidate`, `association_rule`, `variant_consolidation_candidate`.
- **Complexity**: 9/10
- **Dependencies**: Sprint 1, Sprint 2
- **Validation**:
  - `tests/plugins/test_fpgrowth_itemsets_v1.py`
  - `tests/plugins/test_apriori_rules_v1.py`

## Sprint 4: Sequence Mining + Macro Discovery (Plugins 5–6)
**Goal**: Find repeated chains that should be orchestrated as one job.

**Demo/Validation**:
- Synthetic traces with repeating subsequences produce `orchestration_macro_candidate`.

### Task 4.1: `analysis_sequential_patterns_prefixspan_v1`
- **Description**:
  - Mine frequent sequences from traces (requires a trace key/case id).
  - If trace key missing, degrade with explicit gating reason.
  - Emit `frequent_sequence` + `orchestration_macro_candidate` with estimated round trips reduced.
- **Complexity**: 8/10
- **Dependencies**: `profile_eventlog` data availability, Sprint 2
- **Validation**:
  - `tests/plugins/test_prefixspan_v1.py`

### Task 4.2: `analysis_sequence_grammar_sequitur_v1`
- **Description**:
  - Implement SEQUITUR grammar inference on sequences (bounded by max rules).
  - Emit `sequence_macro_rule` + `orchestration_macro_candidate`.
- **Complexity**: 7/10
- **Dependencies**: Task 4.1 or shared sequence extraction
- **Validation**:
  - `tests/plugins/test_sequitur_v1.py`

## Sprint 5: Clustering & Biclustering (Plugins 7–9)
**Goal**: Cluster executions/processes into batchable groups with constraints.

**Demo/Validation**:
- Emits `dense_execution_cluster` / `constrained_cluster` with evidence and candidate actions.

### Task 5.1: `analysis_biclustering_cheng_church_v1`
- **Description**:
  - Build a process×feature matrix (features from kv tokens, plus runtime/queue aggregates).
  - Implement Cheng–Church (bounded; small matrices only).
  - Emit `bicluster` and `shared_cache_candidate`.
- **Complexity**: 9/10
- **Dependencies**: Sprint 1, Sprint 2
- **Validation**:
  - `tests/plugins/test_biclustering_cheng_church_v1.py`

### Task 5.2: `analysis_density_clustering_hdbscan_v1`
- **Description**:
  - If `hdbscan` available, run HDBSCAN on reduced embeddings; else degrade with reason and (optional) fallback DBSCAN.
  - Emit `dense_execution_cluster` + `batch_refactor_candidate`.
- **Complexity**: 8/10
- **Dependencies**: Sprint 1 embeddings/features
- **Validation**:
  - `tests/plugins/test_hdbscan_v1.py` (skips if dependency missing, but plugin must degrade cleanly).

### Task 5.3: `analysis_constrained_clustering_cop_kmeans_v1`
- **Description**:
  - Implement COP-KMeans with must-link/cannot-link constraints:
    - cannot-link from exclusion patterns
    - must-link from known alias/equivalence sets
  - Emit `constrained_cluster` + `batch_refactor_candidate`.
- **Complexity**: 8/10
- **Dependencies**: Exclusion matcher (see Sprint 7), Sprint 1
- **Validation**:
  - `tests/plugins/test_cop_kmeans_v1.py`

## Sprint 6: Graph Communities + Partitioning (Plugins 10–13)
**Goal**: Identify stable subsystem boundaries and concrete decoupling cuts.

**Demo/Validation**:
- Emits `dependency_community` and `min_cut` artifacts on a synthetic graph.

### Task 6.1: `analysis_dependency_community_louvain_v1` and `analysis_dependency_community_leiden_v1`
- **Description**:
  - Build dependency graph via `graph_builders`.
  - Louvain: prefer `python-louvain`; fallback to networkx greedy modularity if missing.
  - Leiden: prefer `igraph`+`leidenalg`; degrade if missing.
  - Emit `dependency_community` + `decoupling_boundary_candidate`.
- **Complexity**: 9/10
- **Dependencies**: Sprint 1.3
- **Validation**:
  - `tests/plugins/test_dependency_communities_v1.py`

### Task 6.2: `analysis_similarity_graph_spectral_clustering_v1`
- **Description**:
  - Build similarity graph (process or execution level) and run spectral clustering (sklearn).
  - Emit `spectral_cluster` + `batch_refactor_candidate`.
- **Complexity**: 7/10
- **Dependencies**: Sprint 1.2/1.3
- **Validation**:
  - `tests/plugins/test_spectral_clustering_v1.py`

### Task 6.3: `analysis_graph_min_cut_partition_v1`
- **Description**:
  - Use Stoer–Wagner min-cut (networkx) on dependency graph.
  - Emit `min_cut` + `decoupling_boundary_candidate` with cut edges and weights.
- **Complexity**: 6/10
- **Dependencies**: Sprint 1.3
- **Validation**:
  - `tests/plugins/test_min_cut_partition_v1.py`

## Sprint 7: Windowed Shift + Bursts + Daily Shape (Plugins 14–16)
**Goal**: Quantify window gaps (close vs spillover), detect triggers, and find daily pattern anomalies.

**Demo/Validation**:
- Emits `target_window_gap_driver`, `hawkes_trigger`, `shape_anomaly_day`.

### Task 7.1: `analysis_distribution_shift_wasserstein_v1`
- **Description**:
  - Compute 1D Wasserstein distances for duration/queue metrics across windows.
  - Provide “top contributing processes” via decomposed per-process shifts.
  - Emit `wasserstein_shift` + `target_window_gap_driver`.
- **Complexity**: 7/10
- **Dependencies**: close-window definitions + time parsing
- **Validation**:
  - `tests/plugins/test_wasserstein_shift_v1.py`

### Task 7.2: `analysis_burst_modeling_hawkes_v1`
- **Description**:
  - Fit a simplified Hawkes-like discrete-time model per process group (bounded max groups).
  - Emit `hawkes_trigger` + `burst_dampening_candidate` with concrete intervention candidates (batch at trigger, cooldown, throttle).
- **Complexity**: 9/10
- **Dependencies**: stable arrival series extraction
- **Validation**:
  - `tests/plugins/test_hawkes_burst_v1.py`

### Task 7.3: `analysis_daily_pattern_alignment_dtw_v1`
- **Description**:
  - Build binned daily series (arrivals / wait / duration).
  - Compute DTW distances and inferred shifts (bounded pairs).
  - Emit `dtw_alignment` + `shape_anomaly_day`.
- **Complexity**: 8/10
- **Dependencies**: time binning utilities
- **Validation**:
  - `tests/plugins/test_dtw_alignment_v1.py`

## Sprint 8: Action Search + Optimization + Simulation (Plugins 17–19)
**Goal**: Turn levers into concrete, modeled action bundles and validate with simulation.

**Demo/Validation**:
- Produces deterministic `modeled` findings with explicit constraints and reproducible seeds.

### Task 8.1: `analysis_action_search_simulated_annealing_v1`
- **Description**:
  - Use levers from `analysis_actionable_ops_levers_v1` and ideaspace gap findings as action primitives.
  - Search bundles to minimize spillover objective under explicit constraints.
  - Emit `action_bundle_candidate` (modeled).
- **Complexity**: 8/10
- **Dependencies**: lever extraction + ideaspace output
- **Validation**:
  - `tests/plugins/test_action_search_annealing_v1.py`

### Task 8.2: `analysis_action_search_mip_batched_scheduler_v1`
- **Description**:
  - Implement a bounded MIP formulation (or deterministic greedy fallback) to propose batching/moving decisions.
  - Emit `optimized_action_plan` + `batching_decision` / `move_window_decision`.
- **Complexity**: 10/10
- **Dependencies**: solver decision (see Questions)
- **Validation**:
  - `tests/plugins/test_action_search_mip_v1.py`

### Task 8.3: `analysis_discrete_event_queue_simulator_v1`
- **Description**:
  - Discrete-event simulation with deterministic RNG seeded from run_seed.
  - Inputs: arrival distributions + service time estimates + action plan deltas.
  - Emit `simulation_result` + `sensitivity_curve`.
- **Complexity**: 10/10
- **Dependencies**: action plans (8.1/8.2), queue model extraction
- **Validation**:
  - `tests/plugins/test_queue_simulator_v1.py`

## Sprint 9: Shrinkage Ranking (Plugin 20) + Actionable Lever Synthesis
**Goal**: Make “top offenders” stable and convert new plugin findings into business-facing actionable levers.

**Demo/Validation**:
- Recommendations show structural actions (batch/orchestrate/decouple) before generic items.

### Task 9.1: `analysis_empirical_bayes_shrinkage_v1`
- **Description**:
  - Implement James–Stein style shrinkage for per-process estimates (median/p95 wait/duration).
  - Emit `shrinkage_ranked_driver` + `stable_priority_candidate`.
- **Complexity**: 7/10
- **Dependencies**: per-process metric aggregation utilities
- **Validation**:
  - `tests/plugins/test_empirical_bayes_shrinkage_v1.py`

### Task 9.2: Extend `analysis_actionable_ops_levers_v1` to Consume New Kinds
- **Location**: `src/statistic_harness/core/stat_plugins/topo_tda_addon.py`
- **Description**:
  - Add synthesis pass that reads prior plugin results (from `ctx.storage` using `ctx.run_id`) and converts:
    - `batch_refactor_candidate` -> `action_type=batch_input_refactor`
    - `orchestration_macro_candidate` -> `action_type=orchestrate_chain`
    - `decoupling_boundary_candidate` / `min_cut` -> `action_type=decouple_boundary`
  - Ensure plain-English recommendation strings and citeable evidence paths.
  - Update `plugins/analysis_actionable_ops_levers_v1/plugin.yaml` `depends_on` to ensure upstream plugins execute first (no cycles).
- **Complexity**: 9/10
- **Dependencies**: Sprints 3–6 outputs exist
- **Validation**:
  - `tests/plugins/test_actionable_ops_lever_synthesis_from_top20.py`

### Task 9.3: Update Recommendation Ranking/Tiering to Prefer Structural Levers
- **Location**: `src/statistic_harness/core/report.py`
- **Description**:
  - Add action-type prioritization caps (structural > targeted schedule > generic) so “payout-style” findings surface.
- **Complexity**: 6/10
- **Dependencies**: Task 9.2
- **Validation**:
  - `tests/test_recommendation_tiering_structural_first.py`

## Sprint 10: Docs/Matrices/Full Gauntlet
**Goal**: Keep coverage matrices and plugin functionality matrices up to date, and verify the full harness run produces actionable results.

**Demo/Validation**:
- `.venv/bin/python -m pytest -q` passes.
- Full run completes on the 500k dataset and produces `answers_recommendations.md` with structural actions.

### Task 10.1: Update/Regenerate Matrices
- **Location**: `docs/`, `scripts/*matrix.py`
- **Description**:
  - Regenerate:
    - `docs/implementation_matrix.*`
    - `docs/binding_implementation_matrix.*`
    - `docs/plugins_functionality_matrix.*`
    - `docs/plugin_data_access_matrix.*`
  - Add the new Top20 matrix outputs to “generated” exclusions (to avoid self-dependency loops).
- **Complexity**: 4/10
- **Dependencies**: plugin scaffolding exists
- **Validation**:
  - Existing tests `tests/test_docs_implementation_matrix.py` and `tests/test_binding_implementation_matrix.py`.

### Task 10.2: Full Run Smoke (Real Dataset)
- **Location**: `scripts/run_loaded_dataset_full.py` and watcher scripts
- **Description**:
  - Run full plugin suite and confirm:
    - new plugins execute or degrade cleanly with reasons
    - synthesized levers appear in `answers_recommendations.md`
- **Complexity**: 5/10
- **Dependencies**: all above
- **Validation**:
  - Review run artifacts under `appdata/runs/<run_id>/artifacts/`.

## Testing Strategy
- One focused synthetic test per plugin (small data; deterministic).
- Cross-cutting test for lever synthesis + recommendation tiering.
- Keep runtime manageable by:
  - hard caps on max patterns/clusters/rules/cuts
  - deterministic sampling where unavoidable (seed = run_seed)

## Potential Risks & Gotchas
- Heavy optional deps (Leiden/HDBSCAN/MIP solvers) may be painful in CI.
  - Mitigation: implement fallbacks and degrade cleanly without failing the run.
- Trace/case key may be absent, blocking sequence plugins.
  - Mitigation: degrade and point to missing linkage; use dependency columns or row adjacency as fallback if configured.
- Parameter tables may be sparse or missing.
  - Mitigation: fallback param parsing from raw text columns; explicitly mark reduced confidence.
- Runtime explosion on multi-million rows.
  - Mitigation: SQL-first aggregation to reduce to process-level and cohort-level feature tables.

## Rollback Plan
- Each plugin is independent; disable by removing from selected plugin set or via config gating.
- If lever synthesis causes regressions, ship it behind a setting toggle and default to off until validated.

## Questions (One Round)
1. For `analysis_action_search_mip_batched_scheduler_v1`, do you prefer a specific solver stack: `pulp` (CBC if available), OR-Tools, or “deterministic greedy fallback only” (no solver dependency)?
2. For Leiden/HDBSCAN, is “fallback to networkx greedy modularity / DBSCAN” acceptable when the preferred dependencies aren’t installed, as long as the plugin still produces citeable outputs?
