# Statistic Harness: 20 additional analytical-method plugins (research-backed)

This document proposes **20 new analysis plugins** to add to the Statistic Harness to surface *non-generic*, engineering-actionable opportunities (batch/multi-input refactors, dedupe/caching candidates, repeatable subflow macros, dependency-graph partitions, modeled levers, etc.).

It is tailored to the repo’s plugin contract:
- `plugin.yaml` fields (`id`, `name`, `version`, `type`, `entrypoint`, `depends_on`, `settings.defaults`, `capabilities`, `config_schema`, `output_schema`, `sandbox`) match existing plugins such as `profile_eventlog` and `analysis_ideaspace_*`.  
- Findings should include at least `{kind, measurement_type, evidence}` per `docs/report.schema.json` conventions; and the plugin output should include required keys `{status, summary, metrics, findings, artifacts, budget, error, references, debug}` (see existing `output.schema.json` files).

---

## Standard files to reuse (copy once)

### 1) Standard `output.schema.json` (reuse verbatim for every new plugin)
Most existing plugins use the same “Plugin Result” shape. Copy a known-good schema (e.g., from `plugins/analysis_busy_period_segmentation_v2/output.schema.json`) to each new plugin’s `output.schema.json`.

### 2) Minimal `plugin.py` wrapper (recommended pattern)
If you implement logic in `src/statistic_harness/core/stat_plugins/*` and register it (via `run_plugin(plugin_id, ctx)`), your `plugins/<id>/plugin.py` can stay tiny:

```python
from __future__ import annotations
from statistic_harness.core.stat_plugins.registry import run_plugin

class Plugin:
    def run(self, ctx):
        return run_plugin("<PLUGIN_ID>", ctx)
```

If you implement standalone, mirror `plugins/analysis_notears_linear/plugin.py` style: load the dataframe via `ctx.dataset_loader()` and emit artifacts via `ctx.artifacts_dir(plugin_id)`.

### 3) Sandbox defaults
Use the conservative sandbox used across the repo:

```yaml
sandbox:
  no_network: true
  fs_allowlist:
    - appdata
    - plugins
    - run_dir
```

---

## Summary: the 20 new plugins

| # | Plugin ID | Method | What it surfaces (non-generic) |
|---:|---|---|---|
| 1 | `analysis_param_near_duplicate_minhash_v1` | MinHash + LSH | Near-duplicate executions with similar params → **batch/multi-input** candidates |
| 2 | `analysis_param_near_duplicate_simhash_v1` | SimHash | Near-duplicate executions based on tokenized params/log text → batching or caching |
| 3 | `analysis_frequent_itemsets_fpgrowth_v1` | FP-Growth | Frequent parameter bundles → propose *API consolidation* / preset workflows |
| 4 | `analysis_association_rules_apriori_v1` | Apriori rules | “If param A then param B” rules → eliminate redundant runs / pre-join inputs |
| 5 | `analysis_sequential_patterns_prefixspan_v1` | PrefixSpan | Frequent process sequences → convert repeated chains into **one orchestrated job** |
| 6 | `analysis_sequence_grammar_sequitur_v1` | SEQUITUR | Repeated subsequences (“macros”) → consolidate into fewer pipeline steps |
| 7 | `analysis_biclustering_cheng_church_v1` | Biclustering | Coherent process×param submatrices → shared batch endpoints / shared caches |
| 8 | `analysis_density_clustering_hdbscan_v1` | HDBSCAN* | Variable-density clusters of similar runs → batchable groups without choosing k |
| 9 | `analysis_constrained_clustering_cop_kmeans_v1` | COP-KMeans | Clustering with must-link / cannot-link (use exclude-list & domain constraints) |
| 10 | `analysis_dependency_community_louvain_v1` | Louvain | Dependency communities → isolate subsystems; targeted scaling/refactor boundaries |
| 11 | `analysis_dependency_community_leiden_v1` | Leiden | Higher-quality communities than Louvain → more stable module boundaries |
| 12 | `analysis_similarity_graph_spectral_clustering_v1` | Spectral clustering | Cluster similarity graph of processes/params → coherent groups for batching |
| 13 | `analysis_graph_min_cut_partition_v1` | Min-cut (Stoer–Wagner) | Minimal “cut edges” between subsystems → where to decouple/queue-separate |
| 14 | `analysis_distribution_shift_wasserstein_v1` | Wasserstein distance | Quantify workload/latency distribution shift across windows → target levers |
| 15 | `analysis_burst_modeling_hawkes_v1` | Hawkes process | Self-exciting bursts → identify trigger processes and dampening interventions |
| 16 | `analysis_daily_pattern_alignment_dtw_v1` | DTW | Align daily patterns (close vs non-close) → measure “shape” similarity & anomalies |
| 17 | `analysis_action_search_simulated_annealing_v1` | Simulated annealing | Search *non-obvious* action combos that achieve target close window |
| 18 | `analysis_action_search_mip_batched_scheduler_v1` | MIP scheduling/batching | Find best subset/batching plan under constraints → concrete to-do list |
| 19 | `analysis_discrete_event_queue_simulator_v1` | Discrete-event simulation | Stress-test action plans; estimate deltas to hit 20–31 close window target |
| 20 | `analysis_empirical_bayes_shrinkage_v1` | Empirical Bayes / James–Stein | Stable ranking of “true worst” processes (de-noise medians/p95s) |

---

## Plugin specs (ready to paste)

Each spec includes:
- `plugin.yaml` (matches repo conventions)
- Suggested `config.schema.json` keys (plugin-specific; keep `additionalProperties: true`)
- Output conventions (finding `kind`s + minimal evidence expectations)
- Research references (URLs)

> Note: These plugins are intentionally oriented toward **engineering change opportunities** (batch endpoints, consolidation, orchestration macros, decoupling boundaries, modeled action plans), not generic “make it faster” advice.

---

### 1) `analysis_param_near_duplicate_minhash_v1` — MinHash + LSH near-duplicate detection

**Purpose**: Identify repeated executions of the same process whose params are *near-duplicates* (e.g., same keys, small value differences, similar filters), which is a strong signal that the job should accept **batch/multi-input** or a “list of IDs” parameter.

**Research**: MinHash was introduced for resemblance/near-duplicate detection and supports efficient similarity estimation and LSH bucketing.
- https://www.cs.princeton.edu/courses/archive/spring13/cos598C/broder97resemblance.pdf

**plugin.yaml**
```yaml
id: analysis_param_near_duplicate_minhash_v1
name: Param Near-Duplicate Detection (MinHash/LSH)
version: 0.1.0
type: analysis
entrypoint: plugin.py:Plugin
depends_on:
  - profile_eventlog
settings:
  description: Find near-duplicate executions by MinHash+LSH over tokenized params to propose batch/multi-input refactors.
  defaults:
    max_candidates: 100
    shingle_size: 3
    num_hashes: 128
    lsh_bands: 32
    min_cluster_size: 5
capabilities:
  - needs_eventlog
  - needs_text
config_schema: config.schema.json
output_schema: output.schema.json
sandbox:
  no_network: true
  fs_allowlist: [appdata, plugins, run_dir]
```

**Suggested `config.schema.json` keys**
- `process_column` (string|null)
- `params_column` (string|null) — raw params JSON / text
- `normalize_params` (boolean, default true) — canonicalize JSON, sort keys, drop volatile keys
- `ignore_param_keys_regex` (string|null) — drop known volatile keys (timestamps, request ids)
- `min_jaccard` (number, default 0.85)
- `min_cluster_size` (integer, default 5)
- `max_candidates` (integer, default 100)

**Findings**
- `kind: "batch_refactor_candidate"`
  - `measurement_type: "measured"`
  - `evidence.metrics`: `{cluster_size, median_runtime, p95_runtime, est_calls_reduced, est_runtime_saved_seconds}`
  - `where`: `{process, representative_params_hash}`
- Artifact: `minhash_clusters.json` (clusters + exemplars + row_ids)

---

### 2) `analysis_param_near_duplicate_simhash_v1` — SimHash near-duplicate detection

**Purpose**: Similar to (1), but for **high-dimensional sparse tokens** (free text params, log messages) using Hamming-distance similarity on SimHash fingerprints.

**Research**: SimHash is commonly used for near-duplicate detection at web scale.
- https://research.google.com/pubs/archive/33026.pdf

**plugin.yaml**
```yaml
id: analysis_param_near_duplicate_simhash_v1
name: Param Near-Duplicate Detection (SimHash)
version: 0.1.0
type: analysis
entrypoint: plugin.py:Plugin
depends_on:
  - profile_eventlog
settings:
  description: Find near-duplicate executions using SimHash over tokenized params/text; propose batch, cache, or dedupe changes.
  defaults:
    max_candidates: 100
    fingerprint_bits: 64
    max_hamming_distance: 3
    min_cluster_size: 5
capabilities:
  - needs_eventlog
  - needs_text
config_schema: config.schema.json
output_schema: output.schema.json
sandbox:
  no_network: true
  fs_allowlist: [appdata, plugins, run_dir]
```

**Suggested `config.schema.json` keys**
- `text_source_columns` (array of strings) — params/text fields to tokenize
- `tokenizer` (enum: `whitespace`, `json_keys_values`, `ngram`)
- `max_hamming_distance` (integer, default 3)
- `min_cluster_size` (integer, default 5)

**Findings**
- `kind: "near_duplicate_cluster"`
- `kind: "batch_refactor_candidate"` (only when clusters are within same process/module)

Artifact: `simhash_clusters.json`

---

### 3) `analysis_frequent_itemsets_fpgrowth_v1` — FP-Growth frequent itemsets

**Purpose**: Find frequently co-occurring parameter *sets* (e.g., same module + same flags + same period) indicating standard “recipes” that can be consolidated into preset jobs or combined into a single multi-query run.

**Research**
- https://www.cs.sfu.ca/~jpei/publications/sigmod00.pdf

**plugin.yaml**
```yaml
id: analysis_frequent_itemsets_fpgrowth_v1
name: Frequent Param Sets (FP-Growth)
version: 0.1.0
type: analysis
entrypoint: plugin.py:Plugin
depends_on:
  - profile_eventlog
settings:
  description: Mine frequent parameter itemsets (FP-Growth) to identify repeatable recipes and consolidation opportunities.
  defaults:
    min_support: 0.02
    max_itemset_size: 8
    max_rules: 200
capabilities:
  - needs_eventlog
  - needs_text
config_schema: config.schema.json
output_schema: output.schema.json
sandbox:
  no_network: true
  fs_allowlist: [appdata, plugins, run_dir]
```

**Suggested config keys**
- `min_support` (number) — relative or absolute support
- `item_encoding` (enum: `param_key`, `param_key_value_bucketed`)
- `max_itemset_size` (integer)

**Findings**
- `kind: "frequent_param_itemset"`
  - evidence: `{support, count, example_params}`
- `kind: "preset_job_candidate"`
  - evidence: `{estimated_adoption, run_count_covered}`

Artifact: `fpgrowth_itemsets.json`

---

### 4) `analysis_association_rules_apriori_v1` — Apriori association rules

**Purpose**: Produce **actionable conditional rules**, e.g. “whenever `module=X` and `period=close` then `flag=Y`”, highlighting redundancies and opportunities to **pre-join** inputs or remove repeated variants.

**Research**
- https://www.vldb.org/conf/1994/P487.PDF

**plugin.yaml**
```yaml
id: analysis_association_rules_apriori_v1
name: Association Rules (Apriori)
version: 0.1.0
type: analysis
entrypoint: plugin.py:Plugin
depends_on:
  - profile_eventlog
settings:
  description: Mine association rules over params to identify deterministic combinations and eliminate redundant variants.
  defaults:
    min_support: 0.02
    min_confidence: 0.7
    min_lift: 1.2
    max_rules: 200
capabilities:
  - needs_eventlog
  - needs_text
config_schema: config.schema.json
output_schema: output.schema.json
sandbox:
  no_network: true
  fs_allowlist: [appdata, plugins, run_dir]
```

**Findings**
- `kind: "association_rule"`
  - evidence: `{support, confidence, lift, antecedent, consequent}`
- `kind: "variant_consolidation_candidate"`
  - evidence: `{variant_count, rule_coverage}`

Artifact: `apriori_rules.json`

---

### 5) `analysis_sequential_patterns_prefixspan_v1` — PrefixSpan sequential pattern mining

**Purpose**: Detect frequent **process sequences** (per case_id / correlation_id) that repeatedly occur together, implying the flow should become a **single orchestrated job** (or a batch endpoint), not N separate runs.

**Research**
- https://hanj.cs.illinois.edu/pdf/span01.pdf

**plugin.yaml**
```yaml
id: analysis_sequential_patterns_prefixspan_v1
name: Frequent Sequences (PrefixSpan)
version: 0.1.0
type: analysis
entrypoint: plugin.py:Plugin
depends_on:
  - profile_eventlog
settings:
  description: Mine frequent sequences of activities to identify repeated chains suitable for orchestration/merging.
  defaults:
    min_support: 0.01
    max_pattern_length: 8
    max_patterns: 200
capabilities:
  - needs_eventlog
  - needs_timestamp
config_schema: config.schema.json
output_schema: output.schema.json
sandbox:
  no_network: true
  fs_allowlist: [appdata, plugins, run_dir]
```

**Findings**
- `kind: "frequent_sequence"`
  - evidence: `{support, count, pattern}`
- `kind: "orchestration_macro_candidate"`
  - evidence: `{estimated_roundtrips_reduced, affected_processes}`

Artifact: `prefixspan_patterns.json`

---

### 6) `analysis_sequence_grammar_sequitur_v1` — SEQUITUR grammar inference (“macro” discovery)

**Purpose**: Infer repeated subsequences and compress them into a grammar; repeated non-terminal rules are strong candidates for “macro steps” or orchestration endpoints.

**Research**
- https://arxiv.org/abs/cs/9709102

**plugin.yaml**
```yaml
id: analysis_sequence_grammar_sequitur_v1
name: Sequence Macros (SEQUITUR)
version: 0.1.0
type: analysis
entrypoint: plugin.py:Plugin
depends_on:
  - profile_eventlog
settings:
  description: Infer repeated subsequences via SEQUITUR; propose macro/orchestration refactors.
  defaults:
    max_rules: 200
    min_rule_uses: 5
capabilities:
  - needs_eventlog
config_schema: config.schema.json
output_schema: output.schema.json
sandbox:
  no_network: true
  fs_allowlist: [appdata, plugins, run_dir]
```

**Findings**
- `kind: "sequence_macro_rule"`
  - evidence: `{rule, uses, expansion_length}`
- `kind: "orchestration_macro_candidate"`
  - evidence: `{uses, estimated_roundtrips_reduced}`

Artifact: `sequitur_grammar.json`

---

### 7) `analysis_biclustering_cheng_church_v1` — Biclustering (process×param coherence)

**Purpose**: Identify coherent submatrices where a subset of processes and a subset of param features behave similarly (run times, queuing, outcomes). Useful for spotting shared refactor targets (common batch endpoint, shared cache layer).

**Research**
- https://cdn.aaai.org/ISMB/2000/ISMB00-010.pdf

**plugin.yaml**
```yaml
id: analysis_biclustering_cheng_church_v1
name: Biclustering (Cheng–Church)
version: 0.1.0
type: analysis
entrypoint: plugin.py:Plugin
depends_on:
  - profile_eventlog
settings:
  description: Find coherent biclusters across processes and parameter features to propose shared refactor/caching surfaces.
  defaults:
    max_biclusters: 50
    msr_threshold: 300.0
    min_rows: 5
    min_cols: 5
capabilities:
  - needs_eventlog
  - needs_numeric
config_schema: config.schema.json
output_schema: output.schema.json
sandbox:
  no_network: true
  fs_allowlist: [appdata, plugins, run_dir]
```

**Findings**
- `kind: "bicluster"`
  - evidence: `{row_count, col_count, msr, processes, features}`
- `kind: "shared_cache_candidate"`

Artifact: `biclusters.json`

---

### 8) `analysis_density_clustering_hdbscan_v1` — HDBSCAN* density clustering

**Purpose**: Cluster executions in parameter/runtime feature space without choosing k and while handling variable density. Clusters with tight param similarity and high run counts are candidates for batching or caching.

**Research**
- https://arxiv.org/abs/1705.07321  
- https://joss.theoj.org/papers/10.21105/joss.00205

**plugin.yaml**
```yaml
id: analysis_density_clustering_hdbscan_v1
name: Density Clustering (HDBSCAN)
version: 0.1.0
type: analysis
entrypoint: plugin.py:Plugin
depends_on:
  - profile_eventlog
settings:
  description: Use HDBSCAN to find dense clusters of similar executions (params+runtime) indicating batching or caching candidates.
  defaults:
    min_cluster_size: 5
    min_samples: 5
    max_clusters_reported: 50
capabilities:
  - needs_eventlog
  - needs_numeric
config_schema: config.schema.json
output_schema: output.schema.json
sandbox:
  no_network: true
  fs_allowlist: [appdata, plugins, run_dir]
```

**Findings**
- `kind: "dense_execution_cluster"`
- `kind: "batch_refactor_candidate"` (when cluster shares process/module)

Artifact: `hdbscan_clusters.json`

---

### 9) `analysis_constrained_clustering_cop_kmeans_v1` — COP-KMeans (constraints-aware clustering)

**Purpose**: Cluster while respecting constraints:
- **Cannot-link** excluded processes and “already accounted for” process families.
- **Must-link** known equivalent process variants (aliases).
This supports “batch candidate discovery” without false positives across excluded families.

**Research**
- https://www.cs.cmu.edu/~dgovinda/pdf/icml-2001.pdf

**plugin.yaml**
```yaml
id: analysis_constrained_clustering_cop_kmeans_v1
name: Constrained Clustering (COP-KMeans)
version: 0.1.0
type: analysis
entrypoint: plugin.py:Plugin
depends_on:
  - profile_eventlog
settings:
  description: Cluster executions with must-link/cannot-link constraints to align with exclude lists and known equivalences.
  defaults:
    k: 20
    max_iter: 100
    constraint_violation_policy: fail_cluster
capabilities:
  - needs_eventlog
  - needs_numeric
config_schema: config.schema.json
output_schema: output.schema.json
sandbox:
  no_network: true
  fs_allowlist: [appdata, plugins, run_dir]
```

**Findings**
- `kind: "constrained_cluster"`
- `kind: "batch_refactor_candidate"` (clusters that are constraint-valid and tight)

Artifact: `copkmeans_clusters.json`

---

### 10) `analysis_dependency_community_louvain_v1` — Louvain community detection

**Purpose**: Build a dependency graph (process ↔ downstream process, host ↔ process, queue ↔ process) and detect communities. Communities are stable candidates for:
- code ownership boundaries,
- separate queues/pools,
- targeted refactors,
- reducing cross-community coupling.

**Research**
- https://arxiv.org/abs/0803.0476

**plugin.yaml**
```yaml
id: analysis_dependency_community_louvain_v1
name: Dependency Communities (Louvain)
version: 0.1.0
type: analysis
entrypoint: plugin.py:Plugin
depends_on:
  - profile_eventlog
settings:
  description: Detect communities in the dependency graph to propose decoupling and refactor boundaries.
  defaults:
    edge_weight: calls
    max_communities_reported: 50
capabilities:
  - needs_eventlog
config_schema: config.schema.json
output_schema: output.schema.json
sandbox:
  no_network: true
  fs_allowlist: [appdata, plugins, run_dir]
```

**Findings**
- `kind: "dependency_community"`
  - evidence: `{nodes, edges, modularity, top_nodes}`
- `kind: "decoupling_boundary_candidate"`

Artifact: `louvain_communities.json`

---

### 11) `analysis_dependency_community_leiden_v1` — Leiden community detection

**Purpose**: Same as (10) but using Leiden, which provides better-connected communities than Louvain in many settings.

**Research**
- https://www.nature.com/articles/s41598-019-41695-z

**plugin.yaml**
```yaml
id: analysis_dependency_community_leiden_v1
name: Dependency Communities (Leiden)
version: 0.1.0
type: analysis
entrypoint: plugin.py:Plugin
depends_on:
  - profile_eventlog
settings:
  description: Detect high-quality dependency communities (Leiden) to propose stable module boundaries and decoupling actions.
  defaults:
    edge_weight: calls
    max_communities_reported: 50
capabilities:
  - needs_eventlog
config_schema: config.schema.json
output_schema: output.schema.json
sandbox:
  no_network: true
  fs_allowlist: [appdata, plugins, run_dir]
```

**Findings**
- `kind: "dependency_community"`
- `kind: "decoupling_boundary_candidate"`

Artifact: `leiden_communities.json`

---

### 12) `analysis_similarity_graph_spectral_clustering_v1` — Spectral clustering

**Purpose**: Build a similarity graph over processes or executions (based on param tokens, runtime quantiles, queue profile), then cluster via spectral methods. Useful when clusters are non-spherical / graph-structured.

**Research**
- https://snap.stanford.edu/class/cs224w-readings/ng01spectralcluster.pdf

**plugin.yaml**
```yaml
id: analysis_similarity_graph_spectral_clustering_v1
name: Similarity Graph Clustering (Spectral)
version: 0.1.0
type: analysis
entrypoint: plugin.py:Plugin
depends_on:
  - profile_eventlog
settings:
  description: Cluster similarity graphs of executions/processes using spectral clustering to find coherent batching/refactor groups.
  defaults:
    k: 20
    affinity: cosine
    max_clusters_reported: 50
capabilities:
  - needs_eventlog
  - needs_numeric
config_schema: config.schema.json
output_schema: output.schema.json
sandbox:
  no_network: true
  fs_allowlist: [appdata, plugins, run_dir]
```

**Findings**
- `kind: "spectral_cluster"`
- `kind: "batch_refactor_candidate"` (clusters of same process family)

Artifact: `spectral_clusters.json`

---

### 13) `analysis_graph_min_cut_partition_v1` — Minimum cut partitioning (Stoer–Wagner)

**Purpose**: Find a **minimum cut** in the dependency graph to surface the smallest set of edges responsible for most cross-subsystem coupling. This is a concrete “where to cut” candidate to reduce cascading queues.

**Research**
- https://www.semanticscholar.org/paper/A-simple-min-cut-algorithm-Stoer-Wagner/fec86cab0f490e94f62bfed00f6895db7296a9a1

**plugin.yaml**
```yaml
id: analysis_graph_min_cut_partition_v1
name: Graph Partition (Min-Cut)
version: 0.1.0
type: analysis
entrypoint: plugin.py:Plugin
depends_on:
  - profile_eventlog
settings:
  description: Compute min-cut partitions of dependency graphs to propose decoupling boundaries with minimal edge removals.
  defaults:
    graph_kind: process_dependency
    max_cuts_reported: 10
capabilities:
  - needs_eventlog
config_schema: config.schema.json
output_schema: output.schema.json
sandbox:
  no_network: true
  fs_allowlist: [appdata, plugins, run_dir]
```

**Findings**
- `kind: "min_cut"`
  - evidence: `{cut_weight, side_a_nodes, side_b_nodes, cut_edges}`
- `kind: "decoupling_boundary_candidate"`

Artifact: `mincut_partitions.json`

---

### 14) `analysis_distribution_shift_wasserstein_v1` — Optimal transport / Wasserstein distance

**Purpose**: Quantify distribution changes between windows (e.g., non-close vs close period; or “20–31 target window” vs “spillover”). Wasserstein distance captures *shape* changes and supports decomposable “what shifted” diagnostics.

**Research**
- https://www.stat.cmu.edu/~larry/%3Dsml/Opt.pdf

**plugin.yaml**
```yaml
id: analysis_distribution_shift_wasserstein_v1
name: Distribution Shift (Wasserstein)
version: 0.1.0
type: analysis
entrypoint: plugin.py:Plugin
depends_on:
  - profile_eventlog
  - analysis_close_cycle_window_resolver
settings:
  description: Use Wasserstein distance to quantify distribution shift (duration/queue) across windows and isolate drivers.
  defaults:
    metric_columns: [__queue_delay_sec, __duration_sec]
    max_groups: 50
capabilities:
  - needs_eventlog
  - needs_numeric
  - needs_timestamp
config_schema: config.schema.json
output_schema: output.schema.json
sandbox:
  no_network: true
  fs_allowlist: [appdata, plugins, run_dir]
```

**Findings**
- `kind: "wasserstein_shift"`
  - evidence: `{w1_distance, window_a, window_b, sample_sizes, top_contributing_processes}`
- `kind: "target_window_gap_driver"`

Artifact: `wasserstein_shift.json`

---

### 15) `analysis_burst_modeling_hawkes_v1` — Hawkes self-exciting burst model

**Purpose**: Model bursts where events trigger more events (self-excitation). This surfaces “trigger” processes/modules that cause downstream work amplification, and suggests interventions (cooldowns, batching at trigger, throttles).

**Research**
- https://academic.oup.com/biomet/article-abstract/58/1/83/224809  
- https://projecteuclid.org/journals/statistical-science/volume-33/issue-3/A-Review-of-Self-Exciting-Spatio-Temporal-Point-Processes-and/10.1214/17-STS629.pdf

**plugin.yaml**
```yaml
id: analysis_burst_modeling_hawkes_v1
name: Burst Modeling (Hawkes)
version: 0.1.0
type: analysis
entrypoint: plugin.py:Plugin
depends_on:
  - profile_eventlog
settings:
  description: Fit a Hawkes-style self-exciting model to event arrivals to find triggers and amplification paths.
  defaults:
    group_by: process
    max_groups: 50
    time_bin_seconds: 60
capabilities:
  - needs_eventlog
  - needs_timestamp
config_schema: config.schema.json
output_schema: output.schema.json
sandbox:
  no_network: true
  fs_allowlist: [appdata, plugins, run_dir]
```

**Findings**
- `kind: "hawkes_trigger"`
  - evidence: `{baseline_rate, excitation_strength, decay, top_trigger_processes}`
- `kind: "burst_dampening_candidate"`

Artifact: `hawkes_fit.json`

---

### 16) `analysis_daily_pattern_alignment_dtw_v1` — Dynamic Time Warping (DTW) pattern alignment

**Purpose**: Compare daily workload/queue patterns even when shifted/scaled in time (e.g., close window starts earlier/later). DTW helps detect when “close-day shape” looks like a shifted “non-close-day” vs truly different.

**Research**
- https://jeffe.cs.illinois.edu/teaching/compgeom/refs/Sakoe-Chiba-DTW.pdf

**plugin.yaml**
```yaml
id: analysis_daily_pattern_alignment_dtw_v1
name: Daily Pattern Alignment (DTW)
version: 0.1.0
type: analysis
entrypoint: plugin.py:Plugin
depends_on:
  - profile_eventlog
  - analysis_close_cycle_window_resolver
settings:
  description: Use DTW to align and compare daily arrival/queue patterns across days/windows; detect shape anomalies and shifts.
  defaults:
    time_bin_minutes: 5
    series: arrivals
    max_pairs: 50
capabilities:
  - needs_eventlog
  - needs_timestamp
config_schema: config.schema.json
output_schema: output.schema.json
sandbox:
  no_network: true
  fs_allowlist: [appdata, plugins, run_dir]
```

**Findings**
- `kind: "dtw_alignment"`
  - evidence: `{dtw_distance, shift_minutes, day_a, day_b}`
- `kind: "shape_anomaly_day"`

Artifact: `dtw_daily_alignment.json`

---

### 17) `analysis_action_search_simulated_annealing_v1` — Simulated annealing search over action combinations

**Purpose**: Avoid generic “schedule elsewhere” suggestions by directly searching a constrained action space (batching, concurrency caps, pre-stage, routing, etc.) to meet the target close window. Produces a ranked set of action bundles and their modeled deltas.

**Research**
- https://www2.stat.duke.edu/~scs/Courses/Stat376/Papers/TemperAnneal/KirkpatrickAnnealScience1983.pdf

**plugin.yaml**
```yaml
id: analysis_action_search_simulated_annealing_v1
name: Action Bundle Search (Simulated Annealing)
version: 0.1.0
type: analysis
entrypoint: plugin.py:Plugin
depends_on:
  - analysis_actionable_ops_levers_v1
  - analysis_ideaspace_normative_gap
settings:
  description: Search non-obvious combinations of actions that achieve the close-window target under constraints.
  defaults:
    iterations: 2000
    initial_temp: 1.0
    cooling: 0.995
    max_actions_per_bundle: 5
capabilities:
  - needs_eventlog
  - needs_timestamp
config_schema: config.schema.json
output_schema: output.schema.json
sandbox:
  no_network: true
  fs_allowlist: [appdata, plugins, run_dir]
```

**Findings**
- `kind: "action_bundle_candidate"`
  - measurement_type: `modeled`
  - evidence: `{bundle, objective_before, objective_after, expected_delta_seconds, constraint_checks}`

Artifact: `annealing_action_bundles.json`

---

### 18) `analysis_action_search_mip_batched_scheduler_v1` — Mixed Integer Programming for batching/scheduling

**Purpose**: Provide a concrete, constraint-aware plan (what to batch, what to run together, what to move) rather than generic scheduling advice. Even if solved approximately, the MIP formulation makes constraints explicit and outputs a reproducible plan.

**Research**
- https://tidel.mie.utoronto.ca/pubs/JSP_CandOR_2016.pdf  
- https://www.sciencedirect.com/science/article/abs/pii/S037722172200251X

**plugin.yaml**
```yaml
id: analysis_action_search_mip_batched_scheduler_v1
name: Action Plan Optimizer (MIP Batching/Scheduling)
version: 0.1.0
type: analysis
entrypoint: plugin.py:Plugin
depends_on:
  - analysis_actionable_ops_levers_v1
  - analysis_queue_delay_decomposition
settings:
  description: Build a batching/scheduling optimization problem (MIP) to meet the target window under resource constraints.
  defaults:
    time_limit_seconds: 30
    max_decisions: 200
    objective: minimize_spillover_minutes
capabilities:
  - needs_eventlog
  - needs_timestamp
config_schema: config.schema.json
output_schema: output.schema.json
sandbox:
  no_network: true
  fs_allowlist: [appdata, plugins, run_dir]
```

**Findings**
- `kind: "optimized_action_plan"`
  - measurement_type: `modeled`
  - evidence: `{plan, objective_value, constraints_satisfied, solver_status}`
- `kind: "batching_decision"` / `kind: "move_window_decision"`

Artifact: `mip_action_plan.json`

---

### 19) `analysis_discrete_event_queue_simulator_v1` — Discrete-event simulation of queue network

**Purpose**: Validate that proposed changes (batching, routing, concurrency caps) plausibly achieve the 20–31 close target given stochastic arrivals and service times; outputs sensitivity curves rather than a single number.

**Research**
- https://pavandm.wordpress.com/wp-content/uploads/2017/03/discrete-event-system-simulation-jerry-banks_2.pdf

**plugin.yaml**
```yaml
id: analysis_discrete_event_queue_simulator_v1
name: Queue Network Simulator (Discrete-Event)
version: 0.1.0
type: analysis
entrypoint: plugin.py:Plugin
depends_on:
  - analysis_queue_delay_decomposition
  - analysis_capacity_scaling
settings:
  description: Discrete-event simulation to estimate impacts of candidate actions on spillover and close-window completion.
  defaults:
    replications: 200
    horizon_days: 14
    random_seed: 123
capabilities:
  - needs_eventlog
  - needs_timestamp
config_schema: config.schema.json
output_schema: output.schema.json
sandbox:
  no_network: true
  fs_allowlist: [appdata, plugins, run_dir]
```

**Findings**
- `kind: "simulation_result"`
  - measurement_type: `modeled`
  - evidence: `{p50_spillover_minutes, p95_spillover_minutes, utilization, queue_length_stats}`
- `kind: "sensitivity_curve"`

Artifact: `queue_simulation_results.json`

---

### 20) `analysis_empirical_bayes_shrinkage_v1` — Empirical Bayes / James–Stein shrinkage

**Purpose**: Stabilize rankings of process drivers by shrinking noisy per-process estimates toward a global mean. This reduces “top offender churn” and makes recommendations more reliable when sample sizes vary.

**Research**
- https://utstat.toronto.edu/reid/sta2212s/2021/LSIChapter1.pdf  
- https://stat210a.berkeley.edu/fall-2024/reader/jamesstein.html

**plugin.yaml**
```yaml
id: analysis_empirical_bayes_shrinkage_v1
name: Empirical Bayes Shrinkage Ranking
version: 0.1.0
type: analysis
entrypoint: plugin.py:Plugin
depends_on:
  - analysis_attribution
settings:
  description: Apply empirical Bayes shrinkage to per-process estimates (median/p95 queue delay) to produce stable priority rankings.
  defaults:
    metric: __eligible_wait_sec_p95
    min_n: 30
    top_k: 50
capabilities:
  - needs_eventlog
  - needs_numeric
config_schema: config.schema.json
output_schema: output.schema.json
sandbox:
  no_network: true
  fs_allowlist: [appdata, plugins, run_dir]
```

**Findings**
- `kind: "shrinkage_ranked_driver"`
  - evidence: `{raw_estimate, shrunk_estimate, n, rank}`
- `kind: "stable_priority_candidate"`

Artifact: `shrinkage_rankings.json`

---

## Implementation notes (to help Codex CLI produce code quickly)

### A) Shared helper utilities worth adding (one-time)
- `core/tokenize_params.py`: canonicalize JSON-ish params; drop volatile keys; produce token sets.
- `core/similarity.py`: Jaccard, cosine, MinHash signatures, SimHash fingerprints, LSH bucketing.
- `core/graph_builders.py`: build dependency/similarity graphs from eventlog with deterministic edge weights.
- `core/pattern_mining.py`: FP-growth / Apriori / PrefixSpan wrappers (with deterministic caps and time budgets).
- `core/optimization_models.py`: objective definitions and constraint checks used by simulated annealing + MIP plugins.

### B) Output integration (recommended)
To surface these findings in business-facing summaries, extend:
- `analysis_actionable_ops_levers_v1` to consume artifacts/kinds like:
  - `batch_refactor_candidate`
  - `orchestration_macro_candidate`
  - `decoupling_boundary_candidate`
and convert them into `kind: actionable_ops_lever` with `action_type` set to new values (e.g., `batch_input_refactor`, `orchestrate_chain`, `decouple_boundary`).

This preserves one “CEO-grade lever” channel while still allowing detailed technical artifacts.

