# Codex Implementation Spec — OpenPlanter-style Cross‑Dataset Evidence Plugins (statistics_harness)

**Target repo:** `statistics_harness` (your repo)  
**Reference repo:** `shinmegamiboson/OpenPlanter` (3rd-party) — repomix: `repomix-output-ShinMegamiBoson-OpenPlanter.md`  
**Objective:** Add a plugin pack that ingests heterogeneous datasets (CSV/XLSX/SQL dumps), resolves entities across them via an explicit map + deterministic matchers, and surfaces *non-obvious* cross-dataset connections with evidence-backed findings.

---

## 0) Non-negotiables (4 pillars)

### Performance
- Must support very large datasets (observed up to ~11M records) by **streaming / chunking**. Never require full-DF load for unbounded scans.
- Prefer SQLite groupby/join scans (via SQL Assist) over pandas for large joins.

### Accuracy
- Explicit match strategies + confidence tiers + reproducible thresholds.
- Every “connection” must carry an evidence chain (row references + match features).

### Security
- No network access.
- SQL dump ingest must be fail-closed and restricted to safe statements (no ATTACH/PRAGMA, etc.). Prefer parsing + parameterized inserts instead of `executescript`.

### Citeability
- Every finding must include stable IDs and evidence references that allow tracing back to input rows.
- All produced artifacts must be registered (path + sha256 + producer plugin id) and included in report.

> Note: statistics_harness already enforces streaming-first dataset access for large datasets and provides `DatasetAccessor.iter_batches()` and a refusal guard for full loads. Integrations below must use `iter_batches()` for unbounded scans. (See `DatasetAccessor.iter_batches()` and the “Refusing to load full dataset…” guard.)

---

## 1) What we are importing (ideas from OpenPlanter)

OpenPlanter’s approach (as found in its scripts) centers on:

1. **Entity normalization for matching**  
   - Deterministic name normalization (uppercasing, suffix stripping, punctuation removal).  
   - “Aggressive” normalization that sorts unique tokens.

2. **Multi-strategy entity resolution** (confidence tiers)  
   - Exact normalized match (high confidence)  
   - Fuzzy match (token-sort ratio threshold) when available  
   - Token-overlap heuristic as a backstop (low confidence)

3. **Cross-link creation** between datasets  
   - Vendor ↔ donor-employer matches with match_type and confidence labels.
   - Joins from contributions to candidates + contracts to vendors.

4. **Bundling detection**  
   - Detect multiple donations from same employer on same day to same candidate.

5. **Timing analysis via permutation test**  
   - Are donations temporally clustered near contract awards more than chance?

6. **Evidence-first outputs**  
   - Produce multiple CSV/JSON outputs plus a file index (“evidence_file_index”) describing each artifact.

We will implement these as a coherent plugin pack that fits statistics_harness’ plugin contracts.

---

## 2) Plugin pack overview (new plugins)

This spec defines **9 new plugins**.

### Ingest
1. `ingest_sql_dump_v1`  
   Stream-parse SQL dumps into a dataset table (SQLite state DB) with strong allowlist semantics.

### Transform (entity map + linkage substrate)
2. `transform_entity_resolution_map_v1`  
   Build an entity map across multiple dataset versions and configured fields. Emits `entity_map.json` + `entity_aliases.csv`.

3. `transform_cross_dataset_link_graph_v1`  
   Use the entity map + matchers to emit a normalized **edge list** (link graph) with evidence chains: `cross_links.csv` + `cross_links.jsonl`.

### Analysis (non-obvious connections)
4. `analysis_bundled_donations_v1`  
   Bundling (same employer + same day + same candidate) from contributions-like datasets.

5. `analysis_contribution_limit_flags_v1`  
   Potential annual limit exceedance flags (configurable limit + grouping fields).

6. `analysis_vendor_influence_breadth_v1`  
   “Breadth” metrics: vendor → number of unique politicians/candidates reached, donation totals, etc.

7. `analysis_vendor_politician_timing_permutation_v1`  
   Timing permutation test per vendor↔politician pair (min donations threshold).

8. `analysis_red_flags_refined_v1`  
   Multi-factor flags combining (sole-source/limited competition), bundling, limit flags, timing, and breadth.

### Report
9. `report_evidence_index_v1`  
   Generate a single index artifact summarizing produced files + record counts + hashes; also add a concise report block with pointers.

---

## 3) Cross-dataset input model (how plugins find data)

Because statistics_harness runs a pipeline against **one** `dataset_version_id` at a time, cross-dataset plugins must accept **explicit dataset scopes** via config:

- `dataset_version_ids`: list of dataset_version_ids to consider
- per-dataset “roles”: e.g. `contracts`, `contributions`, `candidates`, etc.
- per-role field mappings: vendor name, employer, donor name, candidate id, dates, amounts, contract method.

This avoids core pipeline changes and keeps execution deterministic.

### Minimum config convention

All cross-dataset plugins in this pack accept:

```json
{
  "dataset_version_ids": ["dv_contracts", "dv_contribs", "dv_candidates"],
  "datasets": {
    "contracts": {"dataset_version_id": "dv_contracts"},
    "contributions": {"dataset_version_id": "dv_contribs"},
    "candidates": {"dataset_version_id": "dv_candidates"}
  }
}
```

Each plugin additionally requires role-specific column mappings.

---

## 4) Shared library code to add (core modules)

Add these new shared modules so plugins don’t re-implement brittle logic:

### 4.1 `src/statistic_harness/core/entity_resolution.py`
Responsibilities:
- `normalize_org_name(name: str) -> str`  
  Match OpenPlanter behavior: uppercase, remove quotes, strip suffixes, punctuation → spaces, collapse whitespace.
- `normalize_org_name_aggressive(name: str) -> str`  
  Normalize then sort unique tokens.
- `tokenize(norm: str) -> list[str]` (min token length)
- `build_token_inverted_index(rows_iter, name_field, *, min_token_len=4) -> dict[token, set[key]]`
- Optional rapidfuzz integration (if installed): `token_sort_ratio(a,b)`
- Deterministic candidate search:
  - Use token index to produce candidate set; then compute overlap score or fuzzy score.
  - Deterministic tie-breaker: (score desc, candidate_norm asc).

### 4.2 `src/statistic_harness/core/evidence_links.py`
Responsibilities:
- Stable row reference format:
  - `row_ref(dataset_version_id, row_index) -> "db://{dataset_version_id}#row_index={row_index}"`
- Evidence chain object builder:
  - `evidence_link(match_type, confidence, features, left_ref, right_ref, ...) -> dict`
- Artifact registry helper:
  - `register_artifact(ctx, path, description, mime) -> PluginArtifact`  
    plus sha256, record counts, and stable artifact id.

### 4.3 `src/statistic_harness/core/sql_dump_import.py`
Responsibilities:
- Stream parser for `INSERT INTO ... VALUES (...)` statements
- Allowlist:
  - `CREATE TABLE`, `INSERT INTO`
- Reject/ignore:
  - `DROP`, `ALTER`, `ATTACH`, `PRAGMA`, `VACUUM`, `CREATE VIEW`, `CREATE TRIGGER`, etc.
- Convert inserts into parameterized `executemany` into a target table.

Implementation detail:
- Must support chunk flush: accumulate N rows then bulk insert.
- Must tolerate very long lines by buffering until `;`.
- Must support quoted strings, escaped quotes, NULL, numeric literals.
- If dialect is incompatible (e.g., Postgres COPY), fail closed with actionable error.

---

## 5) New plugin specs (folder layout + manifests)

Each plugin lives in: `plugins/<plugin_id>/` with:
- `plugin.yaml`
- `config.schema.json`
- `output.schema.json`
- `plugin.py`
- `__init__.py`

The manifest must follow the repo’s schema and include `sandbox.no_network: true` and `fs_allowlist` including `run_dir` for artifact writes.

> Use existing plugin manifests as examples (they include `config_schema`, `output_schema`, and `fs_allowlist` entries).

### 5.1 `ingest_sql_dump_v1` (type: ingest)

**Purpose:** Import `.sql` dumps into the dataset store in a streaming/chunking way.

**Config keys**
- `input_path` (string; path to `.sql`)
- `table_name` (string; optional; default derived from dataset_version_id)
- `max_rows` (integer|null; optional)
- `chunk_rows` (integer; default 50_000)
- `encoding` (string; default "utf-8")
- `dialect` (string; enum ["sqlite_like","mysql_like","postgres_like"]; default "sqlite_like")  
  Used only to select minor parsing rules; must still be fail-closed.

**Artifacts**
- `canonical_import_manifest.json` (rows inserted, tables created, statement counts, hashes)
- `canonical.csv` (optional; only if explicitly requested due to size)

**Acceptance**
- Imports 11M-row dump without OOM.
- Creates dataset table and dataset_columns metadata; updates dataset_version stats.
- Deterministic: same input yields same dataset_version_id and same row_index ordering.

### 5.2 `transform_entity_resolution_map_v1` (type: transform)

**Purpose:** Build an entity map across dataset roles (vendors, employers, donors, etc.).

**Config keys**
- `datasets` object (role -> dataset_version_id)
- `fields` array of mapping specs:
  - `{ "role": "contracts", "field": "vendor_name1", "entity_type": "org", "key": "vendor" }`
  - `{ "role": "contributions", "field": "employer", "entity_type": "org", "key": "employer" }`
  - `{ "role": "contributions", "field": "donor_name", "entity_type": "org", "key": "donor_org", "when": {"record_type_in":["211"]} }`
- `match` object:
  - `use_rapidfuzz_if_available` (bool; default true)
  - `fuzzy_threshold` (int; default 82)
  - `min_token_len` (int; default 4)
  - `token_overlap_min_ratio` (number; default 0.6)
  - `min_overlap_tokens` (int; default 2)

**Core algorithm**
- Stream each configured dataset field to extract candidate strings + row refs.
- Normalize; aggregate into alias sets.
- Create a canonical `entity_id` via stable hashing:
  - `entity_id = sha256("entity:v1:"+entity_type+":"+normalized_name)[:16]`
- Emit:
  - `entity_map.json`: entities, aliases, per-alias sources (dataset_version_id + row refs)
  - `entity_aliases.csv`: flattened table for debugging

**Artifacts**
- `entity_map.json`
- `entity_aliases.csv`

**Findings**
- `kind: "entity_map_stats"` with counts of entities, aliases, and per-role coverage.

### 5.3 `transform_cross_dataset_link_graph_v1` (type: transform)

**Purpose:** Build cross-links (edges) between entities across datasets, with evidence.

**Config keys**
- `datasets`: roles + dataset ids
- `edges` array:
  - `{ "left": {"role":"contracts","field":"vendor_name1"}, "right": {"role":"contributions","field":"employer"}, "relation":"vendor_employer", "when": {"record_type_in":["201"]}}`
  - `{ "left": {"role":"contracts","field":"vendor_name1"}, "right": {"role":"contributions","field":"donor_name"}, "relation":"vendor_business_donor", "when": {"record_type_in":["211","202","203"]}}`
- `match` same as above (thresholds)
- `include_row_payload_excerpt` (bool; default false) — if true, include a few non-sensitive fields in evidence (still deterministic)

**Core algorithm (streaming + index)**
- Build an index for the “left” side entities (e.g., vendors):
  - normalized_name -> entity_id
  - token inverted index for candidate lookup
- Stream “right” side rows; for each:
  - normalize and attempt exact; else fuzzy/token overlap
  - emit edge record with:
    - `edge_id` stable hash of (relation, left_entity_id, right_entity_id, right_row_ref)
    - `match_type`, `confidence_tier`, match features (`fuzzy_score`, `overlap_ratio`, tokens)
    - row refs to both sides (where possible)
- Write edges incrementally to `cross_links.csv` and `cross_links.jsonl`

**Artifacts**
- `cross_links.csv`
- `cross_links.jsonl`
- `cross_link_summary.json`

**Findings**
- `kind: "cross_link_summary"` (counts by match_type/confidence, unique vendors, unique candidates)

### 5.4 `analysis_bundled_donations_v1` (type: analysis)

**Purpose:** Find potential bundles: ≥N donations from same employer on same day to same candidate.

**Config keys**
- `contributions_dataset_version_id`
- column mappings:
  - `employer`, `candidate_id` (or candidate name), `donation_date`, `amount`, `donor_name`
- `min_donors` (int; default 3)

**Algorithm**
- Stream contributions; normalize employer; group by (employer_norm, date, candidate_id)
- Maintain counts and sums in SQLite scratch table keyed by those fields (avoid in-memory dict for huge data)
- Output top bundles by total amount.

**Artifacts**
- `bundling_events.csv` (event rows)
- `bundling_events.json` (summary + top-N)

**Findings**
- `kind: "bundling_event"` for top K bundles; evidence includes group key and example row refs.

### 5.5 `analysis_contribution_limit_flags_v1` (type: analysis)

**Purpose:** Flag donors exceeding a configurable annual limit.

**Config keys**
- `contributions_dataset_version_id`
- `donor_id_fields` array (e.g. ["donor_last","donor_first","donor_address"] or a provided donor_id column)
- `amount_field`, `date_field`
- `year_extractor` ("from_date" default)
- `annual_limit` (number; default 1000.0)
- `min_excess` (number; default 0.0)

**Algorithm**
- Stream contributions; compute year; group by (donor_key, year)
- Aggregate sum(amount); emit flags where sum > limit + min_excess
- Output flags with evidence (row refs for a sample of contributing rows + counts).

**Artifacts**
- `contribution_limit_flags.csv`
- `contribution_limit_flags.json`

**Findings**
- `kind: "contribution_limit_flag"` for top K excesses.

### 5.6 `analysis_vendor_influence_breadth_v1` (type: analysis)

**Purpose:** Breadth metrics from the cross-link graph:
- For each vendor entity: number of unique candidates reached, total donations, count of donors, etc.

**Config keys**
- `cross_links_path` (optional; if omitted, locate artifact from `transform_cross_dataset_link_graph_v1` via run_dir)
- `vendor_entity_type` (default "org")
- `top_k` (default 50)

**Algorithm**
- Stream edges; aggregate per vendor entity.
- Output summary table.

**Artifacts**
- `shared_donor_networks.csv` (OpenPlanter naming)
- `vendor_influence_breadth.json`

**Findings**
- `kind: "vendor_influence_breadth"` for top K vendors by breadth and/or total.

### 5.7 `analysis_vendor_politician_timing_permutation_v1` (type: analysis)

**Purpose:** Per vendor↔politician pair, test whether donation dates cluster unusually close to contract award dates.

**Config keys**
- `contracts_dataset_version_id`
- `contributions_dataset_version_id`
- `vendor_field`, `award_date_field` (contracts)
- `candidate_id_field`, `donation_date_field`, `amount_field` (contribs)
- `min_donations` (int; default 3)
- `n_permutations` (int; default 2000)
- `rng_seed` (int; default 0 => derive from run_seed + stable hash)
- `time_window_days` (optional; if set, only consider awards within +/- window)

**Algorithm**
- Use a deterministic RNG: `np.random.default_rng(seed)`.
- For each vendor↔candidate pair:
  - observed statistic: mean absolute days from each donation to nearest award date
  - permutation: sample random award dates uniformly between min/max observed dates (as OpenPlanter does)
  - p-value: fraction where perm_mean <= observed_mean
  - effect size: (null_mean - observed_mean) / null_std
- Streaming:
  - Build vendor→award_dates map from contracts (can be large; store in SQLite scratch table with indexes)
  - Stream cross-links or contributions to group donation dates by vendor↔candidate.

**Artifacts**
- `politician_timing_analysis.json` (OpenPlanter naming)
- `vendor_politician_timing.csv`

**Findings**
- `kind: "timing_clustering_signal"` for top K strongest effects (lowest p).

### 5.8 `analysis_red_flags_refined_v1` (type: analysis)

**Purpose:** Combine signals into multi-factor red flags.

**Inputs (artifacts)**
- `cross_links.csv/jsonl`
- `bundling_events.csv`
- `contribution_limit_flags.csv`
- `timing` outputs
- (optional) contract method fields (sole source / limited competition)

**Config keys**
- thresholds:
  - `sole_source_methods` array (default ["Sole Source","Limited Competition","Emergency","Exempt"])
  - `min_total_contract_value_for_flag`
  - `min_total_donations_for_flag`
  - `max_p_value_for_timing_flag`
  - `min_effect_size_for_timing_flag`

**Output**
- `red_flags_refined.csv` with a row per flagged vendor/candidate combo and an evidence chain.

**Findings**
- `kind: "multi_factor_red_flag"` for top K flags; evidence includes contributing signal refs.

### 5.9 `report_evidence_index_v1` (type: report)

**Purpose:** Generate an evidence index similar to OpenPlanter’s `evidence_file_index` list, but consistent with statistics_harness report schema.

**Behavior**
- Scan `run_dir` for artifacts from this plugin pack, count records where feasible, compute sha256.
- Emit:
  - `evidence_index.json`
  - (optional) `evidence_index.md` short summary

**Findings**
- `kind: "evidence_index"` referencing the evidence_index artifact.

---

## 6) Output schema + finding discipline (how to stay schema-compliant)

Every plugin output must include at minimum:
- `status`, `summary`, `metrics`, `findings`, `artifacts`, `budget`, `error`, `references`, `debug`
(as seen in existing plugin output schemas).

### Stable IDs
Use the repo’s stable finding ID helper (e.g., `claim_id(...)`) to create deterministic finding IDs for:
- entity map stats
- cross link summaries
- top bundling events
- top limit flags
- timing signals
- multi-factor flags

### Evidence objects
For every finding:
- include `dataset_version_id` and `row_index` references where available (use `db://...` row_ref format)
- include match features (fuzzy score, overlap ratio, etc.)
- include the artifact path and a row number / key inside that artifact, when relevant

---

## 7) Performance design (streaming + SQLite scratch)

### General pattern
- For unbounded scans: **always** use `DatasetAccessor.iter_batches(batch_size=...)`.
- Avoid giant in-memory dicts when cardinality may be large; instead:
  - create scratch tables keyed by grouping dimensions,
  - `INSERT OR IGNORE`/`UPDATE` aggregates,
  - use indexed lookups.

### Recommended scratch DB
- Create `run_dir/scratch/<plugin_id>.sqlite` per plugin (or one shared scratch.sqlite with plugin-prefixed tables).
- Never write to normalized/template tables; only to scratch or plugin-owned tables.

---

## 8) Tests (must be added)

Add `pytest` coverage sufficient to keep regressions from shipping.

### Unit tests
- `test_sql_dump_import_parser_*`:
  - parses INSERT statements correctly (strings, NULLs, escapes)
  - rejects disallowed statements
  - chunk flush inserts correct row counts
- `test_entity_normalization_*`:
  - suffix stripping matches spec
  - deterministic normalization (same input => same output)
- `test_matcher_determinism_*`:
  - tie-break determinism (stable ordering)

### Integration tests
- Minimal synthetic datasets:
  - contracts + contributions with known vendor/employer matches
  - validate cross_links.csv has expected rows and match_types
  - validate bundling detection finds the known bundle
  - validate limit flags for known over-limit donor
  - validate timing permutation produces deterministic p-values given seed

### “Fail closed” tests
- ingest_sql_dump_v1 refuses `ATTACH DATABASE` and `PRAGMA` statements.
- cross-dataset plugins error with actionable message if required dataset roles/fields missing.

---

## 9) Suggested defaults (balanced 4 pillars)

- `batch_size` (iter_batches): 100_000
- `chunk_rows` (sql dump ingest): 50_000
- `fuzzy_threshold`: 82
- `min_token_len`: 4
- `token_overlap_min_ratio`: 0.6
- `min_overlap_tokens`: 2
- `min_donations` for timing: 3
- `n_permutations`: 2000 (increase if you want tighter p-value resolution)

---

## 10) 20 actionable repo improvements (inspired by OpenPlanter)

These are improvements to statistics_harness specifically, framed as implementable work items for this pack.

Each item includes:
- Improved pillars
- Risked pillars
- Enforcement location (code path)
- Regression detection (test(s) + **ANY_REGRESS => DO_NOT_SHIP**)

> NOTE: Items 1–12 are required for correctness and scalability; 13–20 are quality upgrades.

| # | Recommendation | Improved | Risked | Enforcement location | Regression detection |
|---:|---|---|---|---|---|
| 1 | Add `sql_dump_import.py` allowlist parser (no executescript) | Perf, Sec, Acc, Cite | Perf (parser overhead) | `core/sql_dump_import.py`, `ingest_sql_dump_v1` | parser denylist tests; if allowed stmt slips => DO_NOT_SHIP |
| 2 | Standardize row references (`db://{dvid}#row_index=...`) and reuse everywhere | Cite, Acc | None | `core/evidence_links.py` | unit test for formatting + roundtrip |
| 3 | Ensure every link has `match_type` + `confidence_tier` + features | Acc, Cite | None | `transform_cross_dataset_link_graph_v1` | schema validation + golden fixture |
| 4 | Use token inverted index to avoid O(N*M) fuzzy matching | Perf, Acc | Acc (missed matches if too strict) | `core/entity_resolution.py` | perf test on synthetic large token sets |
| 5 | Deterministic tie-break rules for matcher candidates | Acc, Cite | None | `core/entity_resolution.py` | determinism test with ties |
| 6 | Persist `cross_links.jsonl` (streamable) in addition to CSV | Perf, Cite | None | `transform_cross_dataset_link_graph_v1` | record-count parity test |
| 7 | Add scratch-db aggregate patterns for bundling + limits | Perf | None | `analysis_*` plugins | streaming integration tests |
| 8 | Deterministic RNG seeding for permutation tests | Acc, Cite | None | `analysis_vendor_politician_timing_permutation_v1` | fixed-seed snapshot test |
| 9 | Store artifact hashes + record counts in `evidence_index.json` | Cite, Sec | None | `report_evidence_index_v1` | hash mismatch test => DO_NOT_SHIP |
| 10 | Emit `cross_link_summary.json` with breakdown stats (match_type/confidence) | Cite, Acc | None | `transform_cross_dataset_link_graph_v1` | fixture comparison |
| 11 | Require config schemas for role/field mappings; fail closed if missing | Sec, Acc | UX friction | each plugin config.schema.json | schema validation tests |
| 12 | Add per-signal “modeled vs measured” tagging in findings (timing is modeled) | Acc, Cite | None | plugin findings | report validation test |

| 13 | Optional rapidfuzz acceleration; deterministic fallback when absent | Perf | Acc (slightly different scores) | `core/entity_resolution.py` | test that fallback still matches exact cases |
| 14 | Add optional “address-based” matcher hook (if dataset includes address) | Acc | Perf | `transform_cross_dataset_link_graph_v1` | unit tests on address parsing |
| 15 | Add vendor “procurement method” normalization and canonical enum | Acc, Cite | None | `analysis_red_flags_refined_v1` | enum validation tests |
| 16 | Add “explain” mode that emits per-edge evidence chains for top findings | Cite | Perf | `analysis_red_flags_refined_v1` | bounded output test |
| 17 | Add “entity alias provenance” (which row contributed the alias) | Cite, Acc | Disk use | `transform_entity_resolution_map_v1` | artifact size + record count test |
| 18 | Add stable “edge_id” and reference it from downstream analyses | Cite | None | `transform_cross_dataset_link_graph_v1` | stability test vs ordering changes |
| 19 | Add a small CLI helper to run the full cross-dataset pack in one command | Perf (workflow), Acc | None | `cli.py` | CLI smoke test |
| 20 | Add end-to-end golden fixture run for this pack (tiny data) | Acc, Cite | CI time | `tests/fixtures/openplanter_pack/*` | `pytest` gate => DO_NOT_SHIP |

---

## 11) Concrete file skeletons (Codex should generate)

For each plugin folder, Codex must generate:
- `plugin.yaml` (conforming to `docs/plugin_manifest.schema.json`)
- `config.schema.json` and `output.schema.json`
- `plugin.py` implementing `run(ctx: PluginContext) -> PluginResult`

### Minimal `output.schema.json` template
Use the standard required keys (already used by existing plugins):

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "additionalProperties": true,
  "required": ["status","summary","metrics","findings","artifacts","budget","error","references","debug"],
  "properties": {
    "status": {"type":"string"},
    "summary": {"type":"string"},
    "metrics": {"type":"object"},
    "findings": {"type":"array"},
    "artifacts": {"type":"array"},
    "budget": {"type":"object"},
    "error": {"type":["object","null"]},
    "references": {"type":"array"},
    "debug": {"type":"object"}
  }
}
```

---

## 12) Determinism checklist (Codex must obey)

- Stable ordering:
  - sort keys and rows when writing JSON/CSV summaries (except JSONL streams which are inherently ordered by processing order).
- Stable hashing:
  - entity_id, edge_id, finding_id must be based on stable normalized strings + fixed prefixes.
- Fixed RNG:
  - permutation tests must use `default_rng(seed)` with derived stable seed when `rng_seed==0`.
- Fail closed:
  - unknown columns / missing datasets / unsafe SQL statements => clear error and `status="error"`.

---

## 13) Example “pack run” (documentation only)

This is *not* a requirement for Codex to implement, but the end-state should allow a user to:

1) Ingest contracts CSV as `dv_contracts` (existing `ingest_tabular`)  
2) Ingest contributions CSV/TSV as `dv_contribs` (existing `ingest_tabular`)  
3) Ingest candidates CSV/TSV as `dv_candidates` (existing `ingest_tabular`)  
4) Run:
- `transform_entity_resolution_map_v1`
- `transform_cross_dataset_link_graph_v1`
- `analysis_bundled_donations_v1`
- `analysis_contribution_limit_flags_v1`
- `analysis_vendor_influence_breadth_v1`
- `analysis_vendor_politician_timing_permutation_v1`
- `analysis_red_flags_refined_v1`
- `report_evidence_index_v1`

All without full-DF loads on large datasets.

---

## Appendix A: Where OpenPlanter logic is mirrored vs inferred

### Mirrored (directly supported by OpenPlanter scripts)
- Normalization rules, fuzzy threshold concept, and match strategy stack (exact → fuzzy → token overlap)
- Bundling definition (same employer + same day + same candidate)
- Permutation test structure (uniform random award dates over observed interval)

### Inferred / generalized
- Politician affinity network construction (OpenPlanter references `politician_shared_network.json` but generator is not present in the repo content we received); we implement breadth metrics and keep network projection optional.
- “Refined red flags” exact scoring formula: implement as a transparent, configurable rule set.
