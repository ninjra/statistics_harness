# Kona-style Energy Ideaspace Architecture for Statistics_Harness

## 0) Goal

Add a **Kona-style “energy-based verifier” layer** on top of the existing **ideaspace** pipeline so the system can:

1. Define **what “correct” looks like** (ideal ideaspace / reference frontier).
2. Compute a deterministic **energy score** (distance-from-ideal + constraint penalties).
3. Rank and select **actions** (levers / changes) by expected **energy reduction**, not by heuristic confidence alone.
4. Surface the 3 known issues as first-class actions:
   - **QEMAIL scheduling reduction**
   - **QPEC(+1) / capacity reduction** (reduce queue delay via capacity)
   - **Payout report: batch/multi-input** (batch run + aggregated reporting)

This document is written as an implementation spec for **Codex CLI**: it enumerates repo edits, new plugin IDs, schemas, algorithms, and test fixtures.

---

## 1) Current repo reality (what already exists)

### 1.1 Plugin runtime architecture

- Plugins live under `plugins/<plugin_id>/` with:
  - `plugin.yaml` (id, type, depends_on, sandbox)
  - `plugin.py` (often a thin wrapper calling `stat_plugins.registry.run_plugin`)
  - optional `config.schema.json`, `output.schema.json`
- The **pipeline** compiles a dependency graph from `depends_on` and runs plugins in subprocess isolation.

Key code paths:
- `src/statistic_harness/core/plugin_manager.py` (loads plugin manifests + schemas)
- `src/statistic_harness/core/plugin_runner.py` (subprocess runner + results)
- `src/statistic_harness/core/stat_plugins/registry.py` (maps plugin_id -> handler)
- `src/statistic_harness/core/types.py` (`PluginResult`, `PluginArtifact`)

### 1.2 Existing “ideaspace” plugins (already in repo)

Already implemented handlers:
- `analysis_ideaspace_normative_gap`  
  Builds ideaspace vectors, selects ideal (baseline or Pareto frontier), emits “gap” findings.
- `analysis_ideaspace_action_planner`  
  Uses `core/lever_library.py` to generate deterministic action recommendations from evidence gates.

Key code paths:
- `src/statistic_harness/core/stat_plugins/ideaspace.py`
- `src/statistic_harness/core/ideaspace_feature_extractor.py`
- `src/statistic_harness/core/lever_library.py`
- `docs/ideaspace_baseline.schema.json`
- `scripts/build_ideaspace_baseline.py`

### 1.3 Existing capacity/scheduling recommendation building blocks

- `analysis_capacity_scaling` (eligible-wait scaling model; “add one server” style outputs)
- `analysis_queue_delay_decomposition` (per-process eligible wait > threshold, plus modeled reductions for host_count+1)

There is also downstream recommendation synthesis/dedupe:
- `plugins/analysis_recommendation_dedupe_v2/`
- `src/statistic_harness/core/report_v2_utils.py` filters exclusions; currently too blunt for QEMAIL/QPEC.

---

## 2) Target behavior contract (“correct looks like”)

### 2.1 Minimal user-facing correctness contract

A run against an ERP process log MUST output:

1. **Energy summary**:
   - `energy_total` (scalar)
   - top contributing terms (e.g., queue_delay_p95 gap, eligible_wait_over_threshold gap)
   - “ideal mode” used: baseline vs frontier
2. **Verified action list** (ranked):
   - action text, target, expected delta (in hours or percent), confidence, and risks
   - explicit mapping to enforcement location (what system is changed)
3. **The 3 known issues appear when evidence exists**:
   - QEMAIL schedule tuning lever (frequency reduction and/or reschedule window)
   - QPEC +1 / capacity scaling lever (explicitly labeled for the QPEC cluster)
   - payout report supports multiple datasets/batches per invocation

### 2.2 Hard invariants (fail-closed)

- No nondeterministic ordering: ties must be broken by stable IDs.
- No raw PII in artifacts by default:
  - If `USER_ID` or freeform params are present, only emit hashed/redacted tokens.
- “No evidence” must be represented explicitly:
  - If a known issue is not supported by evidence, emit `skipped` with a structured gating reason.

---

## 3) Kona-style layer: Energy model + verifier

### 3.1 Concepts mapped to this repo

- **State**: a compact, deterministic feature vector derived from the ERP log  
  (use the ideaspace feature extractor + a small ERP-specific extension vector).
- **Ideal**: either:
  - a signed baseline (`docs/ideaspace_baseline.schema.json` + `scripts/build_ideaspace_baseline.py`), OR
  - an in-dataset Pareto frontier (already implemented for ideaspace gap).
- **Energy**: a deterministic scalar score measuring “distance from correct”:
  - weighted KPI gaps + penalties for constraint violations
- **Verifier**: selects/ranks actions by **predicted energy reduction** while respecting constraints.

### 3.2 Energy definition (deterministic, inspectable)

Let:
- `x` = observed feature vector
- `x*` = ideal feature vector
- `w_i` = nonnegative weights (baseline-provided or defaults)

For each minimize-metric `m` (e.g., duration_p95, queue_delay_p95):
- `gap_i = max(0, (x_i - x*_i) / max(|x*_i|, eps))`

For each maximize-metric `m` (e.g., rate_per_min):
- `gap_i = max(0, (x*_i - x_i) / max(|x*_i|, eps))`

Energy:
- `E_gap = Σ_i w_i * gap_i^2`
- `E_constraints = Σ_j penalty_j` (large constants; see below)
- `E_total = E_gap + E_constraints`

Constraint penalties (examples):
- missing time columns: +1000
- negative durations > small tolerance: +200
- parse rate below threshold for required timestamps: +200
- “action excluded by policy” (e.g., trying to remove a close-critical process): +1000

### 3.3 Action scoring

Each candidate action `a` provides:
- an estimated delta on a subset of features: `Δ_a`
- a modeled next-state: `x' = clamp(x + Δ_a)`
- score: `ΔE(a) = E(x) - E(x')`

Rank by:
1. largest `ΔE(a)` (descending)
2. highest confidence (descending)
3. stable `lever_id` (ascending)

---

## 4) New plugins to add

### 4.1 Plugin: `analysis_ideaspace_energy_ebm_v1`

**Type**: `analysis`  
**Depends on**: `analysis_ideaspace_normative_gap` (optional), `profile_basic` (optional)  
**Purpose**: compute `E_total` and a term-level breakdown for the current run.

#### Inputs
- dataset (df)
- `analysis_ideaspace_normative_gap/artifacts/entities_table.json` if present
- optional baseline file path via settings

#### Outputs
- Findings:
  - `kind = "ideaspace_energy"`
  - `energy_total`, `energy_gap`, `energy_constraints`
  - `top_terms`: `[{"metric": "...", "gap": 0.0, "weight": 1.0, "contribution": 0.0}]`
  - `ideal_mode`: `"baseline"` | `"frontier"`
- Artifacts:
  - `energy_breakdown.json` (full term list + configuration)
  - `energy_state_vector.json` (observed + ideal vectors used)

#### Config schema (add as `plugins/analysis_ideaspace_energy_ebm_v1/config.schema.json`)
- `baseline_path` (string, optional)
- `weights` (object, optional overrides)
- `constraint_penalties` (object with numeric values, optional)

#### Algorithm notes
- Use the same column inference path as ideaspace plugins (`infer_columns` + `pick_columns`).
- If normative gap artifact exists, reuse its `ideal_mode` and `ideal` vectors per entity where possible.
- Emit at least one finding even if only constraints can be evaluated.

#### Performance
- O(n) aggregation; no heavy pairwise computations.

---

### 4.2 Plugin: `analysis_ebm_action_verifier_v1`

**Type**: `analysis`  
**Depends on**:
- `analysis_ideaspace_action_planner`
- `analysis_ideaspace_energy_ebm_v1`

**Purpose**: “Kona layer” that verifies and re-ranks candidate actions by energy reduction and policy constraints.

#### Inputs
- Candidate actions from:
  - `artifacts/analysis_ideaspace_action_planner/recommendations.json`
- Energy vector and weights from:
  - `artifacts/analysis_ideaspace_energy_ebm_v1/energy_state_vector.json`

#### Outputs
- Findings:
  - `kind = "verified_action"`
  - `action`, `lever_id`, `target`
  - `delta_energy`, `energy_before`, `energy_after`
  - `confidence`, `constraints_passed` (bool), `blocked_reason` (optional)
- Artifacts:
  - `verified_actions.json` (full ranked list)
  - `blocked_actions.json` (candidates rejected by constraints)

#### Determinism rules
- No randomness.
- Stable sorting (`delta_energy` desc, `confidence` desc, `lever_id` asc).

---

### 4.3 ERP-specific lever additions (implemented in `core/lever_library.py`)

These are *not* separate plugins; they are new **levers** surfaced via `analysis_ideaspace_action_planner` and then scored by the verifier.

#### Lever A: `tune_schedule_qemail_frequency_v1`  (QEMAIL scheduling reduction)

Trigger (evidence gate):
- Detect `PROCESS_ID == "QEMAIL"` (case-insensitive) AND
- infer schedule interval:
  - sort by queue/start timestamp
  - median inter-arrival ≤ 6 minutes AND
  - enough samples (≥ 500)

Action:
- Recommend increasing the interval (default proposal: 5 → 15 minutes)
- Optional: “close-cycle only” schedule reduction when close window resolver exists

Modeled impact (conservative):
- Reduce `rate_per_min` for QEMAIL by factor `k` (e.g., 3× reduction)
- Reduce a generic “background overhead” term (new feature) proportionally
- Confidence: 0.65 if triggers met; higher only if there is measured slowdown correlation.

Artifacts/evidence:
- Include measured interval stats and sample counts.

#### Lever B: `add_qpec_capacity_plus_one_v1` (QPEC(+1) / capacity reduction)

Trigger:
- Host column inferred (LOCAL_MACHINE_ID / ASSIGNED_MACHINE_ID) AND
- `host_count >= 1` AND
- eligible wait (start - eligible/queue) has p95 above threshold (configurable)

Action:
- “Add one QPEC server (QPEC+1)” or “increase QPEC workers by 1”
- Use capacity scaling model: modeled_wait = baseline_wait * host_count/(host_count+1)

Notes:
- Label should be explicit “QPEC+1” if host names match `/qpec/i` to align UI + dedupe logic.

#### Lever C: `batch_payout_report_inputs_v1` (multi-input payout report)

This is implemented primarily as a **report plugin + CLI** (next section), but you may also add an ideaspace lever that detects:
- payout-like process IDs (config list, default includes JBPREPAY, ADPPAYSTAT, PAY*)
- high parameter variety on pay period / batch keys

Action:
- “Batch payout report generation; run multiple pay periods/batches in a single invocation.”

---

### 4.4 Report plugin + CLI: `report_payout_report_v1` (batch / multi input)

**Type**: `report`  
**Depends on**: analysis stage complete OR at least `ingest_tabular`

**Purpose**: produce a payout-focused report that can merge multiple inputs in one run.

#### Implementation approach

Add a new CLI command in `src/statistic_harness/core/cli.py`:

- `stat-harness payout-report --input file1.csv --input file2.csv ... --out out_dir`

It should:
1. Create a single run directory.
2. Ingest each input as a dataset version (or treat them as “attachments”).
3. Run the pipeline for each input OR stack them into one dataset with `__source_*` columns.
4. Generate consolidated payout report artifacts:
   - `artifacts/report_payout_report_v1/payout_report.csv`
   - `artifacts/report_payout_report_v1/payout_report.json`

Minimal MVP (fastest):
- Stack datasets into one frame with:
  - `__source_file` (basename)
  - `__source_hash` (sha256 of file)
- Then compute payout KPIs per source and overall.

Privacy:
- Never emit raw USER_ID or raw PARAM_DESCR_LIST; emit hashed tokens or derived keys only.

---

## 5) Required repo edits (Codex checklist)

### 5.1 New plugin directories

Create:
- `plugins/analysis_ideaspace_energy_ebm_v1/`
- `plugins/analysis_ebm_action_verifier_v1/`
- `plugins/report_payout_report_v1/`

Each needs:
- `plugin.yaml`
- `plugin.py` (stat wrapper calling `run_plugin("<id>", ctx)`)
- `config.schema.json` + `output.schema.json` (at least minimal)

### 5.2 Register new handlers

Edit:
- `src/statistic_harness/core/stat_plugins/ideaspace.py`
  - add handler functions:
    - `_ideaspace_energy_ebm_v1`
    - `_ebm_action_verifier_v1`
  - update `HANDLERS` dict

No changes required in `registry.py` beyond re-importing handlers if needed.

### 5.3 Extend lever library

Edit:
- `src/statistic_harness/core/lever_library.py`
  - Add 2 levers:
    - QEMAIL schedule frequency tuning
    - QPEC+1 capacity scaling
  - Ensure all triggers are deterministic and have minimum evidence gates.

### 5.4 Fix recommendation filtering so QEMAIL/QPEC actions can surface

Edit:
- `src/statistic_harness/core/report_v2_utils.py`

Problem:
- Process-based exclusion is too blunt; it can drop schedule/capacity actions for QEMAIL/QPEC.

Solution:
- Change `filter_excluded_processes()` to:
  - exclude by process **only** for certain action types (e.g., “remove_process”)
  - allow `tune_schedule` and `add_server` actions even if target is in the excluded list

Also ensure:
- `analysis_recommendation_dedupe_v2` action_type inference recognizes:
  - “QPEC+1” as `add_server`
  - “frequency” / “every 5 minutes” as `tune_schedule`

### 5.5 Add payout-report batch CLI

Edit:
- `src/statistic_harness/core/cli.py`
  - add `cmd_payout_report`
  - parse repeated `--input` args
  - reuse `Pipeline.run()` per dataset OR stack into one dataset then run once

---

## 6) Test + regression plan (ANY_REGRESS=>DO_NOT_SHIP)

### 6.1 Unit tests to add

1. `test_energy_ebm_scoring_is_deterministic`
   - same input → same energy + term ordering
2. `test_verifier_ranks_by_delta_energy_then_confidence`
3. `test_lever_qemail_frequency_detects_5_minute_schedule`
   - use a synthetic fixture with QEMAIL every 5 minutes
4. `test_lever_qpec_plus_one_emits_qpec_label_when_hosts_match`
5. `test_filter_excluded_processes_allows_add_server_and_tune_schedule`
6. `test_payout_report_batch_cli_merges_multiple_inputs`
   - ensure output contains per-source group + overall

### 6.2 Fixtures

Add synthetic fixtures under `tests/fixtures/`:
- `qemail_frequency_5min.csv`  
  Minimal columns: PROCESS_ID, QUEUE_DT, START_DT, END_DT, LOCAL_MACHINE_ID, SCHEDULE_ID.
- `payout_multi_input_a.csv`, `payout_multi_input_b.csv`  
  Include a payout-like process + a batch/period key in a param column.
- Reuse existing `tests/fixtures/quorum_close_cycle.csv` where possible.

---

## 7) Additional ERP-native plugins worth adding (next wave)

If you want better “ops-grade” signal beyond the 3 known issues, introduce these as plugins:

1. `analysis_hold_time_attribution_v1`
   - decomposes wait into: queued-but-ineligible (holds, dependencies) vs eligible-wait
2. `analysis_retry_rate_hotspots_v1`
   - uses EXECUTE_ATTEMPTS + status transitions to locate unstable processes
3. `analysis_dependency_critical_path_v1`
   - computes longest paths via DEP_PROCESS_QUEUE_ID / PARENT_PROCESS_QUEUE_ID
4. `analysis_param_variant_explosion_v1`
   - high unique(param)/runs ratio within a process => caching/batching opportunities
5. `analysis_close_cycle_change_point_v1`
   - change-point detection on queue delay and duration around close windows

These should all follow the same EBM pattern:
- extract feature(s) → compute gap vs baseline/frontier → emit candidate actions → verifier ranks.

---

## Appendix A: Measured properties from `proc log 1-14-26.csv` (sanity anchors)

These numbers are **MEASURED** from the provided log in this workspace (rows=435,616). They are included to anchor thresholds and to give Codex a quick “does this look right?” check when implementing.

### Dataset scope

- Rows: **435,616**
- Unique `PROCESS_ID`: **240**
- Unique `LOCAL_MACHINE_ID`: **2** (`QCPCENQPECZC01, QCPCENQPECZC02`)
- Time span (min `QUEUE_DT` → max `END_DT`): **2025-07-01 00:00:02 → 2026-01-14 10:56:00**

### QEMAIL scheduling (candidate for schedule tuning)

- `PROCESS_ID="QEMAIL"` rows: **56,109**
- Median inter-arrival (by `QUEUE_DT`): **300s (~5.0 min)**  
  (Interpretation: effectively a 5-minute schedule.)

### Largest “eligible wait > 60s” contributors (upper-bound queue pressure)

> Note: this uses `eligible_wait = START_DT - QUEUE_DT` and counts the *full* eligible_wait when it exceeds 60s (matching the harness’ current `analysis_queue_delay_decomposition` definition).

| PROCESS_ID   |   eligible_wait_gt_hours |   runs |   host_count |
|:-------------|-------------------------:|-------:|-------------:|
| LOSEXTCHLD   |              247,022.170 |  11743 |            2 |
| LOSLOADCLD   |               23,149.288 |  19636 |            2 |
| JBCREATEJE   |                4,740.286 |   7886 |            2 |
| JBOACHILD    |                2,591.722 |   4663 |            2 |
| JBVALCDBLK   |                2,087.347 |  11472 |            2 |
| JBINVOICE    |                  507.529 |   2282 |            2 |
| RDIMPAIRJE   |                  458.141 |   2799 |            2 |
| JEPRECOMBO   |                  294.762 |   5174 |            2 |
| RDJRNL       |                  263.088 |   9204 |            2 |
| QLONGJOB     |                  210.321 |   6743 |            2 |



---

## DETERMINISM

VERIFIED (all ordering rules, tie-breaks, and formulas are explicitly specified; no randomness is required in any new method)

