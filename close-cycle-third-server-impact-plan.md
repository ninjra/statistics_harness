# Plan: Close-Cycle Third-Server Impact Detection

**Generated**: 2026-02-02
**Estimated Complexity**: High

## Overview
Assess whether the existing plugins can *prove or refute* the claim: “adding a 3rd process server decreases time‑to‑completion by ~30% during close windows,” with **no false positives** (only false negatives). If current plugins cannot reach the conclusion, design a new **dynamic**, **conservative** plugin that detects this effect in future datasets without pre‑known data or columns. The effect will be defined explicitly as a **relative median time‑to‑completion (start→end)** reduction with a conservative CI gate, while also tracking **queue→end** and **eligible→end** as supporting metrics.

The plan follows four pillars:
- **Precision**: zero false positives (strong evidence gates).
- **Recall**: detect real ≥30% improvements when data supports it.
- **Robustness**: dynamic column inference, varying data sizes/shapes.
- **Explainability**: human‑readable reasons and artifacts.

## Prerequisites
- Access to the uploaded dataset(s) in `appdata/state.sqlite`.
- Working Python environment (venv in `/tmp/stat_harness_venv` or equivalent).
- Known close‑window definition (default day 20–5 unless overridden).
- Confirmation of time‑to‑completion definition (start→end vs queue→end).

## Sprint 1: Evaluate With Existing Plugins (No New Code)
**Goal**: Use current plugins to either reach the conclusion or document precisely why they cannot.
**Demo/Validation**:
- Run selected plugins on the dataset and capture report artifacts.
- Verify whether any plugin yields a *measured* close‑window improvement tied to **host count** or **capacity**.

### Task 1.1: Column mapping and host variation check
- **Location**: `appdata/state.sqlite`, dataset columns, plugin settings
- **Description**: Identify host/server column, start/end columns, queue/eligible columns (if any) **dynamically** using column inference. Compute whether close windows ever include ≥3 distinct hosts (current data may only have 2).
- **Complexity**: 4
- **Dependencies**: None
- **Acceptance Criteria**:
  - Confirm which columns represent server/host and time‑to‑completion (start→end).
  - Record a decision note that **time‑to‑completion = start→end** (primary), with queue→end and eligible→end as secondary diagnostics.
  - Confirm whether host count varies (2 vs 3+) during close windows.
**Validation**:
- Record inferred columns, close‑window definition (inferred vs override), and plugin versions in a run artifact or notes.
- Simple exploratory script or plugin log output captured in report notes.

### Task 1.2: Run `analysis_concurrency_reconstruction`
- **Location**: `plugins/analysis_concurrency_reconstruction/`
- **Description**: Produce host‑level concurrency summary and host count statistics using both **concurrent active hosts** and **unique active hosts** per bucket; check for evidence of a 3rd server in close windows.
- **Complexity**: 3
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Artifact shows host count and concurrency timeline.
  - If host count never reaches 3 in close windows, document “insufficient evidence.”
- **Validation**:
  - Report artifact in run directory.

### Task 1.3: Run `analysis_capacity_scaling` and `analysis_queue_delay_decomposition`
- **Location**: `plugins/analysis_capacity_scaling/`, `plugins/analysis_queue_delay_decomposition/`
- **Description**: Use modeled capacity scaling and eligible‑wait decomposition to see if any modeled ≈30% reduction exists during close windows (as supporting evidence only; not proof).
- **Complexity**: 4
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - Clear statement that these are **modeled** (not measured) and **non‑decisive** for “no false positives.”
- **Validation**:
  - Report artifacts with summaries; note limitation if only modeled.

### Task 1.4: Run `analysis_close_cycle_duration_shift`
- **Location**: `plugins/analysis_close_cycle_duration_shift/`
- **Description**: Check for measured close‑window duration changes by process to rule out that the “improvement” is actually per‑process latency change rather than capacity.
- **Complexity**: 3
- **Dependencies**: Task 1.1
- **Acceptance Criteria**:
  - If no measured per‑process duration drop is detected, record that existing plugins do not prove the claim.
- **Validation**:
  - Report artifact in run directory.

## Sprint 2: Design a New Conservative Plugin (Dynamic, No False Positives)
**Goal**: Define a new plugin that detects **measured** close‑window improvement linked to host count, with strong anti‑false‑positive guards.
**Demo/Validation**:
- Design doc + config schema + output schema ready for implementation.

### Task 2.1: Define detection logic & thresholds
- **Location**: new design notes in `docs/` or plugin README
- **Description**: Specify the statistical test and guards:
  - **Effect definition**: `effect = median_ttc(host>=3) / median_ttc(host<=2) - 1`. Require CI entirely ≤ **-0.30 ± tolerance** (configurable).
  - **Time‑to‑completion** is **start→end** (primary); also compute queue→end and eligible→end as secondary diagnostics.
  - **Bucketization**: configurable `bucket_size` (default daily) with minimum items per bucket and minimum buckets per group.
  - **Host count metrics**: compute both **concurrent active hosts** (primary classifier) and **unique active hosts** (secondary diagnostic) per bucket; report both.
  - Compare buckets with `host_count >= 3` vs `host_count <= 2`.
  - Use a conservative test (bootstrap CI or permutation) with low α (e.g., 0.01).
  - Confounding guards: process‑mix divergence threshold **and** workload volume parity (e.g., total rows or queue backlog similarity).
  - Time‑trend guard: optionally difference‑in‑differences across close‑window weeks.
  - **Close‑window detection**: infer real monthly close windows dynamically (e.g., detect compressed windows and holidays). If a user override is supplied, it **supersedes** inference.
- **Complexity**: 6
- **Dependencies**: Sprint 1 findings
- **Acceptance Criteria**:
  - Formal definition of “time‑to‑completion.”
  - Defined bucket size and minimum bucket counts per group.
  - Documented confounding controls and negative control strategy.
  - Documented close‑window inference method with defaults and overrides.
- **Validation**:
  - Written spec + configuration options.

### Task 2.2: Define plugin interface and artifacts
- **Location**: `plugins/analysis_close_cycle_capacity_impact/`
- **Description**: Specify config schema, output schema, and artifact formats:
  - `results.md` with “Detected / Suppressed / Not‑Applicable”
  - `results.csv` detail table
  - `results.json` full metrics and diagnostics
  - Explicit classification policy: when to emit **Detected** vs **Suppressed** vs **Not Applicable**
  - Surface both host‑count metrics and all three time‑to‑completion measures.
- **Complexity**: 5
- **Dependencies**: Task 2.1
- **Acceptance Criteria**:
  - Schemas enforce strict config validation.
  - Output includes reason summary and evidence fields (rows, columns, bucket stats).
- **Validation**:
  - Schema review + example output JSON.

## Sprint 3: Implement Plugin + Tests (No False Positives)
**Goal**: Implement the plugin and verify conservative behavior with synthetic and real datasets.
**Demo/Validation**:
- Tests pass; plugin returns “not applicable” unless evidence is strong.

### Task 3.1: Implement plugin code
- **Location**: `plugins/analysis_close_cycle_capacity_impact/plugin.py`
- **Description**:
  - Dynamic column inference (host, start, end, queue/eligible optional).
  - Close‑window bucketization; **concurrent host count** and **unique host count** computation per bucket.
  - Median completion time (start→end primary); compare 3+ vs 2‑ host groups.
  - Secondary diagnostics: queue→end and eligible→end effect sizes and CIs.
  - Conservative statistical test + effect size threshold.
  - Confounding guards (process mix, workload parity, time‑trend).
  - Data‑quality gates (missing timestamps, negative durations, inconsistent host IDs).
  - Human‑readable summary + suppression reasons.
- **Complexity**: 7
- **Dependencies**: Sprint 2
- **Acceptance Criteria**:
  - Finding emitted only with strong evidence; otherwise “not applicable.”
  - Fully dynamic across datasets/columns.
- **Validation**:
  - Unit tests (see Task 3.2), manual run on provided dataset.

### Task 3.2: Add tests for no‑false‑positive behavior
- **Location**: `tests/plugins/test_close_cycle_capacity_impact.py`
- **Description**:
  - Synthetic dataset with **no host count variation** → no findings.
  - Synthetic dataset with **3rd host** and ≥30% improvement → finding.
  - Confounded dataset (host count change + process mix shift) → no finding.
  - Negative control (shuffle host labels or compare non‑close windows) → no finding.
- **Complexity**: 6
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - All tests pass, no false positives.
- **Validation**:
  - `pytest -q` with `PYTHONPATH` set.

### Task 3.3: Wire into evaluation/known issues
- **Location**: `appdata/known_issues/*.yaml`
- **Description**: Add/adjust known‑issue rule to expect the new plugin’s finding **only for datasets that meet evidence gates** (e.g., ≥3‑host periods). Otherwise require “not applicable.”
- **Complexity**: 3
- **Dependencies**: Task 3.1
- **Acceptance Criteria**:
  - Known‑issue entry matches plugin output.
- **Validation**:
  - `stat-harness eval` passes against the known issue.

## Testing Strategy
- **Unit tests**: synthetic cases for true positive, false positive, confounded.
- **Integration**: run plugin on the proc log dataset; confirm either:
  - A measured ≥30% improvement is detected, or
  - It is flagged as “not applicable/insufficient evidence” because only 2 hosts exist now.
- **Sensitivity**: rerun with alternative bucket sizes to ensure conclusion stability.
- **Regression**: ensure other plugins still pass current test suite.

## Potential Risks & Gotchas
- **No host count variation** in close windows → cannot prove the claim (false negative).
- **Process mix or volume confounding** can mimic improvement; must reject when mix diverges.
- **Time‑to‑completion definition ambiguity** (start→end vs queue→end) or timezone drift.
- **Host column ambiguity** or inconsistent server IDs.
- **Sparse buckets** (few close‑window days/hours) → insufficient power.

## Rollback Plan
- Remove new plugin directory and schema files.
- Revert any known‑issues entry referencing the new plugin.
- Delete plugin references in docs or tests if added.
