## A. THREAD / CHAT_ID

| Field   | Value                                    |
| ------- | ---------------------------------------- |
| THREAD  | process-log-stat-plugins                 |
| CHAT_ID | ideaspace-normative-recommender-20260205 |

---

## B. Answer (bullets<=8)

* Yes: your “ideal ideaspace → compare → recommend” framing maps cleanly to **normative modeling**, **process mining conformance**, and **control/setpoint tracking** (error = observed − ideal).
* The most deterministic harness addition is a **two-stage** design: (1) build a canonical “ideaspace vector” for each process/window, (2) compute a deterministic **gap vector** to an “ideal” frontier/reference model, then generate recommendations via **rule + sensitivity**.
* “Ideal ideaspace” should be formalized as either (a) a **reference model** derived from a Quorum baseline, or (b) an **in-dataset Pareto frontier** (“best observed” behavior), with deterministic tie-breaks.
* “Subtraction/derivative” becomes: **residual analysis** + **sensitivity / what‑if** (finite differences or closed-form queueing/critical-path approximations), producing “do X ⇒ ~Y% improvement” estimates.
* Recommendations must be emitted only when **evidence thresholds** are satisfied; otherwise return `skipped` or `info` findings (no speculation).
* Keep it self-contained: if time/trace columns are missing, degrade to lower “tiers” (counts/cohorts/text-template drift) instead of failing.
* Determinism comes from: stable column inference, stable hashing sampling, fixed seed, closed-form estimators, and stable ranking.

---

## C. key_claims

| Claim                                                                                                                                           | Label       | Notes                                                             |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | ----------- | ----------------------------------------------------------------- |
| Your description is equivalent to a **normative model**: learn/define “normal/ideal” and score deviations (residuals) to guide interventions.   | INFERENCE   | No dataset needed to map the concept.                             |
| It also aligns with **process mining conformance checking**: compare observed traces/graphs to a reference model, then “enhance” the process.   | INFERENCE   | Works even when “ideal” is learned from best slices.              |
| It aligns with **control theory**: setpoint (ideal), measurement (observed), error signal (gap), and a control law (recommendations).           | INFERENCE   | The “derivative” intuition is sensitivity of objective to levers. |
| A deterministic “ideal ideaspace” can be defined via a **Pareto frontier** (efficient set) over KPIs and/or via deterministic reference priors. | INFERENCE   | No randomness required.                                           |
| There is **no evidence** available here that I have access to your “Quorum dataset” beyond your description.                                    | NO EVIDENCE | I can only propose deterministic plugin designs and tests.        |
| There is **no evidence** available here beyond the two example recommendations you provided (“add server”, “reduce qemail frequency”).          | NO EVIDENCE | I will provide *new* recommendation templates with hard triggers. |

---

## Deterministic harness addition: “Ideaspace” as an inspectable mathematical object

### 1) Formalize the “ideaspace”

Define a canonical feature mapping:

* Let `D` be the raw dataset (unknown schema).
* Let `Φ(D)` produce a **set of entities** (process/module/user/host/sequence/window) each represented by a **feature vector** `x ∈ R^k` and optional graphs.

Recommended entity grain (deterministic, tiered):

* **Tier 0 (always):** per categorical entity counts/rates (events, errors, statuses)
* **Tier 1 (if time exists):** per time-bucket rates, bursts, drift windows
* **Tier 2 (if start/end exist):** durations, queue delays, p50/p95/p99, utilization proxies
* **Tier 3 (if trace columns exist):** variants, DFG/Markov transitions, conformance metrics
* **Tier 4 (if host/capacity exist):** concurrency reconstruction, contention indices

All tiers should run with caps (`max_rows`, `max_cols`, `max_groups`) and deterministic sampling.

### 2) Define the “ideal”

You need an `IdealModel` that yields an “ideal point” `x*` for any entity point `x`.

Two deterministic options (support both):

| Ideal construction                | Deterministic definition                                                                                                                                                   | When used                     |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- |
| **Reference prior** (Quorum)      | `x* = median_or_p10(Quorum, matching_archetype)` + empirical-Bayes shrinkage                                                                                               | When a vetted baseline exists |
| **In-dataset efficient frontier** | Compute Pareto frontier over (maximize throughput, minimize p95 duration, minimize queue delay, minimize error rate). Choose nearest frontier point by deterministic norm. | When no baseline is available |

### 3) Compute “gap” and attribute causes

* Gap vector: `g = x − x*` (signed per metric).
* Contribution ranking: order components by `|g_i| / scale_i` (robust scale).
* Attach evidence: which window, which group slice, which step dominates.

### 4) Convert gap into deterministic actions (“derivative”)

Maintain a library of levers `a ∈ A` with:

* **Trigger**: a deterministic predicate on (features, gap, graphs)
* **Impact estimator**: deterministic `Δobjective(a)`

  * closed-form (queueing/critical path) OR
  * deterministic finite difference using observed parameter estimates

Then select top actions by:

* estimated impact magnitude
* confidence (data volume, stability)
* feasibility (requires columns present)

---

## New actionable recommendation templates (not your two examples)

These are phrased as **templates** the harness can emit *only when triggers are satisfied*. Each can output an estimated benefit `x%` deterministically.

| Lever template (action)                                                 | Deterministic trigger (evidence)                                                                                                                                             | Deterministic impact estimate (example)                                                                                                |
| ----------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| **Workload isolation via priority classes / reserved capacity**         | Close-critical processes show high eligible→start delay AND share the same resource pool with non-critical workload (utilization proxy spikes when non-critical load rises). | Estimate ΔWq using Kingman/VUT or empirical before/after segments when non-critical load is absent.                                    |
| **Close-window blackout of non-critical scheduled jobs**                | A process family not on the close critical path consumes ≥p% of concurrency during high-pressure windows AND correlates with increased queue delays for close-critical work. | Remove that arrival-rate component in the fitted wait model; predict Δp95 wait and resulting makespan reduction.                       |
| **Cap concurrency on a thrashing step**                                 | Throughput per minute decreases when active concurrency increases (negative slope over multiple buckets), suggesting lock/IO contention.                                     | Choose concurrency that maximizes observed throughput curve; predict improvement as (best_throughput − current_throughput)/current.    |
| **Split oversized batches / transactions**                              | Presence of `batch_size/items/lines`-like column AND duration shows superlinear growth or heavy tail with batch_size (quantile regression slope at p90/p99).                 | Predict Δp99 duration when batch_size reduced to target quantile; convert to makespan impact via critical-path share.                  |
| **Introduce retry backoff / circuit breaker for hot failure templates** | Template-mined errors show bursts AND a high retry ratio (repeated template per entity within short time) AND retries contribute materially to arrival rate.                 | Estimate reduced arrival rate λ′ by suppressing retries; predict ΔWq from queue model or empirical low-retry intervals.                |
| **Resource affinity / pinning to reduce cold-start penalties**          | Same step shows materially lower service time when executed on same host/resource shortly after prior related work (host effect).                                            | Estimate service time delta between “warm” vs “cold” cohorts; apply to fraction of affected executions.                                |
| **Parallelize independent branches (dependency/sequence)**              | Dependency graph / sequence model shows long serial chain with independent prerequisites (no shared resource signature) AND critical path dominated by those steps.          | Estimate new makespan as `max(branch)` instead of `sum(branch)` for the identified segment; conservative bound using observed medians. |
| **Pre-stage invariant close prerequisites earlier**                     | Close window exhibits a sharp utilization/service-time spike at start and the spike is driven by steps that also appear outside close (same template/activity).              | Shift a measured fraction of workload earlier; recompute predicted peak utilization and queue delay with the reduced peak.             |

**Important deterministic gate:** each recommendation must include a “counterfactual plausibility” check (e.g., enough samples; effect stable across at least 2 windows; not explained by missingness changes) or it must not emit.

---

## Proposed plugin(s) to add

### Plugin 1: `analysis_ideaspace_normative_gap`

**Purpose:** Build `Φ(D)`, select/construct `IdealModel`, compute gap vectors + top deviations.

**Core artifacts:**

* `entities_table`: one row per (process/module/window/group) with KPIs and gaps
* `frontier_points` or `reference_stats` summary
* `top_gap_components` per entity

**Plugin-specific tests (beyond your common suite):**

* `test_frontier_is_deterministic_with_ties` (stable tie-break)
* `test_gap_zero_on_identical_to_reference`
* `test_tiered_degradation` (remove time/start/end/trace columns; plugin still returns ok/skipped without crashing)

### Plugin 2: `analysis_ideaspace_action_planner`

**Purpose:** Generate the **actionable recommendations** using the lever library above.

**Plugin-specific tests:**

* One synthetic test per lever:

  * `test_reco_priority_isolation_triggered`
  * `test_reco_blackout_triggered`
  * `test_reco_concurrency_cap_triggered`
  * `test_reco_batch_split_triggered`
  * `test_reco_retry_backoff_triggered`
  * `test_reco_affinity_triggered`
  * `test_reco_parallelize_triggered`
  * `test_reco_prestage_triggered`
* Negative controls:

  * `test_no_recos_when_triggers_not_met`
* Determinism:

  * `test_recos_stable_same_seed_same_df`
* Safety:

  * `test_recos_do_not_output_raw_text`
* Performance:

  * `test_action_planner_respects_time_budget` (must skip expensive levers first)

---

## D. per_recommendation

| Recommendation                                                                                              | improved                                                                                                                  | risked                                                                                                                         | enforcement_location                                                                            | regression_detection                                                                                             |
| ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Add `analysis_ideaspace_normative_gap` (tiered feature extraction + ideal frontier/reference + gap vectors) | Accurate: explicit gap attribution; Performant: capped features; Secure: aggregated only; Citable: method tags in output. | Wrong “ideal” if reference missing or frontier chosen poorly; mitigated by dual-mode (reference OR frontier) + debug evidence. | New plugin module + shared `ideaspace_feature_extractor.py`.                                    | Golden synthetic datasets for each tier; tie-break determinism; frontier stability; **ANY_REGRESS=>DO_NOT_SHIP** |
| Add `analysis_ideaspace_action_planner` (lever library + deterministic impact estimators)                   | Actionable outputs in the desired “do X ⇒ ~Y%” form; consistent across unknown datasets.                                  | Overconfident impact claims; mitigate via conservative bounds + confidence gating + “estimated” labeling.                      | New plugin module; shared `lever_library.py` with deterministic triggers.                       | One test per lever + negative controls + determinism + privacy; **ANY_REGRESS=>DO_NOT_SHIP**                     |
| Implement “ideal model” input as a signed, versioned artifact (Quorum baseline JSON) with schema hash       | Accurate: stable baselines; Secure: avoids shipping raw data; Deterministic comparisons across clients.                   | Baseline drift over time; mitigate by versioning and explicit validity windows.                                                | Harness loader + config schema; baseline builder tool (offline).                                | Snapshot tests on baseline files; schema-hash mismatch tests; **ANY_REGRESS=>DO_NOT_SHIP**                       |
| Add a strict “no speculation” emission gate for recommendations                                             | Accurate + Secure: prevents misleading or sensitive outputs.                                                              | Fewer recommendations (more conservative).                                                                                     | Action planner scoring: require minimum N, stability across windows, and effect-size threshold. | Tests ensuring recommendations don’t appear under insufficient evidence; **ANY_REGRESS=>DO_NOT_SHIP**            |

---

## E. DETERMINISM

| Item                                    | DETERMINISM     | Notes                                                                                                                                 |
| --------------------------------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| This response (structure/ordering)      | VERIFIED        | Fixed ordering and stable counts.                                                                                                     |
| Proposed “ideaspace gap → lever” design | PARTIAL(scoped) | Deterministic *if implemented exactly* with stable inference/sampling, fixed seed, and closed-form estimators; not yet executed here. |
| Impact estimation (“x% better”)         | PARTIAL(scoped) | Deterministic given fixed estimator, but accuracy depends on data sufficiency and model fit; enforce confidence gating.               |

---

## F. TS

TS=2026-02-05T16:41:08-07:00

THREAD=process-log-stat-plugins | CHAT_ID=ideaspace-normative-recommender-20260205 | TS=2026-02-05T16:41:08-07:00
