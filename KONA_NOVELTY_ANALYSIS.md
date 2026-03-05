# Kona System Analysis — How It Works & Where Novelty Stalls

**D0:** 2026-03-04  
**Source:** INTERNAL — statistics_harness repo  
**IP Status:** INTERNAL — personal IP (Justin Ram / Infoil LLC)  
**Scope:** Analyze Kona ideaspace pipeline mechanics; identify where novelty is constrained; propose deterministic improvements

---

## How Kona Works (End to End)

The Kona system is a five-stage pipeline layered on top of the analysis plugins:

```
Stage 1: Feature Extraction (ideaspace_feature_extractor.py)
    ↓ Infer columns by name hints → compute KPIs → slice into entity cohorts
Stage 2: Normative Gap (analysis_ideaspace_normative_gap)
    ↓ Build observed vector, select ideal (baseline or Pareto frontier), emit gap findings
Stage 3: Action Planner (analysis_ideaspace_action_planner + lever_library.py)
    ↓ Run 10 hardcoded lever detectors against data → emit LeverRecommendations
Stage 4: Energy Model (analysis_ideaspace_energy_ebm_v1)
    ↓ E_total = Σ w_i × gap_i² + Σ constraint_penalties
Stage 5: Route Planner (ideaspace_route.py → solve_kona_route_plan)
    ↓ Beam search over multi-step action sequences
    ↓ Each step: apply lever → model next-state → compute ΔE → rank by energy reduction
    → Output: ranked, verified, multi-step route plan
```

The design is sound. Energy-based ranking converts "is this insight interesting?" into a measurable scalar (ΔE), which is exactly what you need for deterministic ranking. The beam search over multi-step routes is the right approach for finding compound optimizations.

---

## Where Novelty Is Constrained

The system has **one fundamental bottleneck** and three secondary constraints:

### Primary Bottleneck: The Lever Library Is Static

`lever_library.py` contains exactly 10 lever detectors, each hand-coded with a specific trigger condition:

| Lever | Trigger |
|-------|---------|
| `tune_schedule_qemail_frequency_v1` | PROCESS_ID == "QEMAIL" AND median inter-arrival ≤ 6 min |
| `add_qpec_capacity_plus_one_v1` | Host matches `/qpec/i` AND eligible-wait p95 > threshold |
| `workload_isolation` | Queue delay p95 > 120s |
| `blackout_non_critical` | Scheduled work ratio in peak windows > 0.25 |
| `concurrency_cap` | Negative correlation between active concurrency and throughput |
| `split_batches` | Duration p95 for top-decile batches > 1.5× median-batch p95 |
| `retry_backoff` | Burst count in a time bucket > 10 for same message template |
| `resource_affinity` | Best-host median duration < 90% of median-host duration |
| `parallelize_branches` | Both A→B and B→A orderings observed with balance > 0.25 |
| `prestage_prereqs` | High burst count of single activity in peak-decile buckets |

Every recommendation Kona can ever produce is a combination of these 10 levers. The route planner can *sequence* them (do A then B then C), but it cannot *invent* new actions. A dataset with a novel problem pattern — say, a periodic resource leak causing gradual throughput degradation — gets zero Kona recommendations because no lever detects it.

Meanwhile, the harness runs 256+ analysis plugins that detect hundreds of patterns. Most of those findings never become actionable recommendations because there's no lever that translates them.

### Secondary Constraint 1: Feature Extraction Is Name-Hint-Based

`pick_columns()` infers column roles by matching substrings: "eligible", "queue", "start", "end", "host", "machine", "batch". If a dataset uses `SUBMIT_TIMESTAMP` instead of `QUEUE_DT`, the eligible column is missed and all queue-delay levers become blind. The upstream `infer_columns` helps, but the ideaspace feature extractor does a second pass with its own hint list that can disagree.

### Secondary Constraint 2: No Cross-Plugin Synthesis

The 256+ analysis plugins each produce independent findings. Kona's action planner runs its own 10 detectors independently of what the analysis plugins found. A powerful pattern: "Granger causality detected that process A causes delays in process B → lever: decouple A from B." But Kona doesn't read Granger causality findings.

### Secondary Constraint 3: No Cross-Run Memory

Each run is stateless. If the previous run recommended "split batches for LOSEXTCHLD" and the operator implemented it, the next run doesn't know. It will either re-recommend the same action or recommend something else without acknowledging the prior recommendation.

---

## Proposals for Generating Novel Ideas (All Deterministic, No LLM)

### Proposal 1: Plugin-Finding-to-Lever Bridge (HIGH IMPACT)

**Problem:** 256 plugins produce findings. 10 levers produce recommendations. The translation is missing.

**Solution:** A new plugin `analysis_finding_to_lever_bridge_v1` that reads upstream analysis plugin findings and deterministically maps them to lever candidates via a **pattern registry** — a static mapping from finding kinds/attributes to lever templates:

```python
FINDING_LEVER_MAP = [
    {
        "finding_kind": "granger_causal_link",
        "condition": lambda f: f["evidence"]["metrics"].get("p_value", 1.0) < 0.05,
        "lever_template": {
            "lever_id": "decouple_causal_dependency",
            "title": "Decouple causally linked processes",
            "action_template": "Process {cause} Granger-causes delays in {effect} (p={p_value:.4f}). "
                              "Decouple by buffering, async handoff, or scheduling separation.",
            "confidence_base": 0.55,
        },
    },
    {
        "finding_kind": "changepoint_detected",
        "condition": lambda f: f["evidence"]["metrics"].get("magnitude", 0) > 1.5,
        "lever_template": {
            "lever_id": "investigate_regime_change",
            "title": "Investigate detected regime change",
            "action_template": "Changepoint at {timestamp} with magnitude {magnitude:.2f}. "
                              "Investigate root cause (deployment, config, volume shift).",
            "confidence_base": 0.50,
        },
    },
    {
        "finding_kind": "anomaly_cluster",
        "condition": lambda f: f["evidence"]["metrics"].get("anomaly_rate", 0) > 0.05,
        "lever_template": {
            "lever_id": "address_anomaly_hotspot",
            "title": "Address anomaly hotspot",
            "action_template": "Anomaly rate of {anomaly_rate:.1%} in {scope}. "
                              "Root-cause: configuration drift, data quality, or resource contention.",
            "confidence_base": 0.45,
        },
    },
    {
        "finding_kind": "hidden_markov_regime",
        "condition": lambda f: f["evidence"]["metrics"].get("n_states", 0) >= 2,
        "lever_template": {
            "lever_id": "stabilize_regime_oscillation",
            "title": "Stabilize process regime oscillation",
            "action_template": "HMM detected {n_states} distinct operating regimes. "
                              "Stabilize: pin config during close windows, add circuit breaker "
                              "to prevent regime transitions under load.",
            "confidence_base": 0.50,
        },
    },
    {
        "finding_kind": "transfer_entropy_link",
        "condition": lambda f: f["evidence"]["metrics"].get("transfer_entropy", 0) > 0.1,
        "lever_template": {
            "lever_id": "break_information_flow_bottleneck",
            "title": "Break information-flow bottleneck between processes",
            "action_template": "Process {source} drives {target} with TE={transfer_entropy:.3f}. "
                              "Add buffering or decouple to prevent cascading delays.",
            "confidence_base": 0.50,
        },
    },
    {
        "finding_kind": "spectral_clustering_community",
        "condition": lambda f: f["evidence"]["metrics"].get("modularity", 0) > 0.3,
        "lever_template": {
            "lever_id": "isolate_process_community",
            "title": "Isolate tightly-coupled process community onto dedicated resources",
            "action_template": "Processes {members} form a tightly-coupled community "
                              "(modularity={modularity:.2f}). Isolate onto dedicated resources "
                              "to contain blast radius and reduce cross-community contention.",
            "confidence_base": 0.50,
        },
    },
    {
        "finding_kind": "survival_hazard_spike",
        "condition": lambda f: f["evidence"]["metrics"].get("hazard_ratio", 1.0) > 2.0,
        "lever_template": {
            "lever_id": "mitigate_high_hazard_process",
            "title": "Mitigate high-hazard process timeout risk",
            "action_template": "Hazard ratio {hazard_ratio:.1f}× for {process} — high risk of "
                              "exceeding SLA. Add proactive timeout, early warning, or fallback path.",
            "confidence_base": 0.55,
        },
    },
]
```

This is NOT an LLM. It's a deterministic dispatch table. Each entry specifies: what finding kind it consumes, what conditions must be met, what lever it produces. Lever candidates feed into the energy model and route planner for ranking.

**Why this matters:** It immediately unlocks every analysis plugin's findings as candidate recommendations. When a new plugin is added (e.g., DML causal inference from the gap analysis), add one entry to the map and it flows through the energy system automatically.

**Effort:** ~200 lines for the bridge plugin + ~10 lines per finding-kind mapping. Start with the 20 highest-value finding kinds.

---

### Proposal 2: Compositional Lever Generation (MEDIUM-HIGH IMPACT)

**Problem:** The route planner sequences levers but doesn't combine them into new levers with novel semantics. "Split batches" and "blackout non-critical" are independent. But sometimes the novel insight is: "split batches *only during close windows*" — a conditional combination that neither lever alone recommends.

**Solution:** A static composition table that fires when multiple base levers trigger simultaneously:

```python
COMPOSITIONS = [
    {
        "requires": ["split_batches", "blackout_scheduled_jobs"],
        "produces": {
            "lever_id": "split_batches_during_close_window",
            "title": "Split oversized batches during close-window pressure",
            "action_template": "Split batches for {processes} specifically during close-window "
                              "high-pressure periods to maximize throughput during critical windows.",
            "confidence_boost": 0.10,
        },
    },
    {
        "requires": ["concurrency_cap", "resource_affinity"],
        "produces": {
            "lever_id": "affinity_aware_concurrency_cap",
            "title": "Cap concurrency with host affinity routing",
            "action_template": "Cap concurrency at ~{cap} AND route to affinity-optimal hosts, "
                              "combining thrash reduction with cold-start avoidance.",
            "confidence_boost": 0.05,
        },
    },
    {
        "requires": ["tune_schedule_qemail_frequency_v1", "priority_isolation"],
        "produces": {
            "lever_id": "qemail_deprioritize_during_close",
            "title": "Deprioritize QEMAIL during close-critical windows",
            "action_template": "Rather than reducing QEMAIL frequency globally, assign it to a "
                              "low-priority class during close-window hours so it yields to "
                              "close-critical work without reducing overall email throughput.",
            "confidence_boost": 0.10,
        },
    },
    {
        "requires": ["split_batches", "parallelize_branches"],
        "produces": {
            "lever_id": "split_and_parallelize_heavy_work",
            "title": "Split oversized batches then parallelize the chunks",
            "action_template": "Split {batch_col} heavy runs into smaller chunks, then run chunks "
                              "in parallel across available hosts for maximum throughput.",
            "confidence_boost": 0.10,
        },
    },
]
```

When both prerequisite levers trigger, the composition fires and produces a more nuanced recommendation. The energy model scores it normally — the compound lever gets its own ΔE which may be higher than either base lever alone.

**Effort:** ~80 lines. Static list; generation is a set-intersection check.

---

### Proposal 3: Contrapositive / Fragility Analysis (MEDIUM IMPACT)

**Problem:** The current system only looks for things that ARE bad. It doesn't reason about what COULD go bad.

**Solution:** For each entity where energy is currently LOW, compute which single metric degradation (20%) would cause the largest energy increase. This identifies fragile-good situations.

```python
def fragility_analysis(observed, ideal, weights):
    fragilities = {}
    for metric, value in observed.items():
        perturbed = dict(observed)
        if metric in MINIMIZE_METRICS:
            perturbed[metric] = value * 1.2  # 20% worse
        else:
            perturbed[metric] = value * 0.8
        delta_e = energy(perturbed, ideal, weights) - energy(observed, ideal, weights)
        fragilities[metric] = delta_e
    # Return sorted by impact
    return sorted(fragilities.items(), key=lambda x: -x[1])
```

Produces findings like: "Entity LOSEXTCHLD is currently healthy, but a 20% increase in queue delay would add 450 energy units — the most fragile metric. Monitor and alert."

This is a **novel recommendation type** — not "fix this" but "protect this." No current lever produces it.

**Effort:** ~60 lines in the energy EBM plugin.

---

### Proposal 4: Cross-Entity Transfer (MEDIUM IMPACT)

**Problem:** Each entity is analyzed independently. If entity A has low queue delay because it runs with a concurrency cap, and entity B has high queue delay without one, the system doesn't connect these.

**Solution:** After computing per-entity energy breakdowns, compare entity pairs that are similar on most metrics but differ on one. The differing metric + the differing configuration suggests a lever.

```python
def cross_entity_transfer(entities):
    transfers = []
    for a, b in combinations(entities, 2):
        for metric in shared_metrics(a, b):
            if a.gap[metric] < 0.1 and b.gap[metric] > 0.5:
                # A is good, B is bad on this metric — what's different?
                diffs = {m: a.observed[m] - b.observed[m] 
                         for m in shared_metrics(a, b) if m != metric}
                top_diff = max(diffs.items(), key=lambda x: abs(x[1]))
                transfers.append({
                    "source": a.key, "target": b.key,
                    "metric": metric, "differentiator": top_diff,
                    "suggestion": f"{a.key} achieves low {metric} gap — "
                                  f"key difference is {top_diff[0]}. "
                                  f"Apply similar config to {b.key}."
                })
    return transfers
```

Produces: "Server 01 has 40% lower queue delay than Server 02. The key difference is concurrency cap on 01. Apply to 02."

**Effort:** ~100 lines in energy EBM plugin.

---

### Proposal 5: Dynamic Thresholds from Data Distribution (LOW-MEDIUM IMPACT)

**Problem:** Lever triggers use hardcoded thresholds (p95 > 120s, batch multiplier > 1.5×). These are tuned for one dataset scale. Sub-second transaction data or multi-hour batch data will have triggers always-fire or never-fire.

**Solution:** Replace absolute thresholds with relative ones derived from the data's own distribution:

```python
# Instead of:
if p95 > 120.0:  # hardcoded

# Use:
cross_entity_stats = [e.queue_delay_p95 for e in entities]
threshold = np.median(cross_entity_stats) + 1.5 * np.std(cross_entity_stats)
if p95 > threshold:
```

Self-calibrating: on fast datasets, modest delays get flagged. On slow datasets, only outliers do.

**Effort:** ~40 lines. Modify each lever's trigger to accept a dynamic threshold.

---

### Proposal 6: Signal Coverage Audit (LOW EFFORT, HIGH DIAGNOSTIC VALUE)

**Problem:** We don't know how much signal the 256 plugins produce that Kona ignores.

**Solution:** A diagnostic plugin `analysis_kona_signal_coverage_audit_v1` that, after all analysis plugins complete, compares:
- Total findings by kind across all analysis plugins
- Which finding kinds the lever library's 10 detectors can consume
- Which finding kinds are "stranded signal" — detected but never translated to a lever

Output: a coverage matrix. This directly tells you which finding-to-lever-bridge entries (Proposal 1) to write first.

**Effort:** ~80 lines. `depends_on` all analysis plugins.

---

## What Would Require an LLM (and Why to Avoid It)

Generating natural-language action text for novel levers from Proposals 1–4. Currently each lever has a hand-written `action` string. If the bridge generates 50 new lever types, writing 50 action templates is tedious but finite.

**Why to avoid runtime LLM:** The action text is recommendation output. Non-deterministic text breaks the determinism contract, makes regression testing impossible, and introduces a latency/cost dependency.

**Alternative:** Template strings with metric fill-ins (as shown above). Static templates, deterministic fills. Human-readable without an LLM.

**Where an LLM helps at dev time (not runtime):** Generating the initial `FINDING_LEVER_MAP` entries. Give Claude the full list of 256 plugin finding kinds and ask it to draft the mapping. Review, edit, commit as static code. One-time task.

---

## Priority Ranking

| Rank | Proposal | Novelty Impact | Effort | Implementation |
|------|----------|---------------|--------|---------------|
| 1 | Signal Coverage Audit (#6) | Diagnostic — shows the gap | ~80 LOC | New plugin |
| 2 | Finding-to-Lever Bridge (#1) | HIGH — unlocks 256 plugins as lever sources | ~200+ LOC | New plugin + registry |
| 3 | Compositional Levers (#2) | MEDIUM-HIGH — compound insights | ~80 LOC | Extend action planner |
| 4 | Fragility Analysis (#3) | MEDIUM — novel "protect this" finding type | ~60 LOC | Extend energy EBM |
| 5 | Cross-Entity Transfer (#4) | MEDIUM — within-dataset learning | ~100 LOC | Extend energy EBM |
| 6 | Dynamic Thresholds (#5) | LOW-MEDIUM — self-calibrating triggers | ~40 LOC | Modify lever triggers |

**Recommended order:** #6 first (measure the gap), then #1 (close it), then #2–4 for compound novelty. Total: ~560 LOC for all six proposals.

---

*End of Analysis*
