# Statistics_Harness — Web-updated improvement options (4 pillars) + Codex CLI implementation prompts

## Scope
Repo: https://github.com/ninjra/statistics_harness

This file focuses on making recommendations **actionable and defensible** on large ERP log datasets.

---

## Web-sourced options to implement (Stats-harness focused)

### S1) Process mining toolkit (sequence-aware, actionable)
- Source: PM4Py https://github.com/process-intelligence-solutions/pm4py
- Use: discover process models; conformance checking; bottleneck localization.

### S2) Object-centric event-log root cause analysis (OCEL RCA)
- Source: Springer chapter https://link.springer.com/chapter/10.1007/978-3-031-82225-4_5
- Use: when multiple objects interact (jobs, batches, ledgers), get causal-ish RCA.

### S3) Event log repair for incomplete traces
- Source: MDPI paper https://www.mdpi.com/2078-2489/16/5/390
- Use: repair missing events deterministically before analytics plugins run.

### S4) Systematic event-log imperfection taxonomy + cleaning
- Source: Information Systems article listing https://www.sciencedirect.com/science/article/pii/S0306437925001310
- Use: formalize your normalization layer and test it.

### S5) Causal inference for “what intervention helps” (DoWhy)
- Source: https://github.com/py-why/dowhy
- Use: translate correlations into intervention-shaped recommendations with explicit assumptions.

### S6) Changepoint detection for regime shifts (Ruptures)
- Source: https://github.com/deepcharles/ruptures
- Use: detect close-period regime changes, upgrades, seasonal effects.

### S7) Structured log parsing / template mining (Drain3)
- Source: https://github.com/logpai/Drain3
- Use: convert messy message text into templates + parameters; improves grouping.

### S8) High-performance dataframe execution for millions of rows (Polars lazy + streaming)
- Sources: Polars repo https://github.com/pola-rs/polars and streaming docs https://docs.pola.rs/user-guide/concepts/_streaming/
- Use: larger-than-RAM execution, faster groupbys/joins.

---

## Codex CLI implementation prompts (copy/paste)

### Task S-NORM-01: Imperfection taxonomy + deterministic cleaning transforms
**Goal:** Implement a cleaning pipeline with named imperfection classes.
**Deliverables:**
- `imperfections.md` and `imperfections.json` registry
- transforms: timestamp normalization, dedupe rules, missing-event flags
- audit log of transforms (what changed, counts)
**Acceptance tests:**
- golden raw log → normalized log diff
- “no silent drop” invariants

### Task S-PARSE-01: Drain3-based template miner plugin
**Goal:** Generate templates for message columns; add template_id to normalized table.
**Deliverables:**
- plugin `log_template_miner`
- persistence of mined state
- metrics: template count, coverage %
**Acceptance tests:**
- deterministic output on fixed dataset snapshot

### Task S-PM-01: Process mining plugin family (PM4Py)
**Goal:** Introduce sequence-aware plugins.
**Deliverables:**
- event log builder: `{case_id, activity, ts, resource}`
- conformance + bottleneck plugins
- outputs: model, deviations, actionable recommendations
**Acceptance tests:**
- known bottleneck scenarios detected on fixture data

### Task S-RCA-01: Object-centric RCA stage (OCEL)
**Goal:** Add optional OCEL conversion for multi-object ERP processes.
**Deliverables:**
- OCEL schema adapter
- RCA plugin producing ranked root causes + evidence
**Acceptance tests:**
- stable ranking on fixture dataset

### Task S-CAUSAL-01: DoWhy intervention recommendation plugin
**Goal:** Produce recommendations with explicit causal graph + sensitivity.
**Deliverables:**
- plugin `causal_recommendation`
- outputs: ATE estimate, assumptions, refutations
**Acceptance tests:**
- synthetic causal dataset recovers correct directionality

### Task S-TS-01: Changepoint detection stage
**Goal:** Detect regime shifts and re-baseline analytics.
**Deliverables:**
- plugin `changepoint_detector`
- outputs: changepoints + confidence + affected metrics
**Acceptance tests:**
- seeded synthetic time series: finds true changepoints

### Task S-PERF-01: Polars lazy/streaming execution path (opt-in)
**Goal:** Enable processing for million+ rows with bounded memory.
**Deliverables:**
- opt-in execution engine wrapper
- query plan explain capture
**Acceptance tests:**
- benchmark script shows bounded memory vs baseline

---

## Notes on “agents”
For Stats_Harness: prefer deterministic pipelines. If you add an “agent,” it should only:
- propose which deterministic plugins to run and in what order,
- never fabricate facts (must cite plugin outputs),
- be blocked by an evaluator when evidence is insufficient.

