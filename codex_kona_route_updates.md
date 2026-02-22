# Codex CLI Implementation Task: Kona Space Pathfinding (EBM Current→Ideal Route Planner)

## 0) Your role & non-negotiables

You are running inside the target repository. Implement the changes below **by scanning the repo first** and **updating existing code in-place**. Do **not** introduce duplicate schemas, duplicate helper functions, or duplicate “nearly identical” logic blocks.

### Hard constraints
- **No external references required**: everything you need is in this file + the repo you are scanning.
- **Deterministic outputs**: identical inputs + same seed ⇒ identical artifacts, findings, and recommendation ordering.
- **No duplicate info**: if the repo already has an equivalent schema/helper/logic, reuse it; only add what’s missing.
- **Preserve existing behavior by default**: route planning must be **disabled by default** so existing runs are unchanged unless explicitly enabled via config.
- **Keep existing artifacts stable**: especially `verified_actions.json` produced by `analysis_ebm_action_verifier_v1`.

---

## 1) Definition: “Kona space” in this repo (EBM energy landscape)

In this codebase, “Kona space” is an **energy-based model (EBM) ideaspace** where each entity has:

- **Observed metric vector**: `x = {duration_p95, queue_delay_p95, error_rate, rate_per_min, background_overhead_per_min, ...}`  
- **Ideal metric vector**: `x*` (per ideal mode / policy)
- **Energy** (scalar) computed from normalized gaps + constraint penalties.

### 1.1 Energy equations (as implemented conceptually here)

For each metric key `k`:

- If `k` is a **minimize** metric (lower is better):  
  - If `x_k <= x*_k` then `gap_k = 0`  
  - Else `gap_k = (x_k / max(x*_k, eps)) - 1`

- If `k` is a **maximize** metric (higher is better):  
  - If `x_k >= x*_k` then `gap_k = 0`  
  - Else `gap_k = (x*_k / max(x_k, eps)) - 1`

Then:

- **Gap energy**:  
  `E_gap(x) = Σ_k w_k * (gap_k)^2`

- **Constraint energy**: a scalar penalty already computed upstream and stored in the ideaspace energy vector for each entity:  
  `E_constraints(x) >= 0` (often non-zero only for entity `"ALL"`)

- **Total energy**:  
  `E_total(x) = E_gap(x) + E_constraints(x)`

### 1.2 Pathfinding objective (current → ideal)
We treat each recommended “lever” as a **discrete operator** that transforms the current modeled metrics `x` into `x' = T_a(x)`.

A route is an ordered sequence of actions `a_1..a_d` and a sequence of states:
`x_0 → x_1 → ... → x_d` where `x_{i+1} = T_{a_{i+1}}(x_i)`.

Goal: find a route that **reduces energy**:
`ΔE_total = E_total(x_0) - E_total(x_d)` (maximize; must be ≥ 0).

We also track a route-level confidence `C_route ∈ [0, 1]` aggregated from step confidences (use a deterministic aggregation like geometric mean).

---

## 2) What you will add: a deterministic route planner with a beam/A*-style score

### 2.1 Why not “any” pathfinding equation?
Any graph search can run on Kona space, but **optimality** depends on the scoring function and constraints (confidence thresholds, max-uses, incompatibilities, target scoping).

This implementation will use a **deterministic beam search** with an A*-style priority derived from energy:

- Node/state priority uses a lexicographic tuple (deterministic):
  1) Higher total energy reduction (ΔE_total)  
  2) Higher route confidence (C_route)  
  3) Fewer steps (shorter route)  
  4) Stable tie-break (lexicographic signature of lever IDs + targets)

This gives stable, inspectable behavior and avoids nondeterministic “best effort” planning.

### 2.2 Route planning must be OFF by default
Route planning is only enabled when config explicitly sets:
- `route_max_depth >= 2` **and**
- `route_beam_width >= 1`

If disabled: do not emit any new route artifacts/findings (preserves current behavior).

---

## 3) Repository preflight scan (do this before editing)

Run a quick scan to avoid duplicates:

- Search for any existing route artifacts / schema / kinds:
  - `rg -n "kona_route_plan" -S`
  - `rg -n "verified_route_action_plan" -S`
  - `rg -n "route_max_depth|route_beam_width|route_min_delta_energy" -S`
  - `rg -n "ideaspace_route" -S`

If something already exists (file or partial implementation), **extend it** instead of creating a second version.

---

## 4) Implementations required

### 4.1 Add a new artifact schema: `docs/schemas/kona_route_plan.schema.json`

Create **exactly one** new schema file at:
- `docs/schemas/kona_route_plan.schema.json`

If a file with this name already exists, update it to match the spec below (do not create a second schema).

#### 4.1.1 Schema (copy verbatim)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Kona Route Plan Artifact Schema",
  "type": "object",
  "additionalProperties": false,
  "required": [
    "schema_version",
    "plugin_id",
    "generated_at",
    "decision",
    "config",
    "steps",
    "totals"
  ],
  "properties": {
    "schema_version": { "const": "kona_route_plan.v1" },
    "plugin_id": { "type": "string" },
    "generated_at": { "type": "string" },

    "decision": {
      "type": "string",
      "enum": ["modeled", "not_applicable"]
    },

    "target_signature": {
      "anyOf": [
        { "type": "null" },
        { "type": "string", "minLength": 1 }
      ]
    },

    "config": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "route_max_depth",
        "route_beam_width",
        "route_min_delta_energy",
        "route_min_confidence",
        "route_allow_cross_target_steps",
        "route_stop_energy_threshold",
        "route_candidate_limit",
        "route_time_budget_ms",
        "route_disallowed_lever_ids",
        "route_disallowed_action_types"
      ],
      "properties": {
        "route_max_depth": { "type": "integer", "minimum": 0 },
        "route_beam_width": { "type": "integer", "minimum": 0 },
        "route_min_delta_energy": { "type": "number", "minimum": 0.0 },
        "route_min_confidence": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "route_allow_cross_target_steps": { "type": "boolean" },
        "route_stop_energy_threshold": { "type": "number", "minimum": 0.0 },
        "route_candidate_limit": { "type": "integer", "minimum": 1 },
        "route_time_budget_ms": { "type": "integer", "minimum": 0 },
        "route_disallowed_lever_ids": {
          "type": "array",
          "items": { "type": "string" },
          "default": []
        },
        "route_disallowed_action_types": {
          "type": "array",
          "items": { "type": "string" },
          "default": []
        }
      }
    },

    "not_applicable": {
      "anyOf": [
        { "type": "null" },
        {
          "type": "object",
          "additionalProperties": false,
          "required": ["reason_code", "message"],
          "properties": {
            "reason_code": { "type": "string", "minLength": 1 },
            "message": { "type": "string", "minLength": 1 },
            "details": { "type": "object" }
          }
        }
      ]
    },

    "steps": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "step_index",
          "lever_id",
          "action_type",
          "title",
          "action",
          "confidence",
          "target_entity_keys",
          "target_process_ids",
          "energy_before",
          "energy_after",
          "delta_energy",
          "modeled_metrics_after"
        ],
        "properties": {
          "step_index": { "type": "integer", "minimum": 1 },
          "lever_id": { "type": "string", "minLength": 1 },
          "action_type": { "type": "string", "minLength": 1 },
          "title": { "type": "string" },
          "action": { "type": "string", "minLength": 1 },
          "confidence": { "type": "number", "minimum": 0.0, "maximum": 1.0 },

          "target_entity_keys": {
            "type": "array",
            "minItems": 1,
            "items": { "type": "string", "minLength": 1 }
          },
          "target_process_ids": {
            "type": "array",
            "items": { "type": "string", "minLength": 1 }
          },

          "energy_before": { "type": "number", "minimum": 0.0 },
          "energy_after": { "type": "number", "minimum": 0.0 },
          "delta_energy": { "type": "number", "minimum": 0.0 },

          "modeled_metrics_after": {
            "type": "object",
            "additionalProperties": { "type": "number" }
          }
        }
      }
    },

    "totals": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "energy_before",
        "energy_after",
        "total_delta_energy",
        "route_confidence",
        "stop_reason",
        "expanded_states"
      ],
      "properties": {
        "energy_before": { "type": "number", "minimum": 0.0 },
        "energy_after": { "type": "number", "minimum": 0.0 },
        "total_delta_energy": { "type": "number", "minimum": 0.0 },
        "route_confidence": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "stop_reason": { "type": "string", "minLength": 1 },
        "expanded_states": { "type": "integer", "minimum": 0 }
      }
    },

    "debug": { "type": "object" }
  },
  "allOf": [
    {
      "if": {
        "properties": { "decision": { "const": "modeled" } },
        "required": ["decision"]
      },
      "then": {
        "properties": {
          "not_applicable": { "type": "null" },
          "steps": { "minItems": 1 }
        }
      }
    },
    {
      "if": {
        "properties": { "decision": { "const": "not_applicable" } },
        "required": ["decision"]
      },
      "then": {
        "required": ["not_applicable"],
        "properties": {
          "not_applicable": {
            "type": "object",
            "required": ["reason_code", "message"]
          },
          "steps": { "maxItems": 0 }
        }
      }
    }
  ]
}
```

---

### 4.2 Add a new core module: `src/statistic_harness/core/ideaspace_route.py`

Create this file if missing (otherwise extend the existing one):

- `src/statistic_harness/core/ideaspace_route.py`

This module must be **pure + deterministic** and must not import `statistic_harness.core.stat_plugins.*` to avoid circular imports.

#### 4.2.1 Public API (implement exactly these signatures)

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

@dataclass(frozen=True)
class RouteCandidate:
    lever_id: str
    title: str
    action: str
    action_type: str
    confidence: float
    target_entity_keys: tuple[str, ...]
    target_process_ids: tuple[str, ...]
    evidence_metrics: dict[str, Any]
    # Single-step reference (used only for sorting/heuristics; must not be trusted as multi-step truth):
    delta_energy_single: float
    energy_before_single: float

@dataclass(frozen=True)
class RouteConfig:
    route_max_depth: int
    route_beam_width: int
    route_min_delta_energy: float
    route_min_confidence: float
    route_allow_cross_target_steps: bool
    route_stop_energy_threshold: float
    route_candidate_limit: int
    route_time_budget_ms: int
    route_disallowed_lever_ids: tuple[str, ...]
    route_disallowed_action_types: tuple[str, ...]

@dataclass(frozen=True)
class RouteStep:
    step_index: int
    lever_id: str
    action_type: str
    title: str
    action: str
    confidence: float
    target_entity_keys: tuple[str, ...]
    target_process_ids: tuple[str, ...]
    energy_before: float
    energy_after: float
    delta_energy: float
    modeled_metrics_after: dict[str, float]

def solve_kona_route_plan(
    *,
    plugin_id: str,
    generated_at: str,
    entities: Mapping[str, Mapping[str, Any]],
    weights: Mapping[str, float],
    candidates: Sequence[RouteCandidate],
    config: RouteConfig,
    apply_lever: Callable[[str, Mapping[str, float], Mapping[str, Any]], dict[str, float]],
    energy_gap: Callable[[Mapping[str, float], Mapping[str, float], Mapping[str, float]], float],
    update_constraints: Callable[[str, Mapping[str, float], Mapping[str, float], float, Mapping[str, Any]], float],
    time_now_ms: Callable[[], float],
) -> dict[str, Any]:
    ...
```

Notes:
- `entities` uses the same structure already present in `energy_state_vector.json` artifacts: each entity has `observed`, `ideal`, and optionally `energy_constraints`.
- `apply_lever()` must be treated as a pure function: **do not mutate** the input metric mapping.
- `energy_gap()` computes `E_gap(x)` given `(observed_metrics, ideal_metrics, weights)`.
- `update_constraints()` returns updated scalar constraint energy, given lever id, before/after metrics, previous constraints, evidence metrics.
- `time_now_ms()` must be used for deterministic time budget checks (do not call `time.time()` inside this module).

#### 4.2.2 Core algorithm (deterministic beam search)

Implement beam search with these semantics:

- **Enablement**:
  - If `config.route_max_depth < 2` OR `config.route_beam_width < 1`: return `decision="not_applicable"` with reason_code `"ROUTE_DISABLED"`.
- **Candidate filtering**:
  - Drop candidates where `confidence < route_min_confidence`.
  - Drop candidates in `route_disallowed_lever_ids`.
  - Drop candidates where `action_type` in `route_disallowed_action_types`.
  - Sort candidates deterministically by:
    1) `-delta_energy_single`
    2) `-confidence`
    3) `lever_id`
    4) `",".join(target_process_ids)`
  - Limit to `route_candidate_limit` (after sorting).
- **State representation**:
  - Keep a per-entity modeled metrics map and per-entity constraints scalar (only `"ALL"` typically non-zero).
  - For scoring, you must compute route energy as the **sum** of `E_total` across the current route’s target entity keys.
- **Cross-target behavior**:
  - If `route_allow_cross_target_steps` is `False`, all steps must share the **exact same `target_entity_keys` set** as the first step (stable signature).
- **Expansion**:
  - Expand from depth 1..max_depth.
  - Maintain a beam list of best states at each depth (size <= beam_width).
  - Use a stable priority tuple for ordering states (best-first):
    1) higher `total_delta_energy`
    2) higher `route_confidence` (use geometric mean of step confidences)
    3) fewer steps
    4) lexicographic signature: `lever_id@target_signature` joined by `|`
- **Stop conditions**:
  - If best state’s `energy_after <= route_stop_energy_threshold`, stop with stop_reason `"threshold_reached"`.
  - If time budget exceeded (`route_time_budget_ms > 0`), stop with stop_reason `"time_budget_exceeded"` and return best found so far (if any), else not_applicable `"TIME_BUDGET_EXCEEDED"`.
  - If reach max depth: stop_reason `"max_depth"`.
- **Return object**:
  - Must conform to `docs/schemas/kona_route_plan.schema.json`.
  - Always populate `config` snapshot and `totals.expanded_states`.
  - If modeled: include `steps[]` and `totals.{energy_before, energy_after, total_delta_energy, route_confidence}`.
  - If not_applicable: include `not_applicable.reason_code` + message, and steps must be empty.

---

### 4.3 Extend verifier plugin configuration surface (no duplicates)

Update these files (modify, do not duplicate):

- `plugins/analysis_ebm_action_verifier_v1/config.schema.json`
- `plugins/analysis_ebm_action_verifier_v1/plugin.yaml`

#### 4.3.1 Add these config keys (with defaults that keep route OFF)

Add these keys to the schema + defaults:

- `route_max_depth` (int, default **0**)
- `route_beam_width` (int, default **0**)
- `route_min_delta_energy` (number, default **0.0**)
- `route_min_confidence` (number, default **0.0**)
- `route_allow_cross_target_steps` (bool, default **false**)
- `route_stop_energy_threshold` (number, default **1.0**)
- `route_candidate_limit` (int, default **50**)
- `route_time_budget_ms` (int, default **0**)  # 0 = “use plugin budget only / no extra limit”
- `route_disallowed_lever_ids` (array[str], default **[]**)
- `route_disallowed_action_types` (array[str], default **[]**)

Make sure the schema remains valid JSON schema and defaults are applied by existing default-injection logic.

---

### 4.4 Integrate route planning into `analysis_ebm_action_verifier_v1` (Kona plugin)

Modify (do not fork/duplicate) this handler:

- `src/statistic_harness/core/stat_plugins/ideaspace.py`

Specifically, enhance `_ebm_action_verifier_v1(...)`:

#### 4.4.1 Inputs already available in `_ebm_action_verifier_v1`
The plugin already reads:
- `energy_state_vector.json` (entities with observed/ideal/energy_total/constraints)
- `recommendations.json` from the action planner (lever recommendations)

It already produces:
- `verified_actions.json` artifact (keep unchanged)
- `verified_action` findings (keep unchanged)

#### 4.4.2 Add: route candidate normalization (no new duplicated logic blocks)
After you compute `verified` (list of dicts for verified single actions):

- Build `RouteCandidate` objects from the top `route_candidate_limit` verified items.
- Derive `action_type` and canonical `target_process_ids` using the same mapping already used in `report.py` for these lever IDs (keep it local; do NOT import report.py):
  - `tune_schedule_qemail_frequency_v1` ⇒ `action_type="tune_schedule"`, target_process_ids=("qemail",)
  - `add_qpec_capacity_plus_one_v1` ⇒ `action_type="add_server"`, target_process_ids=("qpec",)
  - `split_batches` ⇒ `action_type="batch_input_refactor"`
  - `priority_isolation` ⇒ `action_type="orchestrate_macro"`
  - `retry_backoff` ⇒ `action_type="dedupe_or_cache"`
  - `cap_concurrency` ⇒ `action_type="cap_concurrency"`
  - `blackout_scheduled_jobs` ⇒ `action_type="schedule_shift_target"`
  - else: `action_type="ideaspace_action"` (fallback)
- `target_entity_keys` should be parsed from the existing verified record’s `target` (comma-separated string), preserving order but normalizing whitespace. If empty, default to ("ALL",).

#### 4.4.3 Add: call the route solver (only when enabled)
If route is enabled (`route_max_depth >= 2` and `route_beam_width >= 1`):

1) Construct `RouteConfig` from config values.
2) Call `solve_kona_route_plan(...)` in `statistic_harness.core.ideaspace_route` using callbacks that reuse existing modeling & energy logic:
   - `apply_lever`: reuse the **existing** lever modeling transform logic already present in `_ebm_action_verifier_v1` (extract to a helper inside ideaspace.py if needed, but do not duplicate it).
   - `energy_gap`: reuse the same logic used to compute energy in `_ideaspace_energy_ebm_v1` (you can call existing helper(s) already in ideaspace.py; if none exist, add a small helper there and reuse it for both verifier scoring and route solver callbacks).
   - `update_constraints`: reuse existing `_constraint_after_for_action(...)` behavior.
   - `time_now_ms`: use a deterministic monotonic source (e.g., `time.monotonic()` wrapper) and compute ms. (It is acceptable to pass in a lambda that converts monotonic to ms; do not rely on wall clock time.)
3) Write the returned dict to a new artifact:
   - `route_plan.json` in the verifier plugin’s artifact directory.
4) If decision == `"modeled"`: emit a new finding with kind **`verified_route_action_plan`** (see next section).

If route is disabled (defaults), do nothing extra (no new artifact, no new finding).

#### 4.4.4 Add: new finding kind `verified_route_action_plan`
When a modeled route exists, emit a finding dict with at least:

- `kind`: `"verified_route_action_plan"`
- `title`: something stable like `"Kona current→ideal route plan"`
- `what`: short summary (e.g., `"{n} steps; ΔE={...}; confidence={...}"`)
- `decision`: `"modeled"`
- `target`: a stable target string (use the route’s `target_signature` if you set it; else join process IDs or entity keys)
- `total_delta_energy`, `energy_before`, `energy_after`, `route_confidence`
- `steps`: a list of step dicts (mirror the artifact step objects enough for report rendering; do not force report.py to read artifacts)
- `evidence`: include at least one evidence record pointing to `route_plan.json` artifact path/name

Also ensure the new finding has a stable id (use the same stable-id helper patterns used elsewhere in ideaspace.py).

---

### 4.5 Wire route plan into report recommendations (so it shows up)

Modify (do not duplicate) this file:

- `src/statistic_harness/core/report.py`

There is already a block that converts `analysis_ebm_action_verifier_v1` findings of kind `"verified_action"` into discovery recommendation items.

Extend that block to also process `"verified_route_action_plan"`:

- Only include route plan recommendations when:
  - `decision == "modeled"` and `steps` is a non-empty list.
- Render the recommendation text as markdown with numbered steps, e.g.:

```
Route plan (2 steps, ΔE=0.73, confidence=0.84):
1. Tune qemail schedule to reduce queue pressure… (confidence 0.90)
2. Split large batches to reduce tail duration… (confidence 0.78)
```

- Set:
  - `kind = "verified_route_action_plan"`
  - `plugin_id = "analysis_ebm_action_verifier_v1"`
  - `action_type = "route_process"` (this is the semantic meaning: a multi-step route)
  - `target` = canonical target if single (e.g., `"qemail"`), else `None`
  - `modeled_percent_hint` computed like verified_action does: `(total_delta_energy / energy_before) * 100` when possible.
  - `unit` = `"percent"` when modeled_percent_hint present else `"energy_points"`
  - `measurement_type` = `"modeled"`
  - `relevance_score` should be deterministic and based on the same pattern as verified_action (use total_delta_energy + confidence weights).

---

### 4.6 Update dataset runner script to prefer route_plan.json when present

Modify (do not duplicate) this file:

- `scripts/run_loaded_dataset_full.py`

There is a function `_extract_ideaspace_route_map(...)` that currently fabricates a “route” from top 3 single-step verified actions.

Update it to:
1) Look for `route_plan.json` artifact from `analysis_ebm_action_verifier_v1`.
2) If present and decision == modeled and steps non-empty, use that step list as the route actions.
3) Otherwise, fall back to current behavior (top verified actions).

This avoids UI/scripts showing stale “fake routes” once real routing exists.

---

## 5) Tests you must add (and keep deterministic)

### 5.1 Schema contract tests

Add a new test file (if missing):

- `tests/test_kona_route_plan_schema.py`

Test cases:
1) **Valid modeled payload** validates against the schema.
2) **Valid not_applicable payload** validates against the schema.
3) **Invalid**: modeled with empty steps must fail.
4) **Invalid**: not_applicable missing reason_code/message must fail.

Use `jsonschema.validate` and `pytest.raises(ValidationError)` pattern already present in repo tests.

### 5.2 Route integration test (minimal but real)

Extend `tests/test_kona_energy_ideaspace_architecture.py` (preferred) or add a new test file to verify:

- When route is enabled via settings:
  - `analysis_ebm_action_verifier_v1` emits `route_plan.json` artifact.
  - `route_plan.json` validates against `kona_route_plan.schema.json`.
  - Plugin findings include kind `verified_route_action_plan` with non-empty steps.

Suggested test approach:
- Reuse the existing fixture pattern that writes:
  - `energy_state_vector.json` under `analysis_ideaspace_energy_ebm_v1` artifacts
  - `recommendations.json` under `analysis_ideaspace_action_planner` artifacts
- Enable route settings:
  - `route_max_depth = 2`
  - `route_beam_width = 4`
  - `route_candidate_limit = 10`
- Provide at least two lever recommendations whose transforms both reduce energy (so a 2-step route exists).

Your assertion should not depend on wall-clock time, and should only depend on deterministic sorting/tie-break rules.

---

## 6) Acceptance criteria (must all pass)

- `python -m pytest -q` passes locally.
- Default behavior unchanged:
  - With default settings (route disabled), verifier plugin output matches prior behavior:
    - `verified_actions.json` unchanged
    - no new route artifacts/findings are emitted
- With route enabled:
  - `route_plan.json` emitted and schema-valid
  - `verified_route_action_plan` finding emitted
  - report recommendations include the route plan item, rendered as a multi-step list

---

## 7) Optional but strongly recommended repo-wide optimization (justify + implement)

### 7.1 Ensure plugin cache invalidation covers the new module
This repo has an effective code-hash mechanism for stat-plugin wrappers. If you add a new module `src/statistic_harness/core/ideaspace_route.py` and the cache hash does not include it, behavior changes may not invalidate cache.

Update:
- `src/statistic_harness/core/stat_plugins/code_hash.py`

So that when resolved plugin is in `IDEASPACE_HANDLERS`, the hashed file list includes:
- `.../stat_plugins/ideaspace.py` (already)
- `.../core/ideaspace_route.py` (new)

Do this without duplicating hash logic; just append the path if it exists.

---

## 8) Justifications (for human review)

1) **Beam/A*-style deterministic search**
   - Benefit: stable, explainable, regression-testable route outputs; no randomness; bounded cost.
   - Tradeoff: not guaranteed globally optimal beyond the beam width; acceptable under strict time budgets.

2) **Route planning off by default**
   - Benefit: zero regression risk for existing users/runs.
   - Tradeoff: feature must be explicitly enabled by config; acceptable and expected.

3) **Schema-first artifact contract**
   - Benefit: prevents ambiguous/half-baked routes; makes integration and future extensions safe.
   - Tradeoff: small upfront work and tests; repays via stability.

4) **code_hash includes ideaspace_route.py**
   - Benefit: avoids stale cached results when route solver changes.
   - Tradeoff: none meaningful; only slightly more hashing work.
