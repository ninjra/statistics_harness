# Frozen Surfaces

Status: active

This document tracks which surfaces are frozen and under change control. Changes to frozen files require explicit justification and must be accompanied by updated evidence in the same commit.

## 1. Plugin Surface Hashes (existing)

- **Contract**: `docs/frozen_surfaces_contract.md`
- **Manifest**: `docs/frozen_plugin_surfaces.contract.json`
- **Verifier**: `python3 scripts/verify_frozen_plugin_surfaces.py`
- **Runtime modes**: off / warn / enforce (`STAT_HARNESS_FROZEN_SURFACES_MODE`)

## 2. Ranking Weight Contract

- **Frozen file**: `config/recommendation_weights.yaml`
- **Frozen values**: `modeled_delta_hours_close_dynamic: 0.40`, `modeled_delta_hours: 0.25`, `manual_touch_reduction_count: 0.20`, `close_contention_reduction_pct: 0.15`
- **Constraint**: Weights must sum to 1.0
- **Change control**: Any weight change requires re-running evaluator harness against ground truth and updating `docs/redteam-status.md` or relevant plan.

## 3. Report Schema

- **Frozen file**: `docs/report.schema.json`
- **Constraint**: Additive-only changes. No field removals or type changes to existing fields.
- **Reason**: Downstream consumers (modeling harness, evaluator, slide kit) depend on the schema.

## 4. Plugin Protocol

- **Frozen surface**: `Plugin.run(ctx: PluginContext) -> PluginResult` signature
- **Files**: `src/statistic_harness/core/types.py`
- **Constraint**: No breaking changes to PluginContext or PluginResult without version bump.

## 5. Actionability Thresholds

- **Frozen file**: `config/actionability_thresholds.yaml`
- **Change control**: Threshold changes require evaluator harness validation.

## 6. Configuration Contracts

- **Frozen files**: `config/reporting.yaml`, `config/chunk_invariance_tolerances.yaml`
- **Change control**: Changes require corresponding test updates.

## Change Control Process

1. Identify which frozen surface is affected.
2. Justify the change (bug fix, spec update, or redteam finding).
3. Update the frozen surface.
4. Run the relevant verifier.
5. Update evidence in `docs/redteam-status.md` or `docs/implementation-status-matrix.md`.
6. Commit frozen surface change + evidence update together.
