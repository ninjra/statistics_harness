# 4 Pillars Scoring Spec (0.0-4.0)

## Purpose
Define a deterministic, balanced scoring model for the four pillars:
- `performant`
- `accurate`
- `secure`
- `citable`

The scorecard is computed from full run telemetry and report outputs, not from Kona/ideaspace alone.

## Scale
- Every pillar score is clamped to `0.0..4.0`.
- Overall score is also on `0.0..4.0`.

## Inputs
- Run telemetry (runtime, max RSS, plugin execution status)
- Report findings (measurement tags, evidence coverage)
- Recommendation quality (known-issue confirmation + modeled coverage)
- Traceability and reproducibility lineage fields
- Security/policy findings and fail-closed indicators

## Balance Policy
- Objective: maximize the weakest pillar while keeping pillar spread small.
- Constraints:
  - `min_floor`: minimum allowed pillar score before veto.
  - `max_spread`: maximum allowed `max(pillar)-min(pillar)` before veto.
  - `degradation_tolerance`: high one-pillar gains cannot compensate for a weak pillar.
- If a constraint is violated, result status is vetoed and the veto reason is recorded.

## Output Contract
- `pillars.<name>.score_0_4`
- `pillars.<name>.components`
- `pillars.<name>.rationale`
- `balance`:
  - `min_pillar`, `max_pillar`, `spread`
  - `balance_index_0_4`, `balanced_score_0_4`
  - `vetoes[]`, `status`
- `summary.overall_0_4`

## Determinism
- No randomness is used in scoring.
- Same report payload and run telemetry must always produce the same scorecard.
