# Autonomous Insight Quality Contract

## Purpose
Define deterministic rules for what counts as an actionable autonomous insight and what fails the run in strict autonomous mode.

## Required Recommendation Fields
A recommendation is considered structurally valid only when all of these fields are present:
- `plugin_id`
- `kind`
- `recommendation`
- `scope_class`
- `modeled_percent` or `not_modeled_reason`
- `modeled_basis_hours` or `not_modeled_reason`
- `modeled_delta_hours` or `not_modeled_reason`
- location context in at least one of:
  - `where`
  - `target_process_ids`
  - `evidence.target_process_ids`

## Actionable Insight Definition
An insight is actionable when all of the following are true:
- `status == "ok"`
- recommendation text is non-empty
- plugin is an analysis plugin or explicitly marked as an action-generator
- at least one concrete target is present (process id, process name, host, route, or cohort)
- modeled impact is present, or a deterministic `not_modeled_reason` is present

## Insight Signature (For Diffing)
Use this normalized signature to compare runs deterministically:
- `plugin_id`
- `kind`
- `action_type` (or `action`)
- normalized target scope (`where` + `target_process_ids`)
- SHA-256 hash of normalized recommendation text

This signature is the unit for:
- `new_count`
- `dropped_count`
- `unchanged_count`
- Jaccard similarity

## Autonomous Mode Fail Conditions (`known_issues_mode=off`)
Fail the run if any condition is true:
- zero discovery recommendations
- zero actionable plugins
- unexplained plugin count > 0
- blank finding kind count > 0
- explanation items missing plain-English reason
- non-decision plugins missing downstream dependency list
- autonomous novelty gate fails against reference run

## Non-Actionable Plugin Contract
A plugin that is not expected to emit direct recommendations must provide:
- deterministic `reason_code`
- plain-English explanation
- downstream plugin list consuming its outputs

Absence of this explanation lane is a failure.

## Determinism Requirements
- fixed `run_seed`
- stable sort order when generating recommendation lists
- stable signature hash implementation
- stable reference-run selection policy

## Reference Run Policy
For novelty gates, reference run is the latest successful run matching:
- same dataset version
- same plugin set hash
- same autonomous mode (`known_issues_mode=off`)
- same seed policy

