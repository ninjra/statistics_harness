# Repo TOC (Start Here)

This file is the deterministic entry point for working in this repo.
Use it before broad searches.

## 1) First Files To Open

1. `AGENTS.md` (non-negotiables + workflow gates)
2. `README.md` (setup + runtime flags)
3. `REPO_TOC.md` (this file)

## 2) Top-Level Map

- `src/statistic_harness/`: core application and plugin runtime.
- `plugins/`: plugin spec/artifact support files.
- `tests/`: unit/integration/contract suites.
- `docs/`: implementation matrices, runbooks, release evidence.
- `scripts/`: repo automation, matrix generation, reporting helpers.
- `config/`: thresholds and runtime configuration.
- `appdata/`: local runtime outputs/state (gitignored).
- `unimplemented/`: deferred or parked work items.

## 3) Canonical Evidence + Matrices

- `docs/implementation_matrix.md`
- `docs/binding_implementation_matrix.md`
- `docs/plugins_functionality_matrix.md`
- `docs/plugin_data_access_matrix.md`
- `docs/sql_assist_adoption_matrix.md`
- `docs/_codex_plugin_catalog.md`
- `docs/release_evidence/` (run diffs and release artifacts)

## 4) Canonical Commands

- Tests: `python -m pytest -q`
- Plugin list: `stat-harness list-plugins`
- Release gate (docs + pytest + optional run bundle): `python scripts/run_release_gate.py --run-id <run_id>`
- Post-run evidence bundle: `python scripts/build_post_run_bundle.py --run-id <run_id> --strict`
- Reason-code burndown (single run): `python scripts/actionability_burndown.py --run-id <run_id>`
- Reason-code burndown delta (before/after): `python scripts/actionability_burndown.py --run-id <after_run_id> --before-run-id <before_run_id>`
- Respectful full-run launcher: `bash scripts/start_full_loaded_dataset_bg.sh`
- Regenerate core repo docs:
  - `python scripts/generate_codex_repo_manifest.py`
  - `python scripts/generate_codex_plugin_catalog.py`
  - `python scripts/plugins_functionality_matrix.py`
  - `python scripts/plugin_data_access_matrix.py`
  - `python scripts/sql_assist_adoption_matrix.py`
  - `python scripts/docs_coverage_matrix.py`
  - `python scripts/binding_implementation_matrix.py`

## 5) Task Routing (Open This, Not The Whole Repo)

- "What is implemented vs missing?": `docs/implementation_matrix.md`
- "Which plugin touches what data?": `docs/plugin_data_access_matrix.md`
- "Which plugins are actionable/non-actionable?": `docs/plugins_functionality_matrix.md`
- "What changed between runs?": files in `docs/release_evidence/`
- "What should be fixed next?": `docs/repo_improvements_catalog_v3.json`

## 6) Agent Workflow Rule

Before any wide scan, read `REPO_TOC.md` and route directly to the relevant artifact.
