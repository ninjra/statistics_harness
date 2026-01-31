# PROJECT SPEC — Statistic Harness (Phase 1: local-only)

Repository: `ninjra/statistic_harness`

## 0) Objective
Build a local-only, plugin-first data analysis harness that:
1) Accepts uploaded tabular files (CSV/TSV/TXT delimited, XLSX, JSON).
2) Runs a configurable set of statistical analysis plugins to detect hidden patterns.
3) Optionally builds an OFFLINE LLM prompt artifact (no network calls) to summarize findings.
4) Stores state in SQLite and artifacts on disk (self-contained).
5) Produces both:
   - human report: `report.md`
   - machine report: `report.json` validating against `docs/report.schema.json`
6) Includes a test suite that passes (`pytest -q`) with synthetic fixtures and an evaluator harness.

## 1) Phase constraints
### Phase 1 (this repo must implement fully)
- Local-only: no auth, no multi-tenant.
- No external services: no Qdrant, no server DBs.
- No network calls during analysis runs.
- Portable across Windows 11 + WSL2 (code must be OS-agnostic; scripts provided for PS + bash).

### Phase 2 (future stub only)
- Multi-tenant + auth
- isolated per-tenant storage namespace
- NOT implemented now; only leave clear TODOs and a minimal `core/tenancy.py` placeholder.

## 2) Non-negotiables
- All pipeline steps must be plugins/modules, including ingest + report.
- Plugin system must support:
  - Python plugins (executable)
  - Markdown plugins (static report sections/templates)
- All plugins are configurable:
  - enable/disable per run
  - per-plugin settings via JSON/YAML
- Deterministic runs:
  - a `run_seed` is assigned and recorded
  - all RNG usage must derive from it
- Every run produces a `report.md` and `report.json` even if some plugins fail.

## 3) Repo structure (must match exactly)
Use `src/` layout.

```

.
├─ AGENTS.md
├─ README.md
├─ LICENSE (MIT)
├─ pyproject.toml
├─ .gitignore
├─ config/
│  └─ app.yaml
├─ docs/
│  ├─ report.schema.json
│  ├─ plugin_dev_guide.md
│  ├─ evaluation.md
│  └─ references.md
├─ scripts/
│  ├─ bootstrap.ps1
│  ├─ bootstrap.sh
│  ├─ run_ui.ps1
│  ├─ run_ui.sh
│  ├─ run_cli_example.ps1
│  └─ run_cli_example.sh
├─ src/statistic_harness/
│  ├─ **init**.py
│  ├─ cli.py
│  ├─ core/
│  │  ├─ **init**.py
│  │  ├─ types.py
│  │  ├─ plugin_manager.py
│  │  ├─ pipeline.py
│  │  ├─ storage.py
│  │  ├─ dataset_io.py
│  │  ├─ evaluation.py
│  │  ├─ report.py
│  │  ├─ utils.py
│  │  ├─ vector_store.py
│  │  └─ tenancy.py  (stub only)
│  └─ ui/
│     ├─ **init**.py
│     ├─ server.py
│     ├─ templates/
│     │  ├─ index.html
│     │  ├─ run.html
│     │  └─ plugins.html
│     └─ static/
│        └─ app.css
├─ plugins/
│  ├─ ingest_tabular/
│  │  ├─ plugin.yaml
│  │  ├─ plugin.py
│  │  └─ README.md
│  ├─ profile_basic/
│  │  ├─ plugin.yaml
│  │  ├─ plugin.py
│  │  └─ README.md
│  ├─ analysis_conformal_feature_prediction/
│  ├─ analysis_online_conformal_changepoint/
│  ├─ analysis_gaussian_knockoffs/
│  ├─ analysis_knockoff_wrapper_rf/
│  ├─ analysis_notears_linear/
│  ├─ analysis_bocpd_gaussian/
│  ├─ analysis_scan_statistics/
│  ├─ analysis_graph_topology_curves/
│  ├─ analysis_dp_gmm/
│  ├─ analysis_gaussian_copula_shift/
│  ├─ report_bundle/
│  └─ llm_prompt_builder/
│
└─ tests/
├─ test_plugin_discovery.py
├─ test_pipeline_integration.py
├─ test_report_schema.py
├─ test_security_paths.py
├─ plugins/
│  ├─ test_conformal_feature_prediction.py
│  ├─ test_online_conformal_changepoint.py
│  ├─ test_gaussian_knockoffs.py
│  ├─ test_knockoff_wrapper_rf.py
│  ├─ test_notears_linear.py
│  ├─ test_bocpd_gaussian.py
│  ├─ test_scan_statistics.py
│  ├─ test_graph_topology_curves.py
│  ├─ test_dp_gmm.py
│  └─ test_gaussian_copula_shift.py
└─ fixtures/
├─ make_synth_data.py
├─ synth_linear.csv
├─ synth_timeseries.csv
├─ synth_clusters.csv
├─ synth_shift_corr.csv
└─ ground_truth_synth.yaml

```

## 4) Dependencies (pyproject.toml)
Must be Windows-friendly wheels and common.
Required runtime deps:
- fastapi
- uvicorn
- jinja2
- python-multipart
- pydantic
- pyyaml
- numpy
- pandas
- openpyxl
- scikit-learn
- jsonschema

Dev deps:
- pytest
- ruff (optional)
- mypy (optional)

NOTE: no external DB services.

## 5) Data + storage model
### 5.1 App data directory (gitignored)
- `./appdata/state.sqlite`
- `./appdata/uploads/<upload_id>/<original_filename>`
- `./appdata/runs/<run_id>/`
  - `dataset/canonical.csv`
  - `artifacts/<plugin_id>/...`
  - `report.json`
  - `report.md`
  - `logs/run.log`

### 5.2 SQLite schema (core/storage.py)
Tables:
- `runs`:
  - run_id TEXT PK
  - created_at TEXT (ISO8601)
  - status TEXT (queued|running|completed|failed)
  - upload_id TEXT
  - input_filename TEXT
  - canonical_path TEXT
  - settings_json TEXT
  - error_json TEXT NULL
- `plugin_results`:
  - run_id TEXT
  - plugin_id TEXT
  - status TEXT (ok|skipped|error)
  - summary TEXT
  - metrics_json TEXT
  - findings_json TEXT
  - artifacts_json TEXT
  - error_json TEXT NULL
  - PRIMARY KEY (run_id, plugin_id)

All JSON stored as TEXT.

## 6) Plugin system (core/plugin_manager.py)
### 6.1 Plugin manifest
Each plugin directory contains `plugin.yaml`:

Fields:
- id: string (unique)
- name: string
- version: semver string
- type: one of: ingest|profile|analysis|report|llm|markdown
- entrypoint: `plugin.py:Plugin` (class name)
- depends_on: list of plugin ids
- settings:
  - description: string
  - schema: JSON Schema object OR pydantic import path
  - defaults: dict

### 6.2 Plugin API
Define a base protocol in `core/types.py`:

- `PluginContext`:
  - run_id, run_dir, artifacts_dir(plugin_id)
  - sqlite connection handle (or Storage object)
  - dataset accessor (loads canonical.csv lazily)
  - run_seed
  - logger
  - settings dict (validated)
  - helpers to write artifacts

- `PluginResult`:
  - status
  - summary (short human text)
  - metrics: dict[str, float|int|str]
  - findings: list[dict]  (MACHINE fields; stable)
  - artifacts: list[{path,type,description}]
  - error: optional {type,message,traceback}

Plugin class must implement:
- `run(ctx: PluginContext) -> PluginResult`

### 6.3 Plugin discovery
- Scan `./plugins/*/plugin.yaml`
- Validate manifests.
- Import entrypoint.
- Return a registry list for UI and CLI.

## 7) Pipeline execution (core/pipeline.py)
Pipeline responsibilities:
- Create run_id + run_dir
- Call ingest plugin first to create canonical dataset
- Execute selected plugins in dependency order
- Always run report plugin at end
- Continue execution even if a plugin errors (record error and proceed)
- Ensure report files exist regardless of plugin errors

## 8) CLI (src/statistic_harness/cli.py)
Implement with argparse.

Commands:
- `stat-harness list-plugins`
- `stat-harness serve --host 127.0.0.1 --port 8000`
- `stat-harness run --file <path> --plugins <csv> --settings <json_or_yaml_path> --run-seed <int optional>`
- `stat-harness eval --report <report.json> --ground-truth <ground_truth.yaml>`
- `stat-harness make-ground-truth-template --report <report.json> -o <yaml>`

All commands must work on Windows + WSL.

## 9) UI (FastAPI + Jinja2)
Endpoints:
- GET `/` upload + plugin selection UI
- GET `/plugins` shows plugin list + descriptions
- POST `/api/upload` returns upload_id
- POST `/api/runs` create a run from upload_id + selected plugins + settings
- GET `/runs/{run_id}` status page (polling)
- GET `/api/runs/{run_id}/report.json`
- GET `/api/runs/{run_id}/report.md`
- GET `/api/runs/{run_id}/artifacts/{plugin_id}/{path}` (MUST be safe against traversal)

Run execution should be background thread; store status updates in sqlite.

## 10) Report format
### 10.1 report.json
Must conform to `docs/report.schema.json`.

Minimum fields:
- run_id, created_at, status
- input: {filename, rows, cols, inferred_types}
- plugins: object keyed by plugin_id, each has:
  - status, summary, metrics, findings, artifacts, error

### 10.2 report.md
Human readable:
- dataset summary
- executed plugins + summaries
- top findings list with links to artifacts
- error section (if any)

## 11) Implement the Phase 1 plugin catalog (100% required)

### ingest_tabular (type=ingest)
Goal: parse CSV/TSV/TXT/XLSX/JSON into pandas DataFrame and write canonical.csv.
Settings:
- delimiter: null|string
- sheet_name: null|string (xlsx)
- encoding: default utf-8
- max_rows: int (safety)
Outputs:
- dataset/canonical.csv
- dataset/schema.json: column names + inferred dtype + missingness

Tests:
- ingest csv fixture
- ingest xlsx fixture (generate on the fly in test)

### profile_basic (type=profile)
Compute:
- per-column: dtype, missing %, unique count, min/max/mean/std for numeric
- correlation matrix (numeric only) if <= N columns
Outputs:
- artifacts/profile_basic/columns.json
- artifacts/profile_basic/correlation.csv (optional)
Tests:
- validate basic stats exist

### analysis_conformal_feature_prediction (type=analysis)
Split conformal anomaly detection per numeric column:
- For each selected column y (default: all numeric up to max_cols):
  - Fit model y ~ X (Ridge regression)
  - Split train/calib/test by index (deterministic)
  - q = quantile(|resid_calib|, ceil((n+1)*(1-alpha))/n)
  - interval = pred ± q
  - flag anomalies in test
Settings:
- alpha: float (default 0.1)
- max_target_cols: int
- model: "ridge"
Outputs:
- artifacts/.../anomalies.csv
- artifacts/.../anomalies.json
Findings:
- list of {kind:"anomaly", column, row_index, score, lower, upper}
Test:
- synthetic linear data with injected outliers must be detected.

### analysis_online_conformal_changepoint (type=analysis)
Online conformal scores on a specified value column:
- rolling-mean forecast + calibration window quantile
- compute anomaly indicator
- detect change-point when anomaly-rate in last W exceeds threshold
Settings:
- time_column, value_column
- alpha, forecast_window, calib_window
- alarm_rate_window, alarm_rate_threshold
Outputs:
- artifacts/.../alerts.json
Findings:
- {kind:"changepoint", index, time, score}
Test:
- mean-shift series must produce changepoint within tolerance.

### analysis_gaussian_knockoffs (type=analysis)
Gaussian Model-X knockoffs (estimated covariance) + Lasso importance:
- generate knockoff features
- compute W_j = |beta_j| - |beta_j_knock|
- apply knockoff threshold for FDR q
Settings:
- target_column
- fdr_q (default 0.1)
- lasso_alpha (default 0.01)
Outputs:
- artifacts/.../selection.json
Findings:
- {kind:"feature_discovery", feature, score, selected:true/false}
Test:
- synthetic with 2 true features must select them.

### analysis_knockoff_wrapper_rf (type=analysis)
Same knockoffs generation, but compute importance via RandomForestRegressor:
- W_j = imp_j - imp_j_knock
Settings:
- target_column, fdr_q, n_estimators
Outputs:
- selection.json
Test:
- same synthetic; must recover true features.

### analysis_notears_linear (type=analysis)
Implement simplified NOTEARS-style linear DAG learning:
- minimize (1/2n)||X - XW||^2 + lambda*|W| with acyclicity constraint
- small d cap (default <= 20 numeric cols)
Settings:
- max_cols
- lambda_l1
- max_iter, lr
- weight_threshold
Outputs:
- artifacts/.../graph.json with:
  - nodes: [col names]
  - edges: [{source, target, weight}] AND also edges_compact: [[i,j,weight]]
Findings:
- {kind:"graph_edge", source, target, weight}
Test:
- 3-node chain synthetic should recover edges (within tolerance).

### analysis_bocpd_gaussian (type=analysis)
Bayesian Online Changepoint Detection (Adams & MacKay style):
- univariate series value_column
- constant hazard H
- conjugate normal prior for mean (known variance estimated)
- compute P(r_t=0) and detect peaks above threshold
Settings:
- value_column, time_column (optional)
- hazard (default 1/200)
- peak_threshold
Outputs:
- changepoints.json
Findings:
- {kind:"changepoint", index, prob}
Test:
- mean-shift series must be detected.

### analysis_scan_statistics (type=analysis)
1D scan over windows:
- search over window lengths [Lmin, Lmax]
- statistic = standardized mean difference
- p-value via permutation test (seeded)
Settings:
- value_column
- min_window, max_window
- n_permutations
Outputs:
- results.json includes top windows with score + p_value
Findings:
- {kind:"cluster", start, end, score, p_value}
Test:
- injected elevated segment must be top-ranked.

### analysis_graph_topology_curves (type=analysis)
Approx persistent “Betti-like” curves from threshold graphs:
- sample up to max_points points from numeric feature space
- compute pairwise distances
- for thresholds eps list:
  - build graph edges where dist <= eps
  - compute beta0 via union-find (components)
  - compute beta1 approx via E - V + C
Settings:
- max_points
- n_thresholds
Outputs:
- curves.json {eps:[], beta0:[], beta1:[]}
Findings:
- {kind:"topology", metric:"beta1_peak", value:...}
Test:
- circle-like synthetic should show beta1 peak > 0.

### analysis_dp_gmm (type=analysis)
DP Gaussian mixture (collapsed Gibbs), diagonal / spherical known variance:
- implement CRP assignment sampling with concentration alpha
- compute predictive log probs using conjugate normal prior for mean
Settings:
- alpha
- sigma2 (optional; estimate if null)
- n_iter
- burn_in
Outputs:
- assignments.csv
- summary.json with cluster sizes + means
Findings:
- {kind:"cluster", cluster_id, size}
Test:
- 2-component mixture must yield >=2 clusters and purity threshold.

### analysis_gaussian_copula_shift (type=analysis)
Dependence shift detection via Gaussian copula:
- split data into two segments (first half vs second half) OR rolling
- rank-transform each variable -> U in (0,1)
- probit transform -> Z
- compute corr matrices, delta = max abs diff
- identify top changed pairs; optional permutation p-value (seeded)
Settings:
- segment: "halves" (phase1)
- max_pairs
- n_permutations
Outputs:
- summary.json
Findings:
- {kind:"dependence_shift", pair:[a,b], delta, p_value}
Test:
- synthetic with correlation introduced in second half must flag that pair.

### report_bundle (type=report)
- Reads sqlite plugin_results + artifacts
- Writes report.json + report.md
- Validates JSON against schema; if invalid, fail test
Test:
- pipeline integration test ensures report files exist.

### llm_prompt_builder (type=llm)
OFFLINE ONLY:
- Reads report.json
- Writes:
  - artifacts/llm_prompt_builder/prompt.md (structured prompt)
  - artifacts/llm_prompt_builder/brief.md (short)
No network calls.
Test:
- prompt files exist and contain required sections.

## 12) Evaluator harness (core/evaluation.py)
Implement evaluation against a ground truth YAML schema:
- expected features discovered (by knockoffs plugins)
- expected changepoints within tolerance
- expected dependency shift pairs
- expected anomaly detection (at least K of known anomalies)
Provide:
- docs/evaluation.md
- tests/fixtures/ground_truth_synth.yaml
- a CLI `stat-harness eval ...` returning non-zero exit on failure.

## 13) Security regression tests
- Ensure artifact endpoint cannot read outside run directory (path traversal attempts).
- Ensure uploads are stored only inside `appdata/uploads`.

## 14) Scripts
Provide scripts for Windows PowerShell and bash:
- `scripts/bootstrap.*`:
  - create venv `.venv`
  - install deps `pip install -e ".[dev]"`
- `scripts/run_ui.*`:
  - run local server
- `scripts/run_cli_example.*`:
  - run an example analysis on a fixture dataset

## 15) Docs
Create `docs/references.md` listing conceptual sources (no code copying):
- Conformal prediction: https://arxiv.org/abs/2107.07511
- Model-X knockoffs: https://academic.oup.com/jrsssb/article/80/3/551/7048447
- NOTEARS: https://arxiv.org/abs/1803.01422
- BOCPD: https://arxiv.org/abs/0710.3742
- Spatial scan statistic: https://www.satscan.org/papers/k-cstm1997.pdf
- DP mixture MCMC: https://www2.stat.duke.edu/~scs/Courses/Stat376/Papers/DirichletProc/NealDirichletJCGS2000.pdf
- TDA survey example: https://www.frontiersin.org/articles/10.3389/frai.2021.681108/full
- Copula shift detection example: https://www.tandfonline.com/doi/full/10.1080/08982112.2025.2577437

## 16) Definition of Done (must meet)
- `python -m pytest -q` passes.
- `stat-harness list-plugins` shows all plugins above.
- UI:
  - upload CSV
  - select plugins
  - run completes
  - run page shows results
  - report downloads work
- Integration pipeline always writes report.md + report.json even if a plugin errors.
- report.json validates against docs/report.schema.json.

## 17) Implementation instruction (what Codex must do)
1) Create all files/directories above.
2) Implement core framework + UI + CLI.
3) Implement every plugin listed (Phase 1) with deterministic behavior and tests.
4) Run tests and fix until green.
5) Keep runtime offline (no network calls).
6) Do not copy code from the internet; implement algorithms from scratch based on the ideas only.
