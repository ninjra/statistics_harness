# statistic-harness: `plugins validate` functionality pack

This single file is intended to be **self-contained**: it includes repository context, the current implementation excerpts, and a step-by-step plan to (a) confirm the feature exists, (b) implement it if missing, and (c) verify correctness (including regression tests).

**Generated:** 2026-02-22 (America/Denver)
**Thread:** statistic_harness_plugin_validate
**Chat:** unknown

> Scope assumption: “the plugin” refers to the CLI feature **`stat-harness plugins validate`** (EXT-02), i.e., validating the *plugin system* and plugin entrypoints. If you intended a different plugin (a specific `plugins/<id>`), the check/implement/verify workflow below still applies; replace `plugin_id` accordingly.

## 1) Repository purpose and pipeline (what this repo does)

### 1.1 Purpose (from code & manifests)

- The repo is a **statistics harness** that runs a deterministic pipeline over a dataset and executes a set of **plugins** discovered from `plugins/*/plugin.yaml`.
- In this repo snapshot there are **255 plugin manifests** under `plugins/*/plugin.yaml`.

### 1.2 Pipeline (high level)

At runtime, the CLI creates a `Pipeline`, which:

1. Creates a run directory under `runs/<run_id>/` and a `journal.json` for run state.
2. Resolves the dataset either from an input file (ingest) or from an existing DB dataset_version (`db://...`).
3. Discovers plugin specs via `PluginManager.discover()` and expands requested plugins with dependencies (`depends_on`).
4. Executes plugins (in dependency order) using `run_plugin_subprocess(...)`, persisting each plugin’s output in SQLite and writing artifacts under the run directory.
5. Builds a final report (`build_report`, `write_report`).

**Primary pipeline entrypoint excerpt (`Pipeline.run`):**

```python
        }

    def _expand_selected_with_deps(
        self, specs: dict[str, PluginSpec], selected: set[str]
    ) -> tuple[set[str], list[str], list[str]]:
        """Return (expanded, added, missing_deps).

        - Unknown selected plugin ids are ignored here (handled separately elsewhere).
        - Missing dependencies of known plugins are returned so the pipeline can fail closed.
        """

        expanded = set(selected)
        added: list[str] = []
        missing: list[str] = []
        stack = list(sorted(selected))
        while stack:
            pid = stack.pop()
            spec = specs.get(pid)
            if not spec:
                continue
            for dep in spec.depends_on or []:
                if dep not in specs:
                    missing.append(f"{pid} -> {dep}")
                    continue
                if dep not in expanded:
                    expanded.add(dep)
                    added.append(dep)
                    stack.append(dep)
        return expanded, sorted(set(added)), sorted(set(missing))

    def run(
        self,
        input_file: Path | None,
        plugin_ids: list[str],
        settings: dict[str, Any],
        run_seed: int,
        upload_id: str = "local",
        run_id: str | None = None,
        dataset_version_id: str | None = None,
        project_id: str | None = None,
        dataset_id: str | None = None,
        reuse_cache: bool | None = None,
        force: bool | None = None,
    ) -> str:
        run_id = run_id or make_run_id()
        settings = dict(settings or {})
        system_settings = (
            dict(settings.get("_system") or {})
            if isinstance(settings.get("_system"), dict)
            else {}
        )
        orchestrator_mode = _orchestrator_mode(system_settings.get("orchestrator_mode"))
        system_settings["orchestrator_mode"] = orchestrator_mode
        settings["_system"] = system_settings
        requested_run_seed = int(run_seed)
        tenant_id = self.tenant_id

        def _scope(value: str | None) -> str | None:
            if not value:
                return value
            if tenancy_enabled() and tenant_id != DEFAULT_TENANT_ID:
                return scope_identifier(tenant_id, value)
            return value

        dataset_version_id = _scope(dataset_version_id)
        project_id = _scope(project_id)
        dataset_id = _scope(dataset_id)

        runs_root = self.base_dir / "runs"
        staging_root = runs_root / "_staging"
        run_dir = runs_root / run_id
        def _prepare_staging(dir_path: Path) -> None:
            ensure_dir(dir_path / "dataset")
            ensure_dir(dir_path / "logs")
            write_json(
                dir_path / "journal.json",
                {
                    "run_id": run_id,
                    "status": "staging",
                    "pid": os.getpid(),
                    "created_at": now_iso(),
                },
            )

        atomic_dir(run_dir, staging_root=staging_root, prepare=_prepare_staging)
        write_json(
            run_dir / "journal.json",
            {"run_id": run_id, "status": "running", "pid": os.getpid(), "started_at": now_iso()},
        )
        progress_enabled = os.environ.get("STAT_HARNESS_CLI_PROGRESS", "").lower() in {
            "1",
            "true",
            "yes",
        }
        progress_tty = progress_enabled and sys.stdout.isatty()

        canonical_path = run_dir / "dataset" / "canonical.csv"
        dataset_row_count: int | None = None
        dataset_table_name: str | None = None
        dataset_column_count: int | None = None
        source_classification = normalize_source_classification(None)
        if input_file is None:
            if not dataset_version_id:
                raise ValueError("Dataset version is required for DB-only runs")
            ctx_row = self.storage.get_dataset_version_context(dataset_version_id)
            if not ctx_row:
                raise ValueError("Dataset version not found")
            project_id = ctx_row["project_id"]
            dataset_id = ctx_row["dataset_id"]
            dataset_table_name = str(ctx_row.get("table_name") or "")
            input_hash = ctx_row.get("data_hash") or dataset_version_id
            input_filename = f"db://{dataset_version_id}"
            source_classification = normalize_source_classification(
                str(ctx_row.get("source_classification") or "")
            )
            try:
                dataset_row_count = int(ctx_row.get("row_count") or 0)
            except (TypeError, ValueError):
                dataset_row_count = None
            try:
                dataset_column_count = int(ctx_row.get("column_count") or 0)
            except (TypeError, ValueError):
                dataset_column_count = None
        else:
            upload_row = self.storage.fetch_upload(upload_id) if upload_id else None
            upload_filename = (
                str(upload_row.get("filename") or input_file.name)
                if upload_row
                else input_file.name
            )
            source_classification = normalize_source_classification(
                str(upload_row.get("source_classification") or "") if upload_row else None,
                upload_filename,
            )
            input_hash = file_sha256(input_file)
            project_id = project_id or input_hash
            dataset_id = dataset_id or dataset_version_id or (
                input_hash
                if project_id == input_hash
                else dataset_key(project_id, input_hash)
            )
            dataset_version_id = dataset_version_id or dataset_id
            project_id = _scope(project_id)
            dataset_id = _scope(dataset_id)
            dataset_version_id = _scope(dataset_version_id)
            table_name = f"dataset_{dataset_version_id}"
            dataset_table_name = table_name
            existing = self.storage.get_dataset_version(dataset_version_id)
            if existing:
                existing_hash = existing.get("data_hash")
                same_media = False
                if existing_hash and existing_hash == input_hash:
                    same_media = True
                elif not existing_hash and existing.get("dataset_id") == dataset_id:
                    same_media = True
                if same_media:
                    self.storage.reset_dataset_version(
                        dataset_version_id,
                        table_name=existing.get("table_name") or table_name,
                        data_hash=input_hash,
                        created_at=now_iso(),
```

## 2) Plugin system contract (discovery, schemas, execution)

### 2.1 On-disk plugin layout

Each plugin lives at `plugins/<plugin_id>/` and is expected to include at minimum:

- `plugin.yaml` (manifest; schema-driven)
- `config.schema.json` (JSON Schema for settings; defaults can be applied)
- `output.schema.json` (JSON Schema for the plugin output payload)
- `plugin.py` (entrypoint module containing the entrypoint class in `entrypoint: module:Class`)

### 2.2 Manifest schema (what `plugin.yaml` must look like)

The repo includes a JSON Schema for manifests at `docs/plugin_manifest.schema.json`:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Statistic Harness Plugin Manifest",
  "type": "object",
  "additionalProperties": false,
  "required": [
    "id",
    "name",
    "version",
    "type",
    "entrypoint",
    "capabilities",
    "config_schema",
    "output_schema",
    "sandbox"
  ],
  "properties": {
    "id": {
      "type": "string",
      "pattern": "^[a-z0-9_]+$"
    },
    "name": {
      "type": "string",
      "minLength": 1
    },
    "version": {
      "type": "string",
      "minLength": 1
    },
    "type": {
      "type": "string",
      "enum": [
        "ingest",
        "profile",
        "analysis",
        "report",
        "llm",
        "planner",
        "transform"
      ]
    },
    "entrypoint": {
      "type": "string",
      "pattern": "^[^:]+:[A-Za-z_][A-Za-z0-9_]*$"
    },
    "depends_on": {
      "type": "array",
      "items": { "type": "string" },
      "default": []
    },
    "capabilities": {
      "type": "array",
      "items": { "type": "string" },
      "default": []
    },
    "lane": {
      "type": "string",
      "enum": ["decision", "explanation"]
    },
    "decision_capable": {
      "type": "boolean"
    },
    "requires_downstream_mapping": {
      "type": "boolean"
    },
    "config_schema": {
      "type": "string",
      "minLength": 1
    },
    "output_schema": {
      "type": "string",
      "minLength": 1
    },
    "sandbox": {
      "type": "object",
      "additionalProperties": false,
      "required": ["no_network", "fs_allowlist"],
      "properties": {
        "no_network": { "type": "boolean" },
        "fs_allowlist": {
          "type": "array",
          "items": { "type": "string" },
          "minItems": 1
        }
      }
    },
    "settings": {
      "type": "object",
      "additionalProperties": false,
      "required": ["description", "defaults"],
      "properties": {
        "description": { "type": "string" },
        "defaults": { "type": "object" }
      }
    }
  }
}
```

### 2.3 Plugin discovery and validation logic (`PluginManager`)

Plugin discovery and schema validation is implemented in `src/statistic_harness/core/plugin_manager.py`.

Key behaviors to know:

- `discover()` scans `plugins/*/plugin.yaml`, validates each manifest against `docs/plugin_manifest.schema.json`, checks required files exist, and produces a list of `PluginSpec`.
- `load_plugin(spec)` imports the entrypoint and instantiates the plugin class.
- `resolve_config(spec, settings)` applies JSON Schema defaults (where present) and validates against `config.schema.json`.
- `validate_output(spec, payload)` validates the output payload against `output.schema.json`.
- `health(spec, plugin)` calls optional `plugin.health()` and returns an `ok/unhealthy/error` status object.

**Implementation excerpt (PluginManager):**

```python
from __future__ import annotations

import importlib
import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from jsonschema import ValidationError, validate

from .utils import read_json


@dataclass
class PluginSpec:
    plugin_id: str
    name: str
    version: str
    type: str
    entrypoint: str
    depends_on: list[str]
    settings: dict[str, Any]
    path: Path
    capabilities: list[str]
    config_schema: Path
    output_schema: Path
    sandbox: dict[str, Any]
    lane: str = "explanation"
    decision_capable: bool = False
    requires_downstream_mapping: bool = False


@dataclass(frozen=True)
class PluginDiscoveryError:
    plugin_id: str
    path: Path
    message: str


class PluginManager:
    def __init__(self, plugins_dir: Path) -> None:
        self.plugins_dir = plugins_dir
        self._manifest_schema: dict[str, Any] | None = None
        self._schema_cache: dict[Path, dict[str, Any]] = {}
        self.discovery_errors: list[PluginDiscoveryError] = []

    def _record_discovery_error(
        self, plugin_id: str, manifest: Path, message: str
    ) -> None:
        self.discovery_errors.append(
            PluginDiscoveryError(
                plugin_id=plugin_id or manifest.parent.name,
                path=manifest,
                message=message,
            )
        )

    def discover(self) -> list[PluginSpec]:
        specs: list[PluginSpec] = []
        self.discovery_errors = []
        manifest_schema = self._load_manifest_schema()
        seen: set[str] = set()
        for manifest in sorted(self.plugins_dir.glob("*/plugin.yaml")):
            try:
                data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
            except Exception as exc:  # pragma: no cover - malformed YAML
                self._record_discovery_error(
                    manifest.parent.name,
                    manifest,
                    f"Invalid YAML: {exc}",
                )
                continue
            if not isinstance(data, dict):
                self._record_discovery_error(
                    manifest.parent.name, manifest, "Invalid manifest payload"
                )
                continue
            plugin_id = str(data.get("id") or manifest.parent.name)
            try:
                validate(instance=data, schema=manifest_schema)
            except ValidationError as exc:
                self._record_discovery_error(
                    plugin_id, manifest, f"Invalid manifest: {exc.message}"
                )
                continue
            if plugin_id in seen:
                self._record_discovery_error(
                    plugin_id, manifest, "Duplicate plugin id"
                )
                continue
            config_schema_path = manifest.parent / data["config_schema"]
            output_schema_path = manifest.parent / data["output_schema"]
            if not config_schema_path.exists():
                self._record_discovery_error(
                    plugin_id,
                    manifest,
                    f"Missing config schema: {config_schema_path}",
                )
                continue
            if not output_schema_path.exists():
                self._record_discovery_error(
                    plugin_id,
                    manifest,
                    f"Missing output schema: {output_schema_path}",
                )
                continue
            defaults = data.get("settings", {}).get("defaults", {})
            if defaults is not None:
                try:
                    self.validate_config_schema(config_schema_path, defaults)
                except ValidationError as exc:
                    self._record_discovery_error(
                        plugin_id,
                        manifest,
                        f"Invalid config defaults: {exc.message}",
                    )
                    continue
            capabilities = [str(v).strip() for v in list(data.get("capabilities", []))]
            lane_raw = str(data.get("lane") or "").strip().lower()
            if lane_raw and lane_raw not in {"decision", "explanation"}:
                self._record_discovery_error(
                    plugin_id,
                    manifest,
                    f"Invalid lane value: {lane_raw}",
                )
                continue
            inferred_lane = lane_raw
            if not inferred_lane:
                if data["type"] == "analysis" and "diagnostic_only" not in set(capabilities):
                    inferred_lane = "decision"
                else:
                    inferred_lane = "explanation"
            decision_capable_raw = data.get("decision_capable")
            if decision_capable_raw is None:
                decision_capable = inferred_lane == "decision"
            else:
                decision_capable = bool(decision_capable_raw)
            requires_downstream_raw = data.get("requires_downstream_mapping")
            if requires_downstream_raw is None:
                requires_downstream_mapping = inferred_lane != "decision"
            else:
                requires_downstream_mapping = bool(requires_downstream_raw)
            seen.add(plugin_id)
            specs.append(
                PluginSpec(
                    plugin_id=plugin_id,
                    name=data["name"],
                    version=data["version"],
                    type=data["type"],
                    entrypoint=data["entrypoint"],
                    depends_on=data.get("depends_on", []),
                    settings=data.get("settings", {}),
                    path=manifest.parent,
                    capabilities=capabilities,
                    lane=inferred_lane,
                    decision_capable=decision_capable,
                    requires_downstream_mapping=requires_downstream_mapping,
                    config_schema=config_schema_path,
                    output_schema=output_schema_path,
                    sandbox=dict(data.get("sandbox", {})),
                )
            )
        return specs

    def load_plugin(self, spec: PluginSpec) -> Any:
        module_path, class_name = spec.entrypoint.split(":", 1)
        if module_path.endswith(".py"):
            module_path = module_path[:-3]
        module_name = f"plugins.{spec.plugin_id}.{module_path}"
        module = importlib.import_module(module_name)
        return getattr(module, class_name)()

    def health(self, spec: PluginSpec, plugin: Any | None = None) -> dict[str, Any]:
        """Best-effort plugin health check.

        Contract: a plugin instance may implement `health()` returning one of:
        - dict with a `status` field ("ok"/"unhealthy"/"error")
        - bool (True=ok, False=unhealthy)
        - None (treated as ok)
        """

        if plugin is None:
            try:
                plugin = self.load_plugin(spec)
            except Exception as exc:
                return {"status": "error", "message": f"Load failed: {type(exc).__name__}: {exc}"}
        fn = getattr(plugin, "health", None)
        if not callable(fn):
            return {"status": "ok", "note": "No health() implemented"}
        try:
            result = fn()
        except Exception as exc:
            return {"status": "error", "message": f"health() raised: {type(exc).__name__}: {exc}"}
        if result is None:
            return {"status": "ok"}
        if isinstance(result, bool):
            return {"status": "ok" if result else "unhealthy"}
        if isinstance(result, dict):
            status = str(result.get("status") or "ok")
            payload = dict(result)
            payload["status"] = status
            return payload
        return {"status": "ok", "detail": str(result)}

    def validate_config(self, spec: PluginSpec, config: dict[str, Any]) -> None:
        schema = self._load_schema(spec.config_schema)
        validate(instance=config, schema=schema)

    def resolve_config(self, spec: PluginSpec, config: dict[str, Any]) -> dict[str, Any]:
        """Apply JSONSchema defaults deterministically, then validate."""

        schema = self._load_schema(spec.config_schema)
        resolved: dict[str, Any] = copy.deepcopy(config)
        _apply_jsonschema_defaults(schema, resolved)
        validate(instance=resolved, schema=schema)
        return resolved

    def validate_output(self, spec: PluginSpec, payload: dict[str, Any]) -> None:
        schema = self._load_schema(spec.output_schema)
        try:
            validate(instance=payload, schema=schema)
            return
        except ValidationError as exc:
            # Back-compat: many plugin output schemas still enumerate "skipped" while
            # runtime now persists deterministic not-applicable as status="na".
            status = str(payload.get("status") or "").strip().lower()
            if status != "na":
                raise
            for alias in ("skipped", "degraded", "not_applicable"):
                probe = dict(payload)
                probe["status"] = alias
                try:
                    validate(instance=probe, schema=schema)
                    return
                except ValidationError:
                    continue
            raise exc

    @staticmethod
    def result_payload(result: Any) -> dict[str, Any]:
        return {
            "status": getattr(result, "status", None),
            "summary": getattr(result, "summary", ""),
            "metrics": getattr(result, "metrics", {}),
            "findings": getattr(result, "findings", []),
            "artifacts": [asdict(a) for a in getattr(result, "artifacts", [])],
            "budget": getattr(result, "budget", None),
            "error": asdict(result.error) if getattr(result, "error", None) else None,
            "references": getattr(result, "references", []),
            "debug": getattr(result, "debug", {}),
        }

    def _load_schema(self, path: Path) -> dict[str, Any]:
        if path not in self._schema_cache:
            self._schema_cache[path] = read_json(path)
        return self._schema_cache[path]

    def _load_manifest_schema(self) -> dict[str, Any]:
        if self._manifest_schema is None:
            schema_path = self.plugins_dir.parent / "docs" / "plugin_manifest.schema.json"
            self._manifest_schema = read_json(schema_path)
        return self._manifest_schema

    def validate_config_schema(self, schema_path: Path, defaults: dict[str, Any]) -> None:
        schema = self._load_schema(schema_path)
        validate(instance=defaults, schema=schema)


def _apply_jsonschema_defaults(schema: Any, instance: Any) -> Any:
    """Recursively apply `default` values from a JSONSchema into `instance`.

    This intentionally handles only the common subset used in this repo:
    - object properties + their defaults
    - array items schemas
    - allOf composition
    """

    if not isinstance(schema, dict):
        return instance

    # If the instance is "missing" at this node, apply the node default.
    if instance is None and "default" in schema:
        instance = copy.deepcopy(schema["default"])

    for subschema in schema.get("allOf") or []:
        instance = _apply_jsonschema_defaults(subschema, instance)

    schema_type = schema.get("type")
    if schema_type == "object" and isinstance(instance, dict):
        props = schema.get("properties") or {}
        for key in sorted(props.keys()):
            prop_schema = props.get(key)
            if key not in instance:
                if isinstance(prop_schema, dict) and "default" in prop_schema:
                    instance[key] = copy.deepcopy(prop_schema["default"])
            if key in instance:
                instance[key] = _apply_jsonschema_defaults(prop_schema, instance[key])

        additional = schema.get("additionalProperties")
        if isinstance(additional, dict):
            for key in sorted(instance.keys()):
                if key in props:
                    continue
                instance[key] = _apply_jsonschema_defaults(additional, instance[key])

    if schema_type == "array" and isinstance(instance, list):
        items_schema = schema.get("items")
        if isinstance(items_schema, dict):
            for idx, value in enumerate(list(instance)):
                instance[idx] = _apply_jsonschema_defaults(items_schema, value)

    return instance
```

### 2.4 Note on JSON Schema defaults

JSON Schema’s `default` keyword is an **annotation**, not something validators automatically apply. This repo therefore includes `_apply_jsonschema_defaults(...)` and uses it in `resolve_config(...)` to materialize defaults before validation.

## 3) Target functionality: `stat-harness plugins validate` (EXT-02)

### 3.1 Requirement statement (as written in the repo plan)

From `docs/deprecated/completed_plans/docs/codex_4pillars_golden_release_plan.md` (EXT-02):

```text
134: | EXT-02 | **CLI: `stat-harness plugins validate` (schema + import + smoke + caps)**<br>Rationale: Catch broken plugins early; reduce admin/operator friction.<br>Dependencies: None.<br>Improved: safety and correctness.<br>Risked: validation may be slow without caching.<br>Enforce: plugin_manager + CLI command uses schema validation and imports entrypoint in isolated mode.<br>Regression detection: validation must fail on malformed manifest; if passes => **DO_NOT_SHIP**.<br>Acceptance test: run validate across all plugins; outputs a report with pass/fail and required capabilities.                                                                                                                                                       | P1=0 P2=2 P3=2 P4=2 (Total=6) | S / Low       |
```

### 3.2 Existence check: is it already implemented in this repo snapshot?

Yes. The command is implemented as `cmd_plugins_validate(...)` in `src/statistic_harness/cli.py`, and is wired into the CLI dispatch for `stat-harness plugins validate`.

**CLI entrypoint registration (`pyproject.toml`):**

```toml
  "pytest",
  "ruff",
  "mypy",
  "playwright",
]

[project.scripts]
stat-harness = "statistic_harness.cli:main"

[tool.setuptools.packages.find]
where = ["src"]
```

**CLI implementation excerpt (`cmd_plugins_validate`):**

```python
def cmd_plugins_validate(plugin_id: str | None = None) -> None:
    manager = PluginManager(Path("plugins"))
    specs = manager.discover()
    failures: list[str] = []

    if manager.discovery_errors:
        for err in manager.discovery_errors:
            failures.append(f"{err.plugin_id}: discovery error: {err.message}")

    selected = specs
    if plugin_id:
        selected = [spec for spec in specs if spec.plugin_id == plugin_id]
        if not selected:
            raise SystemExit(f"Unknown plugin id: {plugin_id}")

    for spec in selected:
        try:
            # Load and validate schemas (discovery already checks presence).
            manager.validate_config(spec, dict(spec.settings.get("defaults", {})))
            # Import entrypoint and instantiate plugin.
            plugin = manager.load_plugin(spec)
            if not hasattr(plugin, "run"):
                raise TypeError("Missing run() method")
            health = manager.health(spec, plugin=plugin)
            if str(health.get("status") or "ok").lower() not in {"ok", "healthy"}:
                raise RuntimeError(f"Unhealthy plugin: {health}")
        except Exception as exc:
            failures.append(f"{spec.plugin_id}: {type(exc).__name__}: {exc}")

    for line in sorted(failures):
        print(line)
    if failures:
        raise SystemExit(1)
    print("OK")

```

**CLI subcommand wiring excerpt (argparse + dispatch):**

```python


def _resolve_dimensions(
    store: VectorStore, collection: str, dimensions: int | None
) -> int:
    if dimensions is not None:
        return int(dimensions)
    dims = store.collection_dimensions(collection)
    if not dims:
        raise SystemExit("Unknown collection or dimensions required")
    if len(dims) > 1:
        raise SystemExit("Multiple dimensions found; use --dimensions")
    return int(dims[0])


def cmd_vector_query(
    collection: str,
    text: str | None,
    vector: str | None,
    k: int,
    dimensions: int | None,
) -> None:
    _require_vector_store()
    if auth_enabled():
        _require_cli_api_key()
    tenant_ctx = get_tenant_context()
    store = VectorStore(tenant_ctx.db_path, tenant_ctx.tenant_id)
    if text and vector:
        raise SystemExit("Use either --text or --vector")
    if not text and not vector:
        raise SystemExit("Provide --text or --vector")
    if vector:
        parts = [part for part in vector.replace("\n", ",").split(",") if part.strip()]
        values = [float(part.strip()) for part in parts]
        query_vector = values
    else:
        dims = _resolve_dimensions(store, collection, dimensions)
        query_vector = hash_embedding(text or "", dims)
    results = store.query(collection, query_vector, k=k)
    print(json.dumps(results, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-plugins")

    plugins_parser = sub.add_parser("plugins")
    plugins_sub = plugins_parser.add_subparsers(dest="plugins_command", required=True)
    plugins_validate_parser = plugins_sub.add_parser("validate")
    plugins_validate_parser.add_argument("--plugin-id")

    integrity_parser = sub.add_parser("integrity-check")
    integrity_parser.add_argument("--full", action="store_true")

    backup_parser = sub.add_parser("backup")
    backup_parser.add_argument("--out")
    backup_parser.add_argument("--retention-days", type=int, default=60)

    restore_parser = sub.add_parser("restore")
    restore_parser.add_argument("--path", required=True)

    diag_parser = sub.add_parser("diag")
    diag_parser.add_argument("--run-id", required=True)
    diag_parser.add_argument("--out")

    export_parser = sub.add_parser("export")
    export_parser.add_argument("--run-id", required=True)
    export_parser.add_argument("--out")
    export_parser.add_argument("--no-logs", action="store_true")
    export_parser.add_argument("--no-artifacts", action="store_true")
    export_parser.add_argument("--include-dataset", action="store_true")

    replay_parser = sub.add_parser("replay")
    replay_parser.add_argument("--run-id", required=True)

    db_parser = sub.add_parser("db")
    db_sub = db_parser.add_subparsers(dest="db_command", required=True)
    db_doctor = db_sub.add_parser("doctor")
    db_doctor.add_argument("--full", action="store_true")
    tenant_parser.add_argument("--tenant-id", required=True)
    tenant_parser.add_argument("--name")

    revoke_key_parser = sub.add_parser("revoke-api-key")
    revoke_key_parser.add_argument("--key-id", required=True, type=int)

    disable_user_parser = sub.add_parser("disable-user")
    disable_user_parser.add_argument("--email", required=True)

    sub.add_parser("vector-list")

    vector_query_parser = sub.add_parser("vector-query")
    vector_query_parser.add_argument("--collection", required=True)
    vector_query_parser.add_argument("--text")
    vector_query_parser.add_argument("--vector")
    vector_query_parser.add_argument("--k", type=int, default=10)
    vector_query_parser.add_argument("--dimensions", type=int)

    args = parser.parse_args()

    if args.command == "list-plugins":
        cmd_list_plugins()
    elif args.command == "plugins":
        if args.plugins_command == "validate":
            cmd_plugins_validate(args.plugin_id)
        else:
            raise SystemExit(2)
    elif args.command == "integrity-check":
        cmd_integrity_check(full=bool(args.full))
    elif args.command == "backup":
        cmd_backup(args.out, args.retention_days)
    elif args.command == "restore":
        cmd_restore(args.path)
    elif args.command == "diag":
        cmd_diag(args.run_id, args.out)
    elif args.command == "export":
        cmd_export(
            args.run_id,
            args.out,
            include_logs=not bool(args.no_logs),
            include_artifacts=not bool(args.no_artifacts),
            include_dataset=bool(args.include_dataset),
        )
    elif args.command == "replay":
        cmd_replay(args.run_id)
    elif args.command == "db":
```

**Existing unit test (smoke) for the command:**

```python
from statistic_harness.cli import cmd_plugins_validate


def test_cli_plugins_validate_profile_basic():
    # Keep this fast: validate a single known-good plugin.
    cmd_plugins_validate("profile_basic")
```

### 3.3 What `plugins validate` checks today (behavioral contract in this snapshot)

Based on the code excerpt above, current behavior is:

- **Manifest/schema/discovery validation** via `PluginManager.discover()` (includes manifest schema validation + required file existence).
- **Config schema validation** via `manager.validate_config(spec, dict(spec.settings.get("defaults", {})))`.
  - Note: this validates **only the manifest-provided defaults** (if any). It does *not* call `resolve_config(...)`, so JSON Schema defaults in `config.schema.json` are not materialized before validation.
- **Import/instantiate entrypoint** via `load_plugin(spec)`.
- **Entry point interface check**: ensures plugin instance has `run`.
- **Health check (optional)** via `PluginManager.health(spec, plugin=plugin)`; if plugin implements `health()`, failures are surfaced.
- Prints `OK` on success; prints sorted failure lines and exits with code 1 on failure.

### 3.4 Gaps vs EXT-02 wording (if you treat EXT-02 as the acceptance spec)

If EXT-02 is treated as the acceptance spec, the current implementation is **partial**:

- No true **smoke run** of `plugin.run(ctx)` is performed (only `health()` is checked).
- Entry point import is **not isolated** (it imports in-process, not via the sandboxed subprocess runner).
- It does not emit a structured **report artifact** (only stdout).
- It does not explicitly output/verify **required capabilities** beyond reading them from the manifest.
- It does not apply JSON Schema defaults from `config.schema.json` during validation (unless defaults are duplicated into `plugin.yaml` under `settings.defaults`).

## 4) How to check existence in any working copy (step-by-step)

Run these checks in a real clone of the repository (not the repomix snapshot):

1. **Code existence**
   - `rg "def cmd_plugins_validate" -n src/statistic_harness/cli.py`
   - `rg "plugins validate" -n src/statistic_harness/cli.py`

2. **CLI wiring**
   - `stat-harness --help` should show a `plugins` command and a `validate` subcommand.
   - Or: `python -m statistic_harness.cli --help`

3. **Behavior**
   - Validate a known-good plugin: `stat-harness plugins validate --plugin-id profile_basic`
   - Validate everything: `stat-harness plugins validate`

4. **Existing test**
   - `pytest -q tests/test_cli_plugins_validate.py`

## 5) If missing (or if fixing gaps): implementation recipes

### 5.1 Baseline implementation (matches snapshot)

If you want the exact baseline behavior seen in this snapshot, implement `cmd_plugins_validate` exactly as in §3.2.

### 5.2 Recommended fix: apply JSON Schema defaults from `config.schema.json`

To make validation honor schema defaults (and handle the case where a schema provides defaults for required fields), change the config validation line from:

```python
manager.validate_config(spec, dict(spec.settings.get("defaults", {})))
```

to:

```python
# Merge any manifest-provided defaults (optional) then apply schema defaults deterministically
raw = dict(spec.settings.get("defaults", {}))
resolved = manager.resolve_config(spec, raw)  # applies defaults + validates
# (Optional) manager.validate_config(spec, resolved)  # redundant; resolve_config already validates
```

### 5.3 Add/keep the smoke test

Add `tests/test_cli_plugins_validate.py` (or equivalent) with a fast, single-plugin smoke test:

```python
from statistic_harness.cli import cmd_plugins_validate


def test_cli_plugins_validate_profile_basic():
    # Keep this fast: validate a single known-good plugin.
    cmd_plugins_validate("profile_basic")
```

## 6) How to verify correctness (for the baseline implementation)

### 6.1 Expected pass/fail semantics

The validator should:

- ✅ **Fail** if any plugin manifest is malformed (schema invalid), missing required fields, or required files are missing.
- ✅ **Fail** if a plugin’s entrypoint cannot be imported/instantiated.
- ✅ **Fail** if the instantiated plugin has no `run` method.
- ✅ **Fail** if config validation fails (either manifest defaults fail schema, or—if you implement §5.2—schema defaults cannot be applied/validated).
- ✅ **Fail** if `plugin.health()` exists and returns `status != ok`.
- ✅ **Pass** when all selected plugins satisfy the above.

### 6.2 Existing automated coverage

The repo already has tests that cover the underlying validation machinery:

- `tests/test_plugin_manifest_schema.py` validates manifest schema enforcement and config/output schema validation paths.
- `tests/test_config_defaults.py` validates default materialization logic in `resolve_config` (useful if you implement §5.2).

`tests/test_plugin_manifest_schema.py` excerpt:

```python
from pathlib import Path

import pytest
from jsonschema import ValidationError

from statistic_harness.core.plugin_manager import PluginManager


def test_manifest_schema_validation(tmp_path: Path) -> None:
    schema_src = Path("docs/plugin_manifest.schema.json")
    schema_dst = tmp_path / "docs" / "plugin_manifest.schema.json"
    schema_dst.parent.mkdir(parents=True, exist_ok=True)
    schema_dst.write_text(schema_src.read_text(encoding="utf-8"), encoding="utf-8")

    plugins_dir = tmp_path / "plugins"
    plugin_dir = plugins_dir / "bad_plugin"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "config.schema.json").write_text("{}", encoding="utf-8")
    (plugin_dir / "output.schema.json").write_text("{}", encoding="utf-8")
    (plugin_dir / "plugin.yaml").write_text(
        """id: bad_plugin
name: Bad Plugin
version: 0.1.0
type: analysis
entrypoint: plugin.py:Plugin
""",
        encoding="utf-8",
    )

    manager = PluginManager(plugins_dir)
    specs = manager.discover()
    assert specs == []
    assert manager.discovery_errors
    assert any(
        "Invalid manifest" in err.message for err in manager.discovery_errors
    )


def test_plugin_config_and_output_validation() -> None:
    manager = PluginManager(Path("plugins"))
    specs = {spec.plugin_id: spec for spec in manager.discover()}
    ingest = specs["ingest_tabular"]

    manager.validate_config(
        ingest,
        {"encoding": "utf-8", "delimiter": None, "sheet_name": None, "chunk_size": 10},
    )
    with pytest.raises(ValidationError):
        manager.validate_config(ingest, {"chunk_size": "bad"})

    manager.validate_output(
        ingest,
        {
            "status": "ok",
            "summary": "ok",
            "metrics": {},
            "findings": [],
            "artifacts": [],
            "budget": {
                "row_limit": None,
                "sampled": False,
                "time_limit_ms": None,
                "cpu_limit_ms": None,
            },
            "error": None,
            "references": [],
            "debug": {},
        },
    )
```

### 6.3 Recommended additional regression tests (to make `plugins validate` hard to break)

Add tests that create temporary plugins under `tmp_path/plugins/<id>` and ensure `cmd_plugins_validate()` fails with `SystemExit(1)` and a precise error message. Suggested cases:

1. **Bad entrypoint import**: manifest points to `plugin.py:Plugin` but file missing → must fail.
2. **Missing `run` method**: entrypoint loads but class lacks `run` → must fail.
3. **Health returns unhealthy**: plugin implements `health()` returning `{status:'unhealthy'}` → must fail.
4. **Config schema default required**: schema marks a field required and provides a default; without applying defaults, validation fails. With §5.2 applied, validation should pass (regression guard).
5. **Filter works**: `--plugin-id X` validates only X (does not fail due to unrelated broken plugin).

### 6.4 DO_NOT_SHIP conditions

Treat these as hard regressions:

- If a malformed manifest plugin is present and `plugins validate` exits success → **DO_NOT_SHIP**.
- If an entrypoint that cannot be imported passes validation → **DO_NOT_SHIP**.
- If a plugin with `health()` returning unhealthy passes validation → **DO_NOT_SHIP**.
- If the command becomes non-deterministic in ordering/output across runs → **DO_NOT_SHIP** (sort failures, stable traversal order).

## 7) Optional: bring `plugins validate` up to the full EXT-02 spec

If you want to match the EXT-02 wording more literally (schema + import + smoke + caps, isolated), here is a design that fits the repo’s architecture.

### 7.1 Add an isolated (subprocess) import/health mode

Today, `cmd_plugins_validate` imports plugins in-process. The repo already has a sandboxed subprocess runner used by the pipeline. You can extend validation to run in an isolated subprocess by adding a lightweight **`action: health`** mode to the runner:

1. Extend the plugin runner request dict to include `action` in `{run, health}` (default `run`).
2. In the subprocess, if `action == health`, import the plugin and call `plugin.health()` without requiring a dataset or SQLite state.
3. Return a small JSON payload with pass/fail + capability info.

This preserves security properties: sandbox + optional no-network enforcement.

### 7.2 Add a smoke run option

Add a `--smoke` flag that runs each plugin’s `run(ctx)` with a tiny synthetic dataset and validates the produced output payload against `output.schema.json` via `PluginManager.validate_output(...)`.

Implementation options:

- **Fast / simple (in-process)**: mirror the `tests/conftest.py::make_context` approach to build a temporary `Storage` and `PluginContext`, then call `plugin.run(ctx)` directly. Easiest, but not sandboxed.
- **More secure (subprocess)**: create a temporary appdata directory containing a minimal `state.sqlite` with the required tables + dataset version, then run `run_plugin_subprocess` (the same path used by pipeline). Matches production execution, more work.

### 7.3 Caps report

At minimum, include in the output report per plugin:

- `capabilities` (from manifest)
- `sandbox.no_network` and filesystem allowlists
- Any derived “requires” you can infer (e.g., needs vector store / sqlite / LLM).

### 7.4 Structured report output

Add `--json <path>` to write a deterministic report like:

```json
{
  "validated_at": "2026-02-22T00:00:00-07:00",
  "summary": {"total": 255, "passed": 255, "failed": 0},
  "plugins": [
    {"plugin_id": "profile_basic", "status": "pass", "checks": ["manifest", "config_schema", "import", "health"], "capabilities": [], "sandbox": {"no_network": true}},
    {"plugin_id": "analysis_xyz", "status": "fail", "error": "ImportError: ..."}
  ]
}
```

## 8) Appendix: 30 additional statistical techniques (with live references)

These are candidate techniques for future plugins or analyses. Each entry includes a short use case and a reference link (paper/book). These links are **optional** reading; the implementation guidance in this file does not require them.

- **Conformal prediction** — Distribution-free prediction intervals / calibrated uncertainty.  
  Reference: https://arxiv.org/abs/0706.3188
- **False Discovery Rate (Benjamini–Hochberg)** — Multiple testing correction controlling expected false discoveries.  
  Reference: https://doi.org/10.1111/j.2517-6161.1995.tb02031.x
- **BCa bootstrap confidence intervals** — Bias-corrected and accelerated bootstrap intervals.  
  Reference: https://doi.org/10.1080/01621459.1987.10478410
- **Knockoff filter** — Controlled variable selection with FDR guarantees.  
  Reference: https://arxiv.org/abs/1404.5609
- **Quantile regression** — Model conditional quantiles; robust to outliers; heteroskedasticity-aware.  
  Reference: https://www.econ.uiuc.edu/~roger/research/rq/jasa.pdf
- **LASSO** — Sparse linear models via L1 regularization.  
  Reference: https://doi.org/10.1111/j.2517-6161.1996.tb02080.x
- **Elastic Net** — Sparse + grouped selection via L1+L2 regularization.  
  Reference: https://doi.org/10.1111/j.1467-9868.2005.00503.x
- **Random Forests** — Ensemble trees for nonlinear prediction + feature importance.  
  Reference: https://doi.org/10.1023/A:1010933404324
- **Gradient Boosting Machine** — Additive model built by stagewise optimization (boosting).  
  Reference: https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation--A-gradient-boosting-machine/10.1214/aos/1013203451.full
- **Isolation Forest** — Anomaly detection by isolation in random partition trees.  
  Reference: https://doi.org/10.1109/ICDM.2008.17
- **Local Outlier Factor (LOF)** — Density-based outlier score relative to neighbors.  
  Reference: https://doi.org/10.1145/335191.335388
- **Minimum Covariance Determinant (MCD)** — Robust covariance estimation (outlier-resistant).  
  Reference: https://wis.kuleuven.be/stat/robust/papers/2010/wire-mcd-review.pdf
- **Huber M-estimator** — Robust estimation reducing influence of outliers.  
  Reference: https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-35/issue-1/Robust-Estimation-of-a-Location-Parameter/10.1214/aoms/1177703732.full
- **Gaussian Process Regression** — Nonparametric Bayesian regression with kernel priors.  
  Reference: https://mitpress.mit.edu/9780262182539/gaussian-processes-for-machine-learning/
- **Generalized Additive Models (GAM)** — Nonlinear additive effects with smooth terms.  
  Reference: https://datamining.cs.ucdavis.edu/~filkov/courses/300/winter06/readings/gam.pdf
- **Mixed-effects (random-effects) models** — Hierarchical modeling for grouped/longitudinal data.  
  Reference: https://doi.org/10.2307/2529876
- **Cox proportional hazards** — Survival modeling with hazard ratios (semi-parametric).  
  Reference: https://doi.org/10.1111/j.2517-6161.1972.tb00899.x
- **Bayesian Additive Regression Trees (BART)** — Bayesian tree ensemble with uncertainty quantification.  
  Reference: https://projecteuclid.org/journals/annals-of-applied-statistics/volume-4/issue-1/BART-Bayesian-additive-regression-trees/10.1214/09-AOAS285.full
- **Dirichlet Process (nonparametric Bayes)** — Flexible mixture modeling with countably infinite components.  
  Reference: https://projecteuclid.org/journals/annals-of-statistics/volume-1/issue-2/A-Bayesian-analysis-of-some-nonparametric-problems/10.1214/aos/1176342360.full
- **Hidden Markov Models (HMM)** — Latent-state sequence models for time series / sequences.  
  Reference: https://www.cs.ubc.ca/~murphyk/Bayes/rabiner.pdf
- **Kalman filter** — Optimal linear filtering for state-space models.  
  Reference: https://doi.org/10.1115/1.3662552
- **Granger causality** — Temporal precedence causality test in time series.  
  Reference: https://doi.org/10.2307/1912791
- **Transfer entropy** — Directed information flow measure (nonlinear dependency).  
  Reference: https://doi.org/10.1103/PhysRevLett.85.461
- **kNN mutual information estimator** — Nonparametric MI estimation for dependence detection.  
  Reference: https://doi.org/10.1103/PhysRevE.69.066138
- **Independent Component Analysis (ICA)** — Blind source separation / latent factor recovery.  
  Reference: https://www.cs.helsinki.fi/u/ahyvarin/papers/NN00new.pdf
- **Non-negative Matrix Factorization (NMF)** — Parts-based latent factorization with non-negativity.  
  Reference: https://doi.org/10.1038/44565
- **t-SNE** — Nonlinear dimensionality reduction for visualization.  
  Reference: https://www.jmlr.org/papers/v9/vandermaaten08a.html
- **UMAP** — Manifold learning for dimension reduction (fast, scalable).  
  Reference: https://arxiv.org/abs/1802.03426
- **NOTEARS** — Gradient-based causal DAG learning without combinatorial search.  
  Reference: https://arxiv.org/abs/1803.01422
- **MICE (chained equations) imputation** — Multiple imputation for missing data via chained regressions.  
  Reference: https://www.jstatsoft.org/article/view/v045i03

## 9) Quick file map (for implementers)

- CLI entrypoint and subcommands: `src/statistic_harness/cli.py`.
- Plugin discovery/validation: `src/statistic_harness/core/plugin_manager.py`.
- Pipeline orchestration & subprocess plugin execution: `src/statistic_harness/core/pipeline.py`, `src/statistic_harness/core/plugin_runner.py`.
- Plugin manifest schema: `docs/plugin_manifest.schema.json`.
- Tests: `tests/test_cli_plugins_validate.py`, `tests/test_plugin_manifest_schema.py`, `tests/test_config_defaults.py`.
