from __future__ import annotations

import hashlib
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os
from typing import Any

from .dataset_io import resolve_dataset_accessor
from .plugin_manager import PluginManager, PluginSpec
from .plugin_runner import run_plugin_subprocess
from .storage import Storage
from .tenancy import get_tenant_context, scope_identifier, tenancy_enabled
from .types import PluginContext, PluginError, PluginResult
from .utils import (
    dataset_key,
    ensure_dir,
    file_sha256,
    json_dumps,
    make_run_id,
    now_iso,
    DEFAULT_TENANT_ID,
)
from .report import build_report, write_report


class Pipeline:
    def __init__(
        self, base_dir: Path, plugins_dir: Path, tenant_id: str | None = None
    ) -> None:
        tenant_ctx = get_tenant_context(tenant_id, base_dir)
        self.base_dir = tenant_ctx.tenant_root
        self.appdata_root = tenant_ctx.appdata_root
        self.tenant_id = tenant_ctx.tenant_id
        self.plugins_dir = plugins_dir
        self.storage = Storage(tenant_ctx.db_path, tenant_ctx.tenant_id)
        self.manager = PluginManager(plugins_dir)

    def _toposort_layers(
        self, specs: list[PluginSpec], selected: set[str]
    ) -> list[list[PluginSpec]]:
        spec_map = {spec.plugin_id: spec for spec in specs}
        deps: dict[str, set[str]] = {}
        indegree: dict[str, int] = {}
        for pid in selected:
            spec = spec_map.get(pid)
            if not spec:
                continue
            dep_set = {dep for dep in spec.depends_on if dep in selected}
            deps[pid] = dep_set
            indegree[pid] = len(dep_set)

        layers: list[list[PluginSpec]] = []
        remaining = set(indegree.keys())
        while remaining:
            ready = sorted(pid for pid in remaining if indegree.get(pid, 0) == 0)
            if not ready:
                raise ValueError("Cycle detected in plugin dependencies")
            layer_specs = [spec_map[pid] for pid in ready]
            layers.append(layer_specs)
            for pid in ready:
                remaining.remove(pid)
            for pid in remaining:
                if deps[pid].intersection(ready):
                    indegree[pid] -= len(deps[pid].intersection(ready))
        return layers

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
    ) -> str:
        run_id = run_id or make_run_id()
        tenant_id = self.tenant_id

        def _scope(value: str | None) -> str | None:
            if not value:
                return value
            if tenancy_enabled() and tenant_id != DEFAULT_TENANT_ID:
                return scope_identifier(tenant_id, value)
            return value

        dataset_version_id = _scope(dataset_version_id)
        project_id = _scope(project_id)

        run_dir = self.base_dir / "runs" / run_id
        ensure_dir(run_dir / "dataset")
        ensure_dir(run_dir / "logs")

        canonical_path = run_dir / "dataset" / "canonical.csv"
        if input_file is None:
            if not dataset_version_id:
                raise ValueError("Dataset version is required for DB-only runs")
            ctx_row = self.storage.get_dataset_version_context(dataset_version_id)
            if not ctx_row:
                raise ValueError("Dataset version not found")
            project_id = ctx_row["project_id"]
            dataset_id = ctx_row["dataset_id"]
            input_hash = ctx_row.get("data_hash") or dataset_version_id
            input_filename = f"db://{dataset_version_id}"
        else:
            input_hash = file_sha256(input_file)
            project_id = project_id or input_hash
            dataset_id = dataset_version_id or (
                input_hash
                if project_id == input_hash
                else dataset_key(project_id, input_hash)
            )
            dataset_version_id = dataset_version_id or dataset_id
            project_id = _scope(project_id)
            dataset_id = _scope(dataset_id)
            dataset_version_id = _scope(dataset_version_id)
            table_name = f"dataset_{dataset_version_id}"
            self.storage.ensure_project(project_id, project_id, now_iso())
            self.storage.ensure_dataset(dataset_id, project_id, dataset_id, now_iso())
            self.storage.ensure_dataset_version(
                dataset_version_id, dataset_id, now_iso(), table_name, input_hash
            )
            input_filename = input_file.name

        self.storage.create_run(
            run_id=run_id,
            created_at=now_iso(),
            status="running",
            upload_id=upload_id,
            input_filename=input_filename,
            canonical_path=str(canonical_path),
            settings=settings,
            error=None,
            run_seed=run_seed,
            project_id=project_id,
            dataset_id=dataset_id,
            dataset_version_id=dataset_version_id,
            input_hash=input_hash,
        )

        specs = self.manager.discover()
        spec_map = {spec.plugin_id: spec for spec in specs}
        for err in self.manager.discovery_errors:
            plugin_id = err.plugin_id
            if plugin_id in spec_map:
                plugin_id = f"{plugin_id}__discovery_error"
            record_missing(
                plugin_id,
                f"Plugin discovery error in {err.path}: {err.message}",
            )

        def record_missing(plugin_id: str, message: str) -> None:
            result = PluginResult(
                status="error",
                summary=message,
                metrics={},
                findings=[],
                artifacts=[],
                error=PluginError(
                    type="MissingPlugin",
                    message=message,
                    traceback="",
                ),
            )
            self.storage.save_plugin_result(
                run_id,
                plugin_id,
                None,
                now_iso(),
                None,
                None,
                input_hash,
                result,
            )
            self.storage.insert_plugin_execution(
                run_id=run_id,
                plugin_id=plugin_id,
                plugin_version=None,
                started_at=now_iso(),
                completed_at=now_iso(),
                duration_ms=0,
                status="error",
                exit_code=None,
                cpu_user=None,
                cpu_system=None,
                max_rss=None,
                warnings_count=None,
                stdout=None,
                stderr=None,
            )

        selected = set(plugin_ids)
        auto_plan = not selected or "auto" in selected
        selected.discard("auto")
        if "all" in selected:
            selected = {spec.plugin_id for spec in specs if spec.type == "analysis"}
            auto_plan = False
        llm_selected = "llm_prompt_builder" in selected

        dataset_accessor, dataset_template = resolve_dataset_accessor(
            self.storage, dataset_version_id
        )

        column_lookup: dict[str, int] | None = None

        def get_column_lookup() -> dict[str, int]:
            nonlocal column_lookup
            if column_lookup is None and dataset_version_id:
                if dataset_template:
                    fields = self.storage.fetch_template_fields(
                        int(dataset_template["template_id"])
                    )
                    column_lookup = {field["name"]: int(field["field_id"]) for field in fields}
                else:
                    columns = self.storage.fetch_dataset_columns(dataset_version_id)
                    column_lookup = {
                        col["original_name"]: int(col["column_id"]) for col in columns
                    }
            return column_lookup or {}

        def infer_column_ids(item: dict[str, Any]) -> list[int]:
            lookup = get_column_lookup()
            ids: list[int] = []
            if "feature" in item and isinstance(item["feature"], str):
                if item["feature"] in lookup:
                    ids.append(lookup[item["feature"]])
            if "pair" in item and isinstance(item["pair"], (list, tuple)):
                for name in item["pair"]:
                    if isinstance(name, str) and name in lookup:
                        ids.append(lookup[name])
            if "columns" in item and isinstance(item["columns"], (list, tuple)):
                for name in item["columns"]:
                    if isinstance(name, str) and name in lookup:
                        ids.append(lookup[name])
            return sorted(set(ids))

        def attach_evidence(findings: list[Any]) -> list[dict[str, Any]]:
            enriched: list[dict[str, Any]] = []
            for item in findings:
                if isinstance(item, dict):
                    entry = dict(item)
                else:
                    entry = {"value": item}
                evidence = dict(entry.get("evidence") or {})
                if isinstance(entry.get("dataset_id"), str):
                    evidence.setdefault("dataset_id", entry["dataset_id"])
                evidence.setdefault("dataset_id", dataset_id or "unknown")
                if isinstance(entry.get("dataset_version_id"), str):
                    evidence.setdefault("dataset_version_id", entry["dataset_version_id"])
                evidence.setdefault("dataset_version_id", dataset_version_id or "unknown")
                row_ids: list[int] = []
                if "row_ids" in entry and isinstance(entry["row_ids"], list):
                    try:
                        row_ids = [int(v) for v in entry["row_ids"]]
                    except (TypeError, ValueError):
                        row_ids = []
                if "row_index" in entry:
                    try:
                        row_ids.append(int(entry["row_index"]))
                    except (TypeError, ValueError):
                        pass
                if row_ids:
                    row_ids = sorted(set(row_ids))
                evidence.setdefault("row_ids", row_ids)
                evidence.setdefault("column_ids", infer_column_ids(entry))
                evidence.setdefault("query", entry.get("query"))
                if "start" in entry and "end" in entry:
                    try:
                        evidence.setdefault(
                            "row_ranges",
                            [
                                {
                                    "start": int(entry["start"]),
                                    "end": int(entry["end"]),
                                }
                            ],
                        )
                    except (TypeError, ValueError):
                        pass
                if "measurement_type" not in entry:
                    entry["measurement_type"] = "measured"
                entry["evidence"] = evidence
                enriched.append(entry)
            return enriched

        def validate_modeled_findings(findings: list[dict[str, Any]]) -> list[str]:
            errors: list[str] = []
            for item in findings:
                if item.get("measurement_type") != "modeled":
                    continue
                scope = item.get("scope")
                assumptions = item.get("assumptions")
                if not isinstance(scope, dict) or not scope:
                    errors.append("modeled finding missing scope")
                if (
                    not isinstance(assumptions, list)
                    or not assumptions
                    or not all(isinstance(a, str) and a.strip() for a in assumptions)
                ):
                    errors.append("modeled finding missing assumptions")
            return errors

        def logger(msg: str) -> None:
            log_path = run_dir / "logs" / "run.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(msg + "\n")

        def run_spec(spec: PluginSpec, include_input: bool = False) -> PluginResult:
            plugin_settings = dict(spec.settings.get("defaults", {}))
            plugin_settings.update(settings.get(spec.plugin_id, {}))
            if include_input and input_file is not None:
                plugin_settings["input_file"] = str(input_file)
            budget = plugin_settings.get("budget")
            if not isinstance(budget, dict):
                budget = {}
            budget = {
                "row_limit": budget.get("row_limit"),
                "sampled": bool(budget.get("sampled", False)),
                "time_limit_ms": budget.get("time_limit_ms"),
                "cpu_limit_ms": budget.get("cpu_limit_ms"),
            }
            execution_id = self.storage.start_plugin_execution(
                run_id,
                spec.plugin_id,
                spec.version,
                now_iso(),
                status="running",
            )
            def dataset_loader(
                columns: list[str] | None = None, row_limit: int | None = None
            ):
                limit = row_limit
                if limit is None:
                    limit = budget.get("row_limit")
                return dataset_accessor.load(columns=columns, row_limit=limit)

            ctx = PluginContext(
                run_id=run_id,
                run_dir=run_dir,
                settings=plugin_settings,
                run_seed=run_seed,
                logger=logger,
                storage=self.storage,
                dataset_loader=dataset_loader,
                budget=budget,
                tenant_id=self.tenant_id,
                project_id=project_id,
                dataset_id=dataset_id,
                dataset_version_id=dataset_version_id,
                input_hash=input_hash,
            )
            try:
                self.manager.validate_config(spec, plugin_settings)
                allow_paths: list[str] = []
                for token in spec.sandbox.get("fs_allowlist", []):
                    if token == "appdata":
                        allow_paths.append(str(self.base_dir))
                    elif token == "plugins":
                        allow_paths.append(str(self.plugins_dir))
                    elif token == "run_dir":
                        allow_paths.append(str(run_dir))
                    else:
                        allow_paths.append(
                            str((self.plugins_dir.parent / token).resolve())
                        )
                allow_paths.append(str(self.plugins_dir.parent / "src"))
                allow_paths.append(str(run_dir))
                allow_paths.append(str(self.appdata_root / "state.sqlite"))
                if include_input and input_file is not None:
                    allow_paths.append(str(input_file))
                request = {
                    "plugin_id": spec.plugin_id,
                    "entrypoint": spec.entrypoint,
                    "settings": plugin_settings,
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "run_seed": run_seed,
                    "dataset_version_id": dataset_version_id,
                    "project_id": project_id,
                    "dataset_id": dataset_id,
                    "input_hash": input_hash,
                    "budget": budget,
                    "tenant_id": self.tenant_id,
                    "appdata_dir": str(self.appdata_root),
                    "root_dir": str(self.plugins_dir.parent.resolve()),
                    "sandbox": spec.sandbox,
                    "allow_paths": allow_paths,
                }
                runner = run_plugin_subprocess(
                    spec, request, run_dir, self.plugins_dir.parent
                )
                result = runner.result
            except Exception as exc:  # pragma: no cover - error flow
                tb = traceback.format_exc()
                result = PluginResult(
                    status="error",
                    summary=f"{spec.plugin_id} failed",
                    metrics={},
                    findings=[],
                    artifacts=[],
                    error=PluginError(
                        type=type(exc).__name__, message=str(exc), traceback=tb
                    ),
                )
            result.findings = attach_evidence(result.findings)
            try:
                payload = self.manager.result_payload(result)
                self.manager.validate_output(spec, payload)
                modeled_errors = validate_modeled_findings(result.findings)
                if modeled_errors:
                    raise ValueError("; ".join(sorted(set(modeled_errors))))
            except Exception as exc:  # pragma: no cover - error flow
                tb = traceback.format_exc()
                result = PluginResult(
                    status="error",
                    summary=f"{spec.plugin_id} output validation failed: {exc}",
                    metrics={},
                    findings=[],
                    artifacts=[],
                    error=PluginError(
                        type=type(exc).__name__, message=str(exc), traceback=tb
                    ),
                )
            if result.error:
                logger(f"[ERROR] {spec.plugin_id}: {result.error.message}")
            if "runner" in locals():
                exec_info = runner.execution
                self.storage.update_plugin_execution(
                    execution_id=execution_id,
                    completed_at=exec_info.get("completed_at"),
                    duration_ms=exec_info.get("duration_ms"),
                    status=result.status,
                    exit_code=runner.exit_code,
                    cpu_user=exec_info.get("cpu_user"),
                    cpu_system=exec_info.get("cpu_system"),
                    max_rss=exec_info.get("max_rss"),
                    warnings_count=exec_info.get("warnings_count"),
                    stdout=runner.stdout,
                    stderr=runner.stderr,
                )
            else:
                self.storage.update_plugin_execution(
                    execution_id=execution_id,
                    completed_at=now_iso(),
                    duration_ms=0,
                    status=result.status,
                    exit_code=None,
                    cpu_user=None,
                    cpu_system=None,
                    max_rss=None,
                    warnings_count=None,
                    stdout=None,
                    stderr=None,
                )
            module_path, _ = spec.entrypoint.split(":", 1)
            if module_path.endswith(".py"):
                module_file = spec.path / module_path
            else:
                module_file = spec.path / f"{module_path}.py"
            code_hash = file_sha256(module_file) if module_file.exists() else None
            settings_hash = hashlib.sha256(
                json_dumps(plugin_settings).encode("utf-8")
            ).hexdigest()
            self.storage.save_plugin_result(
                run_id,
                spec.plugin_id,
                spec.version,
                now_iso(),
                code_hash,
                settings_hash,
                input_hash,
                result,
            )
            return result

        if input_file is not None:
            ingest_spec = spec_map.get("ingest_tabular")
            if ingest_spec:
                run_spec(ingest_spec, include_input=True)
            else:
                record_missing("ingest_tabular", "Missing ingest plugin")

        if auto_plan:
            profile_specs = [spec for spec in specs if spec.type == "profile"]
            if profile_specs:
                for spec in sorted(profile_specs, key=lambda item: item.plugin_id):
                    run_spec(spec)
            else:
                record_missing("profile_basic", "Missing profile plugin")
            planner_spec = spec_map.get("planner_basic")
            if planner_spec:
                plan_result = run_spec(planner_spec)
                planned = plan_result.metrics.get("selected_plugins") or []
                selected.update(planned)
            else:
                record_missing("planner_basic", "Missing planner plugin")

        missing_manual = sorted(pid for pid in selected if pid not in spec_map)
        for pid in missing_manual:
            record_missing(pid, f"Unknown plugin id: {pid}")

        transform_ids = {
            pid
            for pid in selected
            if pid in spec_map and spec_map[pid].type == "transform"
        }
        if transform_ids:
            layers = self._toposort_layers(specs, transform_ids)
            for layer in layers:
                for spec in layer:
                    run_spec(spec)
            dataset_accessor, dataset_template = resolve_dataset_accessor(
                self.storage, dataset_version_id
            )
            column_lookup = None

        analysis_ids = {
            pid
            for pid in selected
            if pid in spec_map and spec_map[pid].type == "analysis"
        }
        layers = self._toposort_layers(specs, analysis_ids)
        for layer in layers:
            if len(layer) == 1:
                run_spec(layer[0])
                continue
            max_workers = min(len(layer), (os.cpu_count() or 1))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(run_spec, spec) for spec in layer]
                for future in futures:
                    future.result()

        report_spec = spec_map.get("report_bundle")
        if report_spec:
            run_spec(report_spec)
            report_json = run_dir / "report.json"
            report_md = run_dir / "report.md"
            if not report_json.exists() or not report_md.exists():
                report = build_report(
                    self.storage, run_id, run_dir, Path("docs/report.schema.json")
                )
                write_report(report, run_dir)
        else:
            record_missing("report_bundle", "Missing report plugin")
            report = build_report(
                self.storage, run_id, run_dir, Path("docs/report.schema.json")
            )
            write_report(report, run_dir)

        if llm_selected:
            llm_spec = spec_map.get("llm_prompt_builder")
            if llm_spec:
                run_spec(llm_spec)
            else:
                record_missing("llm_prompt_builder", "Missing llm plugin")

        self.storage.update_run_status(run_id, "completed")
        return run_id
