from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from .dataset_io import DatasetAccessor
from .plugin_manager import PluginManager, PluginSpec
from .storage import Storage
from .types import PluginContext, PluginError, PluginResult
from .utils import ensure_dir, make_run_id, now_iso


class Pipeline:
    def __init__(self, base_dir: Path, plugins_dir: Path) -> None:
        self.base_dir = base_dir
        self.plugins_dir = plugins_dir
        self.storage = Storage(base_dir / "state.sqlite")
        self.manager = PluginManager(plugins_dir)

    def _toposort(self, specs: list[PluginSpec], selected: set[str]) -> list[PluginSpec]:
        spec_map = {spec.plugin_id: spec for spec in specs}
        ordered: list[PluginSpec] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(pid: str) -> None:
            if pid in visited:
                return
            if pid in visiting:
                raise ValueError(f"Cycle detected at {pid}")
            visiting.add(pid)
            spec = spec_map[pid]
            for dep in spec.depends_on:
                if dep in spec_map and dep in selected:
                    visit(dep)
            visiting.remove(pid)
            visited.add(pid)
            ordered.append(spec)

        for pid in sorted(selected):
            if pid in spec_map:
                visit(pid)
        return ordered

    def run(
        self,
        input_file: Path,
        plugin_ids: list[str],
        settings: dict[str, Any],
        run_seed: int,
        upload_id: str = "local",
        run_id: str | None = None,
    ) -> str:
        run_id = run_id or make_run_id()
        run_dir = self.base_dir / "runs" / run_id
        ensure_dir(run_dir / "dataset")
        ensure_dir(run_dir / "logs")

        canonical_path = run_dir / "dataset" / "canonical.csv"
        self.storage.create_run(
            run_id=run_id,
            created_at=now_iso(),
            status="running",
            upload_id=upload_id,
            input_filename=input_file.name,
            canonical_path=str(canonical_path),
            settings=settings,
            error=None,
        )

        specs = self.manager.discover()
        spec_map = {spec.plugin_id: spec for spec in specs}

        selected = set(plugin_ids)
        selected.add("ingest_tabular")
        selected.add("report_bundle")

        ordered_specs = self._toposort(specs, selected)
        report_specs = [spec for spec in ordered_specs if spec.plugin_id == "report_bundle"]
        llm_specs = [spec for spec in ordered_specs if spec.plugin_id == "llm_prompt_builder"]
        ordered_specs = [
            spec
            for spec in ordered_specs
            if spec.plugin_id not in {\"report_bundle\", \"llm_prompt_builder\"}
        ] + report_specs + llm_specs

        dataset_accessor = DatasetAccessor(canonical_path)

        def logger(msg: str) -> None:
            log_path = run_dir / "logs" / "run.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(msg + "\n")

        for spec in ordered_specs:
            plugin_settings = dict(spec.settings.get("defaults", {}))
            plugin_settings.update(settings.get(spec.plugin_id, {}))
            if spec.plugin_id == "ingest_tabular":
                plugin_settings["input_file"] = str(input_file)
            ctx = PluginContext(
                run_id=run_id,
                run_dir=run_dir,
                settings=plugin_settings,
                run_seed=run_seed,
                logger=logger,
                storage=self.storage,
                dataset_loader=dataset_accessor.load,
            )
            try:
                plugin = self.manager.load_plugin(spec)
                result = plugin.run(ctx)
            except Exception as exc:  # pragma: no cover - error flow
                tb = traceback.format_exc()
                result = PluginResult(
                    status="error",
                    summary=f"{spec.plugin_id} failed",
                    metrics={},
                    findings=[],
                    artifacts=[],
                    error=PluginError(type=type(exc).__name__, message=str(exc), traceback=tb),
                )
            self.storage.save_plugin_result(run_id, spec.plugin_id, result)

        self.storage.update_run_status(run_id, "completed")
        return run_id
