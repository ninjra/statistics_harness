from __future__ import annotations

import importlib
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


class PluginManager:
    def __init__(self, plugins_dir: Path) -> None:
        self.plugins_dir = plugins_dir
        self._manifest_schema: dict[str, Any] | None = None
        self._schema_cache: dict[Path, dict[str, Any]] = {}

    def discover(self) -> list[PluginSpec]:
        specs: list[PluginSpec] = []
        manifest_schema = self._load_manifest_schema()
        for manifest in sorted(self.plugins_dir.glob("*/plugin.yaml")):
            data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
            try:
                validate(instance=data, schema=manifest_schema)
            except ValidationError as exc:
                raise ValueError(
                    f"Invalid manifest {manifest}: {exc.message}"
                ) from exc
            config_schema_path = manifest.parent / data["config_schema"]
            output_schema_path = manifest.parent / data["output_schema"]
            if not config_schema_path.exists():
                raise ValueError(
                    f"Missing config schema for {data['id']}: {config_schema_path}"
                )
            if not output_schema_path.exists():
                raise ValueError(
                    f"Missing output schema for {data['id']}: {output_schema_path}"
                )
            defaults = data.get("settings", {}).get("defaults", {})
            if defaults is not None:
                self.validate_config_schema(config_schema_path, defaults)
            specs.append(
                PluginSpec(
                    plugin_id=data["id"],
                    name=data["name"],
                    version=data["version"],
                    type=data["type"],
                    entrypoint=data["entrypoint"],
                    depends_on=data.get("depends_on", []),
                    settings=data.get("settings", {}),
                    path=manifest.parent,
                    capabilities=list(data.get("capabilities", [])),
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

    def validate_config(self, spec: PluginSpec, config: dict[str, Any]) -> None:
        schema = self._load_schema(spec.config_schema)
        validate(instance=config, schema=schema)

    def validate_output(self, spec: PluginSpec, payload: dict[str, Any]) -> None:
        schema = self._load_schema(spec.output_schema)
        validate(instance=payload, schema=schema)

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
