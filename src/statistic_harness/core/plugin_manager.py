from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


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


class PluginManager:
    def __init__(self, plugins_dir: Path) -> None:
        self.plugins_dir = plugins_dir

    def discover(self) -> list[PluginSpec]:
        specs: list[PluginSpec] = []
        for manifest in sorted(self.plugins_dir.glob("*/plugin.yaml")):
            data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
            for key in ["id", "name", "version", "type", "entrypoint"]:
                if key not in data:
                    raise ValueError(f"Missing {key} in {manifest}")
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
                )
            )
        return specs

    def load_plugin(self, spec: PluginSpec) -> Any:
        module_path, class_name = spec.entrypoint.split(":", 1)
        module_name = f"plugins.{spec.plugin_id}.{module_path}"
        module = importlib.import_module(module_name)
        return getattr(module, class_name)()
