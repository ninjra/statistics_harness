from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol


@dataclass
class PluginArtifact:
    path: str
    type: str
    description: str


@dataclass
class PluginError:
    type: str
    message: str
    traceback: str


@dataclass
class PluginResult:
    status: str
    summary: str
    metrics: dict[str, Any]
    findings: list[dict[str, Any]]
    artifacts: list[PluginArtifact]
    error: PluginError | None = None


@dataclass
class PluginContext:
    run_id: str
    run_dir: Path
    settings: dict[str, Any]
    run_seed: int
    logger: Callable[[str], None]
    storage: Any
    dataset_loader: Callable[[], Any]

    def artifacts_dir(self, plugin_id: str) -> Path:
        path = self.run_dir / "artifacts" / plugin_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_text(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


class Plugin(Protocol):
    def run(self, ctx: PluginContext) -> PluginResult:  # pragma: no cover - protocol
        ...
