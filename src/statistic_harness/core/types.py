from __future__ import annotations

from dataclasses import dataclass, field
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
    references: list[dict[str, Any]] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)
    budget: dict[str, Any] = field(
        default_factory=lambda: {
            "row_limit": None,
            "sampled": False,
            "time_limit_ms": None,
            "cpu_limit_ms": None,
        }
    )


@dataclass
class PluginContext:
    run_id: str
    run_dir: Path
    settings: dict[str, Any]
    run_seed: int
    logger: Callable[[str], None]
    storage: Any
    dataset_loader: Callable[..., Any]
    dataset_iter_batches: Callable[..., Any] | None = None
    # Optional SQL helpers (wired by the harness). These are intentionally typed as Any
    # to keep plugin APIs stable and avoid forcing a dependency on specific SQL classes.
    sql: Any | None = None
    sql_exec: Any | None = None
    scratch_storage: Any | None = None
    scratch_sql: Any | None = None
    sql_schema_snapshot: dict[str, Any] | None = None
    budget: dict[str, Any] = field(
        default_factory=lambda: {
            "row_limit": None,
            "sampled": False,
            "time_limit_ms": None,
            "cpu_limit_ms": None,
            "batch_size": None,
        }
    )
    tenant_id: str | None = None
    project_id: str | None = None
    dataset_id: str | None = None
    dataset_version_id: str | None = None
    input_hash: str | None = None

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
