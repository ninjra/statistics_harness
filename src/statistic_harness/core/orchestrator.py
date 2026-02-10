from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from statistic_harness.core.pipeline import Pipeline


@dataclass(frozen=True)
class OrchestratorConfig:
    plugins: list[str]
    run_seed: int


class Orchestrator:
    """Thin wrapper around Pipeline for scripted multi-dataset execution."""

    def __init__(self, appdata_root: Path, plugins_dir: Path, tenant_id: str | None = None) -> None:
        self.pipeline = Pipeline(appdata_root, plugins_dir, tenant_id=tenant_id)

    def run_for_dataset_version(
        self,
        dataset_version_id: str,
        config: OrchestratorConfig,
        settings: dict[str, Any] | None = None,
    ) -> str:
        return self.pipeline.run(
            input_file=None,
            plugin_ids=list(config.plugins),
            settings=settings or {},
            run_seed=int(config.run_seed),
            dataset_version_id=str(dataset_version_id),
        )

