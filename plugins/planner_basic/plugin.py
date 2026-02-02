from __future__ import annotations

from pathlib import Path

from statistic_harness.core.planner import select_plugins
from statistic_harness.core.plugin_manager import PluginManager
from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json


class Plugin:
    def run(self, ctx) -> PluginResult:
        if not ctx.dataset_version_id:
            return PluginResult(
                "error",
                "Missing dataset version",
                {},
                [],
                [],
                None,
            )
        manager = PluginManager(Path("plugins"))
        specs = manager.discover()
        selected = select_plugins(specs, ctx.storage, ctx.dataset_version_id)
        allow = ctx.settings.get("allow") or []
        deny = ctx.settings.get("deny") or []
        if allow:
            selected = [pid for pid in selected if pid in allow]
        if deny:
            selected = [pid for pid in selected if pid not in deny]
        artifacts_dir = ctx.artifacts_dir("planner_basic")
        plan_path = artifacts_dir / "plan.json"
        write_json(plan_path, {"selected_plugins": selected})
        artifacts = [
            PluginArtifact(
                path=str(plan_path.relative_to(ctx.run_dir)),
                type="json",
                description="Planner output",
            )
        ]
        findings = [{"kind": "plan", "plugin_id": pid} for pid in selected]
        return PluginResult(
            "ok",
            "Planned plugin selection",
            {"selected_plugins": selected},
            findings,
            artifacts,
            None,
        )
