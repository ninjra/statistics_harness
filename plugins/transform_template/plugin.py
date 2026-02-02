from __future__ import annotations

import json

from statistic_harness.core.template import apply_template
from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json


class Plugin:
    def run(self, ctx) -> PluginResult:
        if not ctx.dataset_version_id:
            return PluginResult("error", "Missing dataset version", {}, [], [], None)

        template_id = ctx.settings.get("template_id")
        mapping = ctx.settings.get("mapping")
        mapping_json = ctx.settings.get("mapping_json")
        if mapping is None and mapping_json:
            try:
                mapping = json.loads(mapping_json)
            except json.JSONDecodeError as exc:
                return PluginResult(
                    "error",
                    f"Invalid mapping JSON: {exc}",
                    {},
                    [],
                    [],
                    None,
                )
        if not template_id or not isinstance(mapping, dict):
            return PluginResult(
                "error", "template_id and mapping required", {}, [], [], None
            )

        row_count = apply_template(
            ctx.storage, ctx.dataset_version_id, int(template_id), mapping
        )

        artifacts_dir = ctx.artifacts_dir("transform_template")
        map_path = artifacts_dir / "mapping.json"
        write_json(map_path, mapping)
        artifacts = [
            PluginArtifact(
                path=str(map_path.relative_to(ctx.run_dir)),
                type="json",
                description="Template mapping",
            )
        ]

        return PluginResult(
            "ok",
            "Template mapping applied",
            {"row_count": int(row_count)},
            [],
            artifacts,
            None,
        )
