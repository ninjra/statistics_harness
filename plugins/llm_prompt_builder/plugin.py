from __future__ import annotations

import json

from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import DEFAULT_TENANT_ID


class Plugin:
    def run(self, ctx) -> PluginResult:
        report_path = ctx.run_dir / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        tenant_id = ctx.tenant_id or DEFAULT_TENANT_ID
        pii_entities = ctx.storage.fetch_pii_entities(tenant_id)
        pii_map: dict[str, list[tuple[str, str]]] = {}
        for entry in pii_entities:
            raw = str(entry.get("raw_value", ""))
            pii_type = str(entry.get("pii_type", "pii"))
            value_hash = str(entry.get("value_hash", ""))
            if raw:
                pii_map.setdefault(raw, []).append((pii_type, value_hash))

        def _sanitize_value(value):
            if isinstance(value, str) and value in pii_map:
                types = sorted({t for t, _ in pii_map[value] if t})
                value_hash = pii_map[value][0][1] if pii_map[value] else ""
                label = ",".join(types) if types else "pii"
                return f"pii:{label}:{value_hash}"
            if isinstance(value, dict):
                sanitized = {}
                for key, val in value.items():
                    new_key = (
                        _sanitize_value(key)
                        if isinstance(key, str) and key in pii_map
                        else key
                    )
                    sanitized[new_key] = _sanitize_value(val)
                return sanitized
            if isinstance(value, list):
                return [_sanitize_value(item) for item in value]
            return value

        sanitized_report = _sanitize_value(report)
        artifacts_dir = ctx.artifacts_dir("llm_prompt_builder")
        prompt = "# Analysis Summary\n\n" + json.dumps(sanitized_report, indent=2)
        brief = "# Brief\n\nSummary of findings."
        prompt_path = artifacts_dir / "prompt.md"
        brief_path = artifacts_dir / "brief.md"
        prompt_path.write_text(prompt, encoding="utf-8")
        brief_path.write_text(brief, encoding="utf-8")
        artifacts = [
            PluginArtifact(
                path=str(prompt_path.relative_to(ctx.run_dir)),
                type="markdown",
                description="Prompt",
            ),
            PluginArtifact(
                path=str(brief_path.relative_to(ctx.run_dir)),
                type="markdown",
                description="Brief",
            ),
        ]
        return PluginResult("ok", "Built LLM prompt", {}, [], artifacts, None)
