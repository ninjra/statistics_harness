from __future__ import annotations
import logging
import traceback
import json
from statistic_harness.core.types import PluginResult, PluginArtifact, PluginError
from statistic_harness.core.finding_lever_map import match_finding_to_levers

logger = logging.getLogger(__name__)


class Plugin:
    def run(self, ctx) -> PluginResult:
        try:
            # Read all upstream analysis plugin results
            try:
                results = ctx.storage.fetch_plugin_results(ctx.run_id)
            except Exception:
                results = []

            if not results:
                return PluginResult(
                    "ok",
                    "Finding-to-lever bridge: no upstream results",
                    {"lever_candidates": 0},
                    [{"kind": "role_inference", "measurement_type": "not_applicable", "reason": "no_upstream_results"}],
                    [], None,
                )

            lever_candidates = []
            for row in results:
                status = str(row.get("status") or "").lower()
                if status != "ok":
                    continue
                plugin_id = str(row.get("plugin_id") or "")
                findings_raw = row.get("findings_json") or row.get("findings") or "[]"
                if isinstance(findings_raw, str):
                    try:
                        findings = json.loads(findings_raw)
                    except Exception:
                        continue
                elif isinstance(findings_raw, list):
                    findings = findings_raw
                else:
                    continue
                for f in findings:
                    if not isinstance(f, dict):
                        continue
                    matches = match_finding_to_levers(f)
                    for m in matches:
                        lever = m["lever"]
                        lever_candidates.append({
                            "kind": "role_inference",
                            "measurement_type": "measured",
                            "type": "lever_candidate",
                            "lever_id": lever.lever_id,
                            "lever_title": lever.title,
                            "confidence": lever.confidence_base,
                            "source_plugin": plugin_id,
                            "source_finding_kind": m["finding_kind"],
                        })

            # Deduplicate by lever_id + source_plugin
            seen = set()
            deduped = []
            for lc in lever_candidates:
                key = (lc["lever_id"], lc["source_plugin"])
                if key not in seen:
                    seen.add(key)
                    deduped.append(lc)

            return PluginResult(
                status="ok",
                summary=f"Finding-to-lever bridge: {len(deduped)} lever candidates from {len(seen)} unique sources",
                metrics={
                    "lever_candidates": len(deduped),
                    "unique_lever_types": len({lc["lever_id"] for lc in deduped}),
                },
                findings=deduped[:50],  # Cap findings
                artifacts=[],
                error=None,
            )
        except Exception as e:
            logger.error("Finding-to-lever bridge failed: %s", e, exc_info=True)
            return PluginResult("error", f"Finding-to-lever bridge failed: {e}", {}, [], [],
                PluginError(type=type(e).__name__, message=str(e), traceback=traceback.format_exc()))
