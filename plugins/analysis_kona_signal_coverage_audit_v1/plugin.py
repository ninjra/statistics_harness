from __future__ import annotations
import logging
import traceback
import json
from collections import Counter
from statistic_harness.core.types import PluginResult, PluginArtifact, PluginError

logger = logging.getLogger(__name__)

# Finding kinds that the 10 lever detectors in lever_library.py can consume
LEVER_CONSUMED_KINDS = {
    "queue_delay", "concurrency", "batch_size", "retry_burst",
    "resource_affinity", "branch_ordering", "peak_activity",
    "schedule_frequency", "host_capacity", "workload_isolation",
    # Also: the lever_library operates on raw data, not findings.
    # The bridge plugin (finding_to_lever_bridge) consumes these:
    "granger_causal_link", "changepoint_detected", "anomaly_cluster",
    "hidden_markov_regime", "transfer_entropy_link",
    "spectral_clustering_community", "survival_hazard_spike",
    "causal", "counterfactual", "role_inference",
}


class Plugin:
    def run(self, ctx) -> PluginResult:
        try:
            # Collect all finding kinds from all analysis plugin results
            all_kinds: Counter = Counter()
            stranded_kinds: Counter = Counter()
            covered_kinds: Counter = Counter()

            # Try to read all plugin results from storage
            try:
                results = ctx.storage.fetch_plugin_results(ctx.run_id)
            except Exception:
                results = []

            if not results:
                return PluginResult(
                    "ok",
                    "Signal coverage audit: no upstream results available",
                    {"total_finding_kinds": 0},
                    [{"kind": "role_inference", "measurement_type": "not_applicable", "reason": "no_upstream_results"}],
                    [], None,
                )

            for row in results:
                status = str(row.get("status") or "").lower()
                if status != "ok":
                    continue
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
                    kind = str(f.get("kind", ""))
                    if not kind:
                        continue
                    all_kinds[kind] += 1
                    if kind in LEVER_CONSUMED_KINDS:
                        covered_kinds[kind] += 1
                    else:
                        stranded_kinds[kind] += 1

            total_kinds = len(all_kinds)
            covered_count = len(covered_kinds)
            stranded_count = len(stranded_kinds)
            coverage_ratio = covered_count / max(total_kinds, 1)

            findings = [{
                "kind": "role_inference",
                "measurement_type": "measured",
                "type": "signal_coverage_audit",
                "total_finding_kinds": total_kinds,
                "covered_kinds": covered_count,
                "stranded_kinds": stranded_count,
                "coverage_ratio": round(coverage_ratio, 4),
                "stranded_kind_list": dict(stranded_kinds.most_common(20)),
                "covered_kind_list": dict(covered_kinds.most_common(20)),
                "interpretation": (
                    f"{covered_count}/{total_kinds} finding kinds are consumed by levers "
                    f"({coverage_ratio:.0%} coverage). {stranded_count} kinds are stranded signal."
                ),
            }]

            return PluginResult(
                status="ok",
                summary=f"Signal coverage: {covered_count}/{total_kinds} kinds covered ({coverage_ratio:.0%}), {stranded_count} stranded",
                metrics={
                    "total_finding_kinds": total_kinds,
                    "covered_kinds": covered_count,
                    "stranded_kinds": stranded_count,
                    "coverage_ratio": round(coverage_ratio, 4),
                },
                findings=findings,
                artifacts=[],
                error=None,
            )
        except Exception as e:
            logger.error("Signal coverage audit failed: %s", e, exc_info=True)
            return PluginResult("error", f"Signal coverage audit failed: {e}", {}, [], [],
                PluginError(type=type(e).__name__, message=str(e), traceback=traceback.format_exc()))
