from __future__ import annotations

from pathlib import Path
import json
import traceback

from statistic_harness.core.report import build_report, write_report
from statistic_harness.core.types import PluginArtifact, PluginError, PluginResult
from statistic_harness.core.utils import DEFAULT_TENANT_ID, vector_store_enabled
from statistic_harness.core.vector_store import VectorStore, hash_embedding


_VECTOR_DIMENSIONS = 128


def _index_report(ctx, report: dict) -> int:
    tenant_id = ctx.tenant_id or DEFAULT_TENANT_ID
    store = VectorStore(ctx.storage.db_path, tenant_id=tenant_id)
    run_id = report.get("run_id") or ctx.run_id
    collection = f"report_{run_id}"
    vectors: list[list[float]] = []
    item_ids: list[str] = []
    payloads: list[dict] = []

    plugins = report.get("plugins", {}) or {}
    for plugin_id in sorted(plugins.keys()):
        payload = plugins.get(plugin_id) or {}
        summary = payload.get("summary")
        if summary:
            text = f"{plugin_id} summary: {summary}"
            vectors.append(hash_embedding(text, _VECTOR_DIMENSIONS))
            item_ids.append(f"{run_id}:{plugin_id}:summary")
            payloads.append(
                {
                    "run_id": run_id,
                    "plugin_id": plugin_id,
                    "type": "summary",
                    "text": text,
                }
            )
        findings = payload.get("findings", []) or []
        for idx, finding in enumerate(findings):
            kind = finding.get("kind", "")
            title = finding.get("title", "")
            description = finding.get("description", "")
            text_parts = [plugin_id, str(kind), str(title), str(description)]
            text = " ".join(part for part in text_parts if part).strip()
            if not text:
                text = json.dumps(finding, sort_keys=True)
            vectors.append(hash_embedding(text, _VECTOR_DIMENSIONS))
            item_ids.append(f"{run_id}:{plugin_id}:finding:{idx}")
            payloads.append(
                {
                    "run_id": run_id,
                    "plugin_id": plugin_id,
                    "type": "finding",
                    "index": idx,
                    "kind": kind,
                    "text": text,
                }
            )

    if vectors:
        store.add(collection, vectors, item_ids=item_ids, payloads=payloads)
    return len(vectors)


class Plugin:
    def run(self, ctx) -> PluginResult:
        schema_path = Path("docs/report.schema.json")
        report = build_report(ctx.storage, ctx.run_id, ctx.run_dir, schema_path)
        write_report(report, ctx.run_dir)
        artifacts = [
            PluginArtifact(path="report.json", type="json", description="Report JSON"),
            PluginArtifact(
                path="report.md", type="markdown", description="Report Markdown"
            ),
        ]
        summary = "Report generated"
        status = "ok"
        if vector_store_enabled():
            try:
                count = _index_report(ctx, report)
                summary = f"Report generated; indexed {count} vectors"
            except Exception as exc:  # pragma: no cover - optional path
                # Vector indexing is optional; do not fail the run if sqlite-vec isn't available.
                summary = f"Report generated; vector index skipped ({type(exc).__name__}: {exc})"
                return PluginResult(
                    status="ok",
                    summary=summary,
                    # Keep metrics empty to satisfy plugins/report_bundle/output.schema.json.
                    metrics={},
                    findings=[
                        {
                            "kind": "vector_index_skipped",
                            "measurement_type": "measured",
                            "reason": str(exc),
                            "error_type": type(exc).__name__,
                            "traceback": traceback.format_exc(),
                        }
                    ],
                    artifacts=artifacts,
                    error=None,
                )
        return PluginResult(status, summary, {}, [], artifacts, None)
