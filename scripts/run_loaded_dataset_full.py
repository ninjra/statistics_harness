from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import yaml

from statistic_harness.core.pipeline import Pipeline
from statistic_harness.core.tenancy import get_tenant_context
from statistic_harness.core.utils import make_run_id


REPO_ROOT = Path(__file__).resolve().parents[1]


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _latest_dataset_version_row(db_path: Path) -> dict[str, Any] | None:
    conn = _connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT dataset_version_id, dataset_id, created_at, table_name, row_count, column_count, data_hash
            FROM dataset_versions
            ORDER BY row_count DESC, created_at DESC
            LIMIT 1
            """
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def _dataset_version_row(db_path: Path, dataset_version_id: str) -> dict[str, Any] | None:
    conn = _connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT dataset_version_id, dataset_id, created_at, table_name, row_count, column_count, data_hash
            FROM dataset_versions
            WHERE dataset_version_id = ?
            LIMIT 1
            """,
            (dataset_version_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def _discover_plugin_ids(types: set[str]) -> list[str]:
    plugin_ids: list[str] = []
    for manifest in sorted((REPO_ROOT / "plugins").glob("*/plugin.yaml")):
        data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        pid = data.get("id")
        ptype = data.get("type")
        if isinstance(pid, str) and isinstance(ptype, str) and ptype in types:
            plugin_ids.append(pid)
    return sorted(set(plugin_ids))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _get_plugin_block(report: dict[str, Any], plugin_id: str) -> dict[str, Any]:
    plugins = report.get("plugins")
    if not isinstance(plugins, dict):
        return {}
    block = plugins.get(plugin_id)
    return block if isinstance(block, dict) else {}


def _top_findings(report: dict[str, Any], plugin_id: str, n: int = 8) -> list[dict[str, Any]]:
    block = _get_plugin_block(report, plugin_id)
    items = block.get("findings")
    if not isinstance(items, list):
        return []
    out: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            out.append(item)
        if len(out) >= n:
            break
    return out


def _extract_recommendations(report: dict[str, Any]) -> dict[str, Any]:
    recs = report.get("recommendations")
    if isinstance(recs, dict):
        return recs
    return {"summary": "No recommendations block found.", "items": []}


def _render_recommendations_md(recs: dict[str, Any], max_items: int = 40) -> str:
    summary = str(recs.get("summary") or "").strip()
    lines = []
    lines.append("# Recommendations (From report.json)")
    if summary:
        lines.append("")
        lines.append(summary)

    def _as_items(block: Any) -> list[dict[str, Any]]:
        if isinstance(block, dict) and isinstance(block.get("items"), list):
            return [i for i in block["items"] if isinstance(i, dict)]
        if isinstance(block, list):
            return [i for i in block if isinstance(i, dict)]
        return []

    known = _as_items(recs.get("known"))
    discovery = _as_items(recs.get("discovery"))
    flat = _as_items(recs.get("items"))

    sections: list[tuple[str, list[dict[str, Any]]]] = []
    if known or discovery:
        sections.append(("Discovery", discovery))
        sections.append(("Known", known))
    else:
        sections.append(("Recommendations", flat))

    for title, items in sections:
        if not items:
            continue
        lines.append("")
        lines.append(f"## {title}")
        for item in items[:max_items]:
            txt = str(item.get("recommendation") or item.get("text") or "").strip()
            if not txt:
                continue
            lines.append(f"- {txt}")
            plugin_id = item.get("plugin_id")
            kind = item.get("kind")
            where = item.get("where")
            if isinstance(where, dict) and where:
                proc = (
                    where.get("process_norm")
                    or where.get("process")
                    or where.get("process_id")
                    or where.get("transition")
                )
                if isinstance(proc, str) and proc.strip():
                    lines.append(f"  Applies to: {proc.strip()}")
            impact = item.get("impact_hours")
            if isinstance(impact, (int, float)) and float(impact) > 0:
                lines.append(f"  Potential size (upper bound): ~{float(impact):.2f} hours")
            vsteps = item.get("validation_steps")
            if isinstance(vsteps, list):
                steps = [s.strip() for s in vsteps if isinstance(s, str) and s.strip()]
                if steps:
                    lines.append("  Validation:")
                    for s in steps[:3]:
                        lines.append(f"  - {s}")
            if isinstance(plugin_id, str) and plugin_id:
                src = f"{plugin_id}" + (f":{kind}" if isinstance(kind, str) and kind else "")
                lines.append(f"  Source: {src}")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-version-id", default="")
    parser.add_argument("--run-seed", type=int, default=123)
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional run id. If omitted, a new run id is generated and printed immediately.",
    )
    parser.add_argument(
        "--plugin-set",
        choices=["auto", "full"],
        default="full",
        help="auto=planner-selected; full=run all non-ingest plugins (profile+planner+transform+analysis+report+llm)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if a cache hit exists (ignores STAT_HARNESS_REUSE_CACHE).",
    )
    args = parser.parse_args()

    # User-facing completeness: include known-issue recommendations in report synthesis.
    os.environ.setdefault("STAT_HARNESS_INCLUDE_KNOWN_RECOMMENDATIONS", "1")
    os.environ.setdefault("STAT_HARNESS_CLI_PROGRESS", "1")
    # Default to reuse-cache for operator UX on large datasets. Still safe for "updated plugins"
    # because cache keys include plugin code hash + settings hash + dataset hash.
    os.environ.setdefault("STAT_HARNESS_REUSE_CACHE", "1")

    ctx = get_tenant_context()
    db_path = ctx.appdata_root / "state.sqlite"
    if not db_path.exists():
        raise SystemExit(f"Missing DB: {db_path}")

    requested = str(args.dataset_version_id or "").strip()
    if requested:
        dataset = _dataset_version_row(db_path, requested)
        if not dataset:
            raise SystemExit(f"Dataset version not found: {requested}")
    else:
        dataset = _latest_dataset_version_row(db_path)
        if not dataset:
            raise SystemExit("No dataset_version_id found. Upload data first.")

    dataset_version_id = str(dataset["dataset_version_id"])
    run_id = str(args.run_id or "").strip() or make_run_id()
    # Print early so operators can attach watchers while the run is in progress.
    print(f"RUN_ID={run_id}", flush=True)
    row_count = None
    try:
        row_count = int(dataset.get("row_count") or 0)
    except (TypeError, ValueError):
        row_count = None

    # Large datasets: cap parallelism unless explicitly overridden.
    if row_count is not None and row_count >= 200_000:
        os.environ.setdefault("STAT_HARNESS_MAX_WORKERS_ANALYSIS", "2")

    plugin_ids: list[str]
    if args.plugin_set == "auto":
        plugin_ids = ["auto"]
    else:
        # Full harness run: execute every non-ingest plugin on the loaded dataset.
        # (Ingest is file-driven and is skipped for DB-only runs.)
        profiles = _discover_plugin_ids({"profile"})
        planners = _discover_plugin_ids({"planner"})
        transforms = _discover_plugin_ids({"transform"})
        analyses = _discover_plugin_ids({"analysis"})
        reports = _discover_plugin_ids({"report"})
        llm = _discover_plugin_ids({"llm"})
        plugin_ids = [*profiles, *planners, *transforms, *analyses, *reports, *llm]

    pipeline = Pipeline(ctx.appdata_root, Path("plugins"), tenant_id=ctx.tenant_id)
    run_id = pipeline.run(
        input_file=None,
        plugin_ids=plugin_ids,
        settings={},
        run_seed=int(args.run_seed),
        dataset_version_id=dataset_version_id,
        run_id=run_id,
        force=bool(args.force),
    )
    run_dir = ctx.tenant_root / "runs" / run_id
    report_path = run_dir / "report.json"
    if not report_path.exists():
        raise SystemExit(f"report.json not found for run: {run_id}")

    report = _load_json(report_path)
    recs = _extract_recommendations(report)
    ideaspace_gap = _top_findings(report, "analysis_ideaspace_normative_gap", n=8)
    ideaspace_actions = _top_findings(report, "analysis_ideaspace_action_planner", n=12)

    answers = {
        "dataset": dataset,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "report_path": str(report_path),
        "recommendations": recs,
        "ideaspace": {
            "normative_gap_findings": ideaspace_gap,
            "action_planner_findings": ideaspace_actions,
        },
    }
    _write_json(run_dir / "answers_summary.json", answers)
    _write_text(run_dir / "answers_recommendations.md", _render_recommendations_md(recs))

    print(f"DATASET_VERSION_ID={dataset_version_id}")
    print(f"ROWS={int(dataset.get('row_count') or 0)} COLS={int(dataset.get('column_count') or 0)}")
    print(f"RUN_ID={run_id}")
    print(str(run_dir / "report.md"))
    print(str(run_dir / "answers_recommendations.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
