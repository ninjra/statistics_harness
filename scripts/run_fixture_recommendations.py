from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from statistic_harness.core.pipeline import Pipeline
from statistic_harness.core.report import build_report, write_report
from statistic_harness.core.tenancy import get_tenant_context


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixture",
        default="tests/fixtures/synth_linear.csv",
        help="Path to a local fixture dataset file",
    )
    parser.add_argument("--plugins", default="all")
    parser.add_argument("--run-seed", type=int, default=123)
    parser.add_argument(
        "--appdata",
        default="",
        help="If set, use this directory for STAT_HARNESS_APPDATA (otherwise a temp dir is used).",
    )
    args = parser.parse_args()

    fixture = Path(args.fixture)
    if not fixture.exists():
        raise SystemExit(f"Fixture not found: {fixture}")

    if args.appdata:
        appdata = Path(args.appdata)
        appdata.mkdir(parents=True, exist_ok=True)
    else:
        appdata = Path(tempfile.mkdtemp(prefix="stat_harness_fixture_"))

    os.environ["STAT_HARNESS_APPDATA"] = str(appdata)
    os.environ.setdefault("STAT_HARNESS_CLI_PROGRESS", "0")

    tenant_ctx = get_tenant_context()
    pipeline = Pipeline(tenant_ctx.appdata_root, Path("plugins"), tenant_id=tenant_ctx.tenant_id)

    plugin_ids = [p.strip() for p in str(args.plugins).split(",") if p.strip()]
    run_id = pipeline.run(fixture, plugin_ids, settings={}, run_seed=int(args.run_seed))
    run_dir = tenant_ctx.tenant_root / "runs" / run_id

    report_path = run_dir / "report.json"
    if not report_path.exists():
        report = build_report(pipeline.storage, run_id, run_dir, Path("docs/report.schema.json"))
        write_report(report, run_dir)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    recs = {}
    if isinstance(report, dict):
        recs = report.get("recommendations") or {}
    items = []
    if isinstance(recs, dict):
        items = recs.get("items") or []

    ideaspace = report.get("ideaspace") if isinstance(report, dict) else None
    current = ideaspace.get("insight_index") if isinstance(ideaspace, dict) else None
    ideal = ideaspace.get("landmarks") if isinstance(ideaspace, dict) else None

    out = {
        "run_id": run_id,
        "appdata": str(appdata),
        "run_dir": str(run_dir),
        "recommendations_status": recs.get("status") if isinstance(recs, dict) else None,
        "recommendations_summary": recs.get("summary") if isinstance(recs, dict) else None,
        "recommendations_count": len(items) if isinstance(items, list) else 0,
        "recommendations_items": items if isinstance(items, list) else [],
        "ideaspace_erp_type": ideaspace.get("erp_type") if isinstance(ideaspace, dict) else None,
        "ideaspace_current_count": len(current) if isinstance(current, list) else 0,
        "ideaspace_current": current if isinstance(current, list) else [],
        "ideaspace_ideal_count": len(ideal) if isinstance(ideal, list) else 0,
        "ideaspace_ideal": ideal if isinstance(ideal, list) else [],
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
