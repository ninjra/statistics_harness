from __future__ import annotations

import argparse
from pathlib import Path

from statistic_harness.core.pipeline import Pipeline
from statistic_harness.core.tenancy import get_tenant_context


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-version-id", required=True)
    ap.add_argument("--run-seed", type=int, default=123)
    args = ap.parse_args()

    ctx = get_tenant_context()
    pipeline = Pipeline(ctx.appdata_root, Path("plugins"), tenant_id=ctx.tenant_id)
    run_id = pipeline.run(
        input_file=None,
        # Non-empty selection prevents auto-planner mode from selecting a large plugin set.
        plugin_ids=["report_bundle"],
        settings={},
        run_seed=int(args.run_seed),
        dataset_version_id=str(args.dataset_version_id),
        force=False,
    )
    print(f"RUN_ID={run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
