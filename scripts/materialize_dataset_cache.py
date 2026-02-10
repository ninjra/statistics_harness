#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from statistic_harness.core.dataset_cache import DatasetCache, DatasetCacheKey
from statistic_harness.core.dataset_io import DatasetAccessor
from statistic_harness.core.storage import Storage
from statistic_harness.core.tenancy import get_tenant_context


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-version-id", required=True)
    ap.add_argument("--batch-size", type=int, default=100_000)
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--columns",
        action="append",
        default=None,
        help="Optional original column names to cache (repeatable). Default: infer numeric-ish columns.",
    )
    args = ap.parse_args()

    ctx = get_tenant_context()
    storage = Storage(ctx.db_path, ctx.tenant_id)
    accessor = DatasetAccessor(storage, args.dataset_version_id)

    with storage.connection() as conn:
        version = storage.get_dataset_version(args.dataset_version_id, conn)
        if not version:
            raise SystemExit(f"dataset_version_id not found: {args.dataset_version_id}")
        columns_meta = storage.fetch_dataset_columns(args.dataset_version_id, conn)

    ck = DatasetCacheKey.from_dataset(
        dataset_version_id=args.dataset_version_id,
        data_hash=str(version.get("data_hash") or ""),
        columns=columns_meta,
    )
    cache = DatasetCache(ck)
    manifest = cache.materialize_numeric_cache(
        accessor=accessor,
        columns=list(args.columns) if args.columns else None,
        batch_size=int(args.batch_size),
        force=bool(args.force),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

