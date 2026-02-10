#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from statistic_harness.core.storage import Storage
from statistic_harness.core.tenancy import get_tenant_context


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-version-id", required=True)
    args = ap.parse_args()

    ctx = get_tenant_context()
    storage = Storage(ctx.db_path, ctx.tenant_id)
    row = storage.get_dataset_version_context(args.dataset_version_id)
    if not row:
        raise SystemExit(f"dataset_version_id not found: {args.dataset_version_id}")
    table_name = str(row.get("table_name") or "")
    if not table_name:
        raise SystemExit("missing table_name for dataset version")
    storage.ensure_dataset_row_index_index(table_name)
    print(f"ok: ensured row_index index on {table_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

