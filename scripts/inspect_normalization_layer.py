#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=Path("appdata/state.sqlite"))
    ap.add_argument("--dataset-version-id", required=True)
    args = ap.parse_args()

    dvid = str(args.dataset_version_id).strip()
    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row
    try:
        dv = con.execute(
            "SELECT dataset_version_id, tenant_id, dataset_id, table_name, row_count, column_count, data_hash "
            "FROM dataset_versions WHERE dataset_version_id=? LIMIT 1",
            (dvid,),
        ).fetchone()
        if not dv:
            raise SystemExit(f"dataset_version_id not found: {dvid}")
        print(f"dataset_version_id={dv['dataset_version_id']}")
        print(f"raw_table={dv['table_name']}")
        print(f"raw_row_count={int(dv['row_count'] or 0)}")
        print(f"raw_column_count={int(dv['column_count'] or 0)}")
        print(f"data_hash={dv['data_hash'] or ''}")

        dt = con.execute(
            "SELECT dt.dataset_version_id, dt.template_id, dt.status, dt.mapping_json, dt.mapping_hash, dt.updated_at, "
            "t.name AS template_name, t.table_name AS template_table_name, t.version AS template_version "
            "FROM dataset_templates dt "
            "JOIN templates t ON t.template_id = dt.template_id "
            "WHERE dt.dataset_version_id=? "
            "ORDER BY dt.updated_at DESC LIMIT 1",
            (dvid,),
        ).fetchone()
        if not dt:
            print("normalized_status=missing")
            return 0

        print(f"normalized_status={dt['status']}")
        print(f"template_id={dt['template_id']}")
        print(f"template_name={dt['template_name']}")
        print(f"template_version={dt['template_version'] or ''}")
        print(f"template_table={dt['template_table_name']}")
        print(f"mapping_hash={dt['mapping_hash']}")
        print(f"normalized_updated_at={dt['updated_at'] or ''}")

        # Print template fields (normalized columns).
        fields = con.execute(
            "SELECT field_id, name, safe_name, dtype, role, required "
            "FROM template_fields WHERE template_id=? ORDER BY field_id",
            (int(dt["template_id"]),),
        ).fetchall()
        print(f"normalized_fields={len(fields)}")
        for r in fields:
            fid = int(r["field_id"])
            name = str(r["name"])
            dtype = str(r["dtype"] or "")
            role = str(r["role"] or "")
            req = int(r["required"] or 0)
            print(f"field[{fid}]={name}\tdtype={dtype}\trole={role}\trequired={req}")

        # Mapping coverage (safe_name-based).
        mapping_payload = {}
        try:
            mapping_payload = json.loads(str(dt["mapping_json"] or "{}"))
        except Exception:
            mapping_payload = {}
        mapping = mapping_payload.get("mapping") if isinstance(mapping_payload, dict) else None
        if not isinstance(mapping, dict):
            mapping = {}
        observed_safe = set()
        for _, src in mapping.items():
            if isinstance(src, dict):
                safe = str(src.get("safe_name") or "")
                if safe:
                    observed_safe.add(safe)
        cols = con.execute(
            "SELECT safe_name FROM dataset_columns WHERE dataset_version_id=? ORDER BY column_id",
            (dvid,),
        ).fetchall()
        expected_safe = {str(r["safe_name"]) for r in cols if r["safe_name"]}
        missing = sorted(expected_safe - observed_safe)
        extra = sorted(observed_safe - expected_safe)
        print(f"mapping_expected_safe={len(expected_safe)} mapping_observed_safe={len(observed_safe)}")
        print(f"mapping_missing_safe={len(missing)} mapping_extra_safe={len(extra)}")
        for s in missing[:50]:
            print(f"mapping_missing_safe_item={s}")
        if len(missing) > 50:
            print(f"mapping_missing_safe_more={len(missing)-50}")
        for s in extra[:50]:
            print(f"mapping_extra_safe_item={s}")
        if len(extra) > 50:
            print(f"mapping_extra_safe_more={len(extra)-50}")
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

