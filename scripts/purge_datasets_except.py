#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Any


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    except sqlite3.OperationalError:
        return set()
    return {str(r["name"]) for r in rows}


def _quote_ident(name: str) -> str:
    # Minimal SQLite identifier quoting.
    return '"' + name.replace('"', '""') + '"'


def _iter_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "select name from sqlite_master where type='table' and name not like 'sqlite_%' order by name"
    ).fetchall()
    return [str(r[0]) for r in rows]


def _iter_tables_like(conn: sqlite3.Connection, like_pattern: str) -> list[str]:
    rows = conn.execute(
        "select name from sqlite_master where type='table' and name like ? and name not like 'sqlite_%' order by name",
        (like_pattern,),
    ).fetchall()
    return [str(r[0]) for r in rows]


def _is_raw_dataset_table(name: str) -> bool:
    # Raw ingest tables are named like: dataset_<64hex>
    if not name.startswith("dataset_"):
        return False
    suffix = name[len("dataset_") :]
    if len(suffix) != 64:
        return False
    return all(ch in "0123456789abcdef" for ch in suffix.lower())


def _iter_raw_dataset_tables(conn: sqlite3.Connection) -> list[str]:
    return [t for t in _iter_tables_like(conn, "dataset_%") if _is_raw_dataset_table(t)]


def _iter_triggers_for_table(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(
        "select name from sqlite_master where type='trigger' and tbl_name=? order by name",
        (table,),
    ).fetchall()
    return [str(r[0]) for r in rows]


def _chunked(seq: list[str], n: int = 500) -> list[list[str]]:
    out: list[list[str]] = []
    for i in range(0, len(seq), n):
        out.append(seq[i : i + n])
    return out


def _delete_where_in(
    conn: sqlite3.Connection,
    table: str,
    col: str,
    values: list[str],
) -> int:
    if not values:
        return 0
    total = 0
    for batch in _chunked(values, n=500):
        placeholders = ", ".join(["?"] * len(batch))
        sql = f"delete from {_quote_ident(table)} where {_quote_ident(col)} in ({placeholders})"
        cur = conn.execute(sql, batch)
        total += int(cur.rowcount or 0)
    return total


def _vacuum_or_raise(db_path: Path) -> None:
    # VACUUM requires an exclusive lock and can be sensitive to temp file behavior on mounted filesystems.
    # We try standard VACUUM first; if it fails, we fall back to VACUUM INTO + atomic replace.
    db_path = db_path.resolve()
    if not db_path.exists():
        raise sqlite3.OperationalError(f"DB missing: {db_path}")

    def _connect_autocommit(path: Path) -> sqlite3.Connection:
        con = sqlite3.connect(str(path), timeout=60)
        con.isolation_level = None  # autocommit
        try:
            con.execute("PRAGMA busy_timeout=60000")
        except sqlite3.OperationalError:
            pass
        return con

    con = _connect_autocommit(db_path)
    try:
        try:
            con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.OperationalError:
            pass
        try:
            # Prefer memory temp store to avoid temp file issues.
            con.execute("PRAGMA temp_store=2")
        except sqlite3.OperationalError:
            pass
        con.execute("VACUUM")
        return
    except sqlite3.OperationalError:
        pass
    finally:
        con.close()

    vac_path = db_path.with_name(db_path.name + ".vacuum")
    bak_path = db_path.with_name(db_path.name + ".pre_vacuum.bak")
    if vac_path.exists():
        vac_path.unlink()
    if bak_path.exists():
        bak_path.unlink()

    con2 = _connect_autocommit(db_path)
    try:
        try:
            con2.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.OperationalError:
            pass
        # VACUUM INTO generally wants a literal string; attempt parameterization first for safety,
        # then fall back to a literal if the runtime SQLite rejects it.
        try:
            con2.execute("VACUUM INTO ?", (str(vac_path),))
        except sqlite3.OperationalError:
            escaped = str(vac_path).replace("'", "''")
            con2.execute(f"VACUUM INTO '{escaped}'")
    finally:
        con2.close()

    if not vac_path.exists():
        raise sqlite3.OperationalError("VACUUM INTO failed: output file not created")

    os.replace(db_path, bak_path)
    try:
        os.replace(vac_path, db_path)
    except Exception:
        os.replace(bak_path, db_path)
        raise
    finally:
        try:
            vac_path.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass
    try:
        bak_path.unlink()
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--appdata", type=Path, default=Path("appdata"))
    ap.add_argument("--keep-dataset-version-id", required=True)
    ap.add_argument(
        "--delete-runs",
        action="store_true",
        default=True,
        help="Delete runs (and their DB rows/artifacts) for purged dataset versions.",
    )
    ap.add_argument(
        "--no-delete-runs",
        action="store_false",
        dest="delete_runs",
        help="Do not delete runs for purged dataset versions (not recommended).",
    )
    ap.add_argument(
        "--vacuum",
        action="store_true",
        help="Run VACUUM after deletion (can be slow on large DBs).",
    )
    ap.add_argument(
        "--yes",
        action="store_true",
        help="Apply deletion. Without --yes this prints the plan (dry-run).",
    )
    args = ap.parse_args()

    appdata: Path = args.appdata
    keep_dvid = str(args.keep_dataset_version_id).strip()
    if not keep_dvid:
        raise SystemExit("keep-dataset-version-id is required")

    db_path = appdata / "state.sqlite"
    if not db_path.exists():
        raise SystemExit(f"Missing DB: {db_path}")

    con = _connect(db_path)
    try:
        keep_row = con.execute(
            "select dataset_version_id, tenant_id, table_name, row_count, column_count, created_at from dataset_versions where dataset_version_id=? limit 1",
            (keep_dvid,),
        ).fetchone()
        if keep_row is None:
            raise SystemExit(f"Dataset version not found: {keep_dvid}")
        tenant_id = str(keep_row["tenant_id"] or "default")

        all_rows = con.execute(
            "select dataset_version_id, table_name, row_count, created_at from dataset_versions where tenant_id=? order by created_at desc",
            (tenant_id,),
        ).fetchall()
        all_dvids = [str(r["dataset_version_id"]) for r in all_rows]
        purge_dvids = [d for d in all_dvids if d != keep_dvid]

        raw_tables = []
        for r in all_rows:
            d = str(r["dataset_version_id"])
            if d == keep_dvid:
                continue
            t = str(r["table_name"] or "")
            if t:
                raw_tables.append(t)

        alive_raw_tables = {str(r["table_name"]) for r in all_rows if r["table_name"]}
        # Orphan raw dataset tables can exist if older purges were interrupted or if dataset_versions
        # rows were deleted without dropping the underlying raw table.
        orphan_raw_tables = sorted(set(_iter_raw_dataset_tables(con)) - set(alive_raw_tables))

        # Runs to purge (by dataset_version_id).
        runs_to_delete: list[str] = []
        if args.delete_runs and purge_dvids:
            placeholders = ", ".join(["?"] * len(purge_dvids))
            rows = con.execute(
                f"select run_id from runs where tenant_id=? and dataset_version_id in ({placeholders})",
                [tenant_id, *purge_dvids],
            ).fetchall()
            runs_to_delete = [str(r["run_id"]) for r in rows]

        # Template tables: delete rows for purged dataset_version_ids (but do NOT drop the tables).
        template_tables = [t for t in _iter_tables(con) if t.startswith("template_")]
        template_tables = sorted(set(template_tables))

        print(f"KEEP dataset_version_id={keep_dvid} tenant_id={tenant_id}")
        print(
            f"PURGE dataset_versions={len(purge_dvids)} runs={len(runs_to_delete)} "
            f"raw_tables={len(raw_tables)} orphan_raw_tables={len(orphan_raw_tables)}"
        )
        if purge_dvids:
            for d in purge_dvids:
                row = next((x for x in all_rows if str(x["dataset_version_id"]) == d), None)
                if row:
                    print(f"- {d} rows={row['row_count']} table={row['table_name']} created_at={row['created_at']}")
        if orphan_raw_tables:
            for t in orphan_raw_tables:
                print(f"- ORPHAN table={t}")
        if not purge_dvids and not orphan_raw_tables and not args.vacuum:
            print("Nothing to purge.")
            return 0

        if not args.yes:
            print("Dry-run only (pass --yes to apply).")
            return 0

        # Apply.
        deleted: dict[str, Any] = {"tables": {}, "dropped_tables": [], "dropped_triggers": [], "deleted_run_dirs": 0}
        with con:
            # Delete run-scoped rows first.
            if runs_to_delete:
                # Any table with a run_id column is eligible for purge.
                for table in _iter_tables(con):
                    cols = _table_columns(con, table)
                    if "run_id" not in cols:
                        continue
                    n = _delete_where_in(con, table, "run_id", runs_to_delete)
                    if n:
                        deleted["tables"][f"{table}.run_id"] = n

            # Delete dataset-version scoped rows.
            for table in (
                "dataset_columns",
                "dataset_role_candidates",
                "dataset_templates",
                "template_conversions",
                "dataset_versions",
            ):
                if table not in _iter_tables(con):
                    continue
                cols = _table_columns(con, table)
                if "dataset_version_id" not in cols:
                    continue
                n = _delete_where_in(con, table, "dataset_version_id", purge_dvids)
                if n:
                    deleted["tables"][f"{table}.dataset_version_id"] = n

            # Purge template rows for deleted dataset versions.
            for table in template_tables:
                cols = _table_columns(con, table)
                if "dataset_version_id" not in cols:
                    continue
                n = _delete_where_in(con, table, "dataset_version_id", purge_dvids)
                if n:
                    deleted["tables"][f"{table}.dataset_version_id"] = n

            # Drop raw dataset tables (from purged dataset_versions) and their triggers.
            for table in sorted(set(raw_tables)):
                for trig in _iter_triggers_for_table(con, table):
                    try:
                        con.execute(f"drop trigger if exists {_quote_ident(trig)}")
                        deleted["dropped_triggers"].append(trig)
                    except sqlite3.OperationalError:
                        pass
                try:
                    con.execute(f"drop table if exists {_quote_ident(table)}")
                    deleted["dropped_tables"].append(table)
                except sqlite3.OperationalError:
                    pass

            # Drop any orphan raw dataset tables that are no longer referenced by dataset_versions.
            alive_raw = {
                str(r[0])
                for r in con.execute(
                    "select table_name from dataset_versions where tenant_id=? and table_name is not null",
                    (tenant_id,),
                ).fetchall()
            }
            for table in sorted(set(_iter_raw_dataset_tables(con)) - set(alive_raw)):
                for trig in _iter_triggers_for_table(con, table):
                    try:
                        con.execute(f"drop trigger if exists {_quote_ident(trig)}")
                        deleted["dropped_triggers"].append(trig)
                    except sqlite3.OperationalError:
                        pass
                try:
                    con.execute(f"drop table if exists {_quote_ident(table)}")
                    deleted["dropped_tables"].append(table)
                except sqlite3.OperationalError:
                    pass

            # Clean up orphans.
            try:
                con.execute(
                    "delete from datasets where tenant_id=? and dataset_id not in (select distinct dataset_id from dataset_versions where tenant_id=?)",
                    (tenant_id, tenant_id),
                )
            except sqlite3.OperationalError:
                pass
            try:
                con.execute(
                    "delete from projects where tenant_id=? and project_id not in (select distinct project_id from datasets where tenant_id=?)",
                    (tenant_id, tenant_id),
                )
            except sqlite3.OperationalError:
                pass

        # Delete run directories on disk (best-effort).
        if args.delete_runs and runs_to_delete:
            runs_root = appdata / "runs"
            for rid in runs_to_delete:
                shutil.rmtree(runs_root / rid, ignore_errors=True)
            deleted["deleted_run_dirs"] = len(runs_to_delete)

        if args.vacuum:
            try:
                _vacuum_or_raise(db_path)
            except sqlite3.OperationalError as exc:
                raise SystemExit(f"ERROR: VACUUM failed: {exc}")

        print("OK purged.")
        for k, v in sorted(deleted["tables"].items()):
            print(f"deleted[{k}]={v}")
        print(f"dropped_tables={len(deleted['dropped_tables'])} dropped_triggers={len(deleted['dropped_triggers'])}")
        print(f"deleted_run_dirs={deleted['deleted_run_dirs']}")
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
