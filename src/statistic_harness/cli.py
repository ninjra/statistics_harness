from __future__ import annotations

import argparse
import os
import hashlib
import json
import zipfile
import platform
from pathlib import Path
from typing import Any

import uvicorn
import yaml

from statistic_harness.core.auth import (
    generate_api_key,
    hash_password,
    hash_token,
    normalize_email,
    verify_password,
)
from statistic_harness.core.evaluation import evaluate_report
from statistic_harness.core.pipeline import Pipeline
from statistic_harness.core.plugin_manager import PluginManager
from statistic_harness.core.report import build_report, write_report
from statistic_harness.core.payout_report import build_payout_report
from statistic_harness.core.storage import Storage
from statistic_harness.core.tenancy import get_tenant_context
from statistic_harness.core.utils import (
    auth_enabled,
    json_dumps,
    now_iso,
    file_sha256,
    is_windows_or_wsl,
    vector_store_enabled,
)
from statistic_harness.core.vector_store import VectorStore, hash_embedding


if is_windows_or_wsl() and "STAT_HARNESS_SAFE_RENAME" not in os.environ:
    # Default ON for Windows/WSL to reduce partial artifact risk from transient locks.
    os.environ["STAT_HARNESS_SAFE_RENAME"] = "1"


def load_settings(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    content = Path(path).read_text(encoding="utf-8")
    if path.endswith(".json"):
        return json.loads(content)
    return yaml.safe_load(content)


def cmd_list_plugins() -> None:
    manager = PluginManager(Path("plugins"))
    for spec in manager.discover():
        print(f"{spec.plugin_id}: {spec.name} ({spec.type})")


def cmd_plugins_validate(plugin_id: str | None = None) -> None:
    manager = PluginManager(Path("plugins"))
    specs = manager.discover()
    failures: list[str] = []

    if manager.discovery_errors:
        for err in manager.discovery_errors:
            failures.append(f"{err.plugin_id}: discovery error: {err.message}")

    selected = specs
    if plugin_id:
        selected = [spec for spec in specs if spec.plugin_id == plugin_id]
        if not selected:
            raise SystemExit(f"Unknown plugin id: {plugin_id}")

    for spec in selected:
        try:
            # Load and validate schemas (discovery already checks presence).
            manager.validate_config(spec, dict(spec.settings.get("defaults", {})))
            # Import entrypoint and instantiate plugin.
            plugin = manager.load_plugin(spec)
            if not hasattr(plugin, "run"):
                raise TypeError("Missing run() method")
            health = manager.health(spec, plugin=plugin)
            if str(health.get("status") or "ok").lower() not in {"ok", "healthy"}:
                raise RuntimeError(f"Unhealthy plugin: {health}")
        except Exception as exc:
            failures.append(f"{spec.plugin_id}: {type(exc).__name__}: {exc}")

    for line in sorted(failures):
        print(line)
    if failures:
        raise SystemExit(1)
    print("OK")


def cmd_integrity_check(full: bool = False) -> None:
    tenant_ctx = get_tenant_context()
    storage = Storage(tenant_ctx.db_path, tenant_ctx.tenant_id)
    ok, msg = storage.integrity_check(full=full)
    if not ok:
        raise SystemExit(f"Integrity check failed: {msg}")
    print("OK")


def cmd_backup(out_path: str | None, retention_days: int = 60) -> None:
    tenant_ctx = get_tenant_context()
    storage = Storage(tenant_ctx.db_path, tenant_ctx.tenant_id)
    backups_dir = tenant_ctx.appdata_root / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True)
    if out_path:
        dest = Path(out_path)
    else:
        ts = now_iso().replace(":", "").replace("-", "")
        dest = backups_dir / f"state-{ts}.sqlite"
    storage.backup_to(dest)
    if not out_path:
        # Best-effort retention cleanup (mtime-based).
        try:
            import time

            cutoff = time.time() - (max(1, int(retention_days)) * 86400)
            for path in sorted(backups_dir.glob("state-*.sqlite")):
                try:
                    if path.stat().st_mtime < cutoff:
                        path.unlink()
                except Exception:
                    continue
        except Exception:
            pass
    print(str(dest))


def cmd_restore(source_path: str) -> None:
    tenant_ctx = get_tenant_context()
    storage = Storage(tenant_ctx.db_path, tenant_ctx.tenant_id)
    storage.restore_from(Path(source_path))
    ok, msg = storage.integrity_check(full=False)
    if not ok:
        raise SystemExit(f"Restore completed but integrity check failed: {msg}")
    print("OK")


def cmd_diag(run_id: str, out_path: str | None) -> None:
    tenant_ctx = get_tenant_context()
    run_dir = tenant_ctx.tenant_root / "runs" / run_id
    if not run_dir.exists():
        raise SystemExit("Run not found")
    if out_path:
        dest = Path(out_path)
    else:
        dest = run_dir / "diag.zip"
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Allowlisted contents only.
    allow = [
        "run_manifest.json",
        "report.json",
        "report.md",
        "evaluation.json",
    ]
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in allow:
            path = run_dir / name
            if path.exists() and path.is_file():
                zf.write(path, arcname=name)
        for item in sorted((run_dir / "logs").rglob("*")) if (run_dir / "logs").exists() else []:
            if item.is_file():
                zf.write(item, arcname=str(item.relative_to(run_dir)).replace("\\", "/"))
        for item in sorted((run_dir / "artifacts").rglob("*")) if (run_dir / "artifacts").exists() else []:
            if item.is_file():
                zf.write(item, arcname=str(item.relative_to(run_dir)).replace("\\", "/"))
        # System info (no env secrets).
        info = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "vector_store_enabled": vector_store_enabled(),
            "auth_enabled": auth_enabled(),
        }
        zf.writestr("system.json", json.dumps(info, indent=2, sort_keys=True))
        zf.write(Path("docs") / "report.schema.json", arcname="schemas/report.schema.json")
        zf.write(Path("docs") / "run_manifest.schema.json", arcname="schemas/run_manifest.schema.json")
    print(str(dest))


def cmd_export(
    run_id: str,
    out_path: str | None,
    *,
    include_logs: bool = True,
    include_artifacts: bool = True,
    include_dataset: bool = False,
) -> None:
    """Export a repro pack zip (manifest + schemas + report + optional logs/artifacts)."""

    tenant_ctx = get_tenant_context()
    run_dir = tenant_ctx.tenant_root / "runs" / run_id
    if not run_dir.exists():
        raise SystemExit("Run not found")
    if out_path:
        dest = Path(out_path)
    else:
        dest = run_dir / "repro_pack.zip"
    dest.parent.mkdir(parents=True, exist_ok=True)

    allow = [
        "run_manifest.json",
        "report.json",
        "report.md",
        "evaluation.json",
    ]
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in allow:
            path = run_dir / name
            if path.exists() and path.is_file():
                zf.write(path, arcname=name)
        if include_logs:
            for item in sorted((run_dir / "logs").rglob("*")) if (run_dir / "logs").exists() else []:
                if item.is_file():
                    zf.write(item, arcname=str(item.relative_to(run_dir)).replace("\\", "/"))
        if include_artifacts:
            for item in sorted((run_dir / "artifacts").rglob("*")) if (run_dir / "artifacts").exists() else []:
                if item.is_file():
                    zf.write(item, arcname=str(item.relative_to(run_dir)).replace("\\", "/"))
        if include_dataset:
            for item in sorted((run_dir / "dataset").rglob("*")) if (run_dir / "dataset").exists() else []:
                if item.is_file():
                    zf.write(item, arcname=str(item.relative_to(run_dir)).replace("\\", "/"))

        info = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "vector_store_enabled": vector_store_enabled(),
            "auth_enabled": auth_enabled(),
        }
        zf.writestr("system.json", json.dumps(info, indent=2, sort_keys=True))
        zf.write(Path("docs") / "report.schema.json", arcname="schemas/report.schema.json")
        zf.write(Path("docs") / "run_manifest.schema.json", arcname="schemas/run_manifest.schema.json")
    print(str(dest))


def cmd_db_doctor(full: bool = False) -> None:
    """Operator-oriented DB check wrapper (integrity check + basic status)."""

    tenant_ctx = get_tenant_context()
    storage = Storage(tenant_ctx.db_path, tenant_ctx.tenant_id)
    ok, msg = storage.integrity_check(full=full)
    if not ok:
        raise SystemExit(f"Integrity check failed: {msg}")
    print("OK")


def cmd_replay(run_id: str) -> None:
    tenant_ctx = get_tenant_context()
    run_dir = tenant_ctx.tenant_root / "runs" / run_id
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise SystemExit("Missing run_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = json.loads((Path("docs") / "run_manifest.schema.json").read_text(encoding="utf-8"))
    from jsonschema import validate  # lazy import for CLI

    validate(instance=manifest, schema=schema)
    for art in manifest.get("artifacts") or []:
        if not isinstance(art, dict):
            continue
        rel = str(art.get("path") or "")
        if not rel:
            continue
        path = (run_dir / rel).resolve()
        try:
            path.relative_to(run_dir.resolve())
        except ValueError:
            raise SystemExit(f"Path traversal in manifest artifact: {rel}")
        if not path.exists():
            raise SystemExit(f"Missing artifact: {rel}")
        want = str(art.get("sha256") or "")
        got = file_sha256(path)
        if want and got != want:
            raise SystemExit(f"Hash mismatch for {rel}: expected {want} got {got}")
    print("OK")


def cmd_serve(host: str, port: int) -> None:
    from statistic_harness.ui.server import app

    allow_network = os.environ.get("STAT_HARNESS_ALLOW_NETWORK", "").lower() in {
        "1",
        "true",
        "yes",
    }
    if not allow_network and host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("Network disabled: use localhost or set STAT_HARNESS_ALLOW_NETWORK=1")
    uvicorn.run(app, host=host, port=port)


def _require_cli_api_key(admin_required: bool = False) -> None:
    if not auth_enabled():
        return
    token = os.environ.get("STAT_HARNESS_API_KEY", "").strip()
    if not token:
        raise SystemExit("Missing STAT_HARNESS_API_KEY")
    tenant_ctx = get_tenant_context()
    storage = Storage(tenant_ctx.db_path, tenant_ctx.tenant_id)
    row = storage.fetch_api_key_by_hash(hash_token(token), tenant_id=tenant_ctx.tenant_id)
    if not row or row.get("revoked_at") or row.get("disabled_at"):
        raise SystemExit("Invalid API key")
    membership = storage.fetch_membership(int(row["user_id"]), tenant_ctx.tenant_id)
    if not membership:
        raise SystemExit("API key not authorized for tenant")
    if admin_required and not bool(row.get("is_admin")):
        raise SystemExit("Admin API key required")
    storage.touch_api_key(int(row["key_id"]), now_iso())


def cmd_run(
    file_path: str, plugins: str, settings_path: str | None, run_seed: int
) -> None:
    _require_cli_api_key()
    os.environ.setdefault("STAT_HARNESS_CLI_PROGRESS", "1")
    tenant_ctx = get_tenant_context()
    pipeline = Pipeline(
        tenant_ctx.appdata_root, Path("plugins"), tenant_id=tenant_ctx.tenant_id
    )
    plugin_ids = [p for p in plugins.split(",") if p]
    settings = load_settings(settings_path)
    run_id = pipeline.run(Path(file_path), plugin_ids, settings, run_seed)
    run_dir = tenant_ctx.tenant_root / "runs" / run_id
    report = build_report(
        pipeline.storage, run_id, run_dir, Path("docs/report.schema.json")
    )
    write_report(report, run_dir)
    print(run_id)


def cmd_eval(report_path: str, ground_truth: str) -> None:
    ok, messages = evaluate_report(Path(report_path), Path(ground_truth))
    if not ok:
        for msg in messages:
            print(msg)
        raise SystemExit(1)
    print("Evaluation passed")


def cmd_make_ground_truth(report_path: str, output_path: str) -> None:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    features = [
        f.get("feature")
        for f in report.get("plugins", {})
        .get("analysis_gaussian_knockoffs", {})
        .get("findings", [])
    ]
    template = {
        "strict": False,
        "features": [f for f in features if f],
        "changepoints": [],
        "dependence_shift_pairs": [],
        "anomalies": [],
        "min_anomaly_hits": 0,
        "changepoint_tolerance": 3,
    }
    Path(output_path).write_text(yaml.safe_dump(template), encoding="utf-8")


def cmd_backfill(plugin_id: str, run_seed: int) -> None:
    _require_cli_api_key()
    tenant_ctx = get_tenant_context()
    pipeline = Pipeline(
        tenant_ctx.appdata_root, Path("plugins"), tenant_id=tenant_ctx.tenant_id
    )
    specs = {spec.plugin_id: spec for spec in pipeline.manager.discover()}
    if plugin_id not in specs:
        raise SystemExit(f"Unknown plugin: {plugin_id}")
    spec = specs[plugin_id]
    module_path, _ = spec.entrypoint.split(":", 1)
    if module_path.endswith(".py"):
        module_file = spec.path / module_path
    else:
        module_file = spec.path / f"{module_path}.py"
    code_hash = file_sha256(module_file) if module_file.exists() else None
    settings_hash = hashlib.sha256(json_dumps({}).encode("utf-8")).hexdigest()

    for dataset in pipeline.storage.list_dataset_versions():
        dataset_version_id = dataset["dataset_version_id"]
        pipeline.storage.enqueue_analysis_job(
            dataset_version_id,
            plugin_id,
            spec.version,
            code_hash,
            settings_hash,
            run_seed,
            now_iso(),
        )

    jobs = pipeline.storage.list_analysis_jobs(status="queued")
    for job in jobs:
        job_id = int(job["job_id"])
        pipeline.storage.update_analysis_job_status(
            job_id, "running", started_at=now_iso()
        )
        try:
            pipeline.run(
                None,
                [plugin_id],
                {},
                int(job.get("run_seed") or 0),
                dataset_version_id=job["dataset_version_id"],
            )
            pipeline.storage.update_analysis_job_status(
                job_id, "completed", completed_at=now_iso()
            )
        except Exception as exc:  # pragma: no cover - failure path
            pipeline.storage.update_analysis_job_status(
                job_id, "error", completed_at=now_iso(), error={"message": str(exc)}
            )


def cmd_create_user(email: str, password: str, name: str | None, admin: bool) -> None:
    tenant_ctx = get_tenant_context()
    storage = Storage(tenant_ctx.db_path, tenant_ctx.tenant_id)
    if auth_enabled() and storage.count_users() > 0:
        _require_cli_api_key(admin_required=True)
    normalized = normalize_email(email)
    if storage.fetch_user_by_email(normalized):
        raise SystemExit("User already exists")
    user_id = storage.create_user(
        normalized, hash_password(password), name, admin, now_iso()
    )
    role = "admin" if admin else "member"
    storage.ensure_membership(user_id, role, now_iso())
    print(f"Created user {normalized}")


def cmd_create_api_key(email: str, password: str, name: str | None) -> None:
    tenant_ctx = get_tenant_context()
    storage = Storage(tenant_ctx.db_path, tenant_ctx.tenant_id)
    user = storage.fetch_user_by_email(normalize_email(email))
    if not user or user.get("disabled_at"):
        raise SystemExit("Invalid credentials")
    if not verify_password(password, user.get("password_hash") or ""):
        raise SystemExit("Invalid credentials")
    membership = storage.fetch_membership(int(user["user_id"]), tenant_ctx.tenant_id)
    if not membership:
        raise SystemExit("User not authorized for tenant")
    token = generate_api_key()
    storage.create_api_key(
        int(user["user_id"]), hash_token(token), name, now_iso()
    )
    print(token)


def cmd_payout_report(inputs: list[str], out_dir: str) -> None:
    _require_cli_api_key()
    if not inputs:
        raise SystemExit("Provide at least one --input")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    frames = []
    for path in inputs:
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"Input not found: {path}")
        import pandas as pd

        df = pd.read_csv(p)
        df["__source_file"] = p.name
        df["__source_hash"] = file_sha256(p)
        frames.append(df)

    import pandas as pd

    stacked = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()
    report = build_payout_report(stacked)

    artifacts_dir = out / "artifacts" / "report_payout_report_v1"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    report_json = artifacts_dir / "payout_report.json"
    report_csv = artifacts_dir / "payout_report.csv"
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    # CSV: overall + per source rows
    rows = []
    overall = dict(report.get("metrics") or {})
    rows.append({"scope": "overall", **overall})
    for row in report.get("per_source") or []:
        if isinstance(row, dict):
            rows.append({"scope": "source", **row})
    pd.DataFrame(rows).to_csv(report_csv, index=False)
    print(str(report_json))
    print(str(report_csv))


def cmd_create_tenant(tenant_id: str, name: str | None) -> None:
    tenant_ctx = get_tenant_context()
    storage = Storage(tenant_ctx.db_path, tenant_ctx.tenant_id)
    if auth_enabled():
        _require_cli_api_key(admin_required=True)
    storage.create_tenant(tenant_id.strip(), name, now_iso())
    print(f"Created tenant {tenant_id}")


def cmd_revoke_api_key(key_id: int) -> None:
    tenant_ctx = get_tenant_context()
    storage = Storage(tenant_ctx.db_path, tenant_ctx.tenant_id)
    if auth_enabled():
        _require_cli_api_key(admin_required=True)
    storage.revoke_api_key(int(key_id))
    print("API key revoked")


def cmd_disable_user(email: str) -> None:
    tenant_ctx = get_tenant_context()
    storage = Storage(tenant_ctx.db_path, tenant_ctx.tenant_id)
    if auth_enabled():
        _require_cli_api_key(admin_required=True)
    user = storage.fetch_user_by_email(normalize_email(email))
    if not user:
        raise SystemExit("User not found")
    storage.disable_user(int(user["user_id"]))
    storage.revoke_user_sessions(int(user["user_id"]))
    storage.revoke_user_api_keys(int(user["user_id"]))
    print(f"Disabled user {email}")


def _require_vector_store() -> None:
    if not vector_store_enabled():
        raise SystemExit("Vector store disabled (STAT_HARNESS_ENABLE_VECTOR_STORE=1)")


def cmd_vector_list() -> None:
    _require_vector_store()
    if auth_enabled():
        _require_cli_api_key()
    tenant_ctx = get_tenant_context()
    store = VectorStore(tenant_ctx.db_path, tenant_ctx.tenant_id)
    for row in store.list_collections():
        name = row.get("name")
        dims = row.get("dimensions")
        created = row.get("created_at") or ""
        print(f"{name}\t{dims}\t{created}")


def _resolve_dimensions(
    store: VectorStore, collection: str, dimensions: int | None
) -> int:
    if dimensions is not None:
        return int(dimensions)
    dims = store.collection_dimensions(collection)
    if not dims:
        raise SystemExit("Unknown collection or dimensions required")
    if len(dims) > 1:
        raise SystemExit("Multiple dimensions found; use --dimensions")
    return int(dims[0])


def cmd_vector_query(
    collection: str,
    text: str | None,
    vector: str | None,
    k: int,
    dimensions: int | None,
) -> None:
    _require_vector_store()
    if auth_enabled():
        _require_cli_api_key()
    tenant_ctx = get_tenant_context()
    store = VectorStore(tenant_ctx.db_path, tenant_ctx.tenant_id)
    if text and vector:
        raise SystemExit("Use either --text or --vector")
    if not text and not vector:
        raise SystemExit("Provide --text or --vector")
    if vector:
        parts = [part for part in vector.replace("\n", ",").split(",") if part.strip()]
        values = [float(part.strip()) for part in parts]
        query_vector = values
    else:
        dims = _resolve_dimensions(store, collection, dimensions)
        query_vector = hash_embedding(text or "", dims)
    results = store.query(collection, query_vector, k=k)
    print(json.dumps(results, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-plugins")

    plugins_parser = sub.add_parser("plugins")
    plugins_sub = plugins_parser.add_subparsers(dest="plugins_command", required=True)
    plugins_validate_parser = plugins_sub.add_parser("validate")
    plugins_validate_parser.add_argument("--plugin-id")

    integrity_parser = sub.add_parser("integrity-check")
    integrity_parser.add_argument("--full", action="store_true")

    backup_parser = sub.add_parser("backup")
    backup_parser.add_argument("--out")
    backup_parser.add_argument("--retention-days", type=int, default=60)

    restore_parser = sub.add_parser("restore")
    restore_parser.add_argument("--path", required=True)

    diag_parser = sub.add_parser("diag")
    diag_parser.add_argument("--run-id", required=True)
    diag_parser.add_argument("--out")

    export_parser = sub.add_parser("export")
    export_parser.add_argument("--run-id", required=True)
    export_parser.add_argument("--out")
    export_parser.add_argument("--no-logs", action="store_true")
    export_parser.add_argument("--no-artifacts", action="store_true")
    export_parser.add_argument("--include-dataset", action="store_true")

    replay_parser = sub.add_parser("replay")
    replay_parser.add_argument("--run-id", required=True)

    db_parser = sub.add_parser("db")
    db_sub = db_parser.add_subparsers(dest="db_command", required=True)
    db_doctor = db_sub.add_parser("doctor")
    db_doctor.add_argument("--full", action="store_true")

    serve_parser = sub.add_parser("serve")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)

    run_parser = sub.add_parser("run")
    run_parser.add_argument("--file", required=True)
    run_parser.add_argument("--plugins", default="all")
    run_parser.add_argument("--settings")
    run_parser.add_argument("--run-seed", type=int, default=0)

    eval_parser = sub.add_parser("eval")
    eval_parser.add_argument("--report", required=True)
    eval_parser.add_argument("--ground-truth", required=True)

    gt_parser = sub.add_parser("make-ground-truth-template")
    gt_parser.add_argument("--report", required=True)
    gt_parser.add_argument("-o", "--output", required=True)

    backfill_parser = sub.add_parser("backfill")
    backfill_parser.add_argument("--plugin", required=True)
    backfill_parser.add_argument("--run-seed", type=int, default=0)

    payout_parser = sub.add_parser("payout-report")
    payout_parser.add_argument("--input", action="append", required=True)
    payout_parser.add_argument("--out", required=True)

    user_parser = sub.add_parser("create-user")
    user_parser.add_argument("--email", required=True)
    user_parser.add_argument("--password", required=True)
    user_parser.add_argument("--name")
    user_parser.add_argument("--admin", action="store_true")

    api_key_parser = sub.add_parser("create-api-key")
    api_key_parser.add_argument("--email", required=True)
    api_key_parser.add_argument("--password", required=True)
    api_key_parser.add_argument("--name")

    tenant_parser = sub.add_parser("create-tenant")
    tenant_parser.add_argument("--tenant-id", required=True)
    tenant_parser.add_argument("--name")

    revoke_key_parser = sub.add_parser("revoke-api-key")
    revoke_key_parser.add_argument("--key-id", required=True, type=int)

    disable_user_parser = sub.add_parser("disable-user")
    disable_user_parser.add_argument("--email", required=True)

    sub.add_parser("vector-list")

    vector_query_parser = sub.add_parser("vector-query")
    vector_query_parser.add_argument("--collection", required=True)
    vector_query_parser.add_argument("--text")
    vector_query_parser.add_argument("--vector")
    vector_query_parser.add_argument("--k", type=int, default=10)
    vector_query_parser.add_argument("--dimensions", type=int)

    args = parser.parse_args()

    if args.command == "list-plugins":
        cmd_list_plugins()
    elif args.command == "plugins":
        if args.plugins_command == "validate":
            cmd_plugins_validate(args.plugin_id)
        else:
            raise SystemExit(2)
    elif args.command == "integrity-check":
        cmd_integrity_check(full=bool(args.full))
    elif args.command == "backup":
        cmd_backup(args.out, args.retention_days)
    elif args.command == "restore":
        cmd_restore(args.path)
    elif args.command == "diag":
        cmd_diag(args.run_id, args.out)
    elif args.command == "export":
        cmd_export(
            args.run_id,
            args.out,
            include_logs=not bool(args.no_logs),
            include_artifacts=not bool(args.no_artifacts),
            include_dataset=bool(args.include_dataset),
        )
    elif args.command == "replay":
        cmd_replay(args.run_id)
    elif args.command == "db":
        if args.db_command == "doctor":
            cmd_db_doctor(full=bool(args.full))
        else:
            raise SystemExit(2)
    elif args.command == "serve":
        cmd_serve(args.host, args.port)
    elif args.command == "run":
        cmd_run(args.file, args.plugins, args.settings, args.run_seed)
    elif args.command == "eval":
        cmd_eval(args.report, args.ground_truth)
    elif args.command == "make-ground-truth-template":
        cmd_make_ground_truth(args.report, args.output)
    elif args.command == "backfill":
        cmd_backfill(args.plugin, args.run_seed)
    elif args.command == "payout-report":
        cmd_payout_report(args.input, args.out)
    elif args.command == "create-user":
        cmd_create_user(args.email, args.password, args.name, args.admin)
    elif args.command == "create-api-key":
        cmd_create_api_key(args.email, args.password, args.name)
    elif args.command == "create-tenant":
        cmd_create_tenant(args.tenant_id, args.name)
    elif args.command == "revoke-api-key":
        cmd_revoke_api_key(args.key_id)
    elif args.command == "disable-user":
        cmd_disable_user(args.email)
    elif args.command == "vector-list":
        cmd_vector_list()
    elif args.command == "vector-query":
        cmd_vector_query(
            args.collection, args.text, args.vector, args.k, args.dimensions
        )
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
