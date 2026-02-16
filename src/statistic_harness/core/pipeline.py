from __future__ import annotations

import hashlib
import sys
import threading
import time
import traceback
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os
from typing import Any

from .dataset_io import resolve_dataset_accessor
from .plugin_manager import PluginManager, PluginSpec
from .plugin_runner import run_plugin_subprocess
from .retention import apply_retention
from .storage import Storage
from .tenancy import get_tenant_context, scope_identifier, tenancy_enabled
from .types import PluginArtifact, PluginError, PluginResult
from .utils import (
    atomic_dir,
    dataset_key,
    ensure_dir,
    file_sha256,
    json_dumps,
    make_run_id,
    normalize_source_classification,
    now_iso,
    read_json,
    resolve_env_placeholders,
    safe_join,
    stable_hash,
    write_json,
    DEFAULT_TENANT_ID,
)
from .report import build_report, write_report
from .large_dataset_policy import caps_for as _large_caps_for, as_budget_dict as _large_caps_budget

_GOLDEN_MODE_ENV = "STAT_HARNESS_GOLDEN_MODE"
_GOLDEN_MODES = {"off", "default", "strict"}


def _debug_startup_stage(label: str) -> None:
    raw = os.environ.get("STAT_HARNESS_DEBUG_STARTUP", "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        print(f"PIPELINE_STAGE={label}", flush=True)


def _golden_mode() -> str:
    raw = os.environ.get(_GOLDEN_MODE_ENV, "").strip().lower()
    if raw in _GOLDEN_MODES:
        return raw
    if raw in {"1", "true", "yes", "on"}:
        return "strict"
    return "off"


class Pipeline:
    def __init__(
        self, base_dir: Path, plugins_dir: Path, tenant_id: str | None = None
    ) -> None:
        tenant_ctx = get_tenant_context(tenant_id, base_dir)
        self.base_dir = tenant_ctx.tenant_root
        self.appdata_root = tenant_ctx.appdata_root
        self.tenant_id = tenant_ctx.tenant_id
        self.plugins_dir = plugins_dir
        self.storage = Storage(tenant_ctx.db_path, tenant_ctx.tenant_id)
        self.manager = PluginManager(plugins_dir)
        # Cache keys must reflect stat-plugin handler code, not only tiny wrapper modules in plugins/*.
        # Otherwise updates under src/statistic_harness/core/stat_plugins/* can be incorrectly reused.
        self._stat_plugin_effective_hash_cache: dict[str, str] = {}
        _debug_startup_stage("startup_integrity_check_start")
        self._startup_integrity_check()
        _debug_startup_stage("startup_integrity_check_done")
        _debug_startup_stage("cleanup_upload_quarantine_start")
        self._cleanup_upload_quarantine()
        _debug_startup_stage("cleanup_upload_quarantine_done")
        _debug_startup_stage("apply_retention_policy_start")
        self._apply_retention_policy()
        _debug_startup_stage("apply_retention_policy_done")
        _debug_startup_stage("recover_incomplete_runs_start")
        self._recover_incomplete_runs()
        _debug_startup_stage("recover_incomplete_runs_done")

    def _augment_code_hash_if_stat_wrapper(
        self, plugin_id: str, module_file: Path, code_hash: str | None
    ) -> str | None:
        if not code_hash or not module_file.exists():
            return code_hash
        try:
            text = module_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return code_hash
        # Heuristic: stat wrappers import run_plugin and immediately dispatch to it.
        if "run_plugin(" not in text:
            return code_hash
        if not isinstance(plugin_id, str) or not plugin_id.strip():
            return code_hash
        cached = self._stat_plugin_effective_hash_cache.get(plugin_id)
        if cached is None:
            try:
                from statistic_harness.core.stat_plugins.code_hash import (
                    stat_plugin_effective_code_hash,
                )

                cached = stat_plugin_effective_code_hash(plugin_id)
            except Exception:
                cached = None
            if isinstance(cached, str) and cached:
                self._stat_plugin_effective_hash_cache[plugin_id] = cached
        if not isinstance(cached, str) or not cached:
            return code_hash
        return hashlib.sha256(f"{code_hash}:{cached}".encode("utf-8")).hexdigest()

    def _module_file_for_spec(self, spec: PluginSpec) -> Path:
        module_path, _ = spec.entrypoint.split(":", 1)
        if module_path.endswith(".py"):
            return spec.path / module_path
        return spec.path / f"{module_path}.py"

    def _spec_code_hash(self, spec: PluginSpec) -> str | None:
        module_file = self._module_file_for_spec(spec)
        code_hash = file_sha256(module_file) if module_file.exists() else None
        return self._augment_code_hash_if_stat_wrapper(
            spec.plugin_id, module_file, code_hash
        )

    def _startup_integrity_check(self) -> None:
        mode = os.environ.get("STAT_HARNESS_STARTUP_INTEGRITY", "").strip().lower()
        if not mode:
            mode = "quick"
        if mode in {"0", "off", "false", "no"}:
            return
        full = mode in {"1", "full", "integrity_check", "integrity"}
        ok, msg = self.storage.integrity_check(full=full)
        if not ok:
            raise RuntimeError(f"Integrity check failed: {msg}")

    def _cleanup_upload_quarantine(self) -> None:
        quarantine_root = self.appdata_root / "uploads" / "_quarantine"
        if not quarantine_root.exists():
            return
        for entry in quarantine_root.iterdir():
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)

    def _apply_retention_policy(self) -> None:
        enabled = os.environ.get("STAT_HARNESS_RETENTION_ENABLED", "").strip().lower()
        if enabled in {"0", "false", "no", "off"}:
            return
        raw_days = os.environ.get("STAT_HARNESS_RETENTION_DAYS", "").strip()
        days = 60
        if raw_days:
            try:
                days = max(1, int(raw_days))
            except ValueError:
                days = 60
        apply_retention(
            self.appdata_root,
            self.base_dir / "runs",
            days=days,
        )

    def _recover_incomplete_runs(self) -> None:
        runs_root = self.base_dir / "runs"
        staging_root = runs_root / "_staging"
        if staging_root.exists():
            # Best-effort cleanup; staging dirs are never considered valid runs.
            for entry in staging_root.iterdir():
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)

        def _pid_is_same_process(pid: int, started_at: str | None) -> bool:
            """Best-effort PID identity check to avoid false positives from PID reuse.

            On Linux/WSL, we approximate the process start wall-clock time using /proc.
            If we can compute it, we require it to be close to the run's journal started_at.
            """

            if pid <= 0:
                return False
            try:
                os.kill(pid, 0)
            except OSError:
                return False
            if not started_at:
                return True
            try:
                from datetime import datetime, timezone, timedelta

                # Parse the run start time.
                run_started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                if run_started.tzinfo is None:
                    run_started = run_started.replace(tzinfo=timezone.utc)
                # Compute process start wall-clock time (Linux/WSL only).
                stat_path = Path(f"/proc/{pid}/stat")
                uptime_path = Path("/proc/uptime")
                if not stat_path.exists() or not uptime_path.exists():
                    return True
                stat = stat_path.read_text(encoding="utf-8", errors="ignore")
                parts = stat.split()
                if len(parts) < 22:
                    return True
                start_ticks = int(parts[21])
                hz = int(os.sysconf(os.sysconf_names["SC_CLK_TCK"]))
                uptime_s = float(uptime_path.read_text(encoding="utf-8", errors="ignore").split()[0])
                age_s = max(0.0, uptime_s - (float(start_ticks) / float(hz)))
                proc_started = datetime.now(timezone.utc) - timedelta(seconds=age_s)
                # Tolerance: within 3 minutes of journal started_at.
                return abs((proc_started - run_started).total_seconds()) <= 180.0
            except Exception:
                return True

        # Mark any stale "running" runs as aborted when the runner PID is not alive (or was reused).
        for row in self.storage.list_runs_by_status("running", limit=1000):
            run_id = str(row.get("run_id") or "")
            if not run_id:
                continue
            run_dir = runs_root / run_id
            journal = run_dir / "journal.json"
            pid = None
            started_at = None
            if journal.exists():
                try:
                    payload = read_json(journal)
                    if isinstance(payload, dict):
                        pid = payload.get("pid")
                        started_at = payload.get("started_at") or payload.get("created_at")
                except Exception:
                    pid = None
            if isinstance(pid, int) and pid > 0:
                if _pid_is_same_process(pid, str(started_at) if started_at else None):
                    continue
            # If we can't prove the run is active, fail closed and preserve artifacts/logs.
            self.storage.update_run_status(
                run_id,
                "aborted",
                {"type": "CrashRecovery", "message": "Marked aborted on startup recovery"},
            )
            # Keep the DB consistent: any still-running plugin executions are now stale.
            self.storage.abort_plugin_executions_for_run(
                run_id,
                status="aborted",
                note="Aborted by crash recovery (runner pid not alive).",
            )

    @staticmethod
    def _parse_int_env(name: str) -> int | None:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    @staticmethod
    def _read_meminfo_kb() -> dict[str, int]:
        """Parse /proc/meminfo into a {key: kB} dict (Linux/WSL best-effort)."""

        info: dict[str, int] = {}
        try:
            path = Path("/proc/meminfo")
            if not path.exists():
                return info
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if ":" not in line:
                    continue
                key, rest = line.split(":", 1)
                key = key.strip()
                val = rest.strip().split()
                if not val:
                    continue
                try:
                    n = int(val[0])
                except ValueError:
                    continue
                # /proc/meminfo values are in kB.
                info[key] = n
        except Exception:
            return {}
        return info

    def _memory_governor_wait(self, *, plugin_id: str, plugin_type: str, logger: Any) -> None:
        """Soft governor: block starting new work when system memory is under pressure.

        This does not sample rows or reduce accuracy; it only throttles concurrency.
        """

        stages_raw = os.environ.get("STAT_HARNESS_MEM_GOVERNOR_STAGES", "").strip()
        stages = {"analysis"} if not stages_raw else {s.strip().lower() for s in stages_raw.split(",") if s.strip()}
        if plugin_type.strip().lower() not in stages:
            return

        max_used_pct_raw = os.environ.get("STAT_HARNESS_MEM_GOVERNOR_MAX_USED_PCT", "").strip()
        min_avail_mb_raw = os.environ.get("STAT_HARNESS_MEM_GOVERNOR_MIN_AVAILABLE_MB", "").strip()
        if not max_used_pct_raw and not min_avail_mb_raw:
            return

        try:
            poll_s = float(os.environ.get("STAT_HARNESS_MEM_GOVERNOR_POLL_SECONDS", "5").strip() or "5")
        except ValueError:
            poll_s = 5.0
        poll_s = max(0.5, poll_s)
        try:
            log_s = float(os.environ.get("STAT_HARNESS_MEM_GOVERNOR_LOG_SECONDS", "30").strip() or "30")
        except ValueError:
            log_s = 30.0
        log_s = max(5.0, log_s)

        max_used_pct: float | None
        min_avail_mb: int | None
        try:
            max_used_pct = float(max_used_pct_raw) if max_used_pct_raw else None
        except ValueError:
            max_used_pct = None
        try:
            min_avail_mb = int(min_avail_mb_raw) if min_avail_mb_raw else None
        except ValueError:
            min_avail_mb = None

        last_log = 0.0
        start = time.monotonic()
        while True:
            mem = self._read_meminfo_kb()
            total_kb = int(mem.get("MemTotal") or 0)
            avail_kb = int(mem.get("MemAvailable") or 0)
            if total_kb <= 0 or avail_kb <= 0:
                return
            used_pct = 100.0 * (1.0 - (float(avail_kb) / float(total_kb)))
            avail_mb = int(avail_kb // 1024)

            over_used = (max_used_pct is not None) and (used_pct > float(max_used_pct))
            under_avail = (min_avail_mb is not None) and (avail_mb < int(min_avail_mb))
            if not over_used and not under_avail:
                return

            now = time.monotonic()
            if (now - last_log) >= log_s:
                try:
                    logger(
                        "[GOV] memory pressure: "
                        f"plugin={plugin_id} used_pct={used_pct:.1f} avail_mb={avail_mb} "
                        f"max_used_pct={max_used_pct} min_avail_mb={min_avail_mb} "
                        f"waited_s={int(now - start)}"
                    )
                except Exception:
                    pass
                last_log = now
            time.sleep(poll_s)

    @staticmethod
    def _max_workers_for_stage(stage: str, layer_size: int, dataset_row_count: int | None) -> int:
        """Conservative parallelism cap to avoid OOM on large datasets.

        Override with:
        - STAT_HARNESS_MAX_WORKERS_ANALYSIS
        - STAT_HARNESS_MAX_WORKERS_TRANSFORM
        """

        layer_size = max(1, int(layer_size))
        row_count = int(dataset_row_count or 0)
        stage_key = stage.strip().lower()

        env_name = None
        if stage_key == "analysis":
            env_name = "STAT_HARNESS_MAX_WORKERS_ANALYSIS"
        elif stage_key == "transform":
            env_name = "STAT_HARNESS_MAX_WORKERS_TRANSFORM"
        if env_name:
            raw = os.environ.get(env_name, "").strip()
            if raw:
                try:
                    cap = int(raw)
                    cap = max(1, cap)
                    return min(layer_size, cap)
                except ValueError:
                    pass

        # Heuristic defaults: big datasets => low parallelism (each plugin may load a lot of data).
        #
        # User-facing default: allow 2-way parallelism even on >=1M rows; operators can still
        # force lower via STAT_HARNESS_MAX_WORKERS_ANALYSIS=1 if memory is tight.
        if row_count >= 1_000_000:
            cap = 2
        elif row_count >= 200_000:
            cap = 2
        elif row_count >= 100_000:
            cap = 4
        else:
            cap = os.cpu_count() or 1
        cap = max(1, int(cap))
        return min(layer_size, cap)

    def _toposort_layers(
        self, specs: list[PluginSpec], selected: set[str]
    ) -> list[list[PluginSpec]]:
        spec_map = {spec.plugin_id: spec for spec in specs}
        deps: dict[str, set[str]] = {}
        indegree: dict[str, int] = {}
        for pid in selected:
            spec = spec_map.get(pid)
            if not spec:
                continue
            dep_set = {dep for dep in spec.depends_on if dep in selected}
            deps[pid] = dep_set
            indegree[pid] = len(dep_set)

        layers: list[list[PluginSpec]] = []
        remaining = set(indegree.keys())
        while remaining:
            ready = sorted(pid for pid in remaining if indegree.get(pid, 0) == 0)
            if not ready:
                cycle = _find_cycle_path(deps, remaining) or []
                edges = _format_edges(deps, remaining)
                detail = ""
                if cycle:
                    detail = f" cycle={' -> '.join(cycle)}"
                raise ValueError(
                    f"Cycle detected in plugin dependencies.{detail} edges={edges}"
                )
            layer_specs = [spec_map[pid] for pid in ready]
            layers.append(layer_specs)
            for pid in ready:
                remaining.remove(pid)
            for pid in remaining:
                if deps[pid].intersection(ready):
                    indegree[pid] -= len(deps[pid].intersection(ready))
        return layers

    def _expand_selected_with_deps(
        self, specs: dict[str, PluginSpec], selected: set[str]
    ) -> tuple[set[str], list[str], list[str]]:
        """Return (expanded, added, missing_deps).

        - Unknown selected plugin ids are ignored here (handled separately elsewhere).
        - Missing dependencies of known plugins are returned so the pipeline can fail closed.
        """

        expanded = set(selected)
        added: list[str] = []
        missing: list[str] = []
        stack = list(sorted(selected))
        while stack:
            pid = stack.pop()
            spec = specs.get(pid)
            if not spec:
                continue
            for dep in spec.depends_on or []:
                if dep not in specs:
                    missing.append(f"{pid} -> {dep}")
                    continue
                if dep not in expanded:
                    expanded.add(dep)
                    added.append(dep)
                    stack.append(dep)
        return expanded, sorted(set(added)), sorted(set(missing))

    def run(
        self,
        input_file: Path | None,
        plugin_ids: list[str],
        settings: dict[str, Any],
        run_seed: int,
        upload_id: str = "local",
        run_id: str | None = None,
        dataset_version_id: str | None = None,
        project_id: str | None = None,
        dataset_id: str | None = None,
        reuse_cache: bool | None = None,
        force: bool | None = None,
    ) -> str:
        run_id = run_id or make_run_id()
        requested_run_seed = int(run_seed)
        tenant_id = self.tenant_id

        def _scope(value: str | None) -> str | None:
            if not value:
                return value
            if tenancy_enabled() and tenant_id != DEFAULT_TENANT_ID:
                return scope_identifier(tenant_id, value)
            return value

        dataset_version_id = _scope(dataset_version_id)
        project_id = _scope(project_id)
        dataset_id = _scope(dataset_id)

        runs_root = self.base_dir / "runs"
        staging_root = runs_root / "_staging"
        run_dir = runs_root / run_id
        def _prepare_staging(dir_path: Path) -> None:
            ensure_dir(dir_path / "dataset")
            ensure_dir(dir_path / "logs")
            write_json(
                dir_path / "journal.json",
                {
                    "run_id": run_id,
                    "status": "staging",
                    "pid": os.getpid(),
                    "created_at": now_iso(),
                },
            )

        atomic_dir(run_dir, staging_root=staging_root, prepare=_prepare_staging)
        write_json(
            run_dir / "journal.json",
            {"run_id": run_id, "status": "running", "pid": os.getpid(), "started_at": now_iso()},
        )
        progress_enabled = os.environ.get("STAT_HARNESS_CLI_PROGRESS", "").lower() in {
            "1",
            "true",
            "yes",
        }
        progress_tty = progress_enabled and sys.stdout.isatty()

        canonical_path = run_dir / "dataset" / "canonical.csv"
        dataset_row_count: int | None = None
        dataset_table_name: str | None = None
        dataset_column_count: int | None = None
        source_classification = normalize_source_classification(None)
        if input_file is None:
            if not dataset_version_id:
                raise ValueError("Dataset version is required for DB-only runs")
            ctx_row = self.storage.get_dataset_version_context(dataset_version_id)
            if not ctx_row:
                raise ValueError("Dataset version not found")
            project_id = ctx_row["project_id"]
            dataset_id = ctx_row["dataset_id"]
            dataset_table_name = str(ctx_row.get("table_name") or "")
            input_hash = ctx_row.get("data_hash") or dataset_version_id
            input_filename = f"db://{dataset_version_id}"
            source_classification = normalize_source_classification(
                str(ctx_row.get("source_classification") or "")
            )
            try:
                dataset_row_count = int(ctx_row.get("row_count") or 0)
            except (TypeError, ValueError):
                dataset_row_count = None
            try:
                dataset_column_count = int(ctx_row.get("column_count") or 0)
            except (TypeError, ValueError):
                dataset_column_count = None
        else:
            upload_row = self.storage.fetch_upload(upload_id) if upload_id else None
            upload_filename = (
                str(upload_row.get("filename") or input_file.name)
                if upload_row
                else input_file.name
            )
            source_classification = normalize_source_classification(
                str(upload_row.get("source_classification") or "") if upload_row else None,
                upload_filename,
            )
            input_hash = file_sha256(input_file)
            project_id = project_id or input_hash
            dataset_id = dataset_id or dataset_version_id or (
                input_hash
                if project_id == input_hash
                else dataset_key(project_id, input_hash)
            )
            dataset_version_id = dataset_version_id or dataset_id
            project_id = _scope(project_id)
            dataset_id = _scope(dataset_id)
            dataset_version_id = _scope(dataset_version_id)
            table_name = f"dataset_{dataset_version_id}"
            dataset_table_name = table_name
            existing = self.storage.get_dataset_version(dataset_version_id)
            if existing:
                existing_hash = existing.get("data_hash")
                same_media = False
                if existing_hash and existing_hash == input_hash:
                    same_media = True
                elif not existing_hash and existing.get("dataset_id") == dataset_id:
                    same_media = True
                if same_media:
                    self.storage.reset_dataset_version(
                        dataset_version_id,
                        table_name=existing.get("table_name") or table_name,
                        data_hash=input_hash,
                        created_at=now_iso(),
                    )
            self.storage.ensure_project(project_id, project_id, now_iso())
            self.storage.ensure_dataset(dataset_id, project_id, dataset_id, now_iso())
            self.storage.ensure_dataset_version(
                dataset_version_id,
                dataset_id,
                now_iso(),
                table_name,
                input_hash,
                source_classification=source_classification,
            )
            input_filename = input_file.name
            # Best-effort for worker cap heuristics; row_count is populated after ingest.
            try:
                latest_ctx = self.storage.get_dataset_version_context(dataset_version_id)
                if latest_ctx:
                    dataset_row_count = int(latest_ctx.get("row_count") or 0)
                    dataset_column_count = int(latest_ctx.get("column_count") or 0)
            except (TypeError, ValueError):
                dataset_row_count = None
                dataset_column_count = None

        if requested_run_seed == 0:
            # Deterministic auto seed for "run_seed=0" workflows.
            # Use only stable run inputs (content hash + resolved settings payload).
            run_seed = stable_hash(f"{input_hash}:{json_dumps(settings)}")
        else:
            run_seed = int(requested_run_seed)

        # Cache key salt: plugins operate on the normalized/template layer (not raw).
        # After normalization, we incorporate the mapping_hash so cache hits remain correct
        # across mapping/template fixes without forcing users to "force rerun everything".
        cache_dataset_hash = str(input_hash)

        _debug_startup_stage("run_create_run_start")
        self.storage.create_run(
            run_id=run_id,
            created_at=now_iso(),
            status="running",
            upload_id=upload_id,
            input_filename=input_filename,
            canonical_path=str(canonical_path),
            settings=settings,
            error=None,
            run_seed=run_seed,
            requested_run_seed=requested_run_seed,
            project_id=project_id,
            dataset_id=dataset_id,
            dataset_version_id=dataset_version_id,
            input_hash=input_hash,
        )
        _debug_startup_stage("run_create_run_done")
        self.storage.insert_event(
            kind="run_started",
            created_at=now_iso(),
            run_id=run_id,
            run_fingerprint=None,
            payload={
                "requested_run_seed": requested_run_seed,
                "run_seed": run_seed,
                "input_hash": input_hash,
                "upload_id": upload_id,
            },
        )
        if reuse_cache is None:
            # Default ON (opt-out via env), because cache keys are strict fingerprints
            # and significantly improve repeat-run throughput on large datasets.
            reuse_raw = os.environ.get("STAT_HARNESS_REUSE_CACHE", "").strip().lower()
            reuse_cache = reuse_raw not in {"0", "false", "no", "off"}
        else:
            reuse_cache = bool(reuse_cache)
        force = bool(
            (os.environ.get("STAT_HARNESS_FORCE", "").strip().lower() in {"1", "true", "yes"})
            if force is None
            else force
        )

        _debug_startup_stage("run_discover_start")
        specs_all = self.manager.discover()
        _debug_startup_stage("run_discover_done")
        disabled_plugins: set[str] = set()
        specs: list[PluginSpec] = []
        for spec in specs_all:
            if not self.storage.plugin_enabled(spec.plugin_id):
                disabled_plugins.add(spec.plugin_id)
                continue
            specs.append(spec)
        spec_map = {spec.plugin_id: spec for spec in specs}

        def record_missing(plugin_id: str, message: str) -> None:
            result = PluginResult(
                status="error",
                summary=message,
                metrics={},
                findings=[],
                artifacts=[],
                error=PluginError(
                    type="MissingPlugin",
                    message=message,
                    traceback="",
                ),
            )
            self.storage.save_plugin_result(
                run_id,
                plugin_id,
                None,
                now_iso(),
                None,
                None,
                input_hash,
                result,
            )
            self.storage.insert_plugin_execution(
                run_id=run_id,
                plugin_id=plugin_id,
                plugin_version=None,
                started_at=now_iso(),
                completed_at=now_iso(),
                duration_ms=0,
                status="error",
                exit_code=None,
                cpu_user=None,
                cpu_system=None,
                max_rss=None,
                warnings_count=None,
                stdout=None,
                stderr=None,
            )
        for err in self.manager.discovery_errors:
            plugin_id = err.plugin_id
            if plugin_id in spec_map:
                plugin_id = f"{plugin_id}__discovery_error"
            record_missing(
                plugin_id,
                f"Plugin discovery error in {err.path}: {err.message}",
            )

        selected = set(plugin_ids)
        auto_plan = not selected or "auto" in selected
        selected.discard("auto")
        disabled_selected = sorted(pid for pid in selected if pid in disabled_plugins)
        for pid in disabled_selected:
            record_missing(pid, "Plugin disabled")
            selected.discard(pid)
        if "all" in selected:
            selected = {
                spec.plugin_id
                for spec in specs
                if spec.type in {"analysis", "profile", "transform", "report", "llm", "ingest"}
            }
            auto_plan = False
        dataset_accessor, dataset_template = resolve_dataset_accessor(
            self.storage, dataset_version_id
        )
        # Backfill performance-critical index for already-ingested datasets.
        # This is safe/idempotent, and dramatically reduces per-plugin dataset load time.
        if dataset_table_name:
            try:
                self.storage.ensure_dataset_row_index_index(dataset_table_name)
            except Exception:
                # Fail closed for analysis correctness, but don't crash the pipeline due to an index backfill.
                pass

        column_lookup: dict[str, int] | None = None

        def get_column_lookup() -> dict[str, int]:
            nonlocal column_lookup
            if column_lookup is None and dataset_version_id:
                if dataset_template:
                    fields = self.storage.fetch_template_fields(
                        int(dataset_template["template_id"])
                    )
                    column_lookup = {field["name"]: int(field["field_id"]) for field in fields}
                else:
                    columns = self.storage.fetch_dataset_columns(dataset_version_id)
                    column_lookup = {
                        col["original_name"]: int(col["column_id"]) for col in columns
                    }
            return column_lookup or {}

        def infer_column_ids(item: dict[str, Any]) -> list[int]:
            lookup = get_column_lookup()
            ids: list[int] = []
            if "feature" in item and isinstance(item["feature"], str):
                if item["feature"] in lookup:
                    ids.append(lookup[item["feature"]])
            if "pair" in item and isinstance(item["pair"], (list, tuple)):
                for name in item["pair"]:
                    if isinstance(name, str) and name in lookup:
                        ids.append(lookup[name])
            if "columns" in item and isinstance(item["columns"], (list, tuple)):
                for name in item["columns"]:
                    if isinstance(name, str) and name in lookup:
                        ids.append(lookup[name])
            return sorted(set(ids))

        def attach_evidence(findings: list[Any]) -> list[dict[str, Any]]:
            enriched: list[dict[str, Any]] = []
            allowed_measurements = {"measured", "modeled", "not_applicable", "error"}
            for item in findings:
                if isinstance(item, dict):
                    entry = dict(item)
                else:
                    entry = {"value": item}
                evidence = dict(entry.get("evidence") or {})
                if isinstance(entry.get("dataset_id"), str):
                    evidence.setdefault("dataset_id", entry["dataset_id"])
                evidence.setdefault("dataset_id", dataset_id or "unknown")
                if isinstance(entry.get("dataset_version_id"), str):
                    evidence.setdefault("dataset_version_id", entry["dataset_version_id"])
                evidence.setdefault("dataset_version_id", dataset_version_id or "unknown")
                row_ids: list[int] = []
                if "row_ids" in entry and isinstance(entry["row_ids"], list):
                    try:
                        row_ids = [int(v) for v in entry["row_ids"]]
                    except (TypeError, ValueError):
                        row_ids = []
                if "row_index" in entry:
                    try:
                        row_ids.append(int(entry["row_index"]))
                    except (TypeError, ValueError):
                        pass
                if row_ids:
                    row_ids = sorted(set(row_ids))
                evidence.setdefault("row_ids", row_ids)
                evidence.setdefault("column_ids", infer_column_ids(entry))
                evidence.setdefault("query", entry.get("query"))
                if "start" in entry and "end" in entry:
                    try:
                        evidence.setdefault(
                            "row_ranges",
                            [
                                {
                                    "start": int(entry["start"]),
                                    "end": int(entry["end"]),
                                }
                            ],
                        )
                    except (TypeError, ValueError):
                        pass
                if "measurement_type" not in entry:
                    entry["measurement_type"] = "measured"
                measurement = entry.get("measurement_type")
                if isinstance(measurement, str):
                    mnorm = measurement.strip().lower()
                    if mnorm == "degraded":
                        # Back-compat: some plugins emit "degraded" to signal an optional
                        # dependency/fallback. Normalize to a schema-valid state.
                        entry["measurement_type"] = "not_applicable"
                    elif mnorm not in allowed_measurements:
                        entry["measurement_type"] = "error"
                else:
                    entry["measurement_type"] = "error"
                entry["evidence"] = evidence
                enriched.append(entry)
            return enriched

        def validate_modeled_findings(findings: list[dict[str, Any]]) -> list[str]:
            errors: list[str] = []
            for item in findings:
                if item.get("measurement_type") != "modeled":
                    continue
                scope = item.get("scope")
                assumptions = item.get("assumptions")
                if not isinstance(scope, dict) or not scope:
                    errors.append("modeled finding missing scope")
                if (
                    not isinstance(assumptions, list)
                    or not assumptions
                    or not all(isinstance(a, str) and a.strip() for a in assumptions)
                ):
                    errors.append("modeled finding missing assumptions")
            return errors

        def _not_applicable_finding(
            plugin_id: str,
            reason: str,
            *,
            original_status: str,
            error_type: str | None = None,
        ) -> dict[str, Any]:
            digest = hashlib.sha256(
                f"{plugin_id}:{original_status}:{reason}".encode("utf-8")
            ).hexdigest()[:16]
            return {
                "id": digest,
                "kind": "plugin_not_applicable",
                "severity": "info",
                "confidence": 1.0,
                "title": f"{plugin_id} not applicable",
                "what": reason,
                "why": "Plugin preconditions were not satisfied for this dataset/config; deterministic fallback emitted.",
                "measurement_type": "not_applicable",
                "scope": {"plugin_id": plugin_id},
                "assumptions": [],
                "action_type": "not_applicable",
                "target": None,
                "evidence": {
                    "metrics": {
                        "original_status": original_status,
                        "error_type": error_type or "",
                    }
                },
            }

        def normalize_result_status(spec: PluginSpec, result: PluginResult) -> PluginResult:
            status = str(result.status or "").strip().lower()
            error_type = (
                str(result.error.type).strip() if getattr(result, "error", None) and getattr(result.error, "type", None) else ""
            )
            summary_text = str(result.summary or "").strip()
            debug = dict(result.debug or {})
            findings = list(result.findings or [])
            metrics = dict(result.metrics or {})

            should_soften = status in {"skipped", "degraded"}
            if status == "error":
                summary_lower = summary_text.lower()
                is_validation_error = "output validation failed" in summary_lower
                allowed_validation_fallback = (
                    "modeled finding missing assumptions" in summary_lower
                    or "modeled finding missing scope" in summary_lower
                )
                error_type_lower = error_type.lower()
                soft_error_type = error_type_lower in {
                    "timeouterror",
                    "valueerror",
                    "runtimeerror",
                    "floatingpointerror",
                    "linalgerror",
                }
                soft_error_summary = (
                    summary_lower.endswith(" failed")
                    or summary_lower.endswith(" timed out")
                    or " timed out" in summary_lower
                )
                should_soften = (
                    (soft_error_type or soft_error_summary)
                    and (not is_validation_error or allowed_validation_fallback)
                )

            if not should_soften:
                return result

            reason = summary_text or f"{spec.plugin_id} not applicable"
            debug.setdefault("status_original", status or "unknown")
            if error_type:
                debug.setdefault("error_type_original", error_type)
            debug.setdefault("fallback_mode", "not_applicable")
            metrics.setdefault("fallback_not_applicable", 1)
            if not findings:
                findings = [
                    _not_applicable_finding(
                        spec.plugin_id,
                        reason,
                        original_status=status or "unknown",
                        error_type=error_type or None,
                    )
                ]
            result.status = "ok"
            result.summary = f"{spec.plugin_id} not applicable: {reason}"
            result.error = None
            result.debug = debug
            result.metrics = metrics
            result.findings = findings
            return result

        def logger(msg: str) -> None:
            log_path = run_dir / "logs" / "run.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(msg + "\n")

        def run_spec(
            spec: PluginSpec, include_input: bool = False, progress: bool = True
        ) -> PluginResult:
            # Soft governor before marking a plugin "running" in the DB, to avoid runs that
            # appear active without OS work due to throttling.
            self._memory_governor_wait(
                plugin_id=spec.plugin_id,
                plugin_type=str(spec.type or ""),
                logger=logger,
            )
            plugin_settings = dict(spec.settings.get("defaults", {}))
            plugin_settings.update(settings.get(spec.plugin_id, {}))
            if include_input and input_file is not None:
                plugin_settings["input_file"] = str(input_file)
            execution_id = self.storage.start_plugin_execution(
                run_id,
                spec.plugin_id,
                spec.version,
                now_iso(),
                status="running",
            )
            code_hash = self._spec_code_hash(spec)
            plugin_seed = stable_hash(f"{run_seed}:{spec.plugin_id}")
            settings_hash: str | None = None
            execution_fingerprint: str | None = None
            heartbeat_stop = threading.Event()
            heartbeat_thread: threading.Thread | None = None
            start_wall = time.perf_counter()
            if progress and progress_enabled:
                if progress_tty:
                    sys.stdout.write(f"[RUN] {spec.plugin_id}\n")
                    sys.stdout.flush()
                else:
                    print(f"[RUN] {spec.plugin_id}")

                def heartbeat() -> None:
                    spinner = "|/-\\"
                    idx = 0
                    while not heartbeat_stop.wait(5.0):
                        elapsed = int(time.perf_counter() - start_wall)
                        if progress_tty:
                            sys.stdout.write(
                                f"\r[RUN] {spec.plugin_id} {spinner[idx % 4]} {elapsed}s"
                            )
                            sys.stdout.flush()
                        else:
                            print(f"[RUN] {spec.plugin_id} {elapsed}s")
                        idx += 1
                    if progress_tty:
                        sys.stdout.write("\r")
                        sys.stdout.flush()

                heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
                heartbeat_thread.start()
            try:
                # Apply JSONSchema defaults deterministically before validation and execution.
                plugin_settings = self.manager.resolve_config(spec, plugin_settings)
                # Resolve secret/environment indirection at execution-time only.
                plugin_settings_exec = resolve_env_placeholders(plugin_settings)
                budget = plugin_settings.get("budget")
                if not isinstance(budget, dict):
                    budget = {}
                # Large dataset policy: inject complexity caps (not row sampling).
                policy_enabled = os.environ.get("STAT_HARNESS_LARGE_DATASET_POLICY", "").strip().lower()
                if policy_enabled not in {"0", "false", "no", "off"}:
                    try:
                        caps = _large_caps_for(
                            plugin_id=spec.plugin_id,
                            plugin_type=spec.type,
                            row_count=dataset_row_count,
                            column_count=dataset_column_count,
                        )
                        if caps is not None:
                            caps_budget = _large_caps_budget(caps)
                            # Only fill missing keys; explicit plugin config overrides policy.
                            for k, v in caps_budget.items():
                                budget.setdefault(k, v)
                    except Exception:
                        pass
                budget = {
                    "row_limit": budget.get("row_limit"),
                    "sampled": bool(budget.get("sampled", False)),
                    "time_limit_ms": budget.get("time_limit_ms"),
                    "cpu_limit_ms": budget.get("cpu_limit_ms"),
                    "batch_size": budget.get("batch_size"),
                    "max_cols": budget.get("max_cols"),
                    "max_pairs": budget.get("max_pairs"),
                    "max_groups": budget.get("max_groups"),
                    "max_windows": budget.get("max_windows"),
                    "max_findings": budget.get("max_findings"),
                }
                default_timeout_ms = self._parse_int_env(
                    "STAT_HARNESS_DEFAULT_PLUGIN_TIMEOUT_MS"
                )
                if (
                    spec.type == "analysis"
                    and budget.get("time_limit_ms") is None
                    and default_timeout_ms is not None
                    and default_timeout_ms > 0
                ):
                    budget["time_limit_ms"] = int(default_timeout_ms)
                # Optional: hard memory limit for plugin subprocess via RLIMIT_AS.
                hard_mem_mb = self._parse_int_env("STAT_HARNESS_PLUGIN_RLIMIT_AS_MB")
                if hard_mem_mb is None:
                    hard_mem_mb = self._parse_int_env("STAT_HARNESS_DEFAULT_PLUGIN_RLIMIT_AS_MB")
                if hard_mem_mb is None and spec.type == "analysis":
                    hard_mem_mb = 4096
                if hard_mem_mb is not None and hard_mem_mb > 0:
                    budget["mem_limit_mb"] = int(hard_mem_mb)
                settings_for_hash = dict(plugin_settings)
                if "input_file" in settings_for_hash:
                    # File paths are machine-dependent; treat input identity as input_hash instead.
                    settings_for_hash["input_file"] = None
                settings_hash = hashlib.sha256(
                    json_dumps(settings_for_hash).encode("utf-8")
                ).hexdigest()
                execution_fingerprint = hashlib.sha256(
                    json_dumps(
                        {
                            "plugin_id": spec.plugin_id,
                            "plugin_version": spec.version,
                            "code_hash": code_hash,
                            "settings_hash": settings_hash,
                            "dataset_hash": cache_dataset_hash,
                        }
                    ).encode("utf-8")
                ).hexdigest()
                self.storage.insert_event(
                    kind="plugin_started",
                    created_at=now_iso(),
                    run_id=run_id,
                    plugin_id=spec.plugin_id,
                    run_fingerprint=None,
                    payload={
                        "execution_id": execution_id,
                        "execution_fingerprint": execution_fingerprint,
                        "plugin_seed": plugin_seed,
                    },
                )
                # Cache reuse policy:
                # - Ingest: never reused (materializes raw SQLite content).
                # - Report: never reused (run-scoped outputs must match this run_id).
                # - Transforms: only normalization may be reused, and only when a ready
                #   dataset_template already exists (side effects are already materialized).
                # - Analysis/profile/planner/llm: safe to reuse via strict fingerprint.
                allow_cache_reuse = (
                    bool(reuse_cache)
                    and not force
                    and str(spec.type or "") not in {"ingest", "report"}
                )
                if str(spec.type or "") == "transform":
                    if spec.plugin_id != "transform_normalize_mixed":
                        allow_cache_reuse = False
                    else:
                        ready_template = self.storage.fetch_dataset_template(
                            dataset_version_id
                        )
                        allow_cache_reuse = bool(
                            ready_template
                            and str(ready_template.get("status") or "").lower()
                            == "ready"
                        )

                if allow_cache_reuse:
                    cached = self.storage.fetch_cached_plugin_result(execution_fingerprint)
                else:
                    cached = None
                if cached:
                    # Reuse cached ok result deterministically, copying artifacts into this run.
                    old_run_id = str(cached.get("run_id") or "")
                    artifacts_payload = cached.get("artifacts") or []
                    if old_run_id:
                        old_run_dir = self.base_dir / "runs" / old_run_id
                        # Copy standard plugin artifacts directory (most plugins write here).
                        src_dir = old_run_dir / "artifacts" / spec.plugin_id
                        dst_dir = run_dir / "artifacts" / spec.plugin_id
                        if src_dir.exists() and not dst_dir.exists():
                            shutil.copytree(src_dir, dst_dir)
                        # Also copy any explicitly declared artifacts that live outside artifacts/<plugin_id>
                        # (e.g. report plugins writing into run root or slide_kit/).
                        for item in artifacts_payload:
                            if not isinstance(item, dict):
                                continue
                            rel = str(item.get("path") or "")
                            if not rel:
                                continue
                            try:
                                src = safe_join(old_run_dir, rel)
                                dst = safe_join(run_dir, rel)
                            except Exception:
                                continue
                            if not src.exists() or dst.exists():
                                continue
                            ensure_dir(dst.parent)
                            if src.is_dir():
                                shutil.copytree(src, dst)
                            else:
                                shutil.copy2(src, dst)
                    artifacts = [
                        PluginArtifact(
                            path=str(item.get("path") or ""),
                            type=str(item.get("type") or ""),
                            description=str(item.get("description") or ""),
                        )
                        for item in artifacts_payload
                        if isinstance(item, dict)
                    ]
                    result = PluginResult(
                        status="ok",
                        summary=f"REUSED from {cached.get('run_id')}",
                        metrics=cached.get("metrics") or {},
                        findings=cached.get("findings") or [],
                        artifacts=artifacts,
                        error=None,
                        references=cached.get("references") or [],
                        debug={"reused_from": cached.get("run_id")},
                        budget=cached.get("budget")
                        or {
                            "row_limit": None,
                            "sampled": False,
                            "time_limit_ms": None,
                            "cpu_limit_ms": None,
                        },
                    )
                    exec_info = {
                        "completed_at": now_iso(),
                        "duration_ms": 0,
                        "cpu_user": None,
                        "cpu_system": None,
                        "max_rss": None,
                        "warnings_count": None,
                    }
                    self.storage.update_plugin_execution(
                        execution_id=execution_id,
                        completed_at=exec_info.get("completed_at"),
                        duration_ms=0,
                        status="ok",
                        exit_code=0,
                        cpu_user=None,
                        cpu_system=None,
                        max_rss=None,
                        warnings_count=None,
                        stdout="",
                        stderr="",
                    )
                    self.storage.save_plugin_result(
                        run_id,
                        spec.plugin_id,
                        spec.version,
                        now_iso(),
                        code_hash,
                        settings_hash,
                        cache_dataset_hash,
                        result,
                        execution_fingerprint=execution_fingerprint,
                    )
                    self.storage.insert_event(
                        kind="plugin_reused",
                        created_at=now_iso(),
                        run_id=run_id,
                        plugin_id=spec.plugin_id,
                        run_fingerprint=None,
                        payload={"reused_from": cached.get("run_id")},
                    )
                    return result
                # File sandbox policy:
                # - Read: plugin code + src + run dir + (optional) input file + any declared allowlist tokens.
                # - Write: run dir only (artifacts/logs/tmp).
                read_allow_paths: list[str] = []
                for token in spec.sandbox.get("fs_allowlist", []):
                    if token == "appdata":
                        read_allow_paths.append(str(self.base_dir))
                    elif token == "plugins":
                        read_allow_paths.append(str(self.plugins_dir))
                    elif token == "models":
                        # Local model store (operator-managed). Plugins must still be no-network and
                        # the sandbox will prevent writes unless explicitly allowlisted.
                        read_allow_paths.append("/mnt/d/autocapture/models")
                    elif token == "run_dir":
                        read_allow_paths.append(str(run_dir))
                    else:
                        read_allow_paths.append(
                            str((self.plugins_dir.parent / token).resolve())
                        )
                read_allow_paths.append(str(self.plugins_dir))
                read_allow_paths.append(str(self.plugins_dir.parent / "src"))
                read_allow_paths.append(str(run_dir))
                if include_input and input_file is not None:
                    read_allow_paths.append(str(input_file))
                write_allow_paths = [str(run_dir)]
                request = {
                    "plugin_id": spec.plugin_id,
                    "plugin_type": spec.type,
                    "entrypoint": spec.entrypoint,
                    "settings": plugin_settings_exec,
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "run_seed": run_seed,
                    "plugin_seed": plugin_seed,
                    "dataset_version_id": dataset_version_id,
                    "project_id": project_id,
                    "dataset_id": dataset_id,
                    "input_hash": input_hash,
                    "budget": budget,
                    "tenant_id": self.tenant_id,
                    "appdata_dir": str(self.appdata_root),
                    "root_dir": str(self.plugins_dir.parent.resolve()),
                    "sandbox": spec.sandbox,
                    "read_allow_paths": read_allow_paths,
                    "write_allow_paths": write_allow_paths,
                    # Back-compat for older runner versions (and for tests calling directly).
                    "allow_paths": read_allow_paths,
                }
                runner = run_plugin_subprocess(
                    spec, request, run_dir, self.plugins_dir.parent
                )
                result = runner.result
            except Exception as exc:  # pragma: no cover - error flow
                tb = traceback.format_exc()
                result = PluginResult(
                    status="error",
                    summary=f"{spec.plugin_id} failed",
                    metrics={},
                    findings=[],
                    artifacts=[],
                    error=PluginError(
                        type=type(exc).__name__, message=str(exc), traceback=tb
                    ),
                )
            finally:
                heartbeat_stop.set()
                if heartbeat_thread:
                    heartbeat_thread.join(timeout=1.0)
            result = normalize_result_status(spec, result)
            result.findings = attach_evidence(result.findings)
            try:
                payload = self.manager.result_payload(result)
                self.manager.validate_output(spec, payload)
                modeled_errors = validate_modeled_findings(result.findings)
                if modeled_errors:
                    raise ValueError("; ".join(sorted(set(modeled_errors))))
            except Exception as exc:  # pragma: no cover - error flow
                tb = traceback.format_exc()
                result = PluginResult(
                    status="error",
                    summary=f"{spec.plugin_id} output validation failed: {exc}",
                    metrics={},
                    findings=[],
                    artifacts=[],
                    error=PluginError(
                        type=type(exc).__name__, message=str(exc), traceback=tb
                    ),
                )
                # Keep execution status aligned with normalized fallback behavior.
                result = normalize_result_status(spec, result)
                result.findings = attach_evidence(result.findings)
            if result.error:
                logger(f"[ERROR] {spec.plugin_id}: {result.error.message}")
            if "runner" in locals():
                exec_info = runner.execution
                self.storage.update_plugin_execution(
                    execution_id=execution_id,
                    completed_at=exec_info.get("completed_at"),
                    duration_ms=exec_info.get("duration_ms"),
                    status=result.status,
                    exit_code=runner.exit_code,
                    cpu_user=exec_info.get("cpu_user"),
                    cpu_system=exec_info.get("cpu_system"),
                    max_rss=exec_info.get("max_rss"),
                    warnings_count=exec_info.get("warnings_count"),
                    stdout=runner.stdout,
                    stderr=runner.stderr,
                )
            else:
                self.storage.update_plugin_execution(
                    execution_id=execution_id,
                    completed_at=now_iso(),
                    duration_ms=0,
                    status=result.status,
                    exit_code=None,
                    cpu_user=None,
                    cpu_system=None,
                    max_rss=None,
                    warnings_count=None,
                    stdout=None,
                    stderr=None,
                )
            if progress and progress_enabled:
                duration_ms = 0
                if "runner" in locals():
                    duration_ms = int(exec_info.get("duration_ms") or 0)
                else:
                    duration_ms = int((time.perf_counter() - start_wall) * 1000)
                duration_s = duration_ms / 1000.0
                if result.status == "ok":
                    status_tag = "OK"
                elif result.status == "skipped":
                    status_tag = "SKIP"
                elif result.status == "degraded":
                    status_tag = "DEG"
                else:
                    status_tag = "ERR"
                print(f"[{status_tag}] {spec.plugin_id} ({duration_s:.1f}s)")
            module_path, _ = spec.entrypoint.split(":", 1)
            if module_path.endswith(".py"):
                module_file = spec.path / module_path
            else:
                module_file = spec.path / f"{module_path}.py"
            self.storage.save_plugin_result(
                run_id,
                spec.plugin_id,
                spec.version,
                now_iso(),
                code_hash,
                settings_hash,
                input_hash,
                result,
                execution_fingerprint=execution_fingerprint,
            )
            is_failure = str(result.status or "").lower() in {"error", "aborted"}
            if not is_failure:
                self.storage.insert_event(
                    kind="plugin_completed",
                    created_at=now_iso(),
                    run_id=run_id,
                    plugin_id=spec.plugin_id,
                    run_fingerprint=None,
                    payload={
                        "status": result.status,
                        "duration_ms": int(exec_info.get("duration_ms") or 0)
                        if "runner" in locals()
                        else None,
                    },
                )
            else:
                # Avoid leaving partial artifacts behind when output was invalid / failed.
                shutil.rmtree(run_dir / "artifacts" / spec.plugin_id, ignore_errors=True)
                self.storage.insert_event(
                    kind="plugin_failed",
                    created_at=now_iso(),
                    run_id=run_id,
                    plugin_id=spec.plugin_id,
                    run_fingerprint=None,
                    payload={"status": result.status, "summary": result.summary},
                )
            return result

        if input_file is not None:
            ingest_spec = spec_map.get("ingest_tabular")
            if ingest_spec:
                run_spec(ingest_spec, include_input=True)
            else:
                record_missing("ingest_tabular", "Missing ingest plugin")

        # Normalization is required before any other plugin reads the dataset.
        # This ensures all plugins operate on the normalized (template) layer, not raw.
        pre_ran_transforms: set[str] = set()
        normalize_spec = spec_map.get("transform_normalize_mixed")
        normalize_code_hash = ""
        normalize_settings_hash = ""
        if normalize_spec:
            try:
                normalize_code_hash = str(self._spec_code_hash(normalize_spec) or "")
            except Exception:
                normalize_code_hash = ""
            try:
                norm_settings = dict(normalize_spec.settings.get("defaults", {}))
                norm_settings.update(settings.get(normalize_spec.plugin_id, {}))
                norm_settings = self.manager.resolve_config(normalize_spec, norm_settings)
                norm_for_hash = dict(norm_settings)
                if "input_file" in norm_for_hash:
                    norm_for_hash["input_file"] = None
                normalize_settings_hash = hashlib.sha256(
                    json_dumps(norm_for_hash).encode("utf-8")
                ).hexdigest()
            except Exception:
                normalize_settings_hash = ""
        if normalize_spec:
            norm_result = run_spec(normalize_spec)
            pre_ran_transforms.add("transform_normalize_mixed")
            if norm_result.status != "ok":
                self.storage.update_run_status(
                    run_id,
                    "error",
                    {
                        "type": "NormalizationFailed",
                        "plugin_id": "transform_normalize_mixed",
                        "status": norm_result.status,
                        "summary": norm_result.summary,
                    },
                )
                report = build_report(
                    self.storage, run_id, run_dir, Path("docs/report.schema.json")
                )
                write_report(report, run_dir)
                return run_id
        else:
            record_missing("transform_normalize_mixed", "Missing normalization transform plugin")
            self.storage.update_run_status(
                run_id,
                "error",
                {"type": "NormalizationMissing", "plugin_id": "transform_normalize_mixed"},
            )
            report = build_report(
                self.storage, run_id, run_dir, Path("docs/report.schema.json")
            )
            write_report(report, run_dir)
            return run_id

        # Optional secondary transforms. If not configured, they should skip cleanly.
        template_spec = spec_map.get("transform_template")
        if template_spec:
            run_spec(template_spec)
            pre_ran_transforms.add("transform_template")

        # Enforce that the normalized/template layer is now active for dataset access.
        dataset_accessor, dataset_template = resolve_dataset_accessor(
            self.storage, dataset_version_id
        )
        if not dataset_template or str(dataset_template.get("status") or "") != "ready":
            self.storage.update_run_status(
                run_id,
                "error",
                {"type": "NormalizationNotReady", "dataset_version_id": dataset_version_id},
            )
            report = build_report(
                self.storage, run_id, run_dir, Path("docs/report.schema.json")
            )
            write_report(report, run_dir)
            return run_id

        # Once normalization is ready, salt the dataset hash with the mapping hash and
        # normalization implementation fingerprint so downstream cache keys reflect:
        # - raw input identity
        # - normalization mapping/template identity
        # - normalization code/settings identity
        try:
            mapping_hash = str(dataset_template.get("mapping_hash") or "").strip()
            template_id = str(dataset_template.get("template_id") or "").strip()
            if mapping_hash:
                salt_parts = [input_hash, mapping_hash, template_id]
                if normalize_code_hash:
                    salt_parts.append(normalize_code_hash)
                if normalize_settings_hash:
                    salt_parts.append(normalize_settings_hash)
                cache_dataset_hash = hashlib.sha256(
                    ":".join(salt_parts).encode("utf-8")
                ).hexdigest()
        except Exception:
            cache_dataset_hash = str(input_hash)

        # Critical stop: ensure the normalization/template mapping covers every raw dataset column.
        # This fails closed for ambiguous duplicate headers (must be resolved by safe_name).
        try:
            cols = self.storage.fetch_dataset_columns(dataset_version_id)
            expected_safes = [str(c.get("safe_name") or "") for c in cols if c.get("safe_name")]
            expected_set = set(expected_safes)
            mapping_payload = {}
            raw_mapping = {}
            try:
                import json

                mapping_payload = json.loads(str(dataset_template.get("mapping_json") or "{}"))
            except Exception:
                mapping_payload = {}
            if isinstance(mapping_payload, dict):
                raw_mapping = mapping_payload.get("mapping") if isinstance(mapping_payload.get("mapping"), dict) else {}
            if not isinstance(raw_mapping, dict):
                raw_mapping = {}

            # Build observed safe_names from mapping_json.
            observed: list[str] = []
            # Back-compat: mapping may be {field: original_name}; resolve via dataset_columns.
            orig_to_safes: dict[str, list[str]] = {}
            for c in cols:
                orig = str(c.get("original_name") or "")
                safe = str(c.get("safe_name") or "")
                if orig and safe:
                    orig_to_safes.setdefault(orig, []).append(safe)
            for _, src in raw_mapping.items():
                if isinstance(src, dict):
                    safe = str(src.get("safe_name") or "")
                    if safe:
                        observed.append(safe)
                elif isinstance(src, str):
                    safes = orig_to_safes.get(src) or []
                    if len(safes) == 1:
                        observed.append(safes[0])
                    else:
                        # Ambiguous or unknown original name; treat as missing mapping.
                        pass

            observed_set = set(observed)
            missing = sorted(expected_set - observed_set)
            extras = sorted(observed_set - expected_set)
            if missing or extras:
                self.storage.update_run_status(
                    run_id,
                    "error",
                    {
                        "type": "NormalizationMappingIncomplete",
                        "dataset_version_id": dataset_version_id,
                        "missing_safe_names": missing,
                        "extra_safe_names": extras,
                    },
                )
                report = build_report(
                    self.storage, run_id, run_dir, Path("docs/report.schema.json")
                )
                write_report(report, run_dir)
                return run_id
        except Exception as exc:
            self.storage.update_run_status(
                run_id,
                "error",
                {
                    "type": "NormalizationMappingValidationError",
                    "dataset_version_id": dataset_version_id,
                    "message": str(exc),
                },
            )
            report = build_report(
                self.storage, run_id, run_dir, Path("docs/report.schema.json")
            )
            write_report(report, run_dir)
            return run_id

        if auto_plan:
            profile_specs = [spec for spec in specs if spec.type == "profile"]
            if profile_specs:
                for spec in sorted(profile_specs, key=lambda item: item.plugin_id):
                    run_spec(spec)
            else:
                record_missing("profile_basic", "Missing profile plugin")
            planner_spec = spec_map.get("planner_basic")
            if planner_spec:
                plan_result = run_spec(planner_spec)
                planned = plan_result.metrics.get("selected_plugins") or []
                selected.update(planned)
            else:
                record_missing("planner_basic", "Missing planner plugin")

        # Always include normalization-layer plugins in the run fingerprint / expected list,
        # since they are required and executed for every run.
        if "transform_normalize_mixed" in spec_map:
            selected.add("transform_normalize_mixed")
        if "transform_template" in spec_map:
            selected.add("transform_template")

        expanded, added_deps, missing_deps = self._expand_selected_with_deps(
            spec_map, selected
        )
        if missing_deps:
            for edge in missing_deps:
                record_missing("pipeline_preflight", f"Missing dependency: {edge}")
            self.storage.update_run_status(
                run_id,
                "error",
                {"type": "MissingDependency", "missing": missing_deps},
            )
            report = build_report(
                self.storage, run_id, run_dir, Path("docs/report.schema.json")
            )
            write_report(report, run_dir)
            return run_id
        selected = expanded

        missing_manual = sorted(pid for pid in selected if pid not in spec_map)
        for pid in missing_manual:
            record_missing(pid, f"Unknown plugin id: {pid}")

        executed_ids = sorted(pid for pid in selected if pid in spec_map)
        fingerprint_payload: dict[str, Any] = {
            "input_hash": input_hash,
            "requested_run_seed": requested_run_seed,
            "run_seed": run_seed,
            "plugins": [],
            "settings": settings,
            "features": {
                "network_allowed": os.environ.get("STAT_HARNESS_ALLOW_NETWORK", "").lower()
                in {"1", "true", "yes"},
                "vector_store_enabled": os.environ.get("STAT_HARNESS_ENABLE_VECTOR_STORE", "").lower()
                in {"1", "true", "yes", "on"},
            },
        }
        for pid in executed_ids:
            spec = spec_map[pid]
            module_path, _ = spec.entrypoint.split(":", 1)
            if module_path.endswith(".py"):
                module_file = spec.path / module_path
            else:
                module_file = spec.path / f"{module_path}.py"
            code_hash = file_sha256(module_file) if module_file.exists() else None
            default_settings = dict(spec.settings.get("defaults", {}))
            default_settings.update(settings.get(pid, {}))
            try:
                resolved = self.manager.resolve_config(spec, default_settings)
            except Exception:
                resolved = default_settings
            if "input_file" in resolved:
                resolved["input_file"] = None
            settings_hash = hashlib.sha256(json_dumps(resolved).encode("utf-8")).hexdigest()
            fingerprint_payload["plugins"].append(
                {
                    "plugin_id": pid,
                    "plugin_version": spec.version,
                    "code_hash": code_hash,
                    "settings_hash": settings_hash,
                }
            )
        run_fingerprint = hashlib.sha256(
            json_dumps(fingerprint_payload).encode("utf-8")
        ).hexdigest()
        self.storage.update_run_fingerprint(run_id, run_fingerprint)
        write_json(
            run_dir / "journal.json",
            {"run_id": run_id, "status": "running", "pid": os.getpid(), "run_fingerprint": run_fingerprint},
        )
        self.storage.insert_event(
            kind="run_fingerprint",
            created_at=now_iso(),
            run_id=run_id,
            run_fingerprint=run_fingerprint,
            payload={"plugins": executed_ids},
        )

        transform_ids = {
            pid
            for pid in selected
            if pid in spec_map and spec_map[pid].type == "transform"
        }
        if transform_ids:
            layers = self._toposort_layers(specs, transform_ids)
            for layer in layers:
                for spec in layer:
                    if spec.plugin_id in pre_ran_transforms:
                        continue
                    run_spec(spec)
            dataset_accessor, dataset_template = resolve_dataset_accessor(
                self.storage, dataset_version_id
            )
            column_lookup = None

        if not auto_plan:
            profile_ids = {
                pid
                for pid in selected
                if pid in spec_map and spec_map[pid].type == "profile"
            }
            if profile_ids:
                layers = self._toposort_layers(specs, profile_ids)
                for layer in layers:
                    for spec in layer:
                        run_spec(spec)

        if not auto_plan:
            planner_ids = {
                pid
                for pid in selected
                if pid in spec_map and spec_map[pid].type == "planner"
            }
            if planner_ids:
                layers = self._toposort_layers(specs, planner_ids)
                for layer in layers:
                    for spec in layer:
                        run_spec(spec)

        analysis_ids = {
            pid
            for pid in selected
            if pid in spec_map and spec_map[pid].type == "analysis"
        }
        layers = self._toposort_layers(specs, analysis_ids)
        for layer in layers:
            if len(layer) == 1:
                run_spec(layer[0])
                continue
            max_workers = self._max_workers_for_stage(
                "analysis", len(layer), dataset_row_count
            )
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(run_spec, spec, False, False) for spec in layer
                ]
                for future in futures:
                    future.result()

        # Report stage:
        # - Always produce report.md/report.json (required by project policy).
        # - Some report plugins need report.json as input (e.g. report_plain_english_v1).
        #   Ensure report_bundle runs before those, while still running report_bundle late enough
        #   to include pre-bundle report-plugin results in report.json.
        report_ids = {
            pid
            for pid in selected
            if pid in spec_map and spec_map[pid].type == "report" and pid != "report_bundle"
        }

        report_spec = spec_map.get("report_bundle")
        if not report_spec:
            record_missing("report_bundle", "Missing report plugin")
        else:
            # Partition report plugins into:
            # - pre_bundle: should run before report_bundle (their results should appear in report.json)
            # - post_bundle: explicitly depend on report_bundle (they consume report.json)
            pre_bundle: set[str] = set()
            post_bundle: set[str] = set()
            for pid in report_ids:
                spec = spec_map.get(pid)
                if not spec:
                    continue
                if "report_bundle" in (spec.depends_on or []):
                    post_bundle.add(pid)
                else:
                    pre_bundle.add(pid)

            if pre_bundle:
                layers = self._toposort_layers(specs, pre_bundle)
                for layer in layers:
                    for spec in layer:
                        run_spec(spec)

            # Now generate the canonical report bundle from the DB state.
            run_spec(report_spec)

            if post_bundle:
                layers = self._toposort_layers(specs, post_bundle)
                for layer in layers:
                    for spec in layer:
                        run_spec(spec)

        # Fail-closed: even if report_bundle didn't emit files, synthesize the report from DB.
        report_json = run_dir / "report.json"
        report_md = run_dir / "report.md"
        if not report_json.exists() or not report_md.exists():
            report = build_report(self.storage, run_id, run_dir, Path("docs/report.schema.json"))
            write_report(report, run_dir)

        llm_ids = {
            pid for pid in selected if pid in spec_map and spec_map[pid].type == "llm"
        }
        if llm_ids:
            layers = self._toposort_layers(specs, llm_ids)
            for layer in layers:
                for spec in layer:
                    run_spec(spec)

        plugin_results = self.storage.fetch_plugin_results(run_id)
        plugin_executions = self.storage.fetch_plugin_executions(run_id)
        # A run can be "completed" even if some plugins are skipped or degraded under evidence gates.
        # Treat only true failures as "partial".
        golden_mode = _golden_mode()
        skipped_count = sum(
            1
            for row in plugin_results
            if str(row.get("status") or "").lower() == "skipped"
        )
        any_failures = any(
            str(row.get("status") or "").lower() in {"error", "aborted"} for row in plugin_results
        )
        strict_skip_violation = bool(golden_mode == "strict" and skipped_count > 0)
        if strict_skip_violation:
            any_failures = True
        final_status = "partial" if any_failures else "completed"
        if strict_skip_violation:
            self.storage.insert_event(
                kind="run_policy_violation",
                created_at=now_iso(),
                run_id=run_id,
                run_fingerprint=run_fingerprint,
                payload={
                    "policy": "golden_strict",
                    "reason": "skipped_plugins_not_allowed",
                    "skipped_count": int(skipped_count),
                },
            )

        # Update run status before final report synthesis so report.json/report.md reflect completion.
        self.storage.update_run_status(run_id, final_status)
        try:
            report = build_report(self.storage, run_id, run_dir, Path("docs/report.schema.json"))
            write_report(report, run_dir)
        except Exception:
            pass

        # Build and persist a canonical run manifest for portable provenance.
        artifacts: list[dict[str, Any]] = []
        def _register_artifact(rel_path: str, plugin_id: str | None = None) -> None:
            path = run_dir / rel_path
            if not path.exists() or not path.is_file():
                return
            sha = file_sha256(path)
            size = path.stat().st_size
            entry = {
                "path": rel_path.replace("\\", "/"),
                "sha256": sha,
                "bytes": int(size),
                "plugin_id": plugin_id,
            }
            artifacts.append(entry)
            self.storage.upsert_artifact(
                run_id=run_id,
                path=entry["path"],
                sha256=sha,
                size_bytes=int(size),
                mime=None,
                created_at=now_iso(),
                plugin_id=plugin_id,
            )

        _register_artifact("report.json", plugin_id="report")
        _register_artifact("report.md", plugin_id="report")
        if (run_dir / "evaluation.json").exists():
            _register_artifact("evaluation.json", plugin_id="report")
        logs_dir = run_dir / "logs"
        if logs_dir.exists():
            for item in sorted(logs_dir.rglob("*")):
                if item.is_file():
                    rel = str(item.relative_to(run_dir)).replace("\\", "/")
                    _register_artifact(rel, plugin_id="logs")
        artifacts_dir = run_dir / "artifacts"
        if artifacts_dir.exists():
            for item in sorted(artifacts_dir.rglob("*")):
                if not item.is_file():
                    continue
                rel = str(item.relative_to(run_dir)).replace("\\", "/")
                plugin_id = None
                parts = rel.split("/")
                if len(parts) >= 3 and parts[0] == "artifacts":
                    plugin_id = parts[1]
                _register_artifact(rel, plugin_id=plugin_id)

        manifest = {
            "schema_version": "run_manifest.v1",
            "run_id": run_id,
            "run_fingerprint": run_fingerprint,
            "created_at": now_iso(),
            "requested_run_seed": requested_run_seed,
            "run_seed": run_seed,
            "input": {
                "upload_id": upload_id,
                "input_hash": input_hash,
                "original_filename": input_filename,
            },
            "config": {"settings": settings},
            "plugins": [
                {
                    "plugin_id": row.get("plugin_id"),
                    "status": row.get("status"),
                    "plugin_version": row.get("plugin_version"),
                    "code_hash": row.get("code_hash"),
                    "settings_hash": row.get("settings_hash"),
                    "execution_fingerprint": row.get("execution_fingerprint"),
                }
                for row in sorted(plugin_results, key=lambda r: str(r.get("plugin_id") or ""))
            ],
            "executions": plugin_executions,
            "artifacts": sorted(artifacts, key=lambda a: (str(a.get("plugin_id") or ""), str(a.get("path") or ""))),
            "summary": {"status": final_status},
        }
        manifest["summary"]["golden_mode"] = golden_mode
        manifest["summary"]["skipped_count"] = int(skipped_count)
        manifest["summary"]["strict_skip_violation"] = bool(strict_skip_violation)
        manifest_path = run_dir / "run_manifest.json"
        write_json(manifest_path, manifest)
        manifest_sha = file_sha256(manifest_path)
        self.storage.update_run_manifest_sha256(run_id, manifest_sha)

        write_json(
            run_dir / "journal.json",
            {
                "run_id": run_id,
                "status": final_status,
                "pid": os.getpid(),
                "run_fingerprint": run_fingerprint,
                "completed_at": now_iso(),
            },
        )
        self.storage.insert_event(
            kind="run_completed",
            created_at=now_iso(),
            run_id=run_id,
            run_fingerprint=run_fingerprint,
            payload={"status": final_status},
        )
        if strict_skip_violation:
            raise RuntimeError(
                "STAT_HARNESS_GOLDEN_MODE=strict failed: one or more plugins returned skipped"
            )
        return run_id


def _format_edges(deps: dict[str, set[str]], nodes: set[str]) -> list[str]:
    edges: list[str] = []
    for src in sorted(nodes):
        for dst in sorted(deps.get(src, set())):
            if dst in nodes:
                edges.append(f"{src}->{dst}")
    return edges


def _find_cycle_path(deps: dict[str, set[str]], nodes: set[str]) -> list[str] | None:
    """Best-effort cycle path for actionable errors."""

    visiting: set[str] = set()
    visited: set[str] = set()
    parent: dict[str, str] = {}

    def dfs(node: str) -> list[str] | None:
        visiting.add(node)
        for nxt in sorted(deps.get(node, set())):
            if nxt not in nodes:
                continue
            if nxt in visited:
                continue
            if nxt in visiting:
                # Reconstruct cycle node->...->nxt->nxt
                path = [nxt]
                cur = node
                while cur != nxt and cur in parent:
                    path.append(cur)
                    cur = parent[cur]
                path.append(nxt)
                path.reverse()
                return path
            parent[nxt] = node
            found = dfs(nxt)
            if found:
                return found
        visiting.remove(node)
        visited.add(node)
        return None

    for node in sorted(nodes):
        if node in visited:
            continue
        found = dfs(node)
        if found:
            return found
    return None
