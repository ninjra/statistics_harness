from __future__ import annotations

import builtins
import io
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
import math

from .dataset_io import resolve_dataset_accessor
from .plugin_manager import PluginSpec
from .storage import Storage
from .types import PluginArtifact, PluginContext, PluginError, PluginResult
from .utils import ensure_dir, now_iso, read_json, write_json


_NETWORK_ENV = "STAT_HARNESS_ALLOW_NETWORK"


def _seed_runtime(run_seed: int) -> None:
    import random

    random.seed(run_seed)
    try:
        import numpy as np

        np.random.seed(run_seed)
    except Exception:
        pass


def _deterministic_env(run_seed: int, cwd: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(run_seed)
    env["TZ"] = "UTC"
    env["LC_ALL"] = "C"
    env["LANG"] = "C"

    roots: list[str] = []
    if cwd is not None:
        roots.extend([str(cwd), str(cwd / "src")])
    existing = env.get("PYTHONPATH", "")
    parts = [part for part in existing.split(os.pathsep) if part]
    for root in reversed(roots):
        if root in parts:
            parts.remove(root)
        parts.insert(0, root)
    if parts:
        env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _network_allowed() -> bool:
    return os.environ.get(_NETWORK_ENV, "").lower() in {"1", "true", "yes"}


def _install_network_guard() -> None:
    import socket

    def blocked(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Network disabled by STAT_HARNESS_ALLOW_NETWORK=0")

    base_socket = socket.socket

    class GuardedSocket(base_socket):
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
            blocked(*args, **kwargs)

        def connect(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
            return blocked(*args, **kwargs)

        def connect_ex(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
            return blocked(*args, **kwargs)

    socket.socket = GuardedSocket  # type: ignore[assignment]
    socket.create_connection = blocked  # type: ignore[assignment]


def _install_eval_guard(root_dir: Path | None = None) -> None:
    import inspect

    root = root_dir.resolve() if root_dir else None
    orig_eval = builtins.eval

    def guarded_eval(
        expression: Any, globals: dict[str, Any] | None = None, locals: dict[str, Any] | None = None
    ) -> Any:
        frame = inspect.currentframe()
        caller = frame.f_back if frame else None
        if root is not None and caller:
            filename = caller.f_code.co_filename
            if filename:
                try:
                    caller_path = Path(filename).resolve()
                    try:
                        caller_path.relative_to(root)
                        raise RuntimeError("Eval disabled by policy")
                    except ValueError:
                        pass
                except RuntimeError:
                    raise
                except Exception:
                    pass
        if caller:
            if globals is None:
                globals = caller.f_globals
            if locals is None:
                locals = caller.f_locals
        return orig_eval(expression, globals, locals)

    builtins.eval = guarded_eval  # type: ignore[assignment]


def _install_pickle_guard() -> None:
    import pickle

    def blocked(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Pickle disabled by policy")

    pickle.load = blocked  # type: ignore[assignment]
    pickle.loads = blocked  # type: ignore[assignment]
    pickle.dump = blocked  # type: ignore[assignment]
    pickle.dumps = blocked  # type: ignore[assignment]
    pickle.Pickler = blocked  # type: ignore[assignment]
    pickle.Unpickler = blocked  # type: ignore[assignment]


def _install_shell_guard() -> None:
    import subprocess

    def blocked(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Shell disabled by policy")

    os.system = blocked  # type: ignore[assignment]
    os.popen = blocked  # type: ignore[assignment]
    subprocess.run = blocked  # type: ignore[assignment]
    subprocess.call = blocked  # type: ignore[assignment]
    subprocess.check_call = blocked  # type: ignore[assignment]
    subprocess.check_output = blocked  # type: ignore[assignment]
    subprocess.Popen = blocked  # type: ignore[assignment]


def _apply_resource_limits(budget: dict[str, Any]) -> None:
    cpu_limit_ms = budget.get("cpu_limit_ms")
    if cpu_limit_ms is None:
        return
    try:
        cpu_seconds = max(1, int(math.ceil(float(cpu_limit_ms) / 1000.0)))
    except (TypeError, ValueError):
        return
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    except Exception:
        pass


class FileSandbox:
    def __init__(self, allow_paths: Iterable[str], cwd: Path) -> None:
        self._cwd = cwd
        self._allow = [self._normalize(Path(p)) for p in allow_paths]
        self._readonly_allow = self._library_roots()
        self._orig_open = builtins.open
        self._orig_io_open = io.open
        self._orig_path_open = Path.open

    def _normalize(self, path: Path) -> Path:
        if not path.is_absolute():
            path = (self._cwd / path).resolve()
        else:
            path = path.resolve()
        return path

    def _library_roots(self) -> list[Path]:
        roots: list[Path] = []
        for base in {sys.prefix, sys.base_prefix}:
            if base:
                roots.append(Path(base).resolve())
        return roots

    def _is_readonly_allowed(self, path: Path) -> bool:
        target = self._normalize(path)
        for root in self._readonly_allow:
            try:
                target.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    def _is_allowed(self, path: Path) -> bool:
        target = self._normalize(path)
        for allowed in self._allow:
            if allowed.is_file():
                if target == allowed:
                    return True
            else:
                try:
                    target.relative_to(allowed)
                    return True
                except ValueError:
                    continue
        return False

    def _guarded_open(self, file: Any, *args: Any, **kwargs: Any) -> Any:
        path = Path(file) if not isinstance(file, Path) else file
        mode = "r"
        if args:
            mode = str(args[0])
        if "mode" in kwargs:
            mode = str(kwargs["mode"])
        write_mode = any(token in mode for token in ("w", "a", "x", "+"))
        if write_mode:
            if not self._is_allowed(path):
                raise PermissionError(f"File write denied: {path}")
        else:
            if not (self._is_allowed(path) or self._is_readonly_allowed(path)):
                raise PermissionError(f"File read denied: {path}")
        return self._orig_open(file, *args, **kwargs)

    def __enter__(self) -> "FileSandbox":
        builtins.open = self._guarded_open  # type: ignore[assignment]
        io.open = self._guarded_open  # type: ignore[assignment]
        sandbox = self

        def _path_open(path_self: Path, *args: Any, **kwargs: Any) -> Any:
            return sandbox._guarded_open(path_self, *args, **kwargs)

        Path.open = _path_open  # type: ignore[assignment]
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        builtins.open = self._orig_open  # type: ignore[assignment]
        io.open = self._orig_io_open  # type: ignore[assignment]
        Path.open = self._orig_path_open  # type: ignore[assignment]


def _load_plugin(plugin_id: str, entrypoint: str) -> Any:
    module_path, class_name = entrypoint.split(":", 1)
    if module_path.endswith(".py"):
        module_path = module_path[:-3]
    module_name = f"plugins.{plugin_id}.{module_path}"
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)()


def _result_payload(result: PluginResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "summary": result.summary,
        "metrics": result.metrics,
        "findings": result.findings,
        "artifacts": [asdict(a) for a in result.artifacts],
        "budget": result.budget,
        "error": asdict(result.error) if result.error else None,
    }


def _payload_result(payload: dict[str, Any]) -> PluginResult:
    artifacts = [
        PluginArtifact(**item)
        for item in payload.get("artifacts", [])
        if isinstance(item, dict)
    ]
    error = payload.get("error")
    error_obj = None
    if isinstance(error, dict):
        error_obj = PluginError(
            type=error.get("type", "Error"),
            message=error.get("message", ""),
            traceback=error.get("traceback", ""),
        )
    return PluginResult(
        status=payload.get("status", "error"),
        summary=payload.get("summary", ""),
        metrics=payload.get("metrics", {}),
        findings=payload.get("findings", []),
        artifacts=artifacts,
        budget=payload.get("budget")
        or {
            "row_limit": None,
            "sampled": False,
            "time_limit_ms": None,
            "cpu_limit_ms": None,
        },
        error=error_obj,
    )


@dataclass
class RunnerResponse:
    result: PluginResult
    execution: dict[str, Any]
    stdout: str
    stderr: str
    exit_code: int


def run_plugin_subprocess(
    spec: PluginSpec,
    request: dict[str, Any],
    run_dir: Path,
    cwd: Path,
) -> RunnerResponse:
    ensure_dir(run_dir / "logs")
    request_path = run_dir / "logs" / f"{spec.plugin_id}_request.json"
    response_path = run_dir / "logs" / f"{spec.plugin_id}_response.json"
    write_json(request_path, request)
    start = time.perf_counter()
    run_seed = int(request.get("run_seed", 0))
    budget = request.get("budget") or {}
    timeout_ms = budget.get("time_limit_ms")
    timeout = None
    if isinstance(timeout_ms, (int, float)) and timeout_ms > 0:
        timeout = float(timeout_ms) / 1000.0
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "statistic_harness.core.plugin_runner",
                str(request_path),
                str(response_path),
            ],
            capture_output=True,
            text=True,
            cwd=str(cwd),
            env=_deterministic_env(run_seed, cwd=cwd),
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        duration_ms = int((time.perf_counter() - start) * 1000)
        stdout = (exc.stdout or "")[:4000]
        stderr = (exc.stderr or "")[:4000]
        result = PluginResult(
            status="error",
            summary=f"{spec.plugin_id} timed out",
            metrics={},
            findings=[],
            artifacts=[],
            error=PluginError(
                type="TimeoutError",
                message=f"Plugin exceeded time limit of {timeout_ms}ms",
                traceback="",
            ),
        )
        execution = {
            "started_at": request.get("started_at"),
            "completed_at": now_iso(),
            "duration_ms": duration_ms,
            "cpu_user": None,
            "cpu_system": None,
            "max_rss": None,
            "warnings_count": None,
        }
        return RunnerResponse(
            result=result,
            execution=execution,
            stdout=stdout,
            stderr=stderr,
            exit_code=-1,
        )
    duration_ms = int((time.perf_counter() - start) * 1000)
    stdout = (proc.stdout or "")[:4000]
    stderr = (proc.stderr or "")[:4000]
    execution = {
        "started_at": request.get("started_at"),
        "completed_at": now_iso(),
        "duration_ms": duration_ms,
        "cpu_user": None,
        "cpu_system": None,
        "max_rss": None,
        "warnings_count": None,
    }
    if response_path.exists():
        payload = read_json(response_path)
        result = _payload_result(payload.get("result", {}))
        execution.update(payload.get("execution", {}))
    else:
        result = PluginResult(
            status="error",
            summary=f"{spec.plugin_id} failed to execute",
            metrics={},
            findings=[],
            artifacts=[],
            error=PluginError(
                type="RunnerError",
                message=f"Missing response for {spec.plugin_id}",
                traceback=stderr or "",
            ),
        )
    return RunnerResponse(
        result=result,
        execution=execution,
        stdout=stdout,
        stderr=stderr,
        exit_code=proc.returncode,
    )


def _run_request(request: dict[str, Any]) -> dict[str, Any]:
    started_at = now_iso()
    request["started_at"] = started_at
    execution: dict[str, Any] = {"started_at": started_at}
    warnings_count = 0
    try:
        run_seed = int(request.get("run_seed", 0))
        os.environ.update(_deterministic_env(run_seed))
        try:
            time.tzset()
        except AttributeError:
            pass
        _seed_runtime(run_seed)
        if request.get("sandbox", {}).get("no_network") and not _network_allowed():
            _install_network_guard()
        allow_paths = request.get("allow_paths") or []
        cwd = Path(request.get("root_dir", ".")).resolve()
        run_dir = Path(request["run_dir"]).resolve()
        with FileSandbox(allow_paths, cwd):
            storage = Storage(
                Path(request["appdata_dir"]) / "state.sqlite",
                request.get("tenant_id"),
            )
            dataset_version_id = request.get("dataset_version_id")
            accessor, _ = resolve_dataset_accessor(storage, dataset_version_id)
            budget = request.get("budget") or {}

            def dataset_loader(
                columns: list[str] | None = None, row_limit: int | None = None
            ):
                limit = row_limit
                if limit is None:
                    limit = budget.get("row_limit")
                return accessor.load(columns=columns, row_limit=limit)

            ctx = PluginContext(
                run_id=request["run_id"],
                run_dir=run_dir,
                settings=request.get("settings", {}),
                run_seed=run_seed,
                logger=lambda msg: _write_log(run_dir, request["plugin_id"], msg),
                storage=storage,
                dataset_loader=dataset_loader,
                budget=budget
                or {
                    "row_limit": None,
                    "sampled": False,
                    "time_limit_ms": None,
                    "cpu_limit_ms": None,
                },
                tenant_id=request.get("tenant_id"),
                project_id=request.get("project_id"),
                dataset_id=request.get("dataset_id"),
                dataset_version_id=dataset_version_id,
                input_hash=request.get("input_hash"),
            )
            start = time.perf_counter()
            try:
                import resource

                start_usage = resource.getrusage(resource.RUSAGE_SELF)
            except Exception:  # pragma: no cover - platform specific
                start_usage = None

            try:
                import warnings

                with warnings.catch_warnings(record=True) as caught:
                    _install_eval_guard(cwd)
                    _install_pickle_guard()
                    _install_shell_guard()
                    _apply_resource_limits(budget)
                    plugin = _load_plugin(request["plugin_id"], request["entrypoint"])
                    result = plugin.run(ctx)
                    warnings_count = len(caught)
            except Exception as exc:
                tb = traceback.format_exc()
                result = PluginResult(
                    status="error",
                    summary=f"{request['plugin_id']} failed",
                    metrics={},
                    findings=[],
                    artifacts=[],
                    error=PluginError(
                        type=type(exc).__name__, message=str(exc), traceback=tb
                    ),
                )

            duration_ms = int((time.perf_counter() - start) * 1000)
            execution.update(
                {
                    "completed_at": now_iso(),
                    "duration_ms": duration_ms,
                    "warnings_count": warnings_count,
                }
            )
            try:
                import resource

                end_usage = resource.getrusage(resource.RUSAGE_SELF)
                if start_usage and end_usage:
                    execution.update(
                        {
                            "cpu_user": end_usage.ru_utime - start_usage.ru_utime,
                            "cpu_system": end_usage.ru_stime - start_usage.ru_stime,
                            "max_rss": end_usage.ru_maxrss,
                        }
                    )
            except Exception:  # pragma: no cover - platform specific
                pass
    except Exception as exc:  # pragma: no cover - runner failure
        tb = traceback.format_exc()
        result = PluginResult(
            status="error",
            summary=f"{request.get('plugin_id', 'plugin')} runner failed",
            metrics={},
            findings=[],
            artifacts=[],
            error=PluginError(type=type(exc).__name__, message=str(exc), traceback=tb),
        )
        execution.update({"completed_at": now_iso(), "duration_ms": 0})
    return {"result": _result_payload(result), "execution": execution}


def _write_log(run_dir: Path, plugin_id: str, msg: str) -> None:
    log_path = run_dir / "logs" / f"{plugin_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(msg + "\n")


def main(argv: list[str] | None = None) -> int:
    args = argv or sys.argv[1:]
    if len(args) != 2:
        raise SystemExit("Usage: plugin_runner <request.json> <response.json>")
    request_path = Path(args[0])
    response_path = Path(args[1])
    request = read_json(request_path)
    response = _run_request(request)
    write_json(response_path, response)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
