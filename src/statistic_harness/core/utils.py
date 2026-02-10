from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_run_id() -> str:
    return uuid.uuid4().hex


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


_FLOAT_PRECISION = 10
DEFAULT_TENANT_ID = "default"


def _canonicalize(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, _FLOAT_PRECISION)
    if isinstance(value, dict):
        return {key: _canonicalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    if isinstance(value, tuple):
        return [_canonicalize(item) for item in value]
    return value


def json_dumps(data: Any) -> str:
    return json.dumps(
        _canonicalize(data), ensure_ascii=False, indent=2, sort_keys=True
    )


def safe_join(base: Path, *paths: str) -> Path:
    base_resolved = base.resolve()
    joined = base_resolved.joinpath(*paths).resolve()
    try:
        joined.relative_to(base_resolved)
    except ValueError as exc:
        raise ValueError("Path traversal detected") from exc
    return joined


def stable_hash(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def scope_key(scope_type: str, scope_value: str) -> str:
    if scope_type == "sha256":
        return scope_value
    digest = hashlib.sha256(f"{scope_type}:{scope_value}".encode("utf-8")).hexdigest()
    return digest


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def dataset_key(project_id: str, input_hash: str) -> str:
    payload = f"{project_id}:{input_hash}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def quote_identifier(name: str) -> str:
    if not _IDENT_RE.match(name):
        raise ValueError(f"Unsafe identifier: {name}")
    return f"\"{name}\""


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def is_windows_or_wsl() -> bool:
    if os.name == "nt":
        return True
    if os.environ.get("WSL_INTEROP") or os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        txt = Path("/proc/version").read_text(encoding="utf-8", errors="ignore").lower()
        return "microsoft" in txt or "wsl" in txt
    except Exception:
        return False


def safe_rename_enabled() -> bool:
    raw = os.environ.get("STAT_HARNESS_SAFE_RENAME", "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    return is_windows_or_wsl()


def safe_replace(src: Path, dst: Path) -> None:
    """os.replace with best-effort retries to tolerate Windows/WSL transient locks."""

    attempts = 10 if safe_rename_enabled() else 1
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except OSError:
            if i + 1 >= attempts:
                raise
            # Small deterministic backoff.
            time.sleep(0.02 * (i + 1))


def atomic_dir(
    final_dir: Path,
    *,
    staging_root: Path,
    prepare: Callable[[Path], None],
) -> None:
    """Create a directory tree and commit it atomically via rename."""

    staging_root.mkdir(parents=True, exist_ok=True)
    staging_dir = staging_root / final_dir.name
    if staging_dir.exists():
        shutil.rmtree(staging_dir, ignore_errors=True)
    prepare(staging_dir)
    if final_dir.exists():
        raise ValueError(f"Directory already exists: {final_dir}")
    safe_replace(staging_dir, final_dir)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically write bytes to `path` (write temp file then os.replace).

    Temp file is created in the same directory to keep the replace atomic across filesystems.
    Best-effort fsync is used for durability; failures to fsync directories are ignored.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
    try:
        with tmp_path.open("wb") as handle:
            handle.write(data)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                # Some platforms/filesystems do not support fsync; still keep replace atomic.
                pass
        safe_replace(tmp_path, path)
        # Best-effort: fsync directory entry to reduce risk of rename loss on crash.
        try:
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
        except OSError:
            dir_fd = None
        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            except OSError:
                pass
            finally:
                os.close(dir_fd)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, text.encode(encoding))


def write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json_dumps(data), encoding="utf-8")


def file_size_limit(path: Path, max_bytes: int) -> None:
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"File too large: {size} bytes")


def max_upload_bytes() -> int | None:
    raw = os.environ.get("STAT_HARNESS_MAX_UPLOAD_BYTES", "").strip()
    if not raw:
        return None
    try:
        limit = int(raw)
    except ValueError:
        return None
    if limit <= 0:
        return None
    return limit


def auth_enabled() -> bool:
    raw = os.environ.get("STAT_HARNESS_ENABLE_AUTH", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def vector_store_enabled() -> bool:
    raw = os.environ.get("STAT_HARNESS_ENABLE_VECTOR_STORE", "").strip().lower()
    # Default ON (local-first), but allow explicit disable.
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    return True


_ENV_PLACEHOLDER_RE = re.compile(r"^\$\{ENV:([A-Z0-9_]+)\}$")


def resolve_env_placeholders(value: Any) -> Any:
    """Resolve `${ENV:NAME}` strings to their environment variable values.

    This keeps configs schema-friendly (placeholders are still strings) while allowing
    secret indirection without persisting secret values to disk.
    """

    if isinstance(value, str):
        match = _ENV_PLACEHOLDER_RE.match(value.strip())
        if not match:
            return value
        name = match.group(1)
        if name not in os.environ:
            raise ValueError(f"Missing environment variable: {name}")
        return os.environ[name]
    if isinstance(value, list):
        return [resolve_env_placeholders(item) for item in value]
    if isinstance(value, dict):
        return {k: resolve_env_placeholders(v) for k, v in value.items()}
    return value


def get_appdata_dir() -> Path:
    return Path(os.environ.get("STAT_HARNESS_APPDATA", "appdata"))


def infer_close_cycle_window(
    timestamps: Any,
    window_days: int = 17,
    fallback_start: int = 20,
    fallback_end: int = 5,
) -> tuple[int, int]:
    try:
        import pandas as pd  # type: ignore
    except ImportError:
        return fallback_start, fallback_end
    if timestamps is None:
        return fallback_start, fallback_end
    series = pd.to_datetime(timestamps, errors="coerce", utc=False)
    if hasattr(series, "dropna"):
        series = series.dropna()
    if series is None or len(series) == 0:
        return fallback_start, fallback_end
    days = series.dt.day
    counts = days.value_counts().reindex(range(1, 32), fill_value=0)
    total_days = 31
    window = max(1, min(int(window_days), total_days))
    if window >= total_days:
        return 1, total_days
    best_start = 1
    best_sum = -1
    for start in range(1, total_days + 1):
        window_sum = 0
        for offset in range(window):
            day = ((start - 1 + offset) % total_days) + 1
            window_sum += int(counts.get(day, 0))
        if window_sum > best_sum:
            best_sum = window_sum
            best_start = start
    end = ((best_start - 1 + window - 1) % total_days) + 1
    return best_start, end
