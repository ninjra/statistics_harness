from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_dumps(data), encoding="utf-8")


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
    return raw in {"1", "true", "yes", "on"}


def get_appdata_dir() -> Path:
    return Path(os.environ.get("STAT_HARNESS_APPDATA", "appdata"))
