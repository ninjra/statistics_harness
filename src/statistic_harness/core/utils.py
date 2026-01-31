from __future__ import annotations

import hashlib
import json
import os
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


def json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)


def safe_join(base: Path, *paths: str) -> Path:
    joined = base.joinpath(*paths).resolve()
    base_resolved = base.resolve()
    if not str(joined).startswith(str(base_resolved)):
        raise ValueError("Path traversal detected")
    return joined


def stable_hash(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_dumps(data), encoding="utf-8")


def file_size_limit(path: Path, max_bytes: int) -> None:
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"File too large: {size} bytes")


def get_appdata_dir() -> Path:
    return Path(os.environ.get("STAT_HARNESS_APPDATA", "appdata"))
