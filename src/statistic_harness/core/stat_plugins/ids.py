from __future__ import annotations

import hashlib
from typing import Iterable


def stable_id(parts: Iterable[object], prefix: str | None = None) -> str:
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if prefix:
        return f"{prefix}_{digest[:12]}"
    return digest
