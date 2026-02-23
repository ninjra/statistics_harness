from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .types import PluginArtifact, PluginContext
from .utils import file_sha256, now_iso


def row_ref(dataset_version_id: str, row_index: int | str) -> str:
    return f"db://{str(dataset_version_id)}#row_index={int(row_index)}"


def _sha16(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


def stable_entity_id(entity_type: str, normalized_name: str) -> str:
    return _sha16(f"entity:v1:{entity_type}:{normalized_name}")


def stable_edge_id(
    relation: str,
    left_entity_id: str,
    right_entity_id: str,
    right_ref: str,
) -> str:
    return _sha16(f"edge:v1:{relation}:{left_entity_id}:{right_entity_id}:{right_ref}")


def evidence_link(
    *,
    match_type: str,
    confidence_tier: str,
    features: dict[str, Any],
    left_ref: str,
    right_ref: str,
    left_entity_id: str | None = None,
    right_entity_id: str | None = None,
    relation: str | None = None,
) -> dict[str, Any]:
    evidence_id = _sha16(
        f"evidence:v1:{match_type}:{confidence_tier}:{left_ref}:{right_ref}:{left_entity_id or ''}:{right_entity_id or ''}:{relation or ''}"
    )
    return {
        "evidence_id": evidence_id,
        "match_type": str(match_type),
        "confidence_tier": str(confidence_tier),
        "relation": relation,
        "left_ref": left_ref,
        "right_ref": right_ref,
        "left_entity_id": left_entity_id,
        "right_entity_id": right_entity_id,
        "features": dict(features or {}),
    }


@dataclass(frozen=True)
class RegisteredArtifact:
    artifact: PluginArtifact
    metadata: dict[str, Any]


def register_artifact(
    ctx: PluginContext,
    path: Path,
    *,
    description: str,
    mime: str = "application/octet-stream",
    producer_plugin_id: str | None = None,
    record_count: int | None = None,
) -> RegisteredArtifact:
    abs_path = path.resolve()
    rel = str(abs_path.relative_to(ctx.run_dir.resolve()))
    sha = file_sha256(abs_path)
    artifact_id = _sha16(f"artifact:v1:{rel}:{sha}")
    artifact = PluginArtifact(path=rel, type=mime, description=description)
    metadata = {
        "artifact_id": artifact_id,
        "path": rel,
        "sha256": sha,
        "mime": mime,
        "producer_plugin_id": producer_plugin_id,
        "record_count": record_count,
        "size_bytes": int(abs_path.stat().st_size),
        "generated_at_utc": now_iso(),
    }
    return RegisteredArtifact(artifact=artifact, metadata=metadata)

