"""Tenant context helpers for phase 2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re

from .utils import DEFAULT_TENANT_ID, get_appdata_dir


_TENANT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_]{0,63}$")


def tenancy_enabled() -> bool:
    raw = os.environ.get("STAT_HARNESS_ENABLE_TENANCY", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def resolve_tenant_id(requested: str | None = None) -> str:
    env = os.environ.get("STAT_HARNESS_TENANT_ID", "").strip()
    tenant_id = (requested or env or DEFAULT_TENANT_ID).strip() or DEFAULT_TENANT_ID
    if not _TENANT_RE.match(tenant_id):
        raise ValueError("Invalid tenant_id")
    return tenant_id


@dataclass(frozen=True)
class TenantContext:
    tenant_id: str
    appdata_root: Path
    tenant_root: Path
    db_path: Path


def get_tenant_context(
    tenant_id: str | None = None, appdata_root: Path | None = None
) -> TenantContext:
    root = appdata_root or get_appdata_dir()
    resolved = resolve_tenant_id(tenant_id)
    if tenancy_enabled():
        tenant_root = root / "tenants" / resolved
    else:
        tenant_root = root
    db_path = root / "state.sqlite"
    return TenantContext(
        tenant_id=resolved,
        appdata_root=root,
        tenant_root=tenant_root,
        db_path=db_path,
    )


def scope_identifier(tenant_id: str, raw_id: str) -> str:
    prefix = f"{tenant_id}__"
    if raw_id.startswith(prefix):
        return raw_id
    return f"{prefix}{raw_id}"
