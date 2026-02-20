from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

from statistic_harness.core.utils import file_sha256, json_dumps, now_iso, safe_replace

if TYPE_CHECKING:  # pragma: no cover
    from statistic_harness.core.plugin_manager import PluginManager, PluginSpec


CONTRACT_SCHEMA = "frozen_surfaces_contract.v1"
CONTRACT_FILENAME = "frozen_plugin_surfaces.contract.json"


def default_contract_path(root_dir: Path) -> Path:
    return Path(root_dir).resolve() / "docs" / CONTRACT_FILENAME


def plugin_module_file(spec: "PluginSpec") -> Path:
    module_path, _ = str(spec.entrypoint).split(":", 1)
    if module_path.endswith(".py"):
        return spec.path / module_path
    return spec.path / f"{module_path}.py"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_hash_or_none(path: Path) -> str | None:
    try:
        if path.exists():
            return file_sha256(path)
    except Exception:
        return None
    return None


def effective_code_hash_for_spec(spec: "PluginSpec", code_hash: str | None = None) -> str | None:
    module_file = plugin_module_file(spec)
    base_hash = code_hash or _file_hash_or_none(module_file)
    if not base_hash or not module_file.exists():
        return base_hash
    try:
        text = module_file.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return base_hash
    if "run_plugin(" not in text:
        return base_hash
    try:
        from statistic_harness.core.stat_plugins.code_hash import (
            stat_plugin_effective_code_hash,
        )

        effective = stat_plugin_effective_code_hash(str(spec.plugin_id))
    except Exception:
        effective = None
    if not effective:
        return base_hash
    return _sha256_text(f"{base_hash}:{effective}")


def resolved_default_settings_hash(spec: "PluginSpec", manager: "PluginManager") -> str | None:
    try:
        defaults = dict(spec.settings.get("defaults", {}))
        resolved = manager.resolve_config(spec, defaults)
    except Exception:
        return None
    return _sha256_text(json_dumps(resolved))


def build_surface_record(
    spec: "PluginSpec",
    manager: "PluginManager",
    *,
    code_hash: str | None = None,
    settings_hash: str | None = None,
) -> dict[str, Any]:
    manifest_path = spec.path / "plugin.yaml"
    config_schema_path = Path(spec.config_schema)
    output_schema_path = Path(spec.output_schema)

    resolved_code_hash = effective_code_hash_for_spec(spec, code_hash=code_hash)
    resolved_settings_hash = settings_hash or resolved_default_settings_hash(spec, manager)
    manifest_hash = _file_hash_or_none(manifest_path)
    config_schema_hash = _file_hash_or_none(config_schema_path)
    output_schema_hash = _file_hash_or_none(output_schema_path)

    payload = {
        "plugin_id": str(spec.plugin_id),
        "plugin_version": str(spec.version),
        "plugin_type": str(spec.type),
        "entrypoint": str(spec.entrypoint),
        "code_hash": resolved_code_hash,
        "settings_hash": resolved_settings_hash,
        "manifest_hash": manifest_hash,
        "config_schema_hash": config_schema_hash,
        "output_schema_hash": output_schema_hash,
    }
    surface_hash = _sha256_text(json_dumps(payload))
    return {
        **payload,
        "surface_hash": surface_hash,
    }


def contract_plugin_map(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = contract.get("plugins")
    if isinstance(raw, dict):
        out: dict[str, dict[str, Any]] = {}
        for key, value in raw.items():
            if isinstance(value, dict):
                out[str(key)] = dict(value)
        return out
    if isinstance(raw, list):
        out: dict[str, dict[str, Any]] = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            plugin_id = str(item.get("plugin_id") or "").strip()
            if not plugin_id:
                continue
            out[plugin_id] = dict(item)
        return out
    return {}


def load_contract(path: Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"schema": CONTRACT_SCHEMA, "plugins": {}}
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"schema": CONTRACT_SCHEMA, "plugins": {}}
    if not isinstance(payload, dict):
        return {"schema": CONTRACT_SCHEMA, "plugins": {}}
    plugins = contract_plugin_map(payload)
    return {
        **payload,
        "schema": str(payload.get("schema") or CONTRACT_SCHEMA),
        "plugins": plugins,
    }


def save_contract(path: Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    normalized = {
        "schema": CONTRACT_SCHEMA,
        "generated_at": str(payload.get("generated_at") or now_iso()),
        "source_run_id": payload.get("source_run_id"),
        "source_dataset_version_id": payload.get("source_dataset_version_id"),
        "plugins": contract_plugin_map(payload),
    }
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    safe_replace(tmp, target)


def evaluate_locked_surface(
    *,
    plugin_id: str,
    expected_surface_hash: str | None,
    actual_surface_hash: str | None,
) -> dict[str, Any]:
    expected = str(expected_surface_hash or "").strip()
    actual = str(actual_surface_hash or "").strip()
    return {
        "plugin_id": str(plugin_id),
        "locked": bool(expected),
        "ok": bool(expected) and expected == actual,
        "expected_surface_hash": expected or None,
        "actual_surface_hash": actual or None,
    }

