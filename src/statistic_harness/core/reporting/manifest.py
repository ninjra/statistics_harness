from __future__ import annotations

import re
from collections import deque
from pathlib import Path
from typing import Any

import yaml

_PLUGIN_CLASS_TAXONOMY_PATH = Path(__file__).resolve().parents[4] / "docs" / "plugin_class_taxonomy.yaml"
_PLUGIN_CLASS_TAXONOMY_CACHE: dict[str, Any] | None = None


def _manifest_index() -> dict[str, dict[str, Any]]:
    plugins_root = Path(__file__).resolve().parents[4] / "plugins"
    index: dict[str, dict[str, Any]] = {}
    for manifest in sorted(plugins_root.glob("*/plugin.yaml")):
        try:
            payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        plugin_id = str(payload.get("id") or manifest.parent.name)
        deps = payload.get("depends_on")
        depends_on = [str(v).strip() for v in deps] if isinstance(deps, list) else []
        index[plugin_id] = {
            "type": str(payload.get("type") or "").strip().lower(),
            "depends_on": [v for v in depends_on if v],
        }
    return index


def _plugin_class_taxonomy() -> dict[str, Any]:
    global _PLUGIN_CLASS_TAXONOMY_CACHE
    if _PLUGIN_CLASS_TAXONOMY_CACHE is not None:
        return _PLUGIN_CLASS_TAXONOMY_CACHE
    try:
        payload = yaml.safe_load(_PLUGIN_CLASS_TAXONOMY_PATH.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    _PLUGIN_CLASS_TAXONOMY_CACHE = payload if isinstance(payload, dict) else {}
    return _PLUGIN_CLASS_TAXONOMY_CACHE


def _plugin_class_id(plugin_id: str, plugin_type: str) -> str:
    taxonomy = _plugin_class_taxonomy()
    overrides = taxonomy.get("plugin_overrides") if isinstance(taxonomy.get("plugin_overrides"), dict) else {}
    defaults = (
        taxonomy.get("plugin_type_default_class")
        if isinstance(taxonomy.get("plugin_type_default_class"), dict)
        else {}
    )
    pid = str(plugin_id or "").strip()
    ptype = str(plugin_type or "").strip().lower()
    value = overrides.get(pid)
    if isinstance(value, str) and value.strip():
        return value.strip()
    default_value = defaults.get(ptype)
    if isinstance(default_value, str) and default_value.strip():
        return default_value.strip()
    if ptype and ptype != "analysis":
        return "supporting_signal_detectors"
    return "direct_action_generators" if pid == "analysis_actionable_ops_levers_v1" else "supporting_signal_detectors"


def _plugin_expected_output_type(plugin_class: str) -> str:
    taxonomy = _plugin_class_taxonomy()
    classes = taxonomy.get("classes") if isinstance(taxonomy.get("classes"), dict) else {}
    class_meta = classes.get(plugin_class) if isinstance(classes.get(plugin_class), dict) else {}
    value = class_meta.get("expected_output_type")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return ""


def _extract_precondition_inputs(payload: dict[str, Any]) -> tuple[list[str], list[str]]:
    findings = payload.get("findings") if isinstance(payload.get("findings"), list) else []
    debug = payload.get("debug") if isinstance(payload.get("debug"), dict) else {}
    required: set[str] = set()
    missing: set[str] = set()

    def _collect(target: set[str], value: Any) -> None:
        if isinstance(value, str):
            for token in re.split(r"[;,|]", value):
                item = token.strip()
                if item:
                    target.add(item)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                if isinstance(item, str) and item.strip():
                    target.add(item.strip())

    sources: list[dict[str, Any]] = [debug]
    sources.extend(item for item in findings if isinstance(item, dict))
    for item in sources:
        _collect(required, item.get("required_inputs"))
        _collect(required, item.get("required_columns"))
        _collect(required, item.get("required_fields"))
        _collect(missing, item.get("missing_inputs"))
        _collect(missing, item.get("missing_columns"))
        _collect(missing, item.get("missing_fields"))
        prereq = item.get("prerequisites") if isinstance(item.get("prerequisites"), dict) else {}
        if isinstance(prereq, dict):
            _collect(required, prereq.get("required"))
            _collect(required, prereq.get("required_inputs"))
            _collect(required, prereq.get("required_columns"))
            _collect(missing, prereq.get("missing"))
            _collect(missing, prereq.get("missing_inputs"))
            _collect(missing, prereq.get("missing_columns"))

    required_sorted = sorted(required)
    missing_sorted = sorted(missing)
    return required_sorted, missing_sorted


def _downstream_consumers(plugin_ids: set[str], manifest_index: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    reverse: dict[str, set[str]] = {pid: set() for pid in plugin_ids}
    for pid in plugin_ids:
        meta = manifest_index.get(pid) or {}
        deps = meta.get("depends_on") if isinstance(meta.get("depends_on"), list) else []
        for dep in deps:
            dep_id = str(dep or "").strip()
            if dep_id and dep_id in reverse:
                reverse[dep_id].add(pid)
    out: dict[str, list[str]] = {}
    for pid in sorted(plugin_ids):
        seen: set[str] = set()
        queue: deque[str] = deque(sorted(reverse.get(pid) or []))
        while queue:
            cur = queue.popleft()
            if cur in seen:
                continue
            seen.add(cur)
            for nxt in sorted(reverse.get(cur) or []):
                if nxt not in seen:
                    queue.append(nxt)
        out[pid] = sorted(seen)
    return out
