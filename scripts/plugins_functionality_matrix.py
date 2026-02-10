from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PluginMeta:
    plugin_id: str
    name: str
    version: str
    type: str
    entrypoint: str
    depends_on: list[str]
    capabilities: list[str]
    description: str


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_plugins(plugins_root: Path) -> dict[str, PluginMeta]:
    out: dict[str, PluginMeta] = {}
    for child in sorted([p for p in plugins_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        manifest = child / "plugin.yaml"
        if not manifest.exists():
            continue
        data = _read_yaml(manifest)
        pid = str(data.get("id") or child.name)
        settings = data.get("settings") if isinstance(data.get("settings"), dict) else {}
        defaults = ""
        if isinstance(settings, dict):
            defaults = str(settings.get("description") or "")
        meta = PluginMeta(
            plugin_id=pid,
            name=str(data.get("name") or pid),
            version=str(data.get("version") or ""),
            type=str(data.get("type") or ""),
            entrypoint=str(data.get("entrypoint") or ""),
            depends_on=[str(x) for x in (data.get("depends_on") or []) if isinstance(x, (str, int, float))],
            capabilities=[str(x) for x in (data.get("capabilities") or []) if isinstance(x, (str, int, float))],
            description=defaults,
        )
        out[pid] = meta
    return out


def _toposort(nodes: list[str], edges: list[tuple[str, str]]) -> list[str]:
    # Kahn's algorithm; deterministic by sorting ready queue.
    outgoing: dict[str, set[str]] = {n: set() for n in nodes}
    indeg: dict[str, int] = {n: 0 for n in nodes}
    for a, b in edges:
        if a not in outgoing or b not in outgoing:
            continue
        if b in outgoing[a]:
            continue
        outgoing[a].add(b)
        indeg[b] += 1
    ready = sorted([n for n in nodes if indeg[n] == 0])
    out: list[str] = []
    while ready:
        n = ready.pop(0)
        out.append(n)
        for m in sorted(outgoing[n]):
            indeg[m] -= 1
            if indeg[m] == 0:
                ready.append(m)
                ready.sort()
    if len(out) != len(nodes):
        cycle_nodes = sorted([n for n in nodes if indeg[n] > 0])
        raise ValueError(f"Dependency cycle detected among: {cycle_nodes[:20]}")
    return out


def _group_key(plugin_id: str, plugin_type: str) -> str:
    if plugin_type != "analysis":
        return plugin_type
    parts = plugin_id.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:3]) + "_*"
    return plugin_id


def build_matrix(plugins: dict[str, PluginMeta]) -> dict[str, Any]:
    ids = sorted(plugins.keys())
    missing_deps: dict[str, list[str]] = {}
    edges: list[tuple[str, str]] = []
    for pid, meta in plugins.items():
        for dep in meta.depends_on:
            if dep not in plugins:
                missing_deps.setdefault(pid, []).append(dep)
            else:
                edges.append((dep, pid))

    global_order = _toposort(ids, edges)
    order_index = {pid: idx for idx, pid in enumerate(global_order)}

    by_type: dict[str, list[str]] = {}
    for pid in global_order:
        t = plugins[pid].type
        by_type.setdefault(t, []).append(pid)

    # Pipeline stage order (auto_plan run): ingest -> transforms -> profile -> planner -> analysis -> report -> llm.
    pipeline_auto_plan = []
    for pid in global_order:
        if plugins[pid].type == "ingest":
            pipeline_auto_plan.append(pid)
    for pid in global_order:
        if plugins[pid].type == "transform":
            pipeline_auto_plan.append(pid)
    for pid in global_order:
        if plugins[pid].type == "profile":
            pipeline_auto_plan.append(pid)
    for pid in global_order:
        if plugins[pid].type == "planner":
            pipeline_auto_plan.append(pid)
    for pid in global_order:
        if plugins[pid].type == "analysis":
            pipeline_auto_plan.append(pid)
    # Report stage: list non-bundle report plugins first, then bundle last (matches pipeline behavior).
    for pid in global_order:
        if plugins[pid].type == "report" and pid != "report_bundle":
            pipeline_auto_plan.append(pid)
    if "report_bundle" in plugins:
        pipeline_auto_plan.append("report_bundle")
    for pid in global_order:
        if plugins[pid].type == "llm":
            pipeline_auto_plan.append(pid)

    # Group analysis plugins into families/packs for human scanning.
    grouped: dict[str, list[str]] = {}
    for pid, meta in plugins.items():
        grouped.setdefault(_group_key(pid, meta.type), []).append(pid)
    grouped = {
        k: sorted(v, key=lambda pid: order_index.get(pid, 10**9))
        for k, v in sorted(grouped.items(), key=lambda kv: kv[0])
    }

    return {
        "plugins_root": "plugins",
        "plugin_count": len(ids),
        "missing_dep_edges": missing_deps,
        "global_toposort": global_order,
        "by_type_in_global_order": by_type,
        "pipeline_auto_plan_order": pipeline_auto_plan,
        "groups": grouped,
        "plugins": {
            pid: {
                "plugin_id": pid,
                "name": plugins[pid].name,
                "version": plugins[pid].version,
                "type": plugins[pid].type,
                "entrypoint": plugins[pid].entrypoint,
                "depends_on": plugins[pid].depends_on,
                "capabilities": plugins[pid].capabilities,
                "description": plugins[pid].description,
            }
            for pid in ids
        },
    }


def to_markdown(matrix: dict[str, Any]) -> str:
    plugins: dict[str, Any] = matrix.get("plugins") or {}
    lines: list[str] = []
    lines.append("# Plugins Functionality Matrix")
    lines.append("")
    lines.append("Generated by `scripts/plugins_functionality_matrix.py`.")
    lines.append("")
    lines.append(f"- Plugin count: {int(matrix.get('plugin_count') or 0)}")
    lines.append("")
    lines.append("## Pipeline Order (Auto-Plan Runs)")
    lines.append("")
    for idx, pid in enumerate(matrix.get("pipeline_auto_plan_order") or [], start=1):
        meta = plugins.get(pid) or {}
        lines.append(f"{idx}. `{pid}` ({meta.get('type')})")
    lines.append("")
    lines.append("## Groups")
    lines.append("")
    groups: dict[str, list[str]] = matrix.get("groups") or {}
    for key, members in groups.items():
        lines.append(f"### `{key}`")
        for pid in members:
            meta = plugins.get(pid) or {}
            deps = meta.get("depends_on") or []
            dep_txt = f" depends_on={deps}" if deps else ""
            lines.append(f"- `{pid}` ({meta.get('type')}){dep_txt}")
        lines.append("")
    lines.append("## Missing Dependencies")
    lines.append("")
    missing = matrix.get("missing_dep_edges") or {}
    if not missing:
        lines.append("None.")
        lines.append("")
    else:
        for pid, deps in sorted(missing.items()):
            lines.append(f"- `{pid}` missing: {', '.join(f'`{d}`' for d in deps)}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugins-root", default="plugins")
    parser.add_argument("--out-json", default="docs/plugins_functionality_matrix.json")
    parser.add_argument("--out-md", default="docs/plugins_functionality_matrix.md")
    parser.add_argument("--verify", action="store_true", help="Exit non-zero if missing deps exist.")
    args = parser.parse_args()

    plugins_root = (ROOT / args.plugins_root).resolve()
    plugins = _load_plugins(plugins_root)
    matrix = build_matrix(plugins)

    out_json = (ROOT / args.out_json).resolve()
    out_md = (ROOT / args.out_md).resolve()
    json_text = json.dumps(matrix, indent=2, sort_keys=True) + "\n"
    md_text = to_markdown(matrix) + "\n"

    if args.verify:
        # Fail closed if dependencies are missing or generated matrices are stale.
        if matrix.get("missing_dep_edges"):
            return 2
        if not out_json.exists() or out_json.read_text(encoding="utf-8") != json_text:
            return 2
        if not out_md.exists() or out_md.read_text(encoding="utf-8") != md_text:
            return 2
        return 0

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json_text, encoding="utf-8")
    out_md.write_text(md_text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
