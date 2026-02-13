#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
PLUGINS_ROOT = ROOT / "plugins"
DEFAULT_OUT = ROOT / "docs" / "_codex_plugin_catalog.md"


@dataclass(frozen=True)
class Row:
    plugin_id: str
    plugin_type: str
    name: str
    entrypoint: str
    depends_on: list[str]
    capabilities: list[str]


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def build_rows(plugins_root: Path) -> list[Row]:
    rows: list[Row] = []
    for pdir in sorted([p for p in plugins_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        manifest = pdir / "plugin.yaml"
        if not manifest.exists():
            continue
        data = _load_yaml(manifest)
        pid = str(data.get("id") or pdir.name)
        ptype = str(data.get("type") or "")
        rows.append(
            Row(
                plugin_id=pid,
                plugin_type=ptype,
                name=str(data.get("name") or pid),
                entrypoint=str(data.get("entrypoint") or ""),
                depends_on=sorted([str(x) for x in (data.get("depends_on") or []) if isinstance(x, str)]),
                capabilities=sorted([str(x) for x in (data.get("capabilities") or []) if isinstance(x, str)]),
            )
        )
    return rows


def build_markdown(rows: list[Row]) -> str:
    by_type: dict[str, list[Row]] = {}
    for row in rows:
        by_type.setdefault(row.plugin_type or "unknown", []).append(row)
    lines: list[str] = []
    lines.append("# Codex Plugin Catalog")
    lines.append("")
    lines.append(f"- Total plugins: {len(rows)}")
    lines.append("")
    for ptype in sorted(by_type):
        items = sorted(by_type[ptype], key=lambda r: r.plugin_id)
        lines.append(f"## {ptype}")
        lines.append("")
        lines.append("| plugin_id | name | entrypoint | depends_on | capabilities |")
        lines.append("|---|---|---|---:|---:|")
        for row in items:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{row.plugin_id}`",
                        row.name.replace("|", "/"),
                        f"`{row.entrypoint}`",
                        str(len(row.depends_on)),
                        str(len(row.capabilities)),
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugins-root", type=Path, default=PLUGINS_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    rows = build_rows(args.plugins_root.resolve())
    text = build_markdown(rows)
    out = args.out.resolve()

    if args.verify:
        if not out.exists():
            return 2
        return 0 if out.read_text(encoding="utf-8") == text else 2

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"out={out}")
    print(f"plugins={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
