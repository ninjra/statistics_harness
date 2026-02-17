#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PLUGINS = ROOT / "plugins"
EXCLUDED_GENERATED_NAMES = {
    "repomix-output.md",
    "full_instruction_coverage_report.json",
    "full_instruction_coverage_report.md",
    "full_repo_misses.json",
    "full_repo_misses.md",
}

PATH_RE = re.compile(
    r"(?P<path>"
    r"(?:src|plugins|tests|scripts|docs)/[A-Za-z0-9_./\-]+\.(?:py|md|json|yaml|yml|txt|schema)"
    r"|docs\\\\[A-Za-z0-9_.\\\\\\-]+"
    r")"
)
PLUGIN_LIKE_RE = re.compile(r"\b(?:analysis|ingest|profile|transform|report|planner|llm)(?:_[a-z0-9]+)+\b")
ID_RE = re.compile(r"\b[A-Z]{2,8}-\d{2}\b")
REQ_LINE_RE = re.compile(
    r"(?i)\b(must|required|non-negotiable|always|do not|no network|fail closed|tests? must|should)\b"
)
# Only treat explicit actionable markers as incomplete signals.
# Avoid false positives like "TODO closure" in planning prose.
TODO_RE = re.compile(r"(?i)(^\s*-\s*\[\s\])|\b(TBD|FIXME|not implemented|not-implemented)\b")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="replace")


def _norm(path: str) -> str:
    return path.replace("\\\\", "/").replace("\\", "/").strip()


def _collect_sources() -> list[Path]:
    files: list[Path] = []
    for p in DOCS.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".md", ".txt"}:
            files.append(p)
    for p in ROOT.glob("*.md"):
        files.append(p)
    for name in ("AGENTS.md", "README.md"):
        p = ROOT / name
        if p.exists():
            files.append(p)
    # Exclude generated bundles/reports that can create self-referential drift.
    files = [p for p in files if p.name.lower() not in EXCLUDED_GENERATED_NAMES]
    return sorted({p.resolve() for p in files}, key=lambda x: str(x).lower())


def _plugin_ids() -> set[str]:
    return {p.name for p in PLUGINS.iterdir() if p.is_dir()}


def _load_aliases() -> dict[str, str]:
    path = DOCS / "docs_plugin_aliases.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _load_non_tokens() -> set[str]:
    path = DOCS / "docs_non_plugin_tokens.json"
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    items = data.get("tokens") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return set()
    out: set[str] = set()
    for item in items:
        if isinstance(item, str):
            out.add(item)
    return out


@dataclass(frozen=True)
class Item:
    source: str
    line_no: int
    text: str
    item_type: str
    requirement_ids: list[str]
    referenced_paths: list[str]
    missing_paths: list[str]
    plugin_tokens: list[str]
    unresolved_plugin_tokens: list[str]
    has_todo_marker: bool


def _extract_items(path: Path, plugin_ids: set[str], aliases: dict[str, str], non_tokens: set[str]) -> list[Item]:
    text = _read_text(path)
    source = str(path.relative_to(ROOT)).replace("\\", "/")
    items: list[Item] = []
    for idx, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        is_bullet = line.startswith("- ") or line.startswith("* ") or bool(re.match(r"^\d+\.\s+", line))
        item_type = ""
        if TODO_RE.search(line):
            item_type = "todo_marker"
        elif ID_RE.search(line):
            item_type = "requirement_id"
        elif is_bullet and REQ_LINE_RE.search(line):
            item_type = "requirement_text"
        elif is_bullet and "`" in line:
            # Path/plugin refs often live in list bullets even without strict keywords.
            item_type = "reference_text"
        if not item_type:
            continue

        refs = sorted({_norm(m.group("path")) for m in PATH_RE.finditer(line)})
        missing = [r for r in refs if not (ROOT / r).exists()]
        tokens = sorted({m.group(0) for m in PLUGIN_LIKE_RE.finditer(line)})
        unresolved: list[str] = []
        for tok in tokens:
            if tok in non_tokens:
                continue
            if tok in plugin_ids:
                continue
            alias = aliases.get(tok)
            if isinstance(alias, str) and alias in plugin_ids:
                continue
            unresolved.append(tok)
        ids = sorted({m.group(0) for m in ID_RE.finditer(line)})
        items.append(
            Item(
                source=source,
                line_no=idx,
                text=line,
                item_type=item_type,
                requirement_ids=ids,
                referenced_paths=refs,
                missing_paths=missing,
                plugin_tokens=tokens,
                unresolved_plugin_tokens=unresolved,
                has_todo_marker=bool(TODO_RE.search(line)),
            )
        )
    return items


def _load_matrix_status() -> dict[str, Any]:
    def load(name: str) -> dict[str, Any]:
        p = DOCS / name
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    docs_cov = load("implementation_matrix.json")
    binding = load("binding_implementation_matrix.json")
    redteam = load("redteam_ids_matrix.json")
    data_access = load("plugin_data_access_matrix.json")
    sql_adopt = load("sql_assist_adoption_matrix.json")
    plugin_func = load("plugins_functionality_matrix.json")

    required_sql_misses = [
        p.get("plugin_id")
        for p in (sql_adopt.get("plugins") or [])
        if isinstance(p, dict)
        and p.get("sql_intent") == "required"
        and not bool(p.get("uses_sql") or p.get("uses_sql_exec"))
    ]
    unclassified = [
        p.get("plugin_id")
        for p in (data_access.get("plugins") or [])
        if isinstance(p, dict) and bool(p.get("unclassified"))
    ]
    return {
        "docs_missing_any_normative": bool(docs_cov.get("missing_any_normative")),
        "binding_missing_any": bool(binding.get("missing_any")),
        "redteam_missing_required_ids": redteam.get("missing_required_ids") or [],
        "plugin_functionality_missing_dep_edges": plugin_func.get("missing_dep_edges") or {},
        "plugin_data_access_unclassified": sorted([x for x in unclassified if isinstance(x, str)]),
        "sql_required_not_using_sql": sorted([x for x in required_sql_misses if isinstance(x, str)]),
    }


def _to_payload(items: list[Item], matrix_status: dict[str, Any]) -> dict[str, Any]:
    missing_items = [
        i
        for i in items
        if i.missing_paths or i.unresolved_plugin_tokens or i.has_todo_marker
    ]
    by_source: dict[str, int] = {}
    for item in missing_items:
        by_source[item.source] = by_source.get(item.source, 0) + 1
    return {
        "generated_by": "scripts/full_instruction_coverage_report.py",
        "source_count": len(sorted({i.source for i in items})),
        "item_count": len(items),
        "missing_item_count": len(missing_items),
        "matrix_status": matrix_status,
        "missing_by_source": dict(sorted(by_source.items(), key=lambda kv: (-kv[1], kv[0]))),
        "items": [
            {
                "source": i.source,
                "line_no": i.line_no,
                "item_type": i.item_type,
                "text": i.text,
                "requirement_ids": i.requirement_ids,
                "referenced_paths": i.referenced_paths,
                "missing_paths": i.missing_paths,
                "plugin_tokens": i.plugin_tokens,
                "unresolved_plugin_tokens": i.unresolved_plugin_tokens,
                "has_todo_marker": i.has_todo_marker,
            }
            for i in missing_items
        ],
    }


def _to_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Full Instruction Coverage Report")
    lines.append("")
    lines.append(f"- Source files scanned: {payload.get('source_count')}")
    lines.append(f"- Requirement/reference items scanned: {payload.get('item_count')}")
    lines.append(f"- Missing/incomplete items detected: {payload.get('missing_item_count')}")
    lines.append("")
    lines.append("## Matrix Status")
    m = payload.get("matrix_status") or {}
    for k in sorted(m.keys()):
        lines.append(f"- {k}: {m[k]}")
    lines.append("")
    lines.append("## Missing Items")
    lines.append("| source | line | type | missing_paths | unresolved_plugins | todo_marker |")
    lines.append("|---|---:|---|---:|---:|---:|")
    for item in payload.get("items") or []:
        lines.append(
            f"| `{item.get('source')}` | {item.get('line_no')} | {item.get('item_type')} | "
            f"{len(item.get('missing_paths') or [])} | {len(item.get('unresolved_plugin_tokens') or [])} | "
            f"{1 if item.get('has_todo_marker') else 0} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default="docs/full_instruction_coverage_report.json")
    ap.add_argument("--out-md", default="docs/full_instruction_coverage_report.md")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    plugin_ids = _plugin_ids()
    aliases = _load_aliases()
    non_tokens = _load_non_tokens()
    items: list[Item] = []
    for src in _collect_sources():
        items.extend(_extract_items(src, plugin_ids, aliases, non_tokens))
    payload = _to_payload(items, _load_matrix_status())

    out_json = (ROOT / args.out_json).resolve()
    out_md = (ROOT / args.out_md).resolve()
    json_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    md_text = _to_markdown(payload)

    if args.verify:
        if not out_json.exists() or out_json.read_text(encoding="utf-8") != json_text:
            return 2
        if not out_md.exists() or out_md.read_text(encoding="utf-8") != md_text:
            return 2
        return 0

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json_text, encoding="utf-8")
    out_md.write_text(md_text, encoding="utf-8")
    print(f"out_json={out_json}")
    print(f"out_md={out_md}")
    print(f"missing_item_count={payload.get('missing_item_count')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
