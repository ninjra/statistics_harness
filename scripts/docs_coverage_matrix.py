from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


# "Plugin-like" token matcher for docs. Require tokens to end with an alnum so we
# don't mis-detect group labels like `analysis_close_cycle_*` as `analysis_close_cycle_`.
PLUGIN_LIKE_RE = re.compile(
    r"\b(?:analysis|ingest|profile|transform|report|planner|llm)(?:_[a-z0-9]+)+\b"
)
PATH_RE = re.compile(
    r"(?P<path>"
    r"(?:src|plugins|tests|scripts|docs)/[A-Za-z0-9_./\-]+\.(?:py|md|json|yaml|yml|txt)"
    r"|docs\\\\[A-Za-z0-9_.\\\\\\-]+"
    r")"
)

GENERATED_DOC_PATHS = {
    "docs/implementation_matrix.json",
    "docs/implementation_matrix.md",
    "docs/binding_implementation_matrix.json",
    "docs/binding_implementation_matrix.md",
    "docs/plugins_functionality_matrix.json",
    "docs/plugins_functionality_matrix.md",
    "docs/redteam_ids_matrix.json",
    "docs/redteam_ids_matrix.md",
    "docs/full_instruction_coverage_report.json",
    "docs/full_instruction_coverage_report.md",
    "docs/full_repo_misses.json",
    "docs/full_repo_misses.md",
    "docs/repo_improvements_catalog_v3.json",
    "docs/repo_improvements_catalog_v3.canonical.json",
    "docs/repo_improvements_catalog_v3.reduction_report.json",
    "docs/repo_improvements_catalog_v3.normalized.json",
    "docs/repo_improvements_capability_map_v1.json",
    "docs/repo_improvements_execution_plan_v1.json",
    "docs/repo_improvements_execution_plan_v1.md",
    "docs/repo_improvements_dependency_validation_v1.json",
    "docs/repo_improvements_scaffold_plan_v1.json",
    "docs/repo_improvements_status.json",
    "docs/repo_improvements_status.md",
    "docs/plugin_class_actionability_matrix.json",
    "docs/plugin_class_actionability_matrix.md",
    "docs/plugin_example_cards/index.json",
    "docs/plugin_example_cards/direct_action_generators.md",
    "docs/plugin_example_cards/ingest_profile_transform.md",
    "docs/plugin_example_cards/reporting_llm_post_processing.md",
    "docs/plugin_example_cards/supporting_signal_detectors.md",
    "docs/plugin_example_cards/synthesis_verification.md",
    "docs/_codex_repo_manifest.txt",
    "docs/_codex_plugin_catalog.md",
    "docs/codex_statistics_harness_blueprint.md",
}


@dataclass(frozen=True)
class DocScan:
    doc_path: str
    doc_kind: str
    plugin_ids: list[str]
    unresolved_plugin_tokens: list[str]
    referenced_paths: list[str]
    missing_paths: list[str]


def _iter_docs(docs_root: Path) -> list[Path]:
    files = [p for p in docs_root.rglob("*") if p.is_file()]
    return sorted(files, key=lambda p: str(p).lower())


def _read_text_best_effort(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="replace")


def _normalize_doc_rel_path(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _normalize_ref_path(text_path: str) -> str:
    # Normalize windows-style docs\foo\bar.md -> docs/foo/bar.md
    ref = text_path.strip()
    ref = ref.replace("\\\\", "/")
    return ref.replace("\\", "/")


def _extract_plugin_like_tokens(text: str) -> list[str]:
    return sorted({m.group(0) for m in PLUGIN_LIKE_RE.finditer(text)})


def _extract_paths(text: str) -> list[str]:
    refs: set[str] = set()
    for match in PATH_RE.finditer(text):
        refs.add(_normalize_ref_path(match.group("path")))
    return sorted(refs)


def _plugin_ids_on_disk(plugins_root: Path) -> set[str]:
    ids: set[str] = set()
    for child in plugins_root.iterdir():
        if child.is_dir():
            ids.add(child.name)
    return ids


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _path_exists(ref: str) -> bool:
    # Only accept workspace-relative references.
    cand = ROOT / ref
    return cand.exists()


def scan_docs(docs_root: Path, plugins_root: Path) -> list[DocScan]:
    scans: list[DocScan] = []
    plugin_ids = _plugin_ids_on_disk(plugins_root)
    alias_path = docs_root / "docs_plugin_aliases.json"
    non_plugin_path = docs_root / "docs_non_plugin_tokens.json"
    aliases = _load_json(alias_path)
    non_plugin_tokens = set((_load_json(non_plugin_path).get("tokens") or []))
    for doc in _iter_docs(docs_root):
        doc_rel = _normalize_doc_rel_path(doc)
        if doc_rel.startswith("docs/deprecated/"):
            doc_kind = "archived"
        elif doc_rel in GENERATED_DOC_PATHS:
            doc_kind = "generated"
        else:
            doc_kind = "normative"
        # Avoid recursive drift: generated matrix docs contain many workspace-like
        # paths in their JSON bodies, which would cause the matrix to depend on itself.
        if doc_kind == "generated":
            tokens = []
            ref_paths = []
        else:
            text = _read_text_best_effort(doc)
            tokens = _extract_plugin_like_tokens(text)
            ref_paths = _extract_paths(text)

        resolved: set[str] = set()
        unresolved: set[str] = set()
        for tok in tokens:
            if tok in non_plugin_tokens:
                continue
            if tok in plugin_ids:
                resolved.add(tok)
                continue
            mapped = aliases.get(tok)
            if isinstance(mapped, str) and mapped in plugin_ids:
                resolved.add(mapped)
                continue
            unresolved.add(tok)

        missing_paths = [ref for ref in ref_paths if not _path_exists(ref)]
        scans.append(
            DocScan(
                doc_path=doc_rel,
                doc_kind=doc_kind,
                plugin_ids=sorted(resolved),
                unresolved_plugin_tokens=sorted(unresolved),
                referenced_paths=ref_paths,
                missing_paths=missing_paths,
            )
        )
    return scans


def as_json(scans: list[DocScan]) -> dict[str, Any]:
    missing_any_normative = any(
        (s.doc_kind == "normative") and (s.unresolved_plugin_tokens or s.missing_paths)
        for s in scans
    )
    missing_any = any(s.unresolved_plugin_tokens or s.missing_paths for s in scans)
    return {
        "docs_root": "docs",
        "scanned_docs": [s.doc_path for s in scans],
        "missing_any": bool(missing_any),
        "missing_any_normative": bool(missing_any_normative),
        "documents": [
            {
                "doc_path": s.doc_path,
                "doc_kind": s.doc_kind,
                "plugin_ids": s.plugin_ids,
                "unresolved_plugin_tokens": s.unresolved_plugin_tokens,
                "referenced_paths": s.referenced_paths,
                "missing_paths": s.missing_paths,
            }
            for s in scans
        ],
    }


def as_markdown(scans: list[DocScan]) -> str:
    lines: list[str] = []
    lines.append("# Docs Coverage Matrix")
    lines.append("")
    lines.append("This file is generated by `scripts/docs_coverage_matrix.py`.")
    lines.append("")
    lines.append("| Document | Kind | Resolved Plugin IDs | Unresolved Tokens | Referenced Paths | Missing Paths |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for s in scans:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{s.doc_path}`",
                    s.doc_kind,
                    str(len(s.plugin_ids)),
                    str(len(s.unresolved_plugin_tokens)),
                    str(len(s.referenced_paths)),
                    str(len(s.missing_paths)),
                ]
            )
            + " |"
        )
    lines.append("")

    problems = [s for s in scans if s.unresolved_plugin_tokens or s.missing_paths]
    if problems:
        lines.append("## Missing References")
        lines.append("")
        for s in problems:
            lines.append(f"### `{s.doc_path}`")
            if s.unresolved_plugin_tokens:
                lines.append("- Unresolved plugin-like tokens (not present under `plugins/` and not aliased):")
                for tok in s.unresolved_plugin_tokens:
                    lines.append(f"  - `{tok}`")
            if s.missing_paths:
                lines.append("- Missing file paths:")
                for ref in s.missing_paths:
                    lines.append(f"  - `{ref}`")
            lines.append("")
    else:
        lines.append("## Missing References")
        lines.append("")
        lines.append("None. All referenced plugin IDs and file paths exist.")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-root", default="docs")
    parser.add_argument("--plugins-root", default="plugins")
    parser.add_argument("--out-json", default="docs/implementation_matrix.json")
    parser.add_argument("--out-md", default="docs/implementation_matrix.md")
    parser.add_argument("--verify", action="store_true", help="Exit non-zero if missing refs exist.")
    args = parser.parse_args()

    docs_root = (ROOT / args.docs_root).resolve()
    plugins_root = (ROOT / args.plugins_root).resolve()
    scans = scan_docs(docs_root, plugins_root)

    payload = as_json(scans)
    out_json = (ROOT / args.out_json).resolve()
    out_md = (ROOT / args.out_md).resolve()

    json_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    md_text = as_markdown(scans) + "\n"

    if args.verify:
        if payload.get("missing_any_normative"):
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
