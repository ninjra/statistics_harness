from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


# "Plugin-like" token matcher for instruction docs. Require tokens to end with an
# alnum so we don't mis-detect group labels like `analysis_close_cycle_*`.
PLUGIN_LIKE_RE = re.compile(
    r"\b(?:analysis|ingest|profile|transform|report|planner|llm)(?:_[a-z0-9]+)+\b"
)

# Extract explicit workspace-relative file references inside docs.
PATH_RE = re.compile(
    r"(?P<path>"
    r"(?:src|plugins|tests|scripts|docs)/[A-Za-z0-9_./\-]+\.(?:py|md|json|yaml|yml|txt)"
    r"|docs\\\\[A-Za-z0-9_.\\\\\\-]+"
    r")"
)

# Common recommendation IDs in instruction docs, e.g. FND-01, META-07.
REQ_ID_RE = re.compile(r"\b[A-Z]{2,8}-\d{2}\b")
TABLE_ROW_RE = re.compile(r"^\|\s*(?P<id>[A-Z]{2,8}-\d{2})\s*\|\s*(?P<body>.*)\|\s*$")

EXCLUDED_BINDING_DOC_PATHS = {
    "docs/implementation_matrix.json",
    "docs/implementation_matrix.md",
    "docs/plugins_functionality_matrix.json",
    "docs/plugins_functionality_matrix.md",
    "docs/binding_implementation_matrix.json",
    "docs/binding_implementation_matrix.md",
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
    "docs/plugins-validate-runbook-actionable-insights-plan.md",
    "docs/openplanter-cross-dataset-plugins-plan.md",
}

EXCLUDED_ROOT_PLAN_DOC_NAMES = {
    "repo-improvements-catalog-v3-implementation-path-plan.md",
}


@dataclass(frozen=True)
class DocScan:
    doc_path: str
    doc_kind: str
    plugin_ids: list[str]
    unresolved_plugin_tokens: list[str]
    referenced_paths: list[str]
    missing_paths: list[str]
    requirement_ids: list[str]
    requirement_enforcement_paths: dict[str, list[str]]
    missing_requirement_enforcement_paths: dict[str, list[str]]


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


def _extract_requirement_ids(text: str) -> list[str]:
    return sorted({m.group(0) for m in REQ_ID_RE.finditer(text)})


def _plugin_ids_on_disk(plugins_root: Path) -> set[str]:
    return {p.name for p in plugins_root.iterdir() if p.is_dir()}


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


def _extract_requirement_enforcement_paths(
    doc_text: str,
) -> dict[str, list[str]]:
    """Best-effort map of requirement IDs -> referenced enforcement file paths.

    This is primarily to validate that instruction docs point to real code paths
    (and not stale/typoed ones). It does NOT attempt to validate the semantics.
    """

    enforcement: dict[str, set[str]] = {}
    for line in doc_text.splitlines():
        m = TABLE_ROW_RE.match(line.strip())
        if not m:
            continue
        req_id = m.group("id")
        body = m.group("body")
        if "Enforce:" not in body:
            continue
        paths = _extract_paths(body)
        if paths:
            enforcement.setdefault(req_id, set()).update(paths)
    return {k: sorted(v) for k, v in enforcement.items()}


def _iter_binding_docs(
    docs_root: Path,
    extra_docs: list[Path],
    exclude_paths: set[str],
) -> list[Path]:
    files: list[Path] = []
    for p in docs_root.rglob("*"):
        if not p.is_file():
            continue
        rel = _normalize_doc_rel_path(p)
        # Archived docs are historical context and should not gate active binding coverage.
        if rel.startswith("docs/deprecated/"):
            continue
        # Release evidence artifacts are run outputs, not binding spec docs.
        if rel.startswith("docs/release_evidence/"):
            continue
        if rel in exclude_paths:
            continue
        files.append(p)
    for p in extra_docs:
        if p.is_file():
            files.append(p)
    return sorted({p.resolve() for p in files}, key=lambda p: str(p).lower())


def _iter_root_plan_docs(root: Path) -> list[Path]:
    # Per user direction: include all repo-root *-plan.md files as binding inputs.
    return sorted(
        [p for p in root.glob("*-plan.md") if p.is_file() and p.name not in EXCLUDED_ROOT_PLAN_DOC_NAMES],
        key=lambda p: str(p).lower(),
    )


def scan_binding_docs(
    docs_root: Path,
    plugins_root: Path,
    extra_docs: list[Path] | None = None,
) -> list[DocScan]:
    scans: list[DocScan] = []
    plugin_ids = _plugin_ids_on_disk(plugins_root)

    alias_path = docs_root / "docs_plugin_aliases.json"
    non_plugin_path = docs_root / "docs_non_plugin_tokens.json"
    aliases = _load_json(alias_path)
    non_plugin_tokens = set((_load_json(non_plugin_path).get("tokens") or []))

    exclude_paths = set(EXCLUDED_BINDING_DOC_PATHS)

    extra_docs = list(extra_docs or [])
    extra_docs.extend(_iter_root_plan_docs(ROOT))
    for doc in _iter_binding_docs(docs_root, extra_docs, exclude_paths):
        text = _read_text_best_effort(doc)
        doc_rel = _normalize_doc_rel_path(doc)
        doc_kind = "binding"

        tokens = _extract_plugin_like_tokens(text)
        ref_paths = _extract_paths(text)
        req_ids = _extract_requirement_ids(text)
        req_enforce = _extract_requirement_enforcement_paths(text)

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

        missing_req_enforce: dict[str, list[str]] = {}
        for req_id, paths in req_enforce.items():
            missing = [p for p in paths if not _path_exists(p)]
            if missing:
                missing_req_enforce[req_id] = missing

        scans.append(
            DocScan(
                doc_path=doc_rel,
                doc_kind=doc_kind,
                plugin_ids=sorted(resolved),
                unresolved_plugin_tokens=sorted(unresolved),
                referenced_paths=ref_paths,
                missing_paths=missing_paths,
                requirement_ids=req_ids,
                requirement_enforcement_paths=req_enforce,
                missing_requirement_enforcement_paths=missing_req_enforce,
            )
        )
    return scans


def as_json(scans: list[DocScan]) -> dict[str, Any]:
    missing_any = any(
        s.unresolved_plugin_tokens
        or s.missing_paths
        or s.missing_requirement_enforcement_paths
        for s in scans
    )
    return {
        "instruction_roots": ["docs", "repo-root *-plan.md"],
        "scanned_docs": [s.doc_path for s in scans],
        "missing_any": bool(missing_any),
        "documents": [
            {
                "doc_path": s.doc_path,
                "doc_kind": s.doc_kind,
                "plugin_ids": s.plugin_ids,
                "unresolved_plugin_tokens": s.unresolved_plugin_tokens,
                "referenced_paths": s.referenced_paths,
                "missing_paths": s.missing_paths,
                "requirement_ids": s.requirement_ids,
                "requirement_enforcement_paths": s.requirement_enforcement_paths,
                "missing_requirement_enforcement_paths": s.missing_requirement_enforcement_paths,
            }
            for s in scans
        ],
    }


def as_markdown(scans: list[DocScan]) -> str:
    lines: list[str] = []
    lines.append("# Binding Implementation Matrix")
    lines.append("")
    lines.append("This file is generated by `scripts/binding_implementation_matrix.py`.")
    lines.append("")
    lines.append("| Document | Plugin IDs | Unresolved Tokens | Referenced Paths | Missing Paths | Requirement IDs | Missing Enforce Paths |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for s in scans:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{s.doc_path}`",
                    str(len(s.plugin_ids)),
                    str(len(s.unresolved_plugin_tokens)),
                    str(len(s.referenced_paths)),
                    str(len(s.missing_paths)),
                    str(len(s.requirement_ids)),
                    str(len(s.missing_requirement_enforcement_paths)),
                ]
            )
            + " |"
        )
    lines.append("")

    problems = [
        s
        for s in scans
        if s.unresolved_plugin_tokens
        or s.missing_paths
        or s.missing_requirement_enforcement_paths
    ]
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
            if s.missing_requirement_enforcement_paths:
                lines.append("- Missing enforcement paths (referenced in requirement rows):")
                for req_id in sorted(s.missing_requirement_enforcement_paths.keys()):
                    missing = s.missing_requirement_enforcement_paths[req_id]
                    lines.append(f"  - `{req_id}`: {', '.join(f'`{p}`' for p in missing)}")
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
    parser.add_argument(
        "--extra-doc",
        action="append",
        default=["topo-tda-addon-pack-plan.md"],
        help="Workspace-relative paths to include in the binding scan.",
    )
    parser.add_argument("--out-json", default="docs/binding_implementation_matrix.json")
    parser.add_argument("--out-md", default="docs/binding_implementation_matrix.md")
    parser.add_argument("--verify", action="store_true", help="Exit non-zero if missing refs exist.")
    args = parser.parse_args()

    docs_root = (ROOT / args.docs_root).resolve()
    plugins_root = (ROOT / args.plugins_root).resolve()
    extra_docs = [(ROOT / p).resolve() for p in (args.extra_doc or [])]

    scans = scan_binding_docs(docs_root, plugins_root, extra_docs=extra_docs)
    payload = as_json(scans)

    out_json = (ROOT / args.out_json).resolve()
    out_md = (ROOT / args.out_md).resolve()
    json_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    md_text = as_markdown(scans) + "\n"

    if args.verify:
        if payload.get("missing_any"):
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
