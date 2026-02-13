from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]

REQ_ID_RE = re.compile(r"\b[A-Z]{2,8}-\d{2}\b")
TABLE_ROW_RE = re.compile(r"^\|\s*(?P<id>[A-Z]{2,8}-\d{2})\s*\|\s*(?P<body>.*)\|\s*$")
BACKTICK_RE = re.compile(r"`([^`]+)`")
ENV_RE = re.compile(r"\bSTAT_HARNESS_[A-Z0-9_]+\b")
CLI_RE = re.compile(r"`stat-harness\s+([a-z0-9_-]+)(?:\s|`)")

# Workspace-relative file references inside docs.
PATH_RE = re.compile(
    r"(?P<path>"
    r"(?:src|plugins|tests|scripts|docs)/[A-Za-z0-9_./\-]+\.(?:py|md|json|yaml|yml|txt)"
    r"|docs\\\\[A-Za-z0-9_.\\\\\\-]+"
    r")"
)

SYMBOL_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\(\)")


@dataclass(frozen=True)
class RequirementRow:
    req_id: str
    doc_path: str
    line_no: int
    raw_line: str
    body: str


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


def _extract_paths(text: str) -> list[str]:
    refs: set[str] = set()
    for match in PATH_RE.finditer(text):
        refs.add(_normalize_ref_path(match.group("path")))
    return sorted(refs)


def _extract_symbols(text: str) -> list[str]:
    # Prefer explicitly backticked calls (e.g. `atomic_dir()`).
    symbols: set[str] = set()
    for tok in BACKTICK_RE.findall(text):
        m = SYMBOL_CALL_RE.fullmatch(tok.strip())
        if m:
            symbols.add(m.group(1))
    return sorted(symbols)


def _extract_env_vars(text: str) -> list[str]:
    return sorted({m.group(0) for m in ENV_RE.finditer(text)})


def _extract_cli_subcommands(text: str) -> list[str]:
    return sorted({m.group(1) for m in CLI_RE.finditer(text)})


def _iter_requirement_rows(doc_path: Path) -> list[RequirementRow]:
    rel = _normalize_doc_rel_path(doc_path)
    out: list[RequirementRow] = []
    text = _read_text_best_effort(doc_path)
    for idx, line in enumerate(text.splitlines(), start=1):
        m = TABLE_ROW_RE.match(line.strip())
        if not m:
            continue
        req_id = m.group("id")
        body = m.group("body")
        out.append(
            RequirementRow(
                req_id=req_id,
                doc_path=rel,
                line_no=idx,
                raw_line=line,
                body=body,
            )
        )
    return out


def _iter_requirement_mentions(doc_path: Path) -> set[str]:
    text = _read_text_best_effort(doc_path)
    return {m.group(0) for m in REQ_ID_RE.finditer(text)}


def _path_exists(ref: str) -> bool:
    cand = ROOT / ref
    return cand.exists()


def _iter_python_files(roots: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.py"):
            if p.is_file():
                files.append(p)
    return sorted({p.resolve() for p in files}, key=lambda p: str(p).lower())


def _load_text_index(paths: list[Path]) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in paths:
        rel = _normalize_doc_rel_path(p)
        out[rel] = _read_text_best_effort(p)
    return out


def _symbol_exists(symbol: str, text_index: dict[str, str]) -> bool:
    # Best-effort check for a Python-level definition somewhere in src/plugins/scripts/tests.
    pat = re.compile(rf"^\s*(?:def|class)\s+{re.escape(symbol)}\b", re.MULTILINE)
    return any(bool(pat.search(txt)) for txt in text_index.values())


def _cli_subcommand_exists(subcmd: str, cli_text: str) -> bool:
    # Conservative: only count if it is registered as a subparser.
    return f'add_parser("{subcmd}")' in cli_text or f"add_parser('{subcmd}')" in cli_text


def _collect_test_evidence(
    candidates: list[str],
    test_text_index: dict[str, str],
) -> list[str]:
    if not candidates:
        return []
    out: list[str] = []
    for path, txt in test_text_index.items():
        if any(c and (c in txt) for c in candidates):
            out.append(path)
    return sorted(out)


def build_matrix() -> dict[str, Any]:
    sources = [
        ROOT / "docs" / "stat_redteam2-6-26.md",
        ROOT / "docs" / "deprecated" / "redteam2-5-2026.txt",
    ]

    rows: list[RequirementRow] = []
    all_mentions: dict[str, dict[str, Any]] = {}
    for src in sources:
        for r in _iter_requirement_rows(src):
            rows.append(r)
        mentions = _iter_requirement_mentions(src)
        all_mentions[_normalize_doc_rel_path(src)] = {
            "doc_path": _normalize_doc_rel_path(src),
            "requirement_ids": sorted(mentions),
        }

    req_ids = sorted({r.req_id for r in rows} | {rid for v in all_mentions.values() for rid in v["requirement_ids"]})

    # Code/test indices for evidence.
    python_paths = _iter_python_files([ROOT / "src", ROOT / "plugins", ROOT / "scripts", ROOT / "tests"])
    text_index = _load_text_index(python_paths)
    cli_text = text_index.get("src/statistic_harness/cli.py", "")
    test_text_index = {k: v for k, v in text_index.items() if k.startswith("tests/")}

    by_id_rows: dict[str, list[RequirementRow]] = {}
    for r in rows:
        by_id_rows.setdefault(r.req_id, []).append(r)

    items: list[dict[str, Any]] = []
    for rid in req_ids:
        rid_rows = sorted(by_id_rows.get(rid, []), key=lambda r: (r.doc_path, r.line_no))
        required = any(not r.doc_path.startswith("docs/deprecated/") for r in rid_rows) or any(
            (not p.startswith("docs/deprecated/")) and (rid in (all_mentions.get(p) or {}).get("requirement_ids", []))
            for p in all_mentions
        )

        declared_paths: set[str] = set()
        declared_symbols: set[str] = set()
        declared_env: set[str] = set()
        declared_cli: set[str] = set()
        for r in rid_rows:
            declared_paths.update(_extract_paths(r.raw_line))
            declared_symbols.update(_extract_symbols(r.raw_line))
            declared_env.update(_extract_env_vars(r.raw_line))
            declared_cli.update(_extract_cli_subcommands(r.raw_line))

        missing_paths = sorted([p for p in declared_paths if not _path_exists(p)])
        missing_symbols = sorted([s for s in declared_symbols if not _symbol_exists(s, text_index)])
        missing_env = sorted([e for e in declared_env if not any(e in txt for txt in text_index.values())])
        missing_cli = sorted([c for c in declared_cli if not _cli_subcommand_exists(c, cli_text)])

        # Evidence tests: any test containing one of the core tokens.
        candidates: list[str] = [rid]
        candidates.extend(sorted(declared_symbols))
        candidates.extend(sorted(declared_env))
        candidates.extend(sorted(declared_cli))
        candidates.extend([Path(p).name for p in sorted(declared_paths)])
        test_evidence = _collect_test_evidence(candidates, test_text_index)

        if required and (missing_paths or missing_symbols or missing_env or missing_cli):
            status = "missing"
        elif test_evidence:
            status = "implemented"
        else:
            status = "partial"

        items.append(
            {
                "id": rid,
                "required": bool(required),
                "status": status,
                "declared": {
                    "paths": sorted(declared_paths),
                    "symbols": sorted(declared_symbols),
                    "env_vars": sorted(declared_env),
                    "cli_subcommands": sorted(declared_cli),
                },
                "missing": {
                    "paths": missing_paths,
                    "symbols": missing_symbols,
                    "env_vars": missing_env,
                    "cli_subcommands": missing_cli,
                },
                "evidence": {
                    "test_paths": test_evidence,
                },
                "sources": [
                    {
                        "doc_path": r.doc_path,
                        "line_no": r.line_no,
                    }
                    for r in rid_rows
                ],
            }
        )

    missing_required = [x["id"] for x in items if x["required"] and x["status"] == "missing"]
    return {
        "generated_by": "scripts/redteam_ids_matrix.py",
        "sources": [str(_normalize_doc_rel_path(p)) for p in sources],
        "requirement_count": len(items),
        "missing_required_ids": missing_required,
        "items": items,
    }


def to_markdown(matrix: dict[str, Any]) -> str:
    items = matrix.get("items") or []
    lines: list[str] = []
    lines.append("# Redteam IDs Functional Completeness Matrix")
    lines.append("")
    lines.append("Generated by `scripts/redteam_ids_matrix.py`.")
    lines.append("")
    lines.append(f"- Requirement IDs: {int(matrix.get('requirement_count') or 0)}")
    lines.append(f"- Missing required IDs: {len(matrix.get('missing_required_ids') or [])}")
    lines.append("")
    lines.append("| ID | Required | Status | Declared Paths | Missing (paths/symbols/env/cli) | Test Evidence |")
    lines.append("|---|---:|---|---:|---:|---:|")
    for it in items:
        missing = it.get("missing") or {}
        missing_any = sum(
            len(missing.get(k) or [])
            for k in ("paths", "symbols", "env_vars", "cli_subcommands")
        )
        declared_paths = len(((it.get("declared") or {}).get("paths")) or [])
        test_paths = len(((it.get("evidence") or {}).get("test_paths")) or [])
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{it.get('id')}`",
                    "yes" if it.get("required") else "no",
                    str(it.get("status") or ""),
                    str(declared_paths),
                    str(missing_any),
                    str(test_paths),
                ]
            )
            + " |"
        )
    lines.append("")
    if matrix.get("missing_required_ids"):
        lines.append("## Missing Required IDs")
        lines.append("")
        for rid in matrix["missing_required_ids"]:
            lines.append(f"- `{rid}`")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", default="docs/redteam_ids_matrix.json")
    parser.add_argument("--out-md", default="docs/redteam_ids_matrix.md")
    parser.add_argument("--verify", action="store_true", help="Exit non-zero if any required IDs are missing.")
    args = parser.parse_args()

    matrix = build_matrix()
    out_json = (ROOT / args.out_json).resolve()
    out_md = (ROOT / args.out_md).resolve()
    json_text = json.dumps(matrix, indent=2, sort_keys=True) + "\n"
    md_text = to_markdown(matrix) + "\n"

    if args.verify:
        if matrix.get("missing_required_ids"):
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
