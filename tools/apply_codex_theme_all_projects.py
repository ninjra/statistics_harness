#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


THEME_HEADING = "## Codex CLI Theme (Hard Gate)"
THEME_BLOCK = """## Codex CLI Theme (Hard Gate)
- Applies to all assistant prose in Codex CLI for this repo (not only report files).
- Use soft cyberpunk ANSI palette:
  - Header: `38;5;177`
  - Label/key: `38;5;111`
  - Value: `38;5;150`
  - Accounting-month value: `38;5;117`
  - Close-static value: `38;5;81`
  - Close-dynamic value: `38;5;183`
  - Dim/supporting text: `90`
- Separators must be bright white (`97`) and visually emphasized:
  - `/` in triplets (`acct/static/dyn`)
  - `;` between metadata items
  - `=` in key/value pairs (`x=y`)
- `x=y` must render with key and value in different colors; `=` must be bright white.
- If `NO_COLOR` is set or output is non-TTY, fall back to plain text while preserving the same structure.
"""


def _repo_roots(projects_root: Path, maxdepth: int = 3) -> list[Path]:
    cmd = [
        "find",
        str(projects_root),
        "-maxdepth",
        str(maxdepth + 1),
        "-type",
        "d",
        "-name",
        ".git",
    ]
    out = subprocess.check_output(cmd, text=True)
    roots = [Path(line.strip()).parent for line in out.splitlines() if line.strip()]
    return sorted(set(roots))


def _update_agents(path: Path, create_missing: bool) -> tuple[bool, str]:
    if not path.exists():
        if not create_missing:
            return False, "missing"
        text = "# Agent Instructions\n\n" + THEME_BLOCK.strip() + "\n"
        path.write_text(text, encoding="utf-8")
        return True, "created"

    original = path.read_text(encoding="utf-8", errors="replace")
    updated = original
    if THEME_HEADING in original:
        pattern = re.compile(
            r"(?ms)^## Codex CLI Theme \(Hard Gate\)\n.*?(?=^\s*##\s|\Z)"
        )
        updated = pattern.sub(THEME_BLOCK.strip() + "\n\n", original)
    else:
        if not updated.endswith("\n"):
            updated += "\n"
        updated += "\n" + THEME_BLOCK.strip() + "\n"
    if updated != original:
        path.write_text(updated, encoding="utf-8")
        return True, "updated"
    return False, "unchanged"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--projects-root", default="/mnt/d/projects")
    ap.add_argument("--maxdepth", type=int, default=3)
    ap.add_argument("--create-missing", action="store_true")
    args = ap.parse_args()

    root = Path(args.projects_root).resolve()
    repos = _repo_roots(root, maxdepth=int(args.maxdepth))
    summary: dict[str, object] = {
        "projects_root": str(root),
        "repo_count": len(repos),
        "changed": [],
        "missing": [],
        "unchanged": [],
    }
    for repo in repos:
        agents = repo / "AGENTS.md"
        changed, status = _update_agents(agents, create_missing=bool(args.create_missing))
        if changed:
            cast = summary["changed"]
            assert isinstance(cast, list)
            cast.append({"repo": str(repo), "status": status})
        elif status == "missing":
            cast = summary["missing"]
            assert isinstance(cast, list)
            cast.append(str(repo))
        else:
            cast = summary["unchanged"]
            assert isinstance(cast, list)
            cast.append(str(repo))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
