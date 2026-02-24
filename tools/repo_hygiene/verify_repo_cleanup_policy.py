#!/usr/bin/env python3
from __future__ import annotations

import fnmatch
import re
import subprocess
import sys
from pathlib import Path

FORBIDDEN_REFERENCE_PATTERN = re.compile(r"(tools/legacy/|docs/test/|repomix-output\.md)")
TEXT_SUFFIXES = {".json", ".md", ".ps1", ".py", ".sh", ".txt", ".yaml", ".yml", ".toml", ".xml"}
FORBIDDEN_TRACKED_PREFIXES = ("tools/legacy/", "docs/test/")
FORBIDDEN_TRACKED_EXACT = {"generated repository snapshot artifact"}
BLOCKED_BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp",
    ".ico", ".pdf", ".zip", ".tar", ".gz", ".7z", ".bin",
}


def load_allowlist(path: Path) -> list[str]:
    if not path.exists():
        return []
    out = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def is_allowed(path: str, allowlist: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pat) for pat in allowlist)


def git_ls_files(repo_root: Path) -> list[str]:
    out = subprocess.check_output(["git", "-C", str(repo_root), "ls-files"], text=True)
    return [x.strip().replace("\\", "/") for x in out.splitlines() if x.strip()]


def iter_text_files(repo_root: Path, tracked: list[str]) -> list[Path]:
    files = []
    for rel in tracked:
        p = repo_root / rel
        if p.suffix.lower() in TEXT_SUFFIXES and p.is_file():
            files.append(p)
    return files


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    tracked = git_ls_files(repo_root)
    allowlist = load_allowlist(repo_root / "config/repo_cleanup/tracked_binary_allowlist.txt")

    forbidden_refs = {}
    for p in iter_text_files(repo_root, tracked):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        rel = str(p.relative_to(repo_root)).replace("\\", "/")
        for m in FORBIDDEN_REFERENCE_PATTERN.findall(text):
            forbidden_refs.setdefault(m, set()).add(rel)

    forbidden_paths = sorted([p for p in tracked if p in FORBIDDEN_TRACKED_EXACT or any(p.startswith(pref) for pref in FORBIDDEN_TRACKED_PREFIXES)])
    blocked_binaries = sorted([p for p in tracked if Path(p).suffix.lower() in BLOCKED_BINARY_SUFFIXES and not is_allowed(p, allowlist)])

    failures = 0
    if forbidden_refs:
        failures += 1
        print("FAIL: forbidden references detected")
        for k in sorted(forbidden_refs):
            print(f"  {k} <- {', '.join(sorted(forbidden_refs[k]))}")
    if forbidden_paths:
        failures += 1
        print("FAIL: forbidden tracked paths detected")
        for p in forbidden_paths:
            print(f"  {p}")
    if blocked_binaries:
        failures += 1
        print("FAIL: tracked binary/blob files detected")
        for p in blocked_binaries:
            print(f"  {p}")
    if failures:
        return 2

    allowlisted = sum(1 for p in tracked if Path(p).suffix.lower() in BLOCKED_BINARY_SUFFIXES and is_allowed(p, allowlist))
    print(f"PASS: repo hygiene policy verified (tracked={len(tracked)} binary_allowlisted={allowlisted})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
