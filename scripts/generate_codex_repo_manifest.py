#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "_codex_repo_manifest.txt"


def _git_ls_files() -> list[str]:
    try:
        out = subprocess.check_output(
            ["git", "ls-files"],
            cwd=ROOT,
            text=True,
        )
    except Exception:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def _fallback_manifest() -> list[str]:
    out: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        parts = set(rel.parts)
        if ".git" in parts or ".venv" in parts or "__pycache__" in parts:
            continue
        out.append(str(rel).replace("\\", "/"))
    return sorted(out)


def build_manifest_lines() -> list[str]:
    git_lines = set(_git_ls_files())
    fs_lines = set(_fallback_manifest())
    lines = sorted(git_lines | fs_lines)
    return lines


def _text(lines: list[str]) -> str:
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    lines = build_manifest_lines()
    text = _text(lines)
    out = args.out.resolve()

    if args.verify:
        if not out.exists():
            return 2
        return 0 if out.read_text(encoding="utf-8") == text else 2

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"out={out}")
    print(f"files={len(lines)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
