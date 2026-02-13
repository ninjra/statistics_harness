from __future__ import annotations

import json
from pathlib import Path
import sys

from scripts.full_instruction_coverage_report import main as full_instruction_report_main
from scripts.full_repo_misses import main as full_repo_misses_main


def _run_main_with_clean_argv(fn) -> int:
    prev_argv = sys.argv[:]
    try:
        sys.argv = [prev_argv[0]]
        return int(fn())
    finally:
        sys.argv = prev_argv


def test_full_repo_misses_report_is_up_to_date() -> None:
    root = Path(__file__).resolve().parents[1]
    # Ensure dependent report exists/current first.
    assert _run_main_with_clean_argv(full_instruction_report_main) == 0
    assert _run_main_with_clean_argv(full_repo_misses_main) == 0

    existing = json.loads((root / "docs" / "full_repo_misses.json").read_text(encoding="utf-8"))
    assert "hard_gaps" in existing
    assert "soft_gaps" in existing
    assert "summary" in existing
