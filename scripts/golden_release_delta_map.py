#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "docs" / "golden_release_delta_map.json"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _contains(path: str, needle: str) -> bool:
    file_path = ROOT / path
    if not file_path.exists():
        return False
    return needle in _read(file_path)


def _dataset_threshold_is_1m_or_lower() -> bool:
    path = ROOT / "src" / "statistic_harness" / "core" / "dataset_io.py"
    if not path.exists():
        return False
    text = _read(path)
    match = re.search(r"max_rows\s*=\s*([0-9_]+)", text)
    if not match:
        return False
    value = int(match.group(1).replace("_", ""))
    return value <= 1_000_000


@dataclass(frozen=True)
class Requirement:
    req_id: str
    title: str
    paths: tuple[str, ...]
    checks: tuple[Callable[[], bool], ...]


REQUIREMENTS: tuple[Requirement, ...] = (
    Requirement(
        req_id="P0.1",
        title="Tri-mode network policy (off|localhost|on)",
        paths=("src/statistic_harness/core/plugin_runner.py", "tests/test_offline.py"),
        checks=(
            lambda: _contains("src/statistic_harness/core/plugin_runner.py", "STAT_HARNESS_NETWORK_MODE"),
            lambda: _contains("src/statistic_harness/core/plugin_runner.py", "_is_loopback_destination"),
        ),
    ),
    Requirement(
        req_id="P0.2",
        title="Streaming-first full dataframe guard for large datasets",
        paths=("src/statistic_harness/core/dataset_io.py", "tests/test_dataset_streaming.py"),
        checks=(
            _dataset_threshold_is_1m_or_lower,
            lambda: _contains("src/statistic_harness/core/dataset_io.py", "ctx.dataset_iter_batches"),
            lambda: _contains("tests/test_dataset_streaming.py", "large_dataset_load_requires_override"),
        ),
    ),
    Requirement(
        req_id="P0.3",
        title="Golden execution policy modes",
        paths=("src/statistic_harness/core/pipeline.py", "tests/test_pipeline_golden_mode.py"),
        checks=(
            lambda: _contains("src/statistic_harness/core/pipeline.py", "STAT_HARNESS_GOLDEN_MODE"),
            lambda: _contains("src/statistic_harness/core/pipeline.py", "strict_skip_violation"),
            lambda: _contains("tests/test_pipeline_golden_mode.py", "test_golden_mode_truthy_maps_to_strict"),
        ),
    ),
    Requirement(
        req_id="P0.4",
        title="4 pillars scorecard attached to report output",
        paths=("src/statistic_harness/core/four_pillars.py", "src/statistic_harness/core/report.py"),
        checks=(
            lambda: _contains("src/statistic_harness/core/four_pillars.py", "build_four_pillars_scorecard"),
            lambda: _contains("src/statistic_harness/core/report.py", 'report["four_pillars"]'),
        ),
    ),
    Requirement(
        req_id="DoD.1",
        title="Gauntlet includes plugin discovery gate",
        paths=("scripts/run_gauntlet.sh",),
        checks=(lambda: _contains("scripts/run_gauntlet.sh", "stat-harness list-plugins"),),
    ),
)


def _status(check_results: list[bool]) -> str:
    if check_results and all(check_results):
        return "implemented"
    if any(check_results):
        return "partial"
    return "missing"


def build_delta_map() -> dict[str, object]:
    items = []
    for req in REQUIREMENTS:
        check_values = [bool(chk()) for chk in req.checks]
        items.append(
            {
                "requirement_id": req.req_id,
                "title": req.title,
                "status": _status(check_values),
                "paths": list(req.paths),
                "checks_passed": int(sum(1 for ok in check_values if ok)),
                "checks_total": int(len(check_values)),
            }
        )
    summary = {
        "implemented": sum(1 for x in items if x["status"] == "implemented"),
        "partial": sum(1 for x in items if x["status"] == "partial"),
        "missing": sum(1 for x in items if x["status"] == "missing"),
    }
    return {
        "schema_version": "golden_release_delta_map.v1",
        "source_plan": "docs/codex_4pillars_golden_release_plan.md",
        "items": items,
        "summary": summary,
    }


def main() -> int:
    payload = build_delta_map()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(str(OUT_PATH))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
