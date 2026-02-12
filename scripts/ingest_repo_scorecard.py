#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Input must be a JSON object")
    return data


def _validate_scorecard(payload: dict[str, Any]) -> dict[str, Any]:
    pillars = payload.get("pillars")
    if not isinstance(pillars, dict):
        raise ValueError("scorecard.pillars is required")
    for name in ("performant", "accurate", "secure", "citable"):
        node = pillars.get(name)
        if not isinstance(node, dict):
            raise ValueError(f"Missing pillar: {name}")
        score = node.get("score_0_4")
        if not isinstance(score, (int, float)):
            raise ValueError(f"{name}.score_0_4 must be numeric")
        if float(score) < 0.0 or float(score) > 4.0:
            raise ValueError(f"{name}.score_0_4 must be within 0.0..4.0")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest repository-level 4-pillar scorecards into an append-only JSONL ledger."
    )
    parser.add_argument("--repo-id", required=True, help="Stable repository identifier")
    parser.add_argument("--scorecard-json", required=True, help="Path to scorecard JSON")
    parser.add_argument(
        "--out-jsonl",
        default="appdata/repo_scorecards.jsonl",
        help="Output ledger path (append-only JSONL)",
    )
    args = parser.parse_args()

    scorecard_path = Path(args.scorecard_json).resolve()
    out_path = Path(args.out_jsonl).resolve()
    scorecard = _validate_scorecard(_load_json(scorecard_path))

    row = {
        "ingested_at": _now_iso(),
        "repo_id": str(args.repo_id).strip(),
        "source_path": str(scorecard_path),
        "scorecard": scorecard,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
