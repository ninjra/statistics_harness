#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from statistic_harness.core.sql_pack import validate_sql_pack


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema-snapshot", required=True, type=Path)
    ap.add_argument("--intents", required=True, type=Path)
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()

    schema = json.loads(args.schema_snapshot.read_text(encoding="utf-8"))
    intents_payload = json.loads(args.intents.read_text(encoding="utf-8"))

    prompt = (
        "You are generating a SQL pack for SQLite.\n"
        "Output MUST be a single JSON object that validates against docs/sql_pack.schema.json.\n"
        "Rules:\n"
        "- Single-statement SQL only (no semicolons).\n"
        "- Deterministic ordering (ORDER BY) for any top-N.\n"
        "- mode MUST be 'ro' for every query. Do not emit DDL/DML.\n"
        "- DO NOT use PRAGMA/ATTACH/DETACH/VACUUM.\n\n"
        "Schema snapshot (JSON):\n"
        + _stable_json(schema)
        + "\nQuery intents (JSON):\n"
        + _stable_json(intents_payload)
    )

    try:
        from vllm import LLM, SamplingParams  # type: ignore
    except Exception as exc:
        raise SystemExit(f"vLLM not available: {type(exc).__name__}: {exc}")

    llm = LLM(model=str(args.model_dir), trust_remote_code=False)
    params = SamplingParams(
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_tokens),
    )
    outputs = llm.generate([prompt], params)
    text = ""
    if outputs and outputs[0].outputs:
        text = str(outputs[0].outputs[0].text or "").strip()
    if not text:
        raise SystemExit("Model produced empty output")

    try:
        pack = json.loads(text)
    except Exception:
        cleaned = text
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.replace("json", "", 1).strip()
        pack = json.loads(cleaned)

    if not isinstance(pack, dict):
        raise SystemExit("Output is not a JSON object")

    pack.setdefault("schema_hash", str(schema.get("schema_hash") or ""))
    pack.setdefault("dialect", "sqlite")
    pack.setdefault("model", {"name": args.model_dir.name, "source": "local"})
    pack.setdefault(
        "decode",
        {
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "max_tokens": int(args.max_tokens),
        },
    )

    validate_sql_pack(pack, Path("docs/sql_pack.schema.json"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(_stable_json(pack), encoding="utf-8")
    print(str(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
