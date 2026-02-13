#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-root", default="/mnt/d/autocapture/models")
    args = ap.parse_args()

    report: dict[str, object] = {"python": None, "imports": {}, "models": {}}
    import sys

    report["python"] = sys.version

    def _try(mod: str) -> str:
        try:
            m = __import__(mod)
            return str(getattr(m, "__version__", "ok"))
        except Exception as exc:
            return f"missing ({type(exc).__name__}: {exc})"

    report["imports"] = {
        "vllm": _try("vllm"),
        "torch": _try("torch"),
        "transformers": _try("transformers"),
        "huggingface_hub": _try("huggingface_hub"),
    }

    root = Path(str(args.models_root))
    candidates = {
        "SQLCoder-7B-2": root / "sql" / "text2sql" / "defog" / "sqlcoder-7b-2",
        "Snowflake-Arctic-Text2SQL-R1-7B": root / "sql" / "text2sql" / "snowflake" / "arctic-text2sql-r1-7b",
    }
    models: dict[str, object] = {}
    for name, path in candidates.items():
        models[name] = {"path": str(path), "exists": bool(path.exists())}
    report["models"] = models
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
