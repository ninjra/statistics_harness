#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


MODELS = {
    # Target directory layout matches plugins/llm_text2sql_local_generate_v1/plugin.py
    "SQLCoder-7B-2": {
        "repo_id": "defog/sqlcoder-7b-2",
        "dest_rel": "sql/text2sql/defog/sqlcoder-7b-2",
    },
    "Snowflake-Arctic-Text2SQL-R1-7B": {
        "repo_id": "Snowflake/Arctic-Text2SQL-R1-7B",
        "dest_rel": "sql/text2sql/snowflake/arctic-text2sql-r1-7b",
    },
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-root", default="/mnt/d/autocapture/models", type=Path)
    ap.add_argument(
        "--model",
        action="append",
        default=[],
        help=f"Model name(s) to download. Supported: {sorted(MODELS.keys())}. If omitted, downloads all.",
    )
    ap.add_argument("--revision", default=None, help="Optional HF revision (branch/tag/commit).")
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as exc:
        raise SystemExit(
            f"huggingface_hub is required to download models ({type(exc).__name__}: {exc}). "
            "Install it in the environment you are running this script with."
        )

    want = [str(x) for x in (args.model or []) if str(x).strip()]
    names = want or sorted(MODELS.keys())

    for name in names:
        spec = MODELS.get(name)
        if not spec:
            raise SystemExit(f"Unknown model: {name} (supported: {sorted(MODELS.keys())})")
        dest = (Path(args.models_root) / spec["dest_rel"]).resolve()
        dest.mkdir(parents=True, exist_ok=True)
        print(f"downloading {name} from {spec['repo_id']} -> {dest}")
        snapshot_download(
            repo_id=str(spec["repo_id"]),
            revision=args.revision,
            local_dir=str(dest),
            local_dir_use_symlinks=False,
        )
        # Minimal signal for callers/scripts.
        (dest / ".download_complete").write_text("ok\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

