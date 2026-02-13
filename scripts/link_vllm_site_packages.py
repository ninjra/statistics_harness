#!/usr/bin/env python3
from __future__ import annotations

import argparse
import site
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--venv-site-packages",
        default="",
        help="Optional explicit venv site-packages dir; default uses the current interpreter's venv.",
    )
    ap.add_argument("--external-site", default="/mnt/d/autocapture/venvs/vllm/lib/python3.12/site-packages")
    ap.add_argument("--pth-name", default="stat_harness_external_vllm.pth")
    args = ap.parse_args()

    external = Path(str(args.external_site)).resolve()
    if not external.exists():
        raise SystemExit(f"external site-packages not found: {external}")

    if str(args.venv_site_packages).strip():
        sp = Path(str(args.venv_site_packages)).resolve()
    else:
        # Run this script with the target venv interpreter for correctness.
        paths = site.getsitepackages()
        if not paths:
            raise SystemExit("Could not determine site-packages for current interpreter")
        sp = Path(paths[0]).resolve()
    sp.mkdir(parents=True, exist_ok=True)
    pth = sp / str(args.pth_name)
    pth.write_text(str(external) + "\n", encoding="utf-8")

    # Validate import in a fresh interpreter so .pth is applied.
    import subprocess

    try:
        ver = subprocess.check_output(
            [sys.executable, "-c", "import vllm;print(getattr(vllm,'__version__','unknown'))"],
            text=True,
        ).strip()
    except Exception as exc:
        raise SystemExit(f"Linked .pth but vllm import failed in fresh interpreter: {type(exc).__name__}: {exc}")
    print(f"linked: {external} -> {pth}")
    print(f"vllm_version: {ver}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
