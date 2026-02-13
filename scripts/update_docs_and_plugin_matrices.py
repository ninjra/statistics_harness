#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys


def main() -> int:
    # Order matters: some docs are scanned by other matrix generators.
    steps = [
        [sys.executable, "scripts/run_repo_improvements_pipeline.py"],
        [sys.executable, "scripts/plugin_data_access_matrix.py"],
        [sys.executable, "scripts/plugins_functionality_matrix.py"],
        [sys.executable, "scripts/sql_assist_adoption_matrix.py"],
        [sys.executable, "scripts/redteam_ids_matrix.py"],
        [sys.executable, "scripts/docs_coverage_matrix.py"],
        [sys.executable, "scripts/binding_implementation_matrix.py"],
    ]
    for cmd in steps:
        rc = subprocess.call(cmd)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
