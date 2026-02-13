from __future__ import annotations

import subprocess
import sys


def main() -> int:
    steps = [
        [sys.executable, "scripts/binding_implementation_matrix.py", "--verify"],
        [sys.executable, "scripts/docs_coverage_matrix.py", "--verify"],
        [sys.executable, "scripts/plugin_data_access_matrix.py", "--verify"],
        [sys.executable, "scripts/plugins_functionality_matrix.py", "--verify"],
        [sys.executable, "scripts/sql_assist_adoption_matrix.py", "--verify"],
        [sys.executable, "scripts/redteam_ids_matrix.py", "--verify"],
    ]
    for cmd in steps:
        rc = subprocess.call(cmd)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
