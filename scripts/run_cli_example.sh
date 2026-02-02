#!/usr/bin/env bash
set -euo pipefail
. .venv/bin/activate
VEC_PATH="$(python - <<'PY'
import importlib.util
import pathlib

spec = importlib.util.find_spec("sqlite_vec")
if spec and spec.origin:
    base = pathlib.Path(spec.origin).resolve().parent
    for path in base.rglob("vec0.*"):
        if path.suffix in {".so", ".dylib", ".dll"}:
            print(path)
            break
PY
)"
if [[ -n "${VEC_PATH}" ]]; then
  export STAT_HARNESS_ENABLE_VECTOR_STORE=1
  export STAT_HARNESS_SQLITE_VEC_PATH="${VEC_PATH}"
fi
stat-harness run --file tests/fixtures/synth_linear.csv --plugins ingest_tabular,profile_basic,report_bundle --run-seed 42
