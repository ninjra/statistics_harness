#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck source=/dev/null
. "$ROOT_DIR/.venv/bin/activate"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

DATASET_VERSION_ID="${1:-3246cc7cd7d57a317ddc05e80e6f6f5bfe7f50deb0ee7af8db50d04bae180e1a}"
INTERVAL_SECONDS="${2:-30}"

RUN_ID="$(
python -c "import sqlite3; from pathlib import Path; db=Path('appdata/state.sqlite'); con=sqlite3.connect(db); con.row_factory=sqlite3.Row; row=con.execute(\"select run_id from runs where status='running' and dataset_version_id=? order by created_at desc limit 1\", ('${DATASET_VERSION_ID}',)).fetchone(); print(row['run_id'] if row else ''); con.close()"
)"

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(
  python -c "import sqlite3; from pathlib import Path; db=Path('appdata/state.sqlite'); con=sqlite3.connect(db); con.row_factory=sqlite3.Row; row=con.execute(\"select run_id from runs where dataset_version_id=? order by created_at desc limit 1\", ('${DATASET_VERSION_ID}',)).fetchone(); print(row['run_id'] if row else ''); con.close()"
  )"
fi

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(
  python -c "import sqlite3; from pathlib import Path; db=Path('appdata/state.sqlite'); con=sqlite3.connect(db); con.row_factory=sqlite3.Row; row=con.execute(\"select run_id from runs order by created_at desc limit 1\").fetchone(); print(row['run_id'] if row else ''); con.close()"
  )"
fi

if [[ -z "$RUN_ID" ]]; then
  echo "ERROR: no run found"
  exit 1
fi

echo "RUN_ID=$RUN_ID"
exec bash scripts/watch_run_status.sh "$RUN_ID" "$INTERVAL_SECONDS"
