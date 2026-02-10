from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "docs" / "codex_stat_plugins_spec_pack.md"
TESTS_DIR = ROOT / "tests" / "plugins"


def extract_plugin_ids(text: str) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("##"):
            continue
        for token in re.findall(r"analysis_[a-z0-9_]+", line):
            if token not in seen:
                ids.append(token)
                seen.add(token)
    return ids


def main() -> None:
    spec_text = SPEC_PATH.read_text(encoding="utf-8")
    plugin_ids = extract_plugin_ids(spec_text)
    TESTS_DIR.mkdir(parents=True, exist_ok=True)

    for plugin_id in plugin_ids:
        test_path = TESTS_DIR / f"test_{plugin_id}.py"
        if test_path.exists():
            continue
        content = f"""import datetime as dt

import pandas as pd

from plugins.{plugin_id}.plugin import Plugin
from tests.conftest import make_context


def _sample_df() -> pd.DataFrame:
    rows = 200
    return pd.DataFrame(
        {{
            "metric": [0.1] * 100 + [2.0] * 100,
            "metric2": [1.0] * 50 + [3.0] * 150,
            "category": ["A"] * 100 + ["B"] * 100,
            "text": ["Error code 500"] * 100 + ["Timeout at step 3"] * 100,
            "case_id": [i // 4 for i in range(rows)],
            "ts": [
                dt.datetime(2026, 1, 1) + dt.timedelta(minutes=i)
                for i in range(rows)
            ],
        }}
    )


def test_{plugin_id}_smoke(run_dir):
    df = _sample_df()
    df["ts"] = df["ts"].astype(str)
    ctx = make_context(run_dir, df, {{}})
    result = Plugin().run(ctx)
    assert result.status in ("ok", "skipped")
"""
        test_path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
