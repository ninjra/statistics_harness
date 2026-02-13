from __future__ import annotations

import re
from pathlib import Path


SPEC_PATH = Path(__file__).resolve().parents[1] / "docs" / "codex_stat_plugins_spec_pack.md"
PLUGINS_DIR = Path(__file__).resolve().parents[1] / "plugins"


ALIAS_WRAPPERS = {
    "analysis_isolation_forest": "analysis_isolation_forest_anomaly",
    "analysis_log_template_drain": "analysis_log_template_mining_drain",
    "analysis_robust_pca_pcp": "analysis_robust_pca_sparse_outliers",
}


def extract_plugin_ids(text: str) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("##"):
            continue
        matches = re.findall(r"analysis_[a-z0-9_]+", line)
        for token in matches:
            if token not in seen:
                ids.append(token)
                seen.add(token)
    return ids


def existing_plugins() -> set[str]:
    return {path.name for path in PLUGINS_DIR.glob("*") if path.is_dir()}


def main() -> None:
    text = SPEC_PATH.read_text(encoding="utf-8")
    spec_ids = extract_plugin_ids(text)
    existing = existing_plugins()
    missing = [pid for pid in spec_ids if pid not in existing]

    print(f"spec_count={len(spec_ids)}")
    print(f"existing_count={len(existing)}")
    print(f"missing_count={len(missing)}")
    if missing:
        print("missing:")
        for pid in missing:
            print(f"- {pid}")
    else:
        print("missing: none")

    if ALIAS_WRAPPERS:
        print("alias_wrappers:")
        for alias, base in sorted(ALIAS_WRAPPERS.items()):
            status = "exists" if alias in existing else "missing"
            print(f"- {alias} -> {base} ({status})")


if __name__ == "__main__":
    main()
