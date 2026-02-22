from __future__ import annotations

import re
from pathlib import Path

from build_plugins_validate_runbook_map import build_payload


def _slug(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    rows = build_payload(repo_root)
    out_dir = repo_root / "docs" / "release_evidence" / "plugins_validate_runbook_technique_specs"
    out_dir.mkdir(parents=True, exist_ok=True)
    index_lines = [
        "# Technique Specs (Plugins Validate Runbook)",
        "",
        "| # | Technique | Plugin ID | Spec |",
        "|---:|---|---|---|",
    ]
    for row in rows:
        ordinal = int(row["ordinal"])
        technique = str(row["technique"])
        plugin_id = str(row["plugin_id"])
        refs = list(row.get("references") or [])
        filename = f"{ordinal:02d}_{_slug(technique)}.md"
        path = out_dir / filename
        lines = [
            f"# Technique Spec {ordinal:02d}: {technique}",
            "",
            f"- Plugin ID: `{plugin_id}`",
            f"- Implemented: `{'Y' if bool(row.get('implemented')) else 'N'}`",
            "- Deterministic seed: `0`",
            "- Required output contract: actionability metrics triplet (`delta_h`, `eff_%`, `eff_idx`)",
            "",
            "## Inputs",
            "- Required: dataset rows in normalized or source-compatible tabular form.",
            "- Preferred signals: process identifier, timestamp columns, and numeric covariates.",
            "",
            "## Acceptance",
            "- Returns `ok` with actionable finding, or `na` with deterministic reason code.",
            "- Must include plain-English recommendation text.",
            "- Must include modeled windows for accounting-month, close-static, close-dynamic.",
            "",
            "## References",
        ]
        if refs:
            lines.extend([f"- {ref}" for ref in refs])
        else:
            lines.append("- (none)")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        index_lines.append(
            f"| {ordinal} | {technique} | `{plugin_id}` | `{path.relative_to(repo_root)}` |"
        )
    index_path = out_dir / "README.md"
    index_path.write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    print(f"generated={len(rows)} index={index_path}")


if __name__ == "__main__":
    main()
