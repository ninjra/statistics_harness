from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_plain_report(report_json_path: Path) -> str:
    """Render a plain-English report from report.json.

    Contract:
    - Avoid technical jargon.
    - Point to evidence via artifact paths when available.
    """

    report = _read_json(report_json_path)
    plugins = report.get("plugins") if isinstance(report, dict) else None
    recs = report.get("recommendations") if isinstance(report, dict) else None

    lines: list[str] = []
    lines.append("# Plain-English Report")
    lines.append("")
    lines.append("This is a non-technical summary of what the harness found, with concrete next steps.")

    # Recommendations first.
    if isinstance(recs, dict):
        lines.append("")
        lines.append("## Recommended Next Steps")
        blocks = []
        if "discovery" in recs and isinstance(recs.get("discovery"), dict):
            blocks.append(("Discovery", recs["discovery"].get("items") or []))
        if "known" in recs and isinstance(recs.get("known"), dict):
            blocks.append(("Known", recs["known"].get("items") or []))
        if not blocks:
            blocks.append(("Recommendations", recs.get("items") or []))
        for title, items in blocks:
            if not isinstance(items, list) or not items:
                continue
            lines.append("")
            lines.append(f"### {title}")
            for it in items[:20]:
                if not isinstance(it, dict):
                    continue
                text = str(it.get("recommendation") or it.get("title") or "").strip()
                if not text:
                    continue
                where = it.get("where") if isinstance(it.get("where"), dict) else None
                where_txt = ""
                if where:
                    proc = (
                        where.get("process_norm")
                        or where.get("process")
                        or where.get("process_id")
                        or where.get("transition")
                    )
                    if isinstance(proc, str) and proc.strip():
                        where_txt = f" (applies to: {proc.strip()})"
                lines.append(f"- {text}{where_txt}")
                vsteps = it.get("validation_steps")
                if isinstance(vsteps, list):
                    steps = [s.strip() for s in vsteps if isinstance(s, str) and s.strip()]
                    if steps:
                        lines.append("  Validation:")
                        for s in steps[:3]:
                            lines.append(f"  - {s}")
                ev = it.get("evidence")
                if isinstance(ev, list):
                    paths: list[str] = []
                    for e in ev:
                        if not isinstance(e, dict):
                            continue
                        for v in e.values():
                            if isinstance(v, str) and (v.endswith(".json") or v.endswith(".md")):
                                paths.append(v)
                    paths = sorted(set(paths))
                    if paths:
                        lines.append("  Evidence files:")
                        for p in paths[:3]:
                            lines.append(f"  - {p}")

    if isinstance(plugins, dict):
        lines.append("")
        lines.append("## What Each Check Found")
        for pid in sorted(plugins.keys()):
            payload = plugins.get(pid)
            if not isinstance(payload, dict):
                continue
            summary = str(payload.get("summary") or "").strip()
            if not summary:
                continue
            lines.append("")
            lines.append(f"### {pid}")
            lines.append(summary)
            artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), list) else []
            if artifacts:
                lines.append("")
                lines.append("Evidence files:")
                for a in artifacts[:8]:
                    if not isinstance(a, dict):
                        continue
                    p = a.get("path")
                    if isinstance(p, str) and p:
                        lines.append(f"- {p}")

    return "\n".join(lines).rstrip() + "\n"
