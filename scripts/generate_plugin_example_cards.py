#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "docs" / "plugin_class_actionability_matrix.json"
DEFAULT_OUT_DIR = ROOT / "docs" / "plugin_example_cards"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _pick_example(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    actionable = [r for r in rows if str(r.get("actionability_state") or "") == "actionable"]
    explained = [r for r in rows if str(r.get("actionability_state") or "") == "explained_na"]
    missing = [r for r in rows if str(r.get("actionability_state") or "") == "missing_output"]
    if actionable:
        return actionable[0]
    if explained:
        return explained[0]
    if missing:
        return missing[0]
    return None


def _render_card(class_id: str, class_meta: dict[str, Any], row: dict[str, Any], run_id: str) -> str:
    lines: list[str] = []
    lines.append(f"# Plugin Class Example: {class_id}")
    lines.append("")
    rationale = str(class_meta.get("rationale") or "").strip()
    expected = str(class_meta.get("expected_output_type") or "").strip()
    if rationale:
        lines.append(f"- class_rationale: {rationale}")
    if expected:
        lines.append(f"- expected_output_type: `{expected}`")
    lines.append(f"- run_id: `{run_id}`")
    lines.append("")
    if not row:
        lines.append("- No plugin rows available for this class.")
        lines.append("")
        return "\n".join(lines)

    plugin_id = str(row.get("plugin_id") or "")
    plugin_type = str(row.get("plugin_type") or "")
    state = str(row.get("actionability_state") or "")
    reason_code = str(row.get("reason_code") or "")
    example = row.get("example") if isinstance(row.get("example"), dict) else {}
    lines.append("## Example")
    lines.append("")
    lines.append(f"- plugin_id: `{plugin_id}`")
    lines.append(f"- plugin_type: `{plugin_type}`")
    lines.append(f"- actionability_state: `{state}`")
    if reason_code:
        lines.append(f"- reason_code: `{reason_code}`")
    kind = str(example.get("kind") or "").strip()
    action_type = str(example.get("action_type") or "").strip()
    recommendation = str(example.get("recommendation") or "").strip()
    explanation = str(example.get("plain_english_explanation") or "").strip()
    modeled_percent = example.get("modeled_percent")
    if kind:
        lines.append(f"- finding_kind: `{kind}`")
    if action_type:
        lines.append(f"- action_type: `{action_type}`")
    if isinstance(modeled_percent, (int, float)):
        lines.append(f"- modeled_percent: {float(modeled_percent):.2f}")
    if recommendation:
        lines.append(f"- recommendation: {recommendation}")
    if explanation:
        lines.append(f"- explanation: {explanation}")
    lines.append("")
    lines.append("## Traceability")
    lines.append("")
    lines.append(f"- class `{class_id}` -> plugin `{plugin_id}` -> run `{run_id}`")
    lines.append("- Source artifact: `docs/plugin_class_actionability_matrix.json`")
    lines.append("")
    return "\n".join(lines)


def generate_cards(matrix: dict[str, Any], out_dir: Path, *, write: bool = True) -> tuple[dict[str, Any], dict[str, str]]:
    classes = matrix.get("classes") if isinstance(matrix.get("classes"), dict) else {}
    plugins = matrix.get("plugins") if isinstance(matrix.get("plugins"), list) else []
    run_id = str(matrix.get("run_id") or "")
    by_class: dict[str, list[dict[str, Any]]] = {}
    for row in plugins:
        if not isinstance(row, dict):
            continue
        class_id = str(row.get("plugin_class") or "").strip()
        if not class_id:
            continue
        by_class.setdefault(class_id, []).append(row)

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
    index_cards: list[dict[str, Any]] = []
    rendered_cards: dict[str, str] = {}
    for class_id in sorted(classes.keys()):
        class_meta = classes.get(class_id) if isinstance(classes.get(class_id), dict) else {}
        chosen = _pick_example(by_class.get(class_id, [])) or {}
        card_text = _render_card(class_id, class_meta, chosen, run_id)
        card_path = out_dir / f"{class_id}.md"
        rendered_cards[str(card_path.name)] = card_text
        if write:
            card_path.write_text(card_text, encoding="utf-8")
        try:
            card_path_text = str(card_path.relative_to(ROOT))
        except Exception:
            card_path_text = str(card_path)
        index_cards.append(
            {
                "class_id": class_id,
                "card_path": card_path_text,
                "plugin_id": str(chosen.get("plugin_id") or ""),
                "actionability_state": str(chosen.get("actionability_state") or ""),
                "reason_code": str(chosen.get("reason_code") or ""),
            }
        )
    index_payload = {
        "run_id": run_id,
        "card_count": len(index_cards),
        "cards": index_cards,
    }
    if write:
        (out_dir / "index.json").write_text(json.dumps(index_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return index_payload, rendered_cards


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-json", default=str(DEFAULT_MATRIX))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    matrix_path = Path(str(args.matrix_json)).resolve()
    out_dir = Path(str(args.out_dir)).resolve()
    matrix = _read_json(matrix_path)
    if not matrix:
        raise SystemExit(f"Invalid or missing matrix: {matrix_path}")

    generated, rendered_cards = generate_cards(matrix, out_dir, write=not args.verify)
    if args.verify:
        index_path = out_dir / "index.json"
        if not index_path.exists():
            return 2
        try:
            existing = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            return 2
        if existing != generated:
            return 2
        for filename, content in rendered_cards.items():
            card_path = out_dir / filename
            if not card_path.exists():
                return 2
            if card_path.read_text(encoding="utf-8") != content:
                return 2
        return 0
    print(f"out_dir={out_dir}")
    print(f"card_count={generated.get('card_count')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
