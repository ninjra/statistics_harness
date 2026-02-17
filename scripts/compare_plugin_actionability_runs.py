from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_report(run_id: str) -> dict[str, Any]:
    report_path = Path("appdata") / "runs" / run_id / "report.json"
    if not report_path.exists():
        raise SystemExit(f"Missing report.json for run_id={run_id}: {report_path}")
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Invalid report JSON for run_id={run_id}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"report.json for run_id={run_id} is not an object")
    return payload


def _ids_from_items(items: Any) -> set[str]:
    if not isinstance(items, list):
        return set()
    out: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        pid = str(item.get("plugin_id") or "").strip()
        if pid:
            out.add(pid)
    return out


def _actionability_sets(report: dict[str, Any]) -> dict[str, set[str]]:
    plugins = report.get("plugins") if isinstance(report.get("plugins"), dict) else {}
    plugin_ids = {str(pid).strip() for pid in plugins.keys() if str(pid).strip()}
    recs = report.get("recommendations") if isinstance(report.get("recommendations"), dict) else {}
    actionable = _ids_from_items(recs.get("items"))
    explanations = recs.get("explanations") if isinstance(recs.get("explanations"), dict) else {}
    explained = _ids_from_items(explanations.get("items"))
    unexplained = plugin_ids - actionable - explained
    return {
        "plugins": plugin_ids,
        "actionable": actionable,
        "explained": explained,
        "unexplained": unexplained,
    }


def compare_runs(before_run_id: str, after_run_id: str) -> dict[str, Any]:
    before_sets = _actionability_sets(_load_report(before_run_id))
    after_sets = _actionability_sets(_load_report(after_run_id))
    return {
        "before_run_id": before_run_id,
        "after_run_id": after_run_id,
        "before": {
            "plugin_count": int(len(before_sets["plugins"])),
            "actionable_count": int(len(before_sets["actionable"])),
            "explained_count": int(len(before_sets["explained"])),
            "unexplained_count": int(len(before_sets["unexplained"])),
        },
        "after": {
            "plugin_count": int(len(after_sets["plugins"])),
            "actionable_count": int(len(after_sets["actionable"])),
            "explained_count": int(len(after_sets["explained"])),
            "unexplained_count": int(len(after_sets["unexplained"])),
        },
        "delta": {
            "newly_actionable_plugins": sorted(after_sets["actionable"] - before_sets["actionable"]),
            "newly_explained_plugins": sorted(after_sets["explained"] - before_sets["explained"]),
            "regressed_to_unexplained_plugins": sorted(after_sets["unexplained"] - before_sets["unexplained"]),
            "resolved_unexplained_plugins": sorted(before_sets["unexplained"] - after_sets["unexplained"]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-run-id", required=True)
    parser.add_argument("--after-run-id", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    payload = compare_runs(str(args.before_run_id), str(args.after_run_id))
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if str(args.out or "").strip():
        out_path = Path(str(args.out))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
