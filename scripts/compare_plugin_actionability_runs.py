from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _latest_run_for_dataset(dataset_version_id: str) -> str:
    state_path = Path("appdata") / "state.sqlite"
    if not state_path.exists():
        raise SystemExit(f"Missing state DB: {state_path}")
    import sqlite3

    conn = sqlite3.connect(str(state_path))
    try:
        row = conn.execute(
            """
            SELECT run_id
            FROM runs
            WHERE dataset_version_id = ? AND status = 'completed'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (dataset_version_id,),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        raise SystemExit(f"No completed runs found for dataset_version_id={dataset_version_id}")
    return str(row[0])


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


def _normalized_targets(item: dict[str, Any]) -> str:
    where = item.get("where") if isinstance(item.get("where"), dict) else {}
    target_parts: list[str] = []
    if where:
        for key in sorted(where.keys()):
            value = where.get(key)
            if isinstance(value, (str, int, float, bool)):
                target_parts.append(f"{key}={value}")
            elif isinstance(value, list):
                normalized = ",".join(sorted(str(v).strip() for v in value if str(v).strip()))
                if normalized:
                    target_parts.append(f"{key}=[{normalized}]")
    targets = item.get("target_process_ids")
    if isinstance(targets, list):
        normalized_targets = ",".join(sorted(str(v).strip() for v in targets if str(v).strip()))
        if normalized_targets:
            target_parts.append(f"target_process_ids=[{normalized_targets}]")
    return "|".join(target_parts)


def _recommendation_signature(item: dict[str, Any]) -> str:
    plugin_id = str(item.get("plugin_id") or "").strip()
    kind = str(item.get("kind") or "").strip()
    action_type = str(item.get("action_type") or item.get("action") or "").strip()
    scope = str(item.get("scope_class") or "").strip()
    recommendation = str(item.get("recommendation") or "").strip().lower()
    targets = _normalized_targets(item)
    recommendation_hash = hashlib.sha256(recommendation.encode("utf-8")).hexdigest()[:16]
    return "||".join([plugin_id, kind, action_type, scope, targets, recommendation_hash])


def _recommendation_signatures(items: Any) -> set[str]:
    if not isinstance(items, list):
        return set()
    signatures: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        signatures.add(_recommendation_signature(item))
    return signatures


def _actionability_sets(report: dict[str, Any]) -> dict[str, set[str]]:
    plugins = report.get("plugins") if isinstance(report.get("plugins"), dict) else {}
    plugin_ids = {str(pid).strip() for pid in plugins.keys() if str(pid).strip()}
    recs = report.get("recommendations") if isinstance(report.get("recommendations"), dict) else {}
    all_items = recs.get("items") if isinstance(recs.get("items"), list) else []
    known_items = ((recs.get("known") or {}).get("items") if isinstance(recs.get("known"), dict) else []) or []
    discovery_items = ((recs.get("discovery") or {}).get("items") if isinstance(recs.get("discovery"), dict) else []) or []
    actionable = _ids_from_items(all_items)
    explanations = recs.get("explanations") if isinstance(recs.get("explanations"), dict) else {}
    explained = _ids_from_items(explanations.get("items"))
    unexplained = plugin_ids - actionable - explained
    all_signatures = _recommendation_signatures(all_items)
    known_signatures = _recommendation_signatures(known_items)
    discovery_signatures = _recommendation_signatures(discovery_items)
    return {
        "plugins": plugin_ids,
        "actionable": actionable,
        "explained": explained,
        "unexplained": unexplained,
        "all_signatures": all_signatures,
        "known_signatures": known_signatures,
        "discovery_signatures": discovery_signatures,
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return float(len(left & right)) / float(len(union))


def compare_runs(before_run_id: str, after_run_id: str) -> dict[str, Any]:
    before_report = _load_report(before_run_id)
    after_report = _load_report(after_run_id)
    before_sets = _actionability_sets(before_report)
    after_sets = _actionability_sets(after_report)
    before_dataset_version_id = str((((before_report.get("lineage") or {}).get("dataset") or {}).get("dataset_version_id") or "").strip())
    after_dataset_version_id = str((((after_report.get("lineage") or {}).get("dataset") or {}).get("dataset_version_id") or "").strip())
    new_signatures = sorted(after_sets["all_signatures"] - before_sets["all_signatures"])
    dropped_signatures = sorted(before_sets["all_signatures"] - after_sets["all_signatures"])
    unchanged_signatures = sorted(before_sets["all_signatures"] & after_sets["all_signatures"])
    discovery_new = sorted(after_sets["discovery_signatures"] - before_sets["discovery_signatures"])
    discovery_dropped = sorted(before_sets["discovery_signatures"] - after_sets["discovery_signatures"])
    discovery_unchanged = sorted(before_sets["discovery_signatures"] & after_sets["discovery_signatures"])
    return {
        "before_run_id": before_run_id,
        "after_run_id": after_run_id,
        "before_dataset_version_id": before_dataset_version_id,
        "after_dataset_version_id": after_dataset_version_id,
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
            "new_recommendation_signatures": new_signatures,
            "dropped_recommendation_signatures": dropped_signatures,
            "unchanged_recommendation_signatures": unchanged_signatures,
            "new_discovery_signatures": discovery_new,
            "dropped_discovery_signatures": discovery_dropped,
            "unchanged_discovery_signatures": discovery_unchanged,
        },
        "novelty": {
            "all": {
                "new_count": len(new_signatures),
                "dropped_count": len(dropped_signatures),
                "unchanged_count": len(unchanged_signatures),
                "jaccard": _jaccard(before_sets["all_signatures"], after_sets["all_signatures"]),
            },
            "discovery": {
                "new_count": len(discovery_new),
                "dropped_count": len(discovery_dropped),
                "unchanged_count": len(discovery_unchanged),
                "jaccard": _jaccard(before_sets["discovery_signatures"], after_sets["discovery_signatures"]),
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-run-id", default="")
    parser.add_argument("--after-run-id", default="")
    parser.add_argument("--before-dataset-version-id", default="")
    parser.add_argument("--after-dataset-version-id", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    before_run_id = str(args.before_run_id).strip()
    after_run_id = str(args.after_run_id).strip()
    before_dataset_version_id = str(args.before_dataset_version_id).strip()
    after_dataset_version_id = str(args.after_dataset_version_id).strip()
    if not before_run_id:
        if not before_dataset_version_id:
            raise SystemExit("Provide --before-run-id or --before-dataset-version-id")
        before_run_id = _latest_run_for_dataset(before_dataset_version_id)
    if not after_run_id:
        if not after_dataset_version_id:
            raise SystemExit("Provide --after-run-id or --after-dataset-version-id")
        after_run_id = _latest_run_for_dataset(after_dataset_version_id)

    payload = compare_runs(before_run_id, after_run_id)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if str(args.out or "").strip():
        out_path = Path(str(args.out))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
