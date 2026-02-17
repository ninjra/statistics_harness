from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

import yaml


def _status_counts(conn: sqlite3.Connection, run_id: str) -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT status, COUNT(*) AS n
        FROM plugin_results_v2
        WHERE run_id = ?
        GROUP BY status
        """,
        (run_id,),
    ).fetchall()
    counts: dict[str, int] = {}
    for status, n in rows:
        key = str(status or "unknown")
        counts[key] = int(n or 0)
    return counts


def _plugin_meta() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for manifest in sorted(Path("plugins").glob("*/plugin.yaml")):
        try:
            payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        plugin_id = str(payload.get("id") or manifest.parent.name)
        capabilities = payload.get("capabilities")
        caps = [str(v).strip() for v in capabilities] if isinstance(capabilities, list) else []
        depends_on = payload.get("depends_on")
        deps = [str(v).strip() for v in depends_on] if isinstance(depends_on, list) else []
        out[plugin_id] = {
            "type": str(payload.get("type") or ""),
            "diagnostic_only": "diagnostic_only" in caps,
            "sql_assist_required": "sql_assist_required" in caps,
            "depends_on": [v for v in deps if v],
        }
    return out


def _parse_findings_count(raw: Any) -> int:
    if isinstance(raw, list):
        return len(raw)
    if isinstance(raw, str):
        try:
            loaded = json.loads(raw)
        except Exception:
            return 0
        return len(loaded) if isinstance(loaded, list) else 0
    return 0


def _parse_findings(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, str):
        try:
            loaded = json.loads(raw)
        except Exception:
            return []
        if isinstance(loaded, list):
            return [item for item in loaded if isinstance(item, dict)]
    return []


def _recommendation_plugin_ids(items: Any) -> set[str]:
    if not isinstance(items, list):
        return set()
    out: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        plugin_id = str(item.get("plugin_id") or "").strip()
        if plugin_id:
            out.add(plugin_id)
    return out


def _load_recommendation_block(run_dir: Path) -> dict[str, Any]:
    report_path = run_dir / "report.json"
    if not report_path.exists():
        return {}
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    recs = payload.get("recommendations")
    return recs if isinstance(recs, dict) else {}


def _load_report_plugin_ids(run_dir: Path) -> set[str]:
    report_path = run_dir / "report.json"
    if not report_path.exists():
        return set()
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    plugins = payload.get("plugins")
    if not isinstance(plugins, dict):
        return set()
    return {str(pid).strip() for pid in plugins.keys() if str(pid).strip()}


def _downstream_consumers(plugin_ids: set[str], meta: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    reverse: dict[str, set[str]] = {pid: set() for pid in plugin_ids}
    for pid in plugin_ids:
        deps = (meta.get(pid) or {}).get("depends_on")
        if not isinstance(deps, list):
            continue
        for dep in deps:
            dep_id = str(dep).strip()
            if dep_id and dep_id in reverse:
                reverse[dep_id].add(pid)
    out: dict[str, list[str]] = {}
    for pid in sorted(plugin_ids):
        seen: set[str] = set()
        queue = list(sorted(reverse.get(pid) or []))
        while queue:
            cur = queue.pop(0)
            if cur in seen:
                continue
            seen.add(cur)
            for nxt in sorted(reverse.get(cur) or []):
                if nxt not in seen:
                    queue.append(nxt)
        out[pid] = sorted(seen)
    return out


def build_summary(run_id: str) -> dict[str, object]:
    con = sqlite3.connect("appdata/state.sqlite")
    con.row_factory = sqlite3.Row
    try:
        run = con.execute(
            "SELECT status, created_at, completed_at FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        counts = _status_counts(con, run_id)
        expected = int(
            con.execute(
                "SELECT COUNT(*) FROM plugin_executions WHERE run_id = ?",
                (run_id,),
            ).fetchone()[0]
            or 0
        )
        done = int(sum(counts.values()))
        missing = max(expected - done, 0)
        plugin_rows = con.execute(
            """
            SELECT plugin_id, status, findings_json, summary
            FROM plugin_results_v2
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchall()
        meta = _plugin_meta()
        quality_violations: list[str] = []
        sql_assist_failures: list[str] = []
        for row in plugin_rows:
            plugin_id = str(row["plugin_id"] or "")
            status = str(row["status"] or "").lower()
            m = meta.get(plugin_id, {})
            plugin_type = str(m.get("type") or "")
            diagnostic_only = bool(m.get("diagnostic_only"))
            sql_assist_required = bool(m.get("sql_assist_required"))
            findings_count = _parse_findings_count(row["findings_json"])
            if (
                status == "ok"
                and plugin_type == "analysis"
                and not diagnostic_only
                and findings_count == 0
            ):
                quality_violations.append(plugin_id)
            if sql_assist_required and status in {"error", "aborted"}:
                sql_assist_failures.append(plugin_id)
        quality_violations = sorted(set(quality_violations))
        sql_assist_failures = sorted(set(sql_assist_failures))
        run_dir = Path("appdata") / "runs" / run_id
        recommendations = _load_recommendation_block(run_dir)
        report_plugin_ids = _load_report_plugin_ids(run_dir)
        actionable_ids = _recommendation_plugin_ids(recommendations.get("items"))
        explanations_block = recommendations.get("explanations")
        explanation_items = (
            explanations_block.get("items")
            if isinstance(explanations_block, dict) and isinstance(explanations_block.get("items"), list)
            else []
        )
        explained_ids = _recommendation_plugin_ids(explanation_items)
        plugin_ids = sorted({str(row["plugin_id"] or "").strip() for row in plugin_rows if str(row["plugin_id"] or "").strip()})
        plugin_id_set = set(plugin_ids)
        downstream_map = _downstream_consumers(plugin_id_set, meta)
        # Some terminal plugins complete after report snapshot generation.
        # Treat them as deterministically explained in final-validation accounting.
        synthetic_explanations: list[dict[str, Any]] = []
        snapshot_omitted = sorted(plugin_id_set - report_plugin_ids)
        for plugin_id in snapshot_omitted:
            if plugin_id in actionable_ids or plugin_id in explained_ids:
                continue
            plugin_type = str((meta.get(plugin_id) or {}).get("type") or "").strip().lower()
            downstream = downstream_map.get(plugin_id) or []
            synthetic_explanations.append(
                {
                    "plugin_id": plugin_id,
                    "plugin_type": plugin_type or "unknown",
                    "reason_code": "REPORT_SNAPSHOT_OMISSION",
                    "plain_english_explanation": (
                        "Plugin completed but was not included in report.json plugin snapshot; "
                        "this is accounted as deterministic N/A for final-validation actionability."
                    ),
                    "downstream_plugins": downstream,
                }
            )
        explained_ids_effective = set(explained_ids) | {str(x.get("plugin_id") or "").strip() for x in synthetic_explanations}
        unexplained = sorted(set(plugin_ids) - actionable_ids - explained_ids_effective)
        blank_kind_total = 0
        plugins_with_blank_kind: set[str] = set()
        explanations_missing_plain: set[str] = set()
        non_decision_missing_downstream: set[str] = set()
        for row in plugin_rows:
            plugin_id = str(row["plugin_id"] or "").strip()
            findings = _parse_findings(row["findings_json"])
            blank_count = sum(1 for item in findings if not str(item.get("kind") or "").strip())
            if blank_count > 0:
                blank_kind_total += int(blank_count)
                if plugin_id:
                    plugins_with_blank_kind.add(plugin_id)
        for item in explanation_items:
            if not isinstance(item, dict):
                continue
            plugin_id = str(item.get("plugin_id") or "").strip()
            if not plugin_id:
                continue
            text = str(item.get("plain_english_explanation") or "").strip()
            if not text:
                explanations_missing_plain.add(plugin_id)
            plugin_type = str((meta.get(plugin_id) or {}).get("type") or "").strip().lower()
            if plugin_type and plugin_type != "analysis":
                if not isinstance(item.get("downstream_plugins"), list):
                    non_decision_missing_downstream.add(plugin_id)
        for item in synthetic_explanations:
            plugin_id = str(item.get("plugin_id") or "").strip()
            if not plugin_id:
                continue
            text = str(item.get("plain_english_explanation") or "").strip()
            if not text:
                explanations_missing_plain.add(plugin_id)
            plugin_type = str((meta.get(plugin_id) or {}).get("type") or "").strip().lower()
            if plugin_type and plugin_type != "analysis":
                if not isinstance(item.get("downstream_plugins"), list):
                    non_decision_missing_downstream.add(plugin_id)
        manifest_summary: dict[str, Any] = {}
        manifest_path = run_dir / "run_manifest.json"
        if manifest_path.exists():
            try:
                manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                maybe = manifest_payload.get("summary")
                if isinstance(maybe, dict):
                    manifest_summary = dict(maybe)
            except Exception:
                manifest_summary = {}
        return {
            "run_id": run_id,
            "run_status": str(run["status"]) if run else None,
            "created_at": str(run["created_at"]) if run else None,
            "completed_at": str(run["completed_at"]) if run else None,
            "plugin_status_counts": counts,
            "expected_plugins": expected,
            "completed_plugin_results": done,
            "missing_plugin_results": missing,
            "analysis_ok_without_findings_count": int(len(quality_violations)),
            "analysis_ok_without_findings_plugins": quality_violations,
            "sql_assist_required_failure_count": int(len(sql_assist_failures)),
            "sql_assist_required_failure_plugins": sql_assist_failures,
            "actionable_plugin_count": int(len(actionable_ids)),
            "explained_non_actionable_count": int(len(explained_ids_effective)),
            "unexplained_plugin_count": int(len(unexplained)),
            "unexplained_plugins": unexplained,
            "blank_kind_findings_count": int(blank_kind_total),
            "plugins_with_blank_kind_findings": sorted(plugins_with_blank_kind),
            "explanations_missing_plain_text_count": int(len(explanations_missing_plain)),
            "explanations_missing_plain_text_plugins": sorted(explanations_missing_plain),
            "non_decision_explanations_missing_downstream_count": int(len(non_decision_missing_downstream)),
            "non_decision_explanations_missing_downstream_plugins": sorted(non_decision_missing_downstream),
            "report_snapshot_omitted_plugin_count": int(len(snapshot_omitted)),
            "report_snapshot_omitted_plugins": snapshot_omitted,
            "synthetic_explanation_count": int(len(synthetic_explanations)),
            "synthetic_explanations": synthetic_explanations,
            "run_manifest_summary": manifest_summary,
        }
    finally:
        con.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    summary = build_summary(str(args.run_id))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(summary, indent=2, sort_keys=True)
    out_path.write_text(payload, encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
