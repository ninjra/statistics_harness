#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any
import yaml
from statistic_harness.core.actionability_explanations import NON_ADJUSTABLE_PROCESSES

BAD_PLUGIN_STATUSES = ("skipped", "degraded", "error", "aborted")
ACTIONABLE_SIGNAL_REASON_ALLOWLIST = {
    "PREREQUISITE_UNMET",
    "PLUGIN_PRECONDITION_UNMET",
    "OBSERVATION_ONLY",
    "NO_STATISTICAL_SIGNAL",
    "NO_ACTIONABLE_FINDING_CLASS",
    "NON_DECISION_PLUGIN",
    "NO_FINDINGS",
    "NO_DECISION_SIGNAL",
    "ADAPTER_RULE_MISSING",
    "NO_DIRECT_PROCESS_TARGET",
    "ACTION_TYPE_POLICY_BLOCK",
    "PLUGIN_ERROR",
    "CAPACITY_IMPACT_CONSTRAINT",
    "NO_MODELED_CAPACITY_GAIN",
    "NO_REVENUE_COMPRESSION_PRESSURE",
    "SHARE_SHIFT_BELOW_THRESHOLD",
    "EXCLUDED_BY_PROCESS_POLICY",
}
LEGACY_REASON_CODE_BLOCKLIST = {"NOT_APPLICABLE", "CAPACITY_IMPACT_NOT_APPLICABLE"}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _fetch_run_row(conn: sqlite3.Connection, run_id: str) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT run_id, status, dataset_version_id, run_seed, requested_run_seed, created_at
        FROM runs
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchone()
    if row is None:
        raise SystemExit(f"Run not found: {run_id}")
    return {
        "run_id": str(row["run_id"] or ""),
        "status": str(row["status"] or ""),
        "dataset_version_id": str(row["dataset_version_id"] or ""),
        "run_seed": row["run_seed"],
        "requested_run_seed": row["requested_run_seed"],
        "created_at": str(row["created_at"] or ""),
    }


def _plugin_status_counts(conn: sqlite3.Connection, run_id: str) -> dict[str, int]:
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
    for row in rows:
        key = str(row["status"] or "unknown").strip().lower() or "unknown"
        counts[key] = int(row["n"] or 0)
    return counts


def _expected_plugin_count(conn: sqlite3.Connection, run_id: str) -> int:
    row = conn.execute(
        """
        SELECT COUNT(*) AS n
        FROM plugin_executions
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchone()
    return int(row["n"] or 0) if row else 0


def _completed_result_count(counts: dict[str, int]) -> int:
    return int(sum(int(v or 0) for v in counts.values()))


def _safe_known_items(report: dict[str, Any]) -> list[dict[str, Any]]:
    recs = report.get("recommendations")
    if not isinstance(recs, dict):
        return []
    known = recs.get("known")
    if not isinstance(known, dict):
        return []
    items = known.get("items")
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _non_independent_known_items(report: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in _safe_known_items(report):
        status = str(item.get("status") or "").strip().lower()
        if status != "confirmed":
            continue
        source = str(item.get("evidence_source") or "").strip().lower()
        if source in {"plugin_findings", "direct_plugin_findings"}:
            continue
        out.append(
            {
                "plugin_id": str(item.get("plugin_id") or "").strip(),
                "kind": str(item.get("kind") or "").strip(),
                "status": status,
                "evidence_source": source or "missing",
            }
        )
    return out


def _safe_recommendation_items(report: dict[str, Any]) -> list[dict[str, Any]]:
    recs = report.get("recommendations")
    if not isinstance(recs, dict):
        return []
    items = recs.get("items")
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _safe_explanation_items(report: dict[str, Any]) -> list[dict[str, Any]]:
    recs = report.get("recommendations")
    if not isinstance(recs, dict):
        return []
    block = recs.get("explanations")
    if not isinstance(block, dict):
        return []
    items = block.get("items")
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _load_direct_action_plugins(root: Path) -> set[str]:
    path = root / "docs" / "plugin_class_taxonomy.yaml"
    if not path.exists():
        return set()
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    if not isinstance(payload, dict):
        return set()
    overrides = (
        payload.get("plugin_overrides")
        if isinstance(payload.get("plugin_overrides"), dict)
        else {}
    )
    out: set[str] = set()
    for plugin_id, class_id in overrides.items():
        pid = str(plugin_id or "").strip()
        cid = str(class_id or "").strip()
        if pid and cid == "direct_action_generators":
            out.add(pid)
    return out


def _has_actionable_signal(plugin_id: str, findings: list[dict[str, Any]]) -> bool:
    pid = str(plugin_id or "").strip()
    for item in findings:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "").strip()
        if kind == "actionable_ops_lever":
            return True
        if pid == "analysis_close_cycle_uplift" and kind == "close_cycle_share_shift":
            process_norm = str(item.get("process_norm") or item.get("process") or "").strip().lower()
            if process_norm and process_norm in NON_ADJUSTABLE_PROCESSES:
                continue
            share_delta = item.get("share_delta")
            if isinstance(share_delta, (int, float)) and float(share_delta) > 0.0:
                return True
        if pid == "analysis_close_cycle_capacity_model" and kind == "close_cycle_capacity_model":
            if str(item.get("decision") or "").strip().lower() != "modeled":
                continue
            baseline = item.get("baseline_value")
            modeled = item.get("modeled_value")
            if (
                isinstance(baseline, (int, float))
                and isinstance(modeled, (int, float))
                and float(modeled) < float(baseline)
            ):
                return True
        if pid == "analysis_close_cycle_capacity_impact" and kind == "close_cycle_capacity_impact":
            if str(item.get("decision") or "").strip().lower() != "detected":
                continue
            effect = item.get("effect")
            if isinstance(effect, (int, float)) and float(effect) < 0.0:
                return True
        if pid == "analysis_close_cycle_revenue_compression" and kind == "close_cycle_revenue_compression":
            if str(item.get("decision") or "").strip().lower() != "modeled":
                continue
            baseline = item.get("baseline_value")
            modeled = item.get("modeled_value")
            if (
                isinstance(baseline, (int, float))
                and isinstance(modeled, (int, float))
                and float(modeled) < float(baseline)
            ):
                return True
    return False


def _known_status(report: dict[str, Any], answers_summary: dict[str, Any]) -> str:
    checks = answers_summary.get("known_issue_checks")
    if isinstance(checks, dict):
        value = str(checks.get("status") or "").strip().lower()
        if value:
            return value
    recs = report.get("recommendations")
    if isinstance(recs, dict):
        known = recs.get("known")
        if isinstance(known, dict):
            value = str(known.get("status") or "").strip().lower()
            if value:
                return value
    return "unknown"


def _known_mode_label(known_status: str) -> str:
    if known_status == "suppressed":
        return "off"
    if known_status in {"ok", "none", "partial", "failing", "no_expected_findings"}:
        return "on"
    return "unknown"


def _confirmed_known_signatures(report: dict[str, Any]) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for item in _safe_known_items(report):
        status = str(item.get("status") or "").strip().lower()
        if status and status != "confirmed":
            continue
        plugin_id = str(item.get("plugin_id") or "").strip()
        kind = str(item.get("kind") or "").strip()
        if plugin_id and kind:
            out.add((plugin_id, kind))
    return out


def _parse_signature(raw: str) -> tuple[str, str]:
    text = str(raw or "").strip()
    if ":" not in text:
        raise SystemExit(f"Invalid signature '{raw}'. Use plugin_id:kind")
    plugin_id, kind = text.split(":", 1)
    plugin_id = plugin_id.strip()
    kind = kind.strip()
    if not plugin_id or not kind:
        raise SystemExit(f"Invalid signature '{raw}'. Use plugin_id:kind")
    return plugin_id, kind


def _seed_value(snapshot: dict[str, Any]) -> int | None:
    for key in ("run_seed", "requested_run_seed"):
        value = snapshot.get(key)
        if isinstance(value, int):
            return int(value)
    return None


def _check(
    check_id: str,
    ok: bool,
    expected: Any,
    actual: Any,
    detail: str = "",
) -> dict[str, Any]:
    return {
        "id": str(check_id),
        "ok": bool(ok),
        "expected": expected,
        "actual": actual,
        "detail": str(detail),
    }


def _build_snapshot(root: Path, conn: sqlite3.Connection, run_id: str) -> dict[str, Any]:
    row = _fetch_run_row(conn, run_id)
    run_dir = root / "appdata" / "runs" / run_id
    report = _read_json(run_dir / "report.json")
    answers_summary = _read_json(run_dir / "answers_summary.json")
    report_plugins = report.get("plugins") if isinstance(report.get("plugins"), dict) else {}
    recommendation_items = _safe_recommendation_items(report)
    explanation_items = _safe_explanation_items(report)
    recommendation_plugin_ids = {
        str(item.get("plugin_id") or "").strip()
        for item in recommendation_items
        if str(item.get("plugin_id") or "").strip()
    }
    explanation_reason_by_plugin = {
        str(item.get("plugin_id") or "").strip(): str(item.get("reason_code") or "").strip()
        for item in explanation_items
        if str(item.get("plugin_id") or "").strip()
    }
    legacy_explanation_reason_codes = sorted(
        {
            str(item.get("reason_code") or "").strip()
            for item in explanation_items
            if str(item.get("reason_code") or "").strip() in LEGACY_REASON_CODE_BLOCKLIST
        }
    )
    direct_action_plugins = _load_direct_action_plugins(root)
    unrouted_direct_action_signals: list[dict[str, Any]] = []
    direct_action_not_routed_reasons: list[dict[str, Any]] = []
    direct_action_invalid_non_actionable_reasons: list[dict[str, Any]] = []
    for plugin_id in sorted(direct_action_plugins):
        payload = (
            report_plugins.get(plugin_id)
            if isinstance(report_plugins.get(plugin_id), dict)
            else None
        )
        if not isinstance(payload, dict):
            continue
        findings = payload.get("findings") if isinstance(payload, dict) and isinstance(payload.get("findings"), list) else []
        typed_findings = [item for item in findings if isinstance(item, dict)]
        reason_code = explanation_reason_by_plugin.get(plugin_id, "")
        has_signal = _has_actionable_signal(plugin_id, typed_findings)
        is_routed = plugin_id in recommendation_plugin_ids
        if has_signal and not is_routed:
            unrouted_direct_action_signals.append(
                {
                    "plugin_id": plugin_id,
                    "reason_code": reason_code or "MISSING_EXPLANATION",
                    "finding_count": int(len(typed_findings)),
                }
            )
        if (
            reason_code == "NOT_ROUTED_TO_ACTION"
            and plugin_id not in recommendation_plugin_ids
            and typed_findings
        ):
            direct_action_not_routed_reasons.append(
                {
                    "plugin_id": plugin_id,
                    "reason_code": reason_code,
                    "finding_count": int(len(typed_findings)),
                }
            )
        if (not is_routed) and (not has_signal):
            if reason_code not in ACTIONABLE_SIGNAL_REASON_ALLOWLIST:
                direct_action_invalid_non_actionable_reasons.append(
                    {
                        "plugin_id": plugin_id,
                        "reason_code": reason_code or "MISSING_REASON_CODE",
                        "finding_count": int(len(typed_findings)),
                    }
                )
    known_status = _known_status(report, answers_summary)
    known_mode = _known_mode_label(known_status)
    non_independent_known_items = _non_independent_known_items(report)
    counts = _plugin_status_counts(conn, run_id)
    expected_plugins = _expected_plugin_count(conn, run_id)
    completed_plugins = _completed_result_count(counts)
    bad_status_counts = {
        status: int(counts.get(status, 0))
        for status in BAD_PLUGIN_STATUSES
        if int(counts.get(status, 0)) > 0
    }
    return {
        "run_id": run_id,
        "status": row["status"],
        "dataset_version_id": row["dataset_version_id"],
        "run_seed": row["run_seed"],
        "requested_run_seed": row["requested_run_seed"],
        "created_at": row["created_at"],
        "known_status": known_status,
        "known_issues_mode": known_mode,
        "known_non_independent_items": non_independent_known_items,
        "known_non_independent_count": int(len(non_independent_known_items)),
        "expected_plugin_results": expected_plugins,
        "completed_plugin_results": completed_plugins,
        "missing_plugin_results": max(0, int(expected_plugins - completed_plugins)),
        "plugin_status_counts": counts,
        "bad_plugin_status_counts": bad_status_counts,
        "confirmed_known_signatures": [
            {"plugin_id": plugin_id, "kind": kind}
            for plugin_id, kind in sorted(_confirmed_known_signatures(report))
        ],
        "direct_action_plugin_count": int(len(direct_action_plugins)),
        "direct_action_not_routed_reasons": direct_action_not_routed_reasons,
        "direct_action_invalid_non_actionable_reasons": direct_action_invalid_non_actionable_reasons,
        "unrouted_direct_action_signals": unrouted_direct_action_signals,
        "unrouted_direct_action_signal_count": int(len(unrouted_direct_action_signals)),
        "legacy_explanation_reason_codes": legacy_explanation_reason_codes,
        "legacy_explanation_reason_code_count": int(len(legacy_explanation_reason_codes)),
        "allowed_non_actionable_reason_codes": sorted(ACTIONABLE_SIGNAL_REASON_ALLOWLIST),
    }


def verify_contract(
    primary: dict[str, Any],
    *,
    required_known_signatures: list[tuple[str, str]],
    expected_known_issues_mode: str,
    compare: dict[str, Any] | None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    checks.append(
        _check(
            "run.completed",
            str(primary.get("status") or "").lower() == "completed",
            "completed",
            primary.get("status"),
            "Run must complete before recommendations are trusted.",
        )
    )
    checks.append(
        _check(
            "run.results_complete",
            int(primary.get("missing_plugin_results") or 0) == 0,
            0,
            int(primary.get("missing_plugin_results") or 0),
            "All scheduled plugins must produce a terminal result record.",
        )
    )
    checks.append(
        _check(
            "run.no_bad_plugin_statuses",
            not bool(primary.get("bad_plugin_status_counts")),
            {},
            primary.get("bad_plugin_status_counts"),
            "skipped/degraded/error/aborted are contract violations.",
        )
    )
    if expected_known_issues_mode in {"on", "off"}:
        checks.append(
            _check(
                "run.known_issues_mode",
                str(primary.get("known_issues_mode") or "") == expected_known_issues_mode,
                expected_known_issues_mode,
                primary.get("known_issues_mode"),
                "Known-issues mode must be explicit and match the requested certification lane.",
            )
        )
    if expected_known_issues_mode == "on":
        checks.append(
            _check(
                "run.known_issues_independent",
                int(primary.get("known_non_independent_count") or 0) == 0,
                0,
                int(primary.get("known_non_independent_count") or 0),
                "Confirmed known-issue landmarks must come from direct plugin findings, not synthetic fallback.",
            )
        )
    confirmed = {
        (str(item.get("plugin_id") or ""), str(item.get("kind") or ""))
        for item in (primary.get("confirmed_known_signatures") or [])
        if isinstance(item, dict)
    }
    if required_known_signatures:
        missing = [
            {"plugin_id": pid, "kind": kind}
            for pid, kind in required_known_signatures
            if (pid, kind) not in confirmed
        ]
        checks.append(
            _check(
                "run.required_known_signatures",
                len(missing) == 0,
                [
                    {"plugin_id": pid, "kind": kind}
                    for pid, kind in required_known_signatures
                ],
                {"missing": missing, "confirmed_count": len(confirmed)},
                "Baseline cert requires known signatures to be confirmed, not merely present.",
            )
        )
    checks.append(
        _check(
            "run.direct_action_not_routed_reason_codes",
            int(len(primary.get("direct_action_not_routed_reasons") or [])) == 0,
            [],
            primary.get("direct_action_not_routed_reasons") or [],
            "Direct-action plugins cannot remain in NOT_ROUTED_TO_ACTION once adapters are implemented.",
        )
    )
    checks.append(
        _check(
            "run.direct_action_signals_routed",
            int(primary.get("unrouted_direct_action_signal_count") or 0) == 0,
            0,
            int(primary.get("unrouted_direct_action_signal_count") or 0),
            "Actionable direct-action findings must be surfaced in recommendations, not explanations.",
        )
    )
    checks.append(
        _check(
            "run.direct_action_non_actionable_reason_codes",
            int(len(primary.get("direct_action_invalid_non_actionable_reasons") or [])) == 0,
            primary.get("allowed_non_actionable_reason_codes") or [],
            primary.get("direct_action_invalid_non_actionable_reasons") or [],
            "Direct-action plugins without recommendations must carry deterministic reason codes.",
        )
    )
    checks.append(
        _check(
            "run.no_legacy_not_applicable_reason_codes",
            int(primary.get("legacy_explanation_reason_code_count") or 0) == 0,
            0,
            int(primary.get("legacy_explanation_reason_code_count") or 0),
            "Legacy NOT_APPLICABLE reason codes are disallowed; explanations must use resolved deterministic reasons.",
        )
    )
    if compare is not None:
        checks.append(
            _check(
                "compare.completed",
                str(compare.get("status") or "").lower() == "completed",
                "completed",
                compare.get("status"),
                "Comparison run must also be completed.",
            )
        )
        checks.append(
            _check(
                "compare.same_dataset",
                str(primary.get("dataset_version_id") or "")
                == str(compare.get("dataset_version_id") or ""),
                primary.get("dataset_version_id"),
                compare.get("dataset_version_id"),
                "Run-to-run quality comparisons must use the same dataset_version_id.",
            )
        )
        checks.append(
            _check(
                "compare.same_seed",
                _seed_value(primary) == _seed_value(compare),
                _seed_value(primary),
                _seed_value(compare),
                "Run comparisons must keep seed policy constant.",
            )
        )
        checks.append(
            _check(
                "compare.same_known_issues_mode",
                str(primary.get("known_issues_mode") or "")
                == str(compare.get("known_issues_mode") or ""),
                primary.get("known_issues_mode"),
                compare.get("known_issues_mode"),
                "Mixed known-issues modes are not valid for accuracy claims.",
            )
        )
        checks.append(
            _check(
                "compare.same_expected_plugin_count",
                int(primary.get("expected_plugin_results") or 0)
                == int(compare.get("expected_plugin_results") or 0),
                int(primary.get("expected_plugin_results") or 0),
                int(compare.get("expected_plugin_results") or 0),
                "Comparisons must run the same plugin gauntlet size.",
            )
        )
    payload = {
        "schema_version": "agent_execution_contract.v1",
        "ok": all(bool(item.get("ok")) for item in checks),
        "primary_run": primary,
        "compare_run": compare,
        "required_known_signatures": [
            {"plugin_id": pid, "kind": kind} for pid, kind in required_known_signatures
        ],
        "checks": checks,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deterministic execution-contract verifier for run certification and run-to-run comparisons."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--compare-run-id", default="")
    parser.add_argument("--root", default=".")
    parser.add_argument("--state-db", default="")
    parser.add_argument("--expected-known-issues-mode", choices=("any", "on", "off"), default="any")
    parser.add_argument("--require-known-signature", action="append", default=[])
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    root = Path(str(args.root)).resolve()
    state_db = Path(str(args.state_db)).resolve() if str(args.state_db).strip() else root / "appdata" / "state.sqlite"
    if not state_db.exists():
        raise SystemExit(f"Missing state DB: {state_db}")
    signatures = [_parse_signature(raw) for raw in list(args.require_known_signature or [])]

    conn = sqlite3.connect(str(state_db))
    conn.row_factory = sqlite3.Row
    try:
        primary = _build_snapshot(root, conn, str(args.run_id).strip())
        compare = None
        compare_run_id = str(args.compare_run_id or "").strip()
        if compare_run_id:
            compare = _build_snapshot(root, conn, compare_run_id)
    finally:
        conn.close()

    payload = verify_contract(
        primary,
        required_known_signatures=signatures,
        expected_known_issues_mode=str(args.expected_known_issues_mode).strip().lower(),
        compare=compare,
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    out_arg = str(args.out or "").strip()
    if out_arg:
        out_path = Path(out_arg)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n", encoding="utf-8")
        print(str(out_path))
    else:
        print(rendered)
    return 0 if bool(payload.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
