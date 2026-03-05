from __future__ import annotations

import re
from typing import Any


def _normalize_process_hint(raw: Any) -> str:
    if not isinstance(raw, str):
        return ""
    token = raw.strip().lower()
    if token.startswith("proc:"):
        token = token[5:].strip()
    if token in {"", "(multiple)", "multiple", "all", "any", "global"}:
        return ""
    return token


def _finding_recommendation_text(finding: dict[str, Any]) -> str:
    text = str(finding.get("recommendation") or "").strip()
    if text:
        return text
    recs = finding.get("recommendations")
    if isinstance(recs, list):
        for value in recs:
            token = str(value or "").strip()
            if token:
                return token
    return ""


def _known_process_terms_from_report(report: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    plugins = report.get("plugins")
    if not isinstance(plugins, dict):
        return out
    for payload in plugins.values():
        if not isinstance(payload, dict):
            continue
        findings = payload.get("findings") if isinstance(payload.get("findings"), list) else []
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            for key in ("process_norm", "process", "process_id"):
                token = _normalize_process_hint(finding.get(key))
                if token:
                    out.add(token)
            where = finding.get("where") if isinstance(finding.get("where"), dict) else {}
            for key in ("process_norm", "process", "process_id"):
                token = _normalize_process_hint(where.get(key))
                if token:
                    out.add(token)
            evidence = finding.get("evidence") if isinstance(finding.get("evidence"), dict) else {}
            for key in ("process_norm", "process", "process_id"):
                token = _normalize_process_hint(evidence.get(key))
                if token:
                    out.add(token)
            metrics = evidence.get("metrics") if isinstance(evidence.get("metrics"), dict) else {}
            token = _normalize_process_hint(metrics.get("process"))
            if token:
                out.add(token)
    return out


def _infer_process_from_text(text: str, known_process_terms: set[str]) -> str:
    payload = str(text or "").strip().lower()
    if not payload or not known_process_terms:
        return ""
    scored: list[tuple[int, int, str]] = []
    for token in known_process_terms:
        if len(token) < 3:
            continue
        pattern = rf"\b{re.escape(token)}\b"
        if not re.search(pattern, payload):
            continue
        scored.append((len(token), payload.find(token), token))
    if not scored:
        return ""
    scored.sort(reverse=True)
    return scored[0][2]


def _target_process_ids_for_finding(finding: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for key in ("process_norm", "process", "process_id"):
        token = _normalize_process_hint(finding.get(key))
        if token:
            out.append(token)
    evidence = finding.get("evidence") if isinstance(finding.get("evidence"), dict) else {}
    for bucket in (finding, evidence):
        raw = bucket.get("target_process_ids") if isinstance(bucket, dict) else None
        if isinstance(raw, list):
            for value in raw:
                token = _normalize_process_hint(value)
                if token:
                    out.append(token)
    selected = evidence.get("selected") if isinstance(evidence.get("selected"), list) else []
    for row in selected:
        if not isinstance(row, dict):
            continue
        for key in ("process_norm", "process", "process_id"):
            token = _normalize_process_hint(row.get(key))
            if token:
                out.append(token)
    for key in ("processes", "process_ids"):
        raw_list = evidence.get(key) if isinstance(evidence.get(key), list) else []
        for value in raw_list:
            token = _normalize_process_hint(value)
            if token:
                out.append(token)
    metrics = evidence.get("metrics") if isinstance(evidence.get("metrics"), dict) else {}
    for key in ("process", "process_norm", "process_id"):
        token = _normalize_process_hint(metrics.get(key))
        if token:
            out.append(token)
    top_rows = evidence.get("top_spillover_processes") if isinstance(evidence.get("top_spillover_processes"), list) else []
    for row in top_rows:
        if not isinstance(row, dict):
            continue
        for key in ("process_norm", "process", "process_id"):
            token = _normalize_process_hint(row.get(key))
            if token:
                out.append(token)
    # Extract process hints from where.group labels.
    where = finding.get("where") if isinstance(finding.get("where"), dict) else {}
    group_raw = where.get("group")
    if isinstance(group_raw, dict):
        # Dict-style group: {"PROCESS_ID": "jbcreateje"} or {"MODULE_CD": "qra"}
        for _gv in group_raw.values():
            token = _normalize_process_hint(_gv)
            if token:
                out.append(token)
    elif isinstance(group_raw, str) and "=" in group_raw:
        # String-style group: "process_id=qemail"
        _gl_val = group_raw.split("=", 1)[1].strip()
        token = _normalize_process_hint(_gl_val)
        if token:
            out.append(token)
    for key in ("process_norm", "process", "process_id"):
        token = _normalize_process_hint(where.get(key))
        if token:
            out.append(token)
    deduped: list[str] = []
    seen: set[str] = set()
    for token in out:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped


def _has_backstop_decision_signal(plugin_id: str, findings: list[dict[str, Any]]) -> bool:
    decision_plugin = not str(plugin_id).startswith(
        ("profile_", "transform_", "report_", "llm_", "planner_", "ingest_")
    )
    if not findings:
        return decision_plugin
    non_decision_kinds = {
        "",
        "plugin_not_applicable",
        "plugin_observation",
        "analysis_no_action_diagnostic",
        "profile_overview",
    }
    if plugin_id == "analysis_actionable_ops_levers_v1":
        actionable_rows = [
            row
            for row in findings
            if str(row.get("kind") or "").strip().lower() == "actionable_ops_lever"
        ]
        if actionable_rows and not any(_target_process_ids_for_finding(row) for row in actionable_rows):
            return False
    saw_any_signal = False
    saw_non_decision_signal = False
    for row in findings:
        saw_any_signal = True
        kind = str(row.get("kind") or "").strip().lower()
        if kind in non_decision_kinds:
            saw_non_decision_signal = True
            continue
        if _finding_recommendation_text(row):
            return True
        if kind in {
            "anomaly",
            "changepoint",
            "cluster",
            "process_variant",
            "tail_isolation",
            "sequence_classification",
            "recommendation",
            "chi_square_association",
            "close_cycle_capacity_model",
            "capacity_scale_model",
        }:
            return True
    if decision_plugin and (saw_non_decision_signal or saw_any_signal):
        return True
    return False


def _extract_process_queue_ids_from_finding(finding: dict[str, Any]) -> list[int]:
    texts: list[str] = []
    for key in ("recommendation", "title", "what", "why"):
        value = finding.get(key)
        if isinstance(value, str) and value.strip():
            texts.append(value)
    recs = finding.get("recommendations")
    if isinstance(recs, list):
        for value in recs:
            if isinstance(value, str) and value.strip():
                texts.append(value)
    evidence = finding.get("evidence") if isinstance(finding.get("evidence"), dict) else {}
    for key in ("itemset", "antecedents", "consequents"):
        values = evidence.get(key)
        if isinstance(values, list):
            for token in values:
                if isinstance(token, str) and token.strip():
                    texts.append(token)

    out: list[int] = []
    seen: set[int] = set()
    for text in texts:
        payload = str(text or "")
        matches = re.findall(r"process\s*queue\s*id\((\d+)\)", payload, flags=re.IGNORECASE)
        matches += re.findall(r"process\s*id\((\d+)\)", payload, flags=re.IGNORECASE)
        for raw in matches:
            try:
                value = int(raw)
            except (TypeError, ValueError):
                continue
            if value <= 0 or value in seen:
                continue
            seen.add(value)
            out.append(value)
    return out


def _process_hint(where: dict[str, Any] | None) -> str:
    if not where:
        return ""
    for key in ("process", "process_norm", "process_name", "activity"):
        value = where.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _recommendation_process_hint(item: dict[str, Any]) -> str:
    where = item.get("where") if isinstance(item.get("where"), dict) else None
    contains = item.get("contains") if isinstance(item.get("contains"), dict) else None
    hint = _process_hint(where) or _process_hint(contains)
    if isinstance(hint, str) and hint.strip():
        return hint.strip()
    evidence = item.get("evidence")
    if isinstance(evidence, list):
        for row in evidence:
            if not isinstance(row, dict):
                continue
            for key in ("process", "process_norm", "process_id"):
                val = row.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
    return ""
