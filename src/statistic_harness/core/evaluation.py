from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def _parse_tolerance(value: Any, default_abs: float = 0.0) -> dict[str, float]:
    if value is None:
        return {"absolute": float(default_abs), "relative": 0.0}
    if isinstance(value, (int, float)):
        return {"absolute": float(value), "relative": 0.0}
    if isinstance(value, dict):
        abs_tol = value.get("absolute", value.get("abs", default_abs))
        rel_tol = value.get("relative", value.get("rel", 0.0))
        try:
            abs_val = float(abs_tol)
        except (TypeError, ValueError):
            abs_val = float(default_abs)
        try:
            rel_val = float(rel_tol)
        except (TypeError, ValueError):
            rel_val = 0.0
        return {"absolute": abs_val, "relative": rel_val}
    return {"absolute": float(default_abs), "relative": 0.0}


def _within_tolerance(expected: float, actual: float, tol: dict[str, float]) -> bool:
    diff = abs(float(expected) - float(actual))
    abs_tol = tol.get("absolute", 0.0)
    rel_tol = tol.get("relative", 0.0)
    if diff <= abs_tol:
        return True
    if rel_tol <= 0:
        return False
    return diff <= rel_tol * max(1.0, abs(float(expected)))


def _collect_findings(report: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for plugin in report.get("plugins", {}).values():
        for item in plugin.get("findings", []):
            if item.get("kind") == kind:
                findings.append(item)
    return findings


def _collect_findings_for_plugin(
    report: dict[str, Any], plugin_id: str | None, kind: str | None = None
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    plugins = report.get("plugins", {})
    for pid, plugin in plugins.items():
        if plugin_id and pid != plugin_id:
            continue
        for item in plugin.get("findings", []):
            if kind and item.get("kind") != kind:
                continue
            findings.append(item)
    return findings


def _matches_expected(
    item: dict[str, Any],
    where: dict[str, Any] | None,
    contains: dict[str, Any] | None,
) -> bool:
    if where:
        for key, expected in where.items():
            actual = item.get(key)
            if actual != expected:
                return False
    if contains:
        for key, expected in contains.items():
            actual = item.get(key)
            if isinstance(actual, str):
                if str(expected) not in actual:
                    return False
            elif isinstance(actual, (list, tuple, set)):
                if isinstance(expected, (list, tuple, set)):
                    if not set(expected).issubset(set(actual)):
                        return False
                else:
                    if expected not in actual:
                        return False
            else:
                return False
    return True


def evaluate_report(
    report_path: Path, ground_truth_path: Path
) -> tuple[bool, list[str]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    truth = yaml.safe_load(ground_truth_path.read_text(encoding="utf-8"))
    if not isinstance(truth, dict):
        truth = {}

    messages: list[str] = []
    ok = True

    if "strict" in truth:
        strict = bool(truth.get("strict"))
    else:
        strict = True

    expected_features = set(truth.get("features", []))
    found_features = {
        f.get("feature") for f in _collect_findings(report, "feature_discovery")
    }
    if strict:
        unexpected_features = found_features - expected_features
        if unexpected_features:
            ok = False
            messages.append(f"Unexpected features: {sorted(unexpected_features)}")
    missing_features = expected_features - found_features
    if missing_features:
        ok = False
        messages.append(f"Missing features: {sorted(missing_features)}")

    expected_changepoints = truth.get("changepoints", [])
    found_cp = [f.get("index") for f in _collect_findings(report, "changepoint")]
    tolerance = _parse_tolerance(truth.get("changepoint_tolerance", 3), default_abs=3)
    for expected in expected_changepoints:
        if isinstance(expected, dict):
            expected_value = expected.get("index")
            tolerance = _parse_tolerance(
                expected.get("tolerance", tolerance), default_abs=tolerance["absolute"]
            )
        else:
            expected_value = expected
        if expected_value is None:
            continue
        if not any(
            _within_tolerance(float(expected_value), float(found), tolerance)
            for found in found_cp
            if found is not None
        ):
            ok = False
            messages.append(
                f"Changepoint {expected_value} not found within tolerance"
            )
    if strict and expected_changepoints is not None:
        expected_values = []
        for expected in expected_changepoints:
            if isinstance(expected, dict):
                value = expected.get("index")
                tol = _parse_tolerance(
                    expected.get("tolerance", tolerance), default_abs=tolerance["absolute"]
                )
            else:
                value = expected
                tol = tolerance
            if value is None:
                continue
            expected_values.append((float(value), tol))
        for found in found_cp:
            if found is None:
                continue
            if not any(
                _within_tolerance(float(found), expected, tol)
                for expected, tol in expected_values
            ):
                ok = False
                messages.append(f"Unexpected changepoint: {found}")

    expected_pairs = {tuple(pair) for pair in truth.get("dependence_shift_pairs", [])}
    found_pairs = {
        tuple(f.get("pair", [])) for f in _collect_findings(report, "dependence_shift")
    }
    if strict:
        unexpected_pairs = found_pairs - expected_pairs
        if unexpected_pairs:
            ok = False
            messages.append(
                f"Unexpected dependence shift pairs: {sorted(unexpected_pairs)}"
            )
    missing_pairs = expected_pairs - found_pairs
    if missing_pairs:
        ok = False
        messages.append(f"Missing dependence shift pairs: {sorted(missing_pairs)}")

    expected_anomalies = truth.get("anomalies", [])
    found_anomalies = [f.get("row_index") for f in _collect_findings(report, "anomaly")]
    min_hits = truth.get("min_anomaly_hits", len(expected_anomalies))
    anomaly_tol = _parse_tolerance(
        truth.get("anomaly_tolerance", 0), default_abs=0
    )
    matches = 0
    for expected in expected_anomalies:
        if isinstance(expected, dict):
            expected_value = expected.get("row_index")
            tol = _parse_tolerance(
                expected.get("tolerance", anomaly_tol),
                default_abs=anomaly_tol["absolute"],
            )
        else:
            expected_value = expected
            tol = anomaly_tol
        if expected_value is None:
            continue
        if any(
            _within_tolerance(float(expected_value), float(found), tol)
            for found in found_anomalies
            if found is not None
        ):
            matches += 1
    if matches < min_hits:
        ok = False
        messages.append("Not enough anomalies detected")
    if strict and expected_anomalies is not None:
        expected_rows = []
        for expected in expected_anomalies:
            if isinstance(expected, dict):
                value = expected.get("row_index")
                tol = _parse_tolerance(
                    expected.get("tolerance", anomaly_tol),
                    default_abs=anomaly_tol["absolute"],
                )
            else:
                value = expected
                tol = anomaly_tol
            if value is None:
                continue
            expected_rows.append((float(value), tol))
        for found in found_anomalies:
            if found is None:
                continue
            if not any(
                _within_tolerance(float(found), expected, tol)
                for expected, tol in expected_rows
            ):
                ok = False
                messages.append(f"Unexpected anomaly: {found}")

    expected_findings = truth.get("expected_findings", []) or []
    expected_matchers: list[dict[str, Any]] = []
    for entry in expected_findings:
        if not isinstance(entry, dict):
            continue
        plugin_id = entry.get("plugin_id")
        kind = entry.get("kind")
        if not kind:
            continue
        where = entry.get("where") or {}
        contains = entry.get("contains") or {}
        min_count = entry.get("min_count", 1)
        max_count = entry.get("max_count")
        candidates = _collect_findings_for_plugin(report, plugin_id, kind)
        matches = [
            item
            for item in candidates
            if _matches_expected(item, where, contains)
        ]
        if len(matches) < int(min_count):
            ok = False
            messages.append(
                f"Expected finding missing: kind={kind} plugin={plugin_id or '*'}"
            )
        if max_count is not None and len(matches) > int(max_count):
            ok = False
            messages.append(
                f"Too many matches for kind={kind} plugin={plugin_id or '*'}"
            )
        expected_matchers.append(
            {
                "plugin_id": plugin_id,
                "kind": kind,
                "where": where,
                "contains": contains,
            }
        )

    if strict and expected_matchers:
        for plugin_id, plugin in report.get("plugins", {}).items():
            for item in plugin.get("findings", []):
                kind = item.get("kind")
                if not kind:
                    continue
                if not any(matcher["kind"] == kind for matcher in expected_matchers):
                    continue
                matched = False
                for matcher in expected_matchers:
                    if matcher["kind"] != kind:
                        continue
                    if matcher["plugin_id"] and matcher["plugin_id"] != plugin_id:
                        continue
                    if _matches_expected(item, matcher["where"], matcher["contains"]):
                        matched = True
                        break
                if not matched:
                    ok = False
                    messages.append(
                        f"Unexpected finding: kind={kind} plugin={plugin_id}"
                    )

    expected_metrics = truth.get("expected_metrics", []) or []
    for entry in expected_metrics:
        if not isinstance(entry, dict):
            continue
        plugin_id = entry.get("plugin_id")
        metric_path = entry.get("metric")
        if not plugin_id or not metric_path:
            continue
        plugin = report.get("plugins", {}).get(plugin_id, {})
        metrics = plugin.get("metrics", {})
        if not isinstance(metrics, dict):
            ok = False
            messages.append(f"Metrics missing for plugin={plugin_id}")
            continue
        value = metrics
        for part in str(metric_path).split("."):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                value = None
                break
        if isinstance(value, dict) and "value" in value:
            value = value.get("value")
        if value is None:
            ok = False
            messages.append(f"Expected metric missing: {plugin_id}.{metric_path}")
            continue
        try:
            actual_val = float(value)
        except (TypeError, ValueError):
            ok = False
            messages.append(f"Expected metric not numeric: {plugin_id}.{metric_path}")
            continue
        expected_val = entry.get("value")
        try:
            expected_val_f = float(expected_val)
        except (TypeError, ValueError):
            ok = False
            messages.append(f"Expected metric invalid: {plugin_id}.{metric_path}")
            continue
        tol = _parse_tolerance(entry.get("tolerance", 0), default_abs=0.0)
        if not _within_tolerance(expected_val_f, actual_val, tol):
            ok = False
            messages.append(
                f"Metric out of tolerance: {plugin_id}.{metric_path} "
                f"expected={expected_val_f} actual={actual_val}"
            )

    return ok, messages
