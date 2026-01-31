from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def _collect_findings(report: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for plugin in report.get("plugins", {}).values():
        for item in plugin.get("findings", []):
            if item.get("kind") == kind:
                findings.append(item)
    return findings


def evaluate_report(report_path: Path, ground_truth_path: Path) -> tuple[bool, list[str]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    truth = yaml.safe_load(ground_truth_path.read_text(encoding="utf-8"))

    messages: list[str] = []
    ok = True

    expected_features = set(truth.get("features", []))
    found_features = {f.get("feature") for f in _collect_findings(report, "feature_discovery")}
    missing_features = expected_features - found_features
    if missing_features:
        ok = False
        messages.append(f"Missing features: {sorted(missing_features)}")

    expected_changepoints = truth.get("changepoints", [])
    found_cp = [f.get("index") for f in _collect_findings(report, "changepoint")]
    tolerance = truth.get("changepoint_tolerance", 3)
    for expected in expected_changepoints:
        if not any(abs(int(expected) - int(found)) <= tolerance for found in found_cp if found is not None):
            ok = False
            messages.append(f"Changepoint {expected} not found within tolerance")

    expected_pairs = {tuple(pair) for pair in truth.get("dependence_shift_pairs", [])}
    found_pairs = {tuple(f.get("pair", [])) for f in _collect_findings(report, "dependence_shift")}
    missing_pairs = expected_pairs - found_pairs
    if missing_pairs:
        ok = False
        messages.append(f"Missing dependence shift pairs: {sorted(missing_pairs)}")

    expected_anomalies = set(truth.get("anomalies", []))
    found_anomalies = {f.get("row_index") for f in _collect_findings(report, "anomaly")}
    min_hits = truth.get("min_anomaly_hits", len(expected_anomalies))
    if len(expected_anomalies & found_anomalies) < min_hits:
        ok = False
        messages.append("Not enough anomalies detected")

    return ok, messages
