#!/usr/bin/env python3
"""Verify deterministic output for all 275 plugins.

For each plugin:
  1. Run twice with run_seed=42 -> outputs must be identical (canonical JSON)
  2. Run once with run_seed=99 -> if identical to seed=42, flag hardcoded_seed

Usage:
    .venv/bin/python scripts/verify_determinism.py
"""
from __future__ import annotations

import ast
import importlib
import json
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from plugin_verifier_lib.check_runners import (
    CheckResult,
    VerificationResult,
)
from plugin_verifier_lib.issue_classifier import classify_verification_failure
from plugin_verifier_lib.minimal_context import make_minimal_context
from plugin_verifier_lib.report_builder import build_report
from plugin_verifier_lib.synthetic_datasets import ds_determinism

from statistic_harness.core.stat_plugins.registry import HANDLERS, run_plugin
from statistic_harness.core.types import PluginResult

PLUGINS_DIR = ROOT / "plugins"
SKIP_TYPES = {"ingest", "profile", "planner", "transform", "report"}


def _canonical_json(result: PluginResult) -> str:
    """Serialize PluginResult to canonical JSON for comparison.

    Sorts dict keys, rounds floats to 6 decimals, sorts findings by id.
    """
    d = _result_to_comparable(result)
    return json.dumps(d, sort_keys=True, default=_json_default)


def _result_to_comparable(result: PluginResult) -> dict[str, Any]:
    """Convert PluginResult to a comparable dict, excluding volatile fields."""
    findings = sorted(result.findings, key=lambda f: f.get("id", ""))
    return {
        "status": result.status,
        "summary": result.summary,
        "metrics": _round_dict(result.metrics),
        "findings": [_round_dict(f) for f in findings],
    }


def _round_dict(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: _round_dict(v) for k, v in sorted(d.items())}
    if isinstance(d, list):
        return [_round_dict(v) for v in d]
    if isinstance(d, float):
        if math.isnan(d) or math.isinf(d):
            return str(d)
        return round(d, 6)
    return d


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "__dict__"):
        return str(obj)
    return str(obj)


def _is_thin_wrapper(plugin_dir: Path) -> bool:
    plugin_py = plugin_dir / "plugin.py"
    if not plugin_py.exists():
        return False
    try:
        source = plugin_py.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if "registry" in module:
                for alias in node.names:
                    if alias.name == "run_plugin":
                        return True
    return False


def _get_plugin_type(plugin_dir: Path) -> str | None:
    yaml_path = plugin_dir / "plugin.yaml"
    if not yaml_path.exists():
        return None
    text = yaml_path.read_text(encoding="utf-8")
    for line in text.split("\n"):
        if line.startswith("type:"):
            return line.split(":", 1)[1].strip()
    return "analysis"


def _load_plugin_class(plugin_dir: Path):
    plugin_py = plugin_dir / "plugin.py"
    spec = importlib.util.spec_from_file_location(
        f"plugin_{plugin_dir.name}", plugin_py,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Plugin()


def _run_plugin_safe(plugin_id: str, df, run_seed: int) -> PluginResult | str:
    """Run a plugin and return result or error string."""
    run_dir = Path(tempfile.mkdtemp(prefix=f"det_{plugin_id}_{run_seed}_"))
    ctx = make_minimal_context(df, run_dir=run_dir, run_seed=run_seed)
    try:
        if plugin_id in HANDLERS:
            return run_plugin(plugin_id, ctx)
        else:
            plugin_dir = PLUGINS_DIR / plugin_id
            plugin = _load_plugin_class(plugin_dir)
            return plugin.run(ctx)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


@dataclass
class DeterminismResult:
    plugin_id: str
    status: str  # PASS, FAIL, ERROR
    seed42_identical: bool = True
    seed99_identical: bool = False
    hardcoded_seed_suspected: bool = False
    error: str | None = None


def main() -> None:
    output_dir = ROOT / "appdata" / "verification"
    output_dir.mkdir(parents=True, exist_ok=True)

    df_det, _ = ds_determinism()

    # Collect all analysis plugins
    all_plugins: list[tuple[str, str]] = []  # (plugin_id, source)

    # Registry handlers
    for handler_id in sorted(HANDLERS.keys()):
        all_plugins.append((handler_id, "registry"))

    # Full-impl plugins
    for pdir in sorted(PLUGINS_DIR.iterdir()):
        if not pdir.is_dir() or not (pdir / "plugin.py").exists():
            continue
        if pdir.name == "__pycache__":
            continue
        plugin_id = pdir.name
        ptype = _get_plugin_type(pdir)
        if ptype in SKIP_TYPES:
            continue
        if plugin_id in HANDLERS:
            continue  # Already tested as registry handler
        if _is_thin_wrapper(pdir):
            continue  # Covered by registry handler test
        all_plugins.append((plugin_id, "full_impl"))

    print(f"Determinism verification for {len(all_plugins)} plugins")

    det_results: list[DeterminismResult] = []
    verification_results: list[VerificationResult] = []

    for i, (plugin_id, source) in enumerate(all_plugins):
        print(f"  [{i + 1}/{len(all_plugins)}] {plugin_id} ... ", end="", flush=True)

        # Run 1: seed=42
        r1 = _run_plugin_safe(plugin_id, df_det, run_seed=42)
        if isinstance(r1, str):
            print(f"ERROR (run1)")
            det_results.append(DeterminismResult(plugin_id, "ERROR", error=r1))
            verification_results.append(VerificationResult(
                plugin_id=plugin_id, dataset_name="ds_determinism",
                status="ERROR", error=r1,
            ))
            continue

        # Run 2: seed=42 again
        r2 = _run_plugin_safe(plugin_id, df_det, run_seed=42)
        if isinstance(r2, str):
            print(f"ERROR (run2)")
            det_results.append(DeterminismResult(plugin_id, "ERROR", error=r2))
            verification_results.append(VerificationResult(
                plugin_id=plugin_id, dataset_name="ds_determinism",
                status="ERROR", error=r2,
            ))
            continue

        j1 = _canonical_json(r1)
        j2 = _canonical_json(r2)
        identical_42 = (j1 == j2)

        # Run 3: seed=99
        r3 = _run_plugin_safe(plugin_id, df_det, run_seed=99)
        if isinstance(r3, str):
            # If seed=99 errors but seed=42 works, still check determinism
            hardcoded = False
        else:
            j3 = _canonical_json(r3)
            hardcoded = (j1 == j3) and r1.status == "ok" and len(r1.findings) > 0

        status = "PASS" if identical_42 and not hardcoded else "FAIL"
        dr = DeterminismResult(
            plugin_id=plugin_id,
            status=status,
            seed42_identical=identical_42,
            seed99_identical=hardcoded,
            hardcoded_seed_suspected=hardcoded,
        )
        det_results.append(dr)

        # Build verification result for the report pipeline
        checks = []
        checks.append(CheckResult(
            "determinism_identical",
            identical_42,
            "Identical" if identical_42 else "Outputs differ between same-seed runs",
        ))
        if hardcoded:
            checks.append(CheckResult(
                "no_hardcoded_seed",
                False,
                "Output identical for seed=42 and seed=99 (hardcoded seed suspected)",
            ))
        else:
            checks.append(CheckResult("no_hardcoded_seed", True, "OK"))

        vr = VerificationResult(
            plugin_id=plugin_id,
            dataset_name="ds_determinism",
            status=status,
            check_results=checks,
            plugin_status=r1.status,
        )
        verification_results.append(vr)

        if status == "FAIL":
            label = []
            if not identical_42:
                label.append("NON-DETERMINISTIC")
            if hardcoded:
                label.append("HARDCODED_SEED")
            print(f"FAIL ({', '.join(label)})")
        else:
            print("PASS")

    # Classify issues
    issues = []
    for vr in verification_results:
        if vr.status in ("FAIL", "ERROR"):
            classified = classify_verification_failure(vr)
            issues.extend(classified)

    # Write results
    results_path = output_dir / "determinism_results.json"
    results_data = [
        {
            "plugin_id": dr.plugin_id,
            "status": dr.status,
            "seed42_identical": dr.seed42_identical,
            "seed99_identical": dr.seed99_identical,
            "hardcoded_seed_suspected": dr.hardcoded_seed_suspected,
            "error": dr.error,
        }
        for dr in det_results
    ]
    results_path.write_text(json.dumps(results_data, indent=2, default=str), encoding="utf-8")

    # Build report
    json_path, md_path = build_report(
        verification_results, issues, output_dir,
        report_name="determinism_verification_report",
        extra_meta={"phase": "determinism", "total_plugins": len(all_plugins)},
    )

    # Summary
    passed = sum(1 for r in det_results if r.status == "PASS")
    failed = sum(1 for r in det_results if r.status == "FAIL")
    errored = sum(1 for r in det_results if r.status == "ERROR")
    hardcoded = sum(1 for r in det_results if r.hardcoded_seed_suspected)
    non_det = sum(1 for r in det_results if not r.seed42_identical)
    print(f"\n{'='*60}")
    print(f"Determinism Verification Complete")
    print(f"  Plugins tested: {len(all_plugins)}")
    print(f"  Passed: {passed}  Failed: {failed}  Errors: {errored}")
    print(f"  Non-deterministic: {non_det}")
    print(f"  Hardcoded seed suspected: {hardcoded}")
    print(f"  Results: {results_path}")
    print(f"  Report:  {md_path}")


if __name__ == "__main__":
    main()
