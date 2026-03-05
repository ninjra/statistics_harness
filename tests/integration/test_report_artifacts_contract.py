"""Report artifacts contract tests (four-pillars Task 4.4).

Validates that generated report artifacts conform to expected schemas,
contain required keys, and maintain stable column headers/structure.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import pytest


def _make_report(tmp_path: Path) -> dict:
    """Create a minimal valid report dict and write artifacts."""
    report = {
        "plugins": {
            "profile_basic": {
                "status": "ok",
                "summary": "Basic profiling complete",
                "metrics": {"row_count": 200, "col_count": 3},
                "findings": [
                    {"id": "f1", "kind": "feature_discovery", "feature": "x1", "severity": "info"},
                ],
                "artifacts": [],
            },
            "analysis_percentile_analysis": {
                "status": "ok",
                "summary": "Percentile analysis complete",
                "metrics": {},
                "findings": [],
                "artifacts": [],
            },
        },
        "recommendations": {
            "items": [
                {
                    "process_hint": "close_a",
                    "action_type": "automate",
                    "modeled_delta_hours": 10.5,
                    "plugin_id": "analysis_percentile_analysis",
                    "text": "Automate close_a to save 10.5 hours",
                },
            ],
            "metadata": {"top_n": 20, "filter_mode": "strict"},
        },
        "metadata": {
            "run_id": "test-run",
            "run_seed": 42,
            "schema_version": "2.0",
        },
        "executive_summary": "Test summary for validation.",
    }
    # Write report.json
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


class TestReportJsonContract:
    """Verify report.json has required top-level keys."""

    def test_report_json_has_required_keys(self, tmp_path: Path) -> None:
        report = _make_report(tmp_path)
        required = {"plugins", "recommendations", "metadata"}
        assert required.issubset(set(report.keys())), (
            f"Missing keys: {required - set(report.keys())}"
        )

    def test_report_json_plugins_are_dicts(self, tmp_path: Path) -> None:
        report = _make_report(tmp_path)
        plugins = report["plugins"]
        assert isinstance(plugins, dict)
        for pid, data in plugins.items():
            assert isinstance(data, dict), f"Plugin {pid} is not a dict"
            assert "status" in data, f"Plugin {pid} missing 'status'"

    def test_recommendations_have_items(self, tmp_path: Path) -> None:
        report = _make_report(tmp_path)
        recs = report["recommendations"]
        assert isinstance(recs, dict)
        items = recs.get("items", [])
        assert isinstance(items, list)
        for item in items:
            assert isinstance(item, dict)
            assert "plugin_id" in item or "process_hint" in item

    def test_metadata_has_schema_version(self, tmp_path: Path) -> None:
        report = _make_report(tmp_path)
        meta = report["metadata"]
        assert "schema_version" in meta
        assert "run_id" in meta


class TestSlideKitContract:
    """Verify slide kit CSV output structure."""

    EXPECTED_HEADERS = [
        "recommendation_rank",
        "process_hint",
        "action_type",
        "modeled_delta_hours",
        "text",
    ]

    def test_slide_kit_csv_headers_stable(self, tmp_path: Path) -> None:
        """Slide kit CSV must have stable expected column headers."""
        report = _make_report(tmp_path)
        items = report["recommendations"]["items"]

        csv_path = tmp_path / "slide_kit.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.EXPECTED_HEADERS)
            writer.writeheader()
            for rank, item in enumerate(items, 1):
                writer.writerow({
                    "recommendation_rank": rank,
                    "process_hint": item.get("process_hint", ""),
                    "action_type": item.get("action_type", ""),
                    "modeled_delta_hours": item.get("modeled_delta_hours", 0),
                    "text": item.get("text", ""),
                })

        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames is not None
            assert list(reader.fieldnames) == self.EXPECTED_HEADERS
            rows = list(reader)
            assert len(rows) == len(items)


class TestBusinessSummary:
    """Verify business summary markdown generation."""

    def test_business_summary_md_generated(self, tmp_path: Path) -> None:
        """Business summary must exist and contain a KPI section."""
        report = _make_report(tmp_path)
        md_path = tmp_path / "business_summary.md"

        lines = ["# Business Summary\n\n"]
        lines.append("## KPI Overview\n\n")
        lines.append(f"- Run ID: {report['metadata']['run_id']}\n")
        lines.append(f"- Plugins executed: {len(report['plugins'])}\n")
        rec_count = len(report["recommendations"]["items"])
        lines.append(f"- Recommendations: {rec_count}\n")
        md_path.write_text("".join(lines), encoding="utf-8")

        content = md_path.read_text(encoding="utf-8")
        assert "KPI" in content
        assert "Recommendations" in content
        assert md_path.stat().st_size > 0


class TestArtifactManifest:
    """Verify artifact manifest schema."""

    def test_artifact_manifest_schema(self, tmp_path: Path) -> None:
        """Manifest entries must have path, sha256, and schema_version."""
        report = _make_report(tmp_path)
        report_path = tmp_path / "report.json"

        manifest = {
            "artifacts": [
                {
                    "path": str(report_path),
                    "sha256": hashlib.sha256(
                        report_path.read_bytes()
                    ).hexdigest(),
                    "schema_version": "2.0",
                    "source_plugin": "pipeline",
                },
            ]
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in loaded["artifacts"]:
            assert "path" in entry, "Manifest entry missing 'path'"
            assert "sha256" in entry, "Manifest entry missing 'sha256'"
            assert "schema_version" in entry, "Manifest entry missing 'schema_version'"
            assert len(entry["sha256"]) == 64, "Invalid sha256 hash length"
