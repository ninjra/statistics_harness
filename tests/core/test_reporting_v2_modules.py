"""Tests for the Phase 3 reporting v2 modules: traceability, redaction, guardrails."""
from __future__ import annotations

import csv
import textwrap
from pathlib import Path
from typing import Any

import pytest

from statistic_harness.core.reporting.guardrails import (
    GuardrailViolation,
    check_forbidden_slide_kit_columns,
    check_recommendation_dedupe_conflicts,
    check_unclaimed_numbers,
    check_waterfall_reconciliation,
    run_all_guardrails,
)
from statistic_harness.core.reporting.redaction import (
    FORBIDDEN_COLUMNS,
    check_forbidden_columns,
    pseudonymize,
    redact_dict_values,
    redact_hostnames,
    redact_user_ids,
)
from statistic_harness.core.reporting.traceability import Claim, ClaimRegistry
from statistic_harness.core.report_v2_utils import normalize_scale_factor


# ---------------------------------------------------------------------------
# Traceability tests
# ---------------------------------------------------------------------------


class TestClaimRegistry:
    def test_register_and_get(self):
        reg = ClaimRegistry()
        claim = Claim(
            claim_id="claim_aabbccdd",
            label="MEASURED",
            summary_text="Average wait 2.5h",
            value=2.5,
            unit="hours",
            population_scope="all runs",
            source_plugin="analysis_queue_delay",
            source_kind="metric",
            measurement_type="measured",
            artifact_path="artifacts/queue_delay/summary.json",
            render_targets=["business_summary"],
        )
        reg.register(claim)
        assert reg.get("claim_aabbccdd") is claim
        assert reg.get("nonexistent") is None
        assert "claim_aabbccdd" in reg.claim_ids()

    def test_missing_claim_fails_validation(self):
        """A claim registered for business_summary but absent from rendered text must fail."""
        reg = ClaimRegistry()
        reg.register(
            Claim(
                claim_id="claim_deadbeef",
                label="MEASURED",
                summary_text="Throughput 100/hr",
                value=100,
                unit="per_hour",
                population_scope="all",
                source_plugin="analysis_throughput",
                source_kind="metric",
                measurement_type="measured",
                artifact_path="artifacts/throughput/summary.json",
                render_targets=["business_summary"],
            )
        )
        rendered = "This report has no claim references at all."
        errors = reg.validate_rendered_numbers(rendered)
        assert len(errors) == 1
        assert "claim_deadbeef" in errors[0]

    def test_referenced_claim_passes_validation(self):
        reg = ClaimRegistry()
        reg.register(
            Claim(
                claim_id="claim_12345678",
                label="MODELED",
                summary_text="Delta 1.2h",
                value=1.2,
                unit="hours",
                population_scope="sample",
                source_plugin="analysis_delta",
                source_kind="metric",
                measurement_type="modeled",
                artifact_path="artifacts/delta/summary.json",
                render_targets=["business_summary"],
            )
        )
        rendered = "The modeled delta is 1.2h (claim_12345678)."
        errors = reg.validate_rendered_numbers(rendered)
        assert errors == []

    def test_non_business_summary_claim_not_flagged(self):
        """Claims not targeting business_summary should not be flagged even if unreferenced."""
        reg = ClaimRegistry()
        reg.register(
            Claim(
                claim_id="claim_aabb0011",
                label="MEASURED",
                summary_text="Internal metric",
                value=42,
                unit="count",
                population_scope="all",
                source_plugin="analysis_internal",
                source_kind="metric",
                measurement_type="measured",
                artifact_path="artifacts/internal.json",
                render_targets=["engineering_summary"],
            )
        )
        rendered = "No claims here."
        errors = reg.validate_rendered_numbers(rendered)
        assert errors == []

    def test_to_manifest_sorted(self):
        reg = ClaimRegistry()
        for cid in ["claim_cccccccc", "claim_aaaaaaaa", "claim_bbbbbbbb"]:
            reg.register(
                Claim(
                    claim_id=cid,
                    label="MEASURED",
                    summary_text="test",
                    value=0,
                    unit="",
                    population_scope="",
                    source_plugin="p",
                    source_kind="k",
                    measurement_type="measured",
                    artifact_path="",
                )
            )
        manifest = reg.to_manifest()
        ids = [m["claim_id"] for m in manifest]
        assert ids == ["claim_aaaaaaaa", "claim_bbbbbbbb", "claim_cccccccc"]


# ---------------------------------------------------------------------------
# Redaction tests
# ---------------------------------------------------------------------------


class TestRedaction:
    def test_redact_hostnames_scrubs_fqdn(self):
        text = "Connected to server.example.com on port 443."
        result = redact_hostnames(text)
        assert "server.example.com" not in result
        assert "host_" in result

    def test_redact_hostnames_scrubs_server_pattern(self):
        text = "Alert from srv01 and db-3."
        result = redact_hostnames(text)
        assert "srv01" not in result
        assert "db-3" not in result

    def test_redact_user_ids_scrubs_corporate_id(self):
        text = "User AB12345 submitted the request."
        result = redact_user_ids(text)
        assert "AB12345" not in result
        assert "user_" in result

    def test_redact_user_ids_scrubs_email(self):
        text = "Contact john.doe@company.com for details."
        result = redact_user_ids(text)
        assert "john.doe@company.com" not in result

    def test_pseudonymize_applies_all_passes(self):
        text = "Host srv01 contacted by AB12345 on server.example.com."
        result = pseudonymize(text)
        assert "srv01" not in result
        assert "AB12345" not in result
        assert "server.example.com" not in result

    def test_check_forbidden_columns(self):
        headers = ["process_id", "hostname", "user_email", "delta_hours"]
        forbidden = check_forbidden_columns(headers)
        assert "hostname" in forbidden
        assert "user_email" in forbidden
        assert "process_id" not in forbidden

    def test_redact_dict_values(self):
        data = {
            "hostname": "srv01.prod.example.com",
            "count": 42,
            "nested": {"user_id": "AB12345", "value": 100},
        }
        result = redact_dict_values(data)
        assert result["hostname"] != "srv01.prod.example.com"
        assert result["count"] == 42
        assert result["nested"]["user_id"] != "AB12345"
        assert result["nested"]["value"] == 100


# ---------------------------------------------------------------------------
# Guardrail tests
# ---------------------------------------------------------------------------


class TestGuardrails:
    def test_waterfall_reconciliation_passes(self):
        waterfall = {
            "total_bp_over_threshold_wait_hours": 10.0,
            "top_driver_over_threshold_wait_hours": 7.0,
            "remainder_without_top_driver_hours": 3.0,
        }
        violations = check_waterfall_reconciliation(waterfall)
        assert violations == []

    def test_waterfall_reconciliation_fails(self):
        waterfall = {
            "total_bp_over_threshold_wait_hours": 10.0,
            "top_driver_over_threshold_wait_hours": 7.0,
            "remainder_without_top_driver_hours": 2.0,
        }
        violations = check_waterfall_reconciliation(waterfall)
        assert len(violations) == 1
        assert violations[0].code == "waterfall_reconciliation"

    def test_waterfall_reconciliation_within_tolerance(self):
        waterfall = {
            "total_bp_over_threshold_wait_hours": 10.0,
            "top_driver_over_threshold_wait_hours": 7.005,
            "remainder_without_top_driver_hours": 2.999,
        }
        violations = check_waterfall_reconciliation(waterfall, tolerance_hours=0.01)
        assert violations == []

    def test_waterfall_reconciliation_none_input(self):
        assert check_waterfall_reconciliation(None) == []
        assert check_waterfall_reconciliation({}) == []

    def test_forbidden_slide_kit_columns(self, tmp_path: Path):
        slide_kit = tmp_path / "slide_kit"
        slide_kit.mkdir()
        csv_path = slide_kit / "scenario_summary.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["process_id", "hostname", "delta_hours"])
            writer.writerow(["proc_1", "srv01", "2.5"])
        violations = check_forbidden_slide_kit_columns(slide_kit)
        assert len(violations) == 1
        assert violations[0].code == "forbidden_column"
        assert "hostname" in violations[0].message

    def test_forbidden_slide_kit_columns_clean(self, tmp_path: Path):
        slide_kit = tmp_path / "slide_kit"
        slide_kit.mkdir()
        csv_path = slide_kit / "clean.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["process_id", "delta_hours"])
            writer.writerow(["proc_1", "2.5"])
        violations = check_forbidden_slide_kit_columns(slide_kit)
        assert violations == []

    def test_recommendation_dedupe_conflict_fails(self):
        recs = [
            {"action_type": "tune_schedule", "target": "proc_A", "scenario_id": "s1", "delta_hours": 2.0},
            {"action_type": "tune_schedule", "target": "proc_A", "scenario_id": "s1", "delta_hours": 5.0},
        ]
        violations = check_recommendation_dedupe_conflicts(recs)
        assert len(violations) == 1
        assert violations[0].code == "conflicting_dedupe_delta"

    def test_recommendation_dedupe_no_conflict(self):
        recs = [
            {"action_type": "tune_schedule", "target": "proc_A", "scenario_id": "s1", "delta_hours": 2.0},
            {"action_type": "tune_schedule", "target": "proc_A", "scenario_id": "s1", "delta_hours": 2.005},
        ]
        violations = check_recommendation_dedupe_conflicts(recs)
        assert violations == []

    def test_recommendation_dedupe_different_keys(self):
        recs = [
            {"action_type": "tune_schedule", "target": "proc_A", "scenario_id": "s1", "delta_hours": 2.0},
            {"action_type": "add_server", "target": "proc_A", "scenario_id": "s1", "delta_hours": 5.0},
        ]
        violations = check_recommendation_dedupe_conflicts(recs)
        assert violations == []

    def test_unclaimed_numbers(self):
        reg = ClaimRegistry()
        reg.register(
            Claim(
                claim_id="claim_11111111",
                label="MEASURED",
                summary_text="Test",
                value=1,
                unit="hours",
                population_scope="all",
                source_plugin="test",
                source_kind="metric",
                measurement_type="measured",
                artifact_path="",
                render_targets=["business_summary"],
            )
        )
        violations = check_unclaimed_numbers(reg, "No claims referenced here.")
        assert len(violations) == 1
        assert violations[0].code == "unclaimed_number"

    def test_run_all_guardrails_combined(self, tmp_path: Path):
        """run_all_guardrails aggregates violations from multiple checks."""
        slide_kit = tmp_path / "slide_kit"
        slide_kit.mkdir()
        csv_path = slide_kit / "bad.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ip_address", "delta"])
            writer.writerow(["10.0.0.1", "5"])

        waterfall = {
            "total_bp_over_threshold_wait_hours": 10.0,
            "top_driver_over_threshold_wait_hours": 5.0,
            "remainder_without_top_driver_hours": 2.0,
        }
        violations = run_all_guardrails(
            waterfall=waterfall,
            slide_kit_dir=slide_kit,
            recommendations=[],
        )
        codes = {v.code for v in violations}
        assert "waterfall_reconciliation" in codes
        assert "forbidden_column" in codes


class TestGuardrailViolation:
    def test_to_dict(self):
        v = GuardrailViolation("test_code", "test message", severity="warning")
        d = v.to_dict()
        assert d == {"code": "test_code", "message": "test message", "severity": "warning"}


# ---------------------------------------------------------------------------
# Scale factor normalization tests
# ---------------------------------------------------------------------------


class TestNormalizeScaleFactor:
    def test_no_scale_returns_same(self):
        items = [{"delta_hours": 2.0}]
        assert normalize_scale_factor(items, scale_factor=None) is items
        assert normalize_scale_factor(items, scale_factor=1.0) is items

    def test_scales_values(self):
        items = [
            {"delta_hours": 2.0, "name": "a"},
            {"delta_hours": 3.0, "name": "b"},
        ]
        result = normalize_scale_factor(items, scale_factor=10.0)
        assert result[0]["delta_hours"] == 20.0
        assert result[1]["delta_hours"] == 30.0
        # Original unchanged
        assert items[0]["delta_hours"] == 2.0

    def test_custom_key(self):
        items = [{"impact": 5.0}]
        result = normalize_scale_factor(items, key="impact", scale_factor=2.0)
        assert result[0]["impact"] == 10.0

    def test_missing_key_ignored(self):
        items = [{"name": "a"}]
        result = normalize_scale_factor(items, scale_factor=10.0)
        assert result[0] == {"name": "a"}

    def test_non_numeric_ignored(self):
        items = [{"delta_hours": "N/A"}]
        result = normalize_scale_factor(items, scale_factor=10.0)
        assert result[0]["delta_hours"] == "N/A"
