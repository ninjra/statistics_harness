from __future__ import annotations

from statistic_harness.core.entity_resolution import normalize_org_name


def test_entity_resolution_normalization_contract() -> None:
    assert normalize_org_name("Acme, Inc.") == "ACME"
    assert normalize_org_name("Acme Incorporated") == "ACME"

