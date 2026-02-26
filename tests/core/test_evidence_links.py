from __future__ import annotations

from statistic_harness.core.evidence_links import evidence_link, row_ref, stable_edge_id, stable_entity_id


def test_evidence_link_ids_are_deterministic() -> None:
    left_ref = row_ref("contracts_v1", 10)
    right_ref = row_ref("contrib_v1", 42)
    left_id = stable_entity_id("org", "ACME INC")
    right_id = stable_entity_id("person", "DOE JANE")
    edge_id_a = stable_edge_id("vendor_employer", left_id, right_id, right_ref)
    edge_id_b = stable_edge_id("vendor_employer", left_id, right_id, right_ref)
    assert edge_id_a == edge_id_b

    link_a = evidence_link(
        match_type="exact",
        confidence_tier="high",
        features={"token_overlap": 1.0},
        left_ref=left_ref,
        right_ref=right_ref,
        left_entity_id=left_id,
        right_entity_id=right_id,
        relation="vendor_employer",
    )
    link_b = evidence_link(
        match_type="exact",
        confidence_tier="high",
        features={"token_overlap": 1.0},
        left_ref=left_ref,
        right_ref=right_ref,
        left_entity_id=left_id,
        right_entity_id=right_id,
        relation="vendor_employer",
    )
    assert link_a["evidence_id"] == link_b["evidence_id"]

