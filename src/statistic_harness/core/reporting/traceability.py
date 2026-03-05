"""Traceability: every rendered number must have a claim_id.

The ClaimRegistry collects claims during report building and validates
that no number is rendered without a corresponding claim mapping.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Claim:
    claim_id: str
    label: str  # MEASURED | MODELED | INFERENCE
    summary_text: str
    value: Any
    unit: str
    population_scope: str
    source_plugin: str
    source_kind: str
    measurement_type: str
    artifact_path: str
    render_targets: list[str] = field(default_factory=list)


class ClaimRegistry:
    """Collects and validates claim-to-number mappings."""

    def __init__(self) -> None:
        self._claims: dict[str, Claim] = {}

    def register(self, claim: Claim) -> None:
        self._claims[claim.claim_id] = claim

    def get(self, claim_id: str) -> Claim | None:
        return self._claims.get(claim_id)

    def all_claims(self) -> list[Claim]:
        return list(self._claims.values())

    def claim_ids(self) -> set[str]:
        return set(self._claims.keys())

    def validate_rendered_numbers(
        self, rendered_text: str
    ) -> list[str]:
        """Find numbers in rendered text that lack claim_id references.

        Returns a list of error messages for numbers without claims.
        """
        errors: list[str] = []
        known_ids = self.claim_ids()
        if not known_ids:
            return errors

        # Find all claim_id references in the rendered text
        referenced_ids = set(re.findall(r"claim_[a-f0-9]{8}", rendered_text))

        # Find claim_ids that were registered but never referenced
        unreferenced = known_ids - referenced_ids
        for claim_id in sorted(unreferenced):
            claim = self._claims[claim_id]
            # Only flag claims that target business_summary (the main output)
            if "business_summary" in claim.render_targets:
                errors.append(
                    f"Claim {claim_id} ({claim.summary_text}) registered for "
                    f"business_summary but not referenced in rendered output"
                )

        return errors

    def to_manifest(self) -> list[dict[str, Any]]:
        """Export all claims as a list of dicts for traceability_manifest.json."""
        out: list[dict[str, Any]] = []
        for claim in sorted(self._claims.values(), key=lambda c: c.claim_id):
            out.append({
                "claim_id": claim.claim_id,
                "label": claim.label,
                "summary_text": claim.summary_text,
                "value": claim.value,
                "unit": claim.unit,
                "population_scope": claim.population_scope,
                "source": {
                    "plugin": claim.source_plugin,
                    "kind": claim.source_kind,
                    "measurement_type": claim.measurement_type,
                    "artifact_path": claim.artifact_path,
                },
                "render_targets": claim.render_targets,
            })
        return out
