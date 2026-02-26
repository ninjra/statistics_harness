from __future__ import annotations

from pathlib import Path

import pandas as pd

from tests.conftest import make_context

from plugins.transform_cross_dataset_link_graph_v1.plugin import Plugin as LinkGraphPlugin
from plugins.transform_entity_resolution_map_v1.plugin import Plugin as EntityMapPlugin


def openplanter_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "vendor_name1": "Acme, Inc.",
                "employer": "Acme Inc",
                "candidate_id": "C001",
                "donation_date": "2025-01-05",
                "amount": 250.0,
                "award_date": "2025-01-10",
            },
            {
                "vendor_name1": "Acme Inc",
                "employer": "Acme Incorporated",
                "candidate_id": "C001",
                "donation_date": "2025-01-05",
                "amount": 260.0,
                "award_date": "2025-01-12",
            },
            {
                "vendor_name1": "Beta LLC",
                "employer": "Beta LLC",
                "candidate_id": "C002",
                "donation_date": "2025-01-11",
                "amount": 110.0,
                "award_date": "2025-01-15",
            },
        ]
    )


def make_openplanter_context(run_dir: Path, *, run_seed: int = 7):
    return make_context(run_dir, openplanter_frame(), settings={}, run_seed=run_seed)


def run_openplanter_transforms(ctx) -> None:
    dataset_id = "test_dataset"
    datasets = {
        "contracts": {"dataset_version_id": dataset_id},
        "contributions": {"dataset_version_id": dataset_id},
    }
    ctx.settings = {
        "datasets": datasets,
        "fields": [
            {"role": "contracts", "field": "vendor_name1", "entity_type": "org", "key": "vendor"},
            {"role": "contributions", "field": "employer", "entity_type": "org", "key": "employer"},
        ],
    }
    assert EntityMapPlugin().run(ctx).status == "ok"

    ctx.settings = {
        "datasets": datasets,
        "edges": [
            {
                "left": {"role": "contracts", "field": "vendor_name1"},
                "right": {"role": "contributions", "field": "employer"},
                "relation": "vendor_employer",
            }
        ],
    }
    assert LinkGraphPlugin().run(ctx).status == "ok"

