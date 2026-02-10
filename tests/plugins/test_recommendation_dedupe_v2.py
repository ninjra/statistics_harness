import pandas as pd
import json

from plugins.analysis_recommendation_dedupe_v2.plugin import Plugin
from tests.conftest import make_context


def test_recommendation_dedupe_merges_duplicates(run_dir):
    override = [
        {
            "action": "add_server",
            "title": "Rec A",
            "recommendation": "Add one server",
            "modeled_delta": 1.0,
            "where": {"process_norm": "qemail"},
        },
        {
            "action": "add_server",
            "title": "Rec B",
            "recommendation": "Add one server",
            "modeled_delta": 1.0,
            "where": {"process_norm": "qemail"},
        },
    ]
    ctx = make_context(
        run_dir,
        df=pd.DataFrame(),
        settings={"recommendations_override": override},
        populate=False,
    )
    result = Plugin().run(ctx)
    assert result.status == "ok"
    payload = (run_dir / "artifacts" / "analysis_recommendation_dedupe_v2" / "recommendations.json").read_text(
        encoding="utf-8"
    )
    assert '"recommendations"' in payload


def test_recommendation_dedupe_conflict(run_dir):
    override = [
        {
            "action": "add_server",
            "title": "Rec A",
            "recommendation": "Add one server",
            "modeled_delta": 1.0,
            "where": {"process_norm": "qemail"},
        },
        {
            "action": "add_server",
            "title": "Rec B",
            "recommendation": "Add one server",
            "modeled_delta": 2.0,
            "where": {"process_norm": "qemail"},
        },
    ]
    ctx = make_context(
        run_dir,
        df=pd.DataFrame(),
        settings={"recommendations_override": override},
        populate=False,
    )
    result = Plugin().run(ctx)
    assert result.status == "ok"


def test_recommendation_dedupe_allows_known_only(run_dir):
    override = [
        {
            "action": "add_server",
            "title": "Known Rec",
            "recommendation": "Add one server",
            "modeled_delta": 1.0,
            "where": {"process_norm": "qemail"},
            "category": "known",
        }
    ]
    ctx = make_context(
        run_dir,
        df=pd.DataFrame(),
        settings={"recommendations_override": override},
        populate=False,
    )
    result = Plugin().run(ctx)
    assert result.status == "ok"
    payload_path = (
        run_dir / "artifacts" / "analysis_recommendation_dedupe_v2" / "recommendations.json"
    )
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    assert payload.get("has_discovery") is False
    counts = payload.get("counts") or {}
    assert int(counts.get("source_items_known") or 0) == 1
