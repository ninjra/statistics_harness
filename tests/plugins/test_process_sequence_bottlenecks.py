import pandas as pd

from plugins.analysis_process_sequence_bottlenecks.plugin import Plugin
from tests.conftest import make_context


def test_process_sequence_bottlenecks_detect_gap(run_dir):
    base = pd.Timestamp("2024-01-01 00:00:00")
    rows = []
    rows.append(
        {
            "case_id": "c1",
            "process": "A",
            "start_ts": base.isoformat(),
            "end_ts": (base + pd.Timedelta(seconds=30)).isoformat(),
        }
    )
    rows.append(
        {
            "case_id": "c1",
            "process": "B",
            "start_ts": (base + pd.Timedelta(seconds=400)).isoformat(),
            "end_ts": (base + pd.Timedelta(seconds=430)).isoformat(),
        }
    )
    rows.append(
        {
            "case_id": "c2",
            "process": "A",
            "start_ts": (base + pd.Timedelta(hours=1)).isoformat(),
            "end_ts": (base + pd.Timedelta(hours=1, seconds=20)).isoformat(),
        }
    )
    rows.append(
        {
            "case_id": "c2",
            "process": "B",
            "start_ts": (base + pd.Timedelta(hours=1, seconds=350)).isoformat(),
            "end_ts": (base + pd.Timedelta(hours=1, seconds=380)).isoformat(),
        }
    )
    df = pd.DataFrame(rows)
    ctx = make_context(
        run_dir,
        df=df,
        settings={"min_transition_count": 1, "wait_threshold_seconds": 60},
    )
    result = Plugin().run(ctx)
    assert result.status == "ok"
    assert any(
        isinstance(item, dict) and item.get("kind") == "sequence_bottleneck"
        for item in result.findings
    )
