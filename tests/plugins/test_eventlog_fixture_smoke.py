from pathlib import Path

import pandas as pd

from plugins.analysis_attribution.plugin import Plugin as Attribution
from plugins.analysis_capacity_scaling.plugin import Plugin as CapacityScaling
from plugins.analysis_chain_makespan.plugin import Plugin as ChainMakespan
from plugins.analysis_concurrency_reconstruction.plugin import Plugin as Concurrency
from plugins.analysis_dependency_resolution_join.plugin import Plugin as DependencyJoin
from plugins.analysis_percentile_analysis.plugin import Plugin as Percentile
from plugins.analysis_process_sequence.plugin import Plugin as ProcessSequence
from plugins.analysis_sequence_classification.plugin import Plugin as SequenceClassification
from plugins.analysis_tail_isolation.plugin import Plugin as TailIsolation
from tests.conftest import make_context


def test_quorum_fixture_smoke_for_eventlog_plugins(run_dir):
    df = pd.read_csv(Path("tests/fixtures/quorum_close_cycle.csv"))
    ctx = make_context(run_dir, df, {})

    plugins = [
        Attribution,
        CapacityScaling,
        ChainMakespan,
        Concurrency,
        DependencyJoin,
        Percentile,
        ProcessSequence,
        SequenceClassification,
        TailIsolation,
    ]

    for plugin_cls in plugins:
        result = plugin_cls().run(ctx)
        assert result.status in {"ok", "skipped"}
