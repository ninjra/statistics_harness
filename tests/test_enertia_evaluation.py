from pathlib import Path

from statistic_harness.core.evaluation import evaluate_report
from statistic_harness.core.pipeline import Pipeline


def test_enertia_eventlog_evaluation(tmp_path, monkeypatch):
    appdata = tmp_path / "appdata"
    monkeypatch.setenv("STAT_HARNESS_APPDATA", str(appdata))
    pipeline = Pipeline(appdata, Path("plugins"))
    run_id = pipeline.run(
        Path("tests/fixtures/enertia_eventlog.csv"),
        [
            "analysis_close_cycle_contention",
            "analysis_queue_delay_decomposition",
            "analysis_tail_isolation",
            "analysis_percentile_analysis",
            "analysis_attribution",
            "analysis_dependency_resolution_join",
            "analysis_sequence_classification",
            "analysis_process_sequence",
            "analysis_chain_makespan",
            "analysis_concurrency_reconstruction",
            "analysis_capacity_scaling",
            "analysis_close_cycle_capacity_model",
            "analysis_determinism_discipline",
        ],
        {},
        123,
    )
    report_path = appdata / "runs" / run_id / "report.json"
    ok, messages = evaluate_report(
        report_path, Path("tests/fixtures/ground_truth_enertia.yaml")
    )
    assert ok, messages
