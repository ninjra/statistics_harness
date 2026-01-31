from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import uvicorn
import yaml

from statistic_harness.core.evaluation import evaluate_report
from statistic_harness.core.pipeline import Pipeline
from statistic_harness.core.plugin_manager import PluginManager
from statistic_harness.core.report import build_report, write_report
from statistic_harness.core.utils import get_appdata_dir
from statistic_harness.ui.server import app


def load_settings(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    content = Path(path).read_text(encoding="utf-8")
    if path.endswith(".json"):
        return json.loads(content)
    return yaml.safe_load(content)


def cmd_list_plugins() -> None:
    manager = PluginManager(Path("plugins"))
    for spec in manager.discover():
        print(f"{spec.plugin_id}: {spec.name} ({spec.type})")


def cmd_serve(host: str, port: int) -> None:
    uvicorn.run(app, host=host, port=port)


def cmd_run(file_path: str, plugins: str, settings_path: str | None, run_seed: int) -> None:
    pipeline = Pipeline(get_appdata_dir(), Path("plugins"))
    plugin_ids = [p for p in plugins.split(",") if p]
    settings = load_settings(settings_path)
    run_id = pipeline.run(Path(file_path), plugin_ids, settings, run_seed)
    run_dir = get_appdata_dir() / "runs" / run_id
    report = build_report(pipeline.storage, run_id, run_dir, Path("docs/report.schema.json"))
    write_report(report, run_dir)
    print(run_id)


def cmd_eval(report_path: str, ground_truth: str) -> None:
    ok, messages = evaluate_report(Path(report_path), Path(ground_truth))
    if not ok:
        for msg in messages:
            print(msg)
        raise SystemExit(1)
    print("Evaluation passed")


def cmd_make_ground_truth(report_path: str, output_path: str) -> None:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    features = [f.get("feature") for f in report.get("plugins", {}).get("analysis_gaussian_knockoffs", {}).get("findings", [])]
    template = {
        "features": [f for f in features if f],
        "changepoints": [],
        "dependence_shift_pairs": [],
        "anomalies": [],
        "min_anomaly_hits": 0,
        "changepoint_tolerance": 3,
    }
    Path(output_path).write_text(yaml.safe_dump(template), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-plugins")

    serve_parser = sub.add_parser("serve")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)

    run_parser = sub.add_parser("run")
    run_parser.add_argument("--file", required=True)
    run_parser.add_argument("--plugins", required=True)
    run_parser.add_argument("--settings")
    run_parser.add_argument("--run-seed", type=int, default=0)

    eval_parser = sub.add_parser("eval")
    eval_parser.add_argument("--report", required=True)
    eval_parser.add_argument("--ground-truth", required=True)

    gt_parser = sub.add_parser("make-ground-truth-template")
    gt_parser.add_argument("--report", required=True)
    gt_parser.add_argument("-o", "--output", required=True)

    args = parser.parse_args()

    if args.command == "list-plugins":
        cmd_list_plugins()
    elif args.command == "serve":
        cmd_serve(args.host, args.port)
    elif args.command == "run":
        cmd_run(args.file, args.plugins, args.settings, args.run_seed)
    elif args.command == "eval":
        cmd_eval(args.report, args.ground_truth)
    elif args.command == "make-ground-truth-template":
        cmd_make_ground_truth(args.report, args.output)
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
