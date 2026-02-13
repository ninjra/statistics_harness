from __future__ import annotations

from typing import Any

import re

import pandas as pd

from statistic_harness.core.stat_plugins import (
    BudgetTimer,
    merge_config,
    infer_columns,
    deterministic_sample,
    stable_id,
    build_redactor,
)
from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json


DEFAULTS = {
    "log_templates": {
        "min_count": 3,
        "max_templates": 50,
        "max_findings": 20,
    }
}


UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
HEX_RE = re.compile(r"\b[0-9a-fA-F]{8,}\b")
NUM_RE = re.compile(r"\b\d+\b")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def _template(text: str) -> str:
    text = EMAIL_RE.sub("<EMAIL>", text)
    text = UUID_RE.sub("<UUID>", text)
    text = HEX_RE.sub("<HEX>", text)
    text = NUM_RE.sub("<NUM>", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class Plugin:
    def run(self, ctx) -> PluginResult:
        config = merge_config(ctx.settings)
        config["log_templates"] = {**DEFAULTS["log_templates"], **config.get("log_templates", {})}
        timer = BudgetTimer(config.get("time_budget_ms"))

        df = ctx.dataset_loader()
        if df.empty:
            return PluginResult("skipped", "Empty dataset", {}, [], [], None)

        df, sample_meta = deterministic_sample(df, config.get("max_rows"), seed=config.get("seed", 1337))
        inferred = infer_columns(df, config)
        text_cols = inferred.get("text_columns") or []

        if not text_cols:
            return PluginResult("skipped", "No text/log columns detected", {}, [], [], None)

        privacy = config.get("privacy") if isinstance(config.get("privacy"), dict) else {}
        redactor = build_redactor(privacy)
        allow_snippets = bool(privacy.get("allow_exemplar_snippets", False))
        max_exemplars = int(privacy.get("max_exemplars", 3))

        templates: dict[str, dict[str, Any]] = {}
        for col in text_cols:
            if timer.exceeded():
                break
            series = df[col].dropna().astype(str)
            for value in series:
                if timer.exceeded():
                    break
                tpl = _template(value)
                entry = templates.setdefault(tpl, {"count": 0, "examples": []})
                entry["count"] += 1
                if allow_snippets and len(entry["examples"]) < max_exemplars:
                    entry["examples"].append(redactor(value)[:120])

        if not templates:
            return PluginResult("skipped", "No templates extracted", {}, [], [], None)

        ordered = sorted(templates.items(), key=lambda item: (-item[1]["count"], item[0]))
        max_templates = int(config["log_templates"].get("max_templates", 50))
        min_count = int(config["log_templates"].get("min_count", 3))

        findings: list[dict[str, Any]] = []
        for tpl, info in ordered:
            if info["count"] > min_count:
                continue
            findings.append(
                {
                    "id": stable_id(f"tpl:{tpl}"),
                    "severity": "info",
                    "confidence": 0.5,
                    "title": "Rare log template detected",
                    "what": f"Template occurs {info['count']} times.",
                    "why": "Rare templates can indicate edge cases or errors.",
                    "evidence": {
                        "metrics": {
                            "template": tpl,
                            "count": info["count"],
                            "examples": info.get("examples", []),
                        }
                    },
                    "where": {"column": "text"},
                    "recommendation": "Review rare templates for potential issues.",
                    "measurement_type": "measured",
                    "references": [
                        {
                            "title": "Drain: An Online Log Parsing Approach",
                            "url": "https://doi.org/10.1145/3133956.3134015",
                            "doi": "10.1145/3133956.3134015",
                        }
                    ],
                }
            )
            if len(findings) >= int(config["log_templates"].get("max_findings", 20)):
                break

        artifacts_dir = ctx.artifacts_dir("analysis_log_template_mining_drain")
        out_path = artifacts_dir / "templates.json"
        write_json(
            out_path,
            {
                "templates": [
                    {"template": tpl, "count": info["count"], "examples": info.get("examples", [])}
                    for tpl, info in ordered[:max_templates]
                ]
            },
        )
        artifacts = [
            PluginArtifact(
                path=str(out_path.relative_to(ctx.run_dir)),
                type="json",
                description="Log template summary",
            )
        ]

        metrics = {
            "rows_seen": int(sample_meta.get("rows_total", len(df))),
            "rows_used": int(sample_meta.get("rows_used", len(df))),
            "cols_used": len(text_cols),
            "references": [
                {
                    "title": "Drain: An Online Log Parsing Approach",
                    "url": "https://doi.org/10.1145/3133956.3134015",
                    "doi": "10.1145/3133956.3134015",
                }
            ],
        }

        summary = f"Extracted {len(templates)} templates." if templates else "No templates extracted."

        return PluginResult(
            "ok",
            summary,
            metrics,
            findings,
            artifacts,
            None,
        )
