from __future__ import annotations

from collections import defaultdict

import pandas as pd

from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json


class Plugin:
    def run(self, ctx) -> PluginResult:
        df = ctx.dataset_loader()
        if df.empty:
            return PluginResult("skipped", "Empty dataset", {}, [], [], None)

        case_col = None
        activity_col = None
        timestamp_col = None

        columns_meta = []
        if ctx.dataset_version_id:
            columns_meta = ctx.storage.fetch_dataset_columns(ctx.dataset_version_id)

        role_by_name = {
            col["original_name"]: (col.get("role") or "") for col in columns_meta
        }
        lower_names = {col: col.lower() for col in df.columns}

        for col in df.columns:
            if role_by_name.get(col) == "id":
                case_col = case_col or col
            if role_by_name.get(col) == "timestamp":
                timestamp_col = timestamp_col or col
        for col in df.columns:
            lname = lower_names[col]
            if case_col is None and (lname.endswith("id") or " case" in lname or "session" in lname):
                case_col = col
            if activity_col is None and (
                "activity" in lname
                or "event" in lname
                or "step" in lname
                or "process" in lname
                or "action" in lname
                or "task" in lname
            ):
                activity_col = col
            if timestamp_col is None and ("time" in lname or "date" in lname):
                timestamp_col = col

        if case_col is None or activity_col is None:
            return PluginResult(
                "skipped",
                "No event log columns detected",
                {},
                [],
                [],
                None,
            )

        work = df.copy()
        work = work.reset_index().rename(columns={"index": "row_index"})
        sort_cols = [case_col]
        if timestamp_col and timestamp_col in work.columns:
            sort_cols.append(timestamp_col)
        else:
            sort_cols.append("row_index")
        work = work.sort_values(sort_cols)

        sequences: dict[tuple[str, ...], int] = defaultdict(int)
        examples: dict[tuple[str, ...], list[int]] = {}
        transition_counts: dict[tuple[str, str], int] = defaultdict(int)

        for _, group in work.groupby(case_col, sort=False):
            activities = [str(x) for x in group[activity_col].tolist()]
            if not activities:
                continue
            seq = tuple(activities)
            sequences[seq] += 1
            if seq not in examples:
                examples[seq] = [int(i) for i in group["row_index"].tolist()]
            for a, b in zip(activities, activities[1:]):
                transition_counts[(a, b)] += 1

        if not sequences:
            return PluginResult(
                "skipped",
                "No sequences detected",
                {},
                [],
                [],
                None,
            )

        total_cases = sum(sequences.values())
        max_variants = int(ctx.settings.get("max_variants", 20))
        min_fraction = float(ctx.settings.get("min_variant_fraction", 0.0))
        max_examples = int(ctx.settings.get("max_examples", 25))

        findings = []
        sorted_variants = sorted(
            sequences.items(), key=lambda item: (-item[1], item[0])
        )
        for seq, count in sorted_variants[:max_variants]:
            fraction = count / total_cases if total_cases else 0.0
            if fraction < min_fraction:
                continue
            row_ids = examples.get(seq, [])[:max_examples]
            findings.append(
                {
                    "kind": "process_variant",
                    "variant": list(seq),
                    "count": int(count),
                    "fraction": float(fraction),
                    "columns": [case_col, activity_col]
                    + ([timestamp_col] if timestamp_col else []),
                    "evidence": {
                        "row_ids": row_ids,
                        "query": f"variant={seq}",
                    },
                }
            )

        rare_variants = [
            (seq, count)
            for seq, count in sorted_variants
            if count == 1 and len(seq) > 1
        ]
        for seq, count in rare_variants[:max_variants]:
            row_ids = examples.get(seq, [])[:max_examples]
            findings.append(
                {
                    "kind": "rare_variant",
                    "variant": list(seq),
                    "count": int(count),
                    "fraction": float(count / total_cases),
                    "columns": [case_col, activity_col]
                    + ([timestamp_col] if timestamp_col else []),
                    "evidence": {
                        "row_ids": row_ids,
                        "query": f"variant={seq}",
                    },
                }
            )

        transitions = []
        for (a, b), count in sorted(
            transition_counts.items(), key=lambda item: (-item[1], item[0])
        ):
            transitions.append(
                {
                    "kind": "transition",
                    "from": a,
                    "to": b,
                    "count": int(count),
                }
            )

        artifacts_dir = ctx.artifacts_dir("analysis_process_sequence")
        out_path = artifacts_dir / "sequences.json"
        write_json(out_path, {"variants": findings, "transitions": transitions})
        artifacts = [
            PluginArtifact(
                path=str(out_path.relative_to(ctx.run_dir)),
                type="json",
                description="Sequence mining results",
            )
        ]

        return PluginResult(
            "ok",
            "Computed process sequence variants",
            {"variants": len(findings), "transitions": len(transitions)},
            findings,
            artifacts,
            None,
        )
