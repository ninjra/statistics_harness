from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

import pandas as pd

from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json


def _pick_column(df: pd.DataFrame, preferred: str | None, candidates: list[str]) -> str | None:
    if preferred and preferred in df.columns:
        return preferred
    col_map = {str(c).lower(): str(c) for c in df.columns}
    for cand in candidates:
        if cand in col_map:
            return col_map[cand]
    return None


class Plugin:
    def run(self, ctx) -> PluginResult:
        loader = getattr(ctx, "dataset_loader", None)
        if not callable(loader):
            return PluginResult("skipped", "dataset_loader unavailable", {}, [], [], None)
        df = loader()
        if df is None or len(df) == 0:
            return PluginResult("skipped", "Empty dataset", {"rows": 0}, [], [], None)

        case_col = _pick_column(
            df,
            ctx.settings.get("case_column"),
            ["case_id", "case", "trace_id", "session_id"],
        )
        activity_col = _pick_column(
            df,
            ctx.settings.get("activity_column"),
            ["activity", "event", "process", "step", "task"],
        )
        time_col = _pick_column(
            df,
            ctx.settings.get("time_column"),
            ["timestamp", "ts", "start_ts", "event_time", "datetime"],
        )
        if not case_col or not activity_col:
            return PluginResult(
                "skipped",
                "Missing case/activity columns for process discovery",
                {"rows": int(len(df))},
                [],
                [],
                None,
            )

        data = df[[case_col, activity_col] + ([time_col] if time_col else [])].copy()
        data[activity_col] = data[activity_col].astype(str)
        if time_col:
            data["_t"] = pd.to_datetime(data[time_col], errors="coerce")
            data = data.sort_values([case_col, "_t"], kind="mergesort")
        else:
            data = data.sort_values([case_col], kind="mergesort")

        node_counts: Counter[str] = Counter()
        edge_counts: Counter[tuple[str, str]] = Counter()
        for _, group in data.groupby(case_col, sort=False):
            acts = [str(x) for x in group[activity_col].tolist() if str(x)]
            for act in acts:
                node_counts[act] += 1
            for i in range(len(acts) - 1):
                edge_counts[(acts[i], acts[i + 1])] += 1

        top_k = int(ctx.settings.get("top_k_edges", 20))
        top_edges = sorted(edge_counts.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))[:top_k]
        findings = [
            {
                "kind": "process_pattern",
                "source": src,
                "target": dst,
                "count": int(cnt),
            }
            for (src, dst), cnt in top_edges
        ]
        artifact_payload = {
            "schema_version": "process_mining.v1",
            "plugin_id": "process_mining_discovery_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "nodes": [
                {"activity": act, "count": int(cnt)}
                for act, cnt in sorted(node_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            ],
            "edges": [
                {"source": src, "target": dst, "count": int(cnt)}
                for (src, dst), cnt in top_edges
            ],
        }
        out_dir = ctx.artifacts_dir("process_mining_discovery_v1")
        out_path = out_dir / "process_graph.json"
        write_json(out_path, artifact_payload)
        artifacts = [
            PluginArtifact(
                path=str(out_path.relative_to(ctx.run_dir)),
                type="json",
                description="process transition graph",
            )
        ]
        return PluginResult(
            status="ok",
            summary="Built deterministic process transition graph",
            metrics={
                "rows": int(len(data)),
                "cases": int(data[case_col].nunique()),
                "unique_activities": int(len(node_counts)),
                "unique_edges": int(len(edge_counts)),
            },
            findings=findings,
            artifacts=artifacts,
            references=[],
            debug={"case_column": case_col, "activity_column": activity_col, "time_column": time_col},
        )
