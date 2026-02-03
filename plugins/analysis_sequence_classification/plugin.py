from __future__ import annotations

from typing import Any

import pandas as pd

from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json


INVALID_STRINGS = {"", "nan", "none", "null"}
MAX_SAMPLE_ROWS = 50000
MIN_PROCESS_ALPHA_RATIO = 0.15
MIN_DEP_OVERLAP_RATIO = 0.02
MIN_DEP_OVERLAP_COUNT = 10


def _candidate_columns(
    preferred: str | None,
    columns: list[str],
    role_by_name: dict[str, str],
    roles: set[str],
    patterns: list[str],
    lower_names: dict[str, str],
) -> list[str]:
    seen: set[str] = set()
    candidates: list[str] = []
    if preferred and preferred in columns and preferred not in seen:
        candidates.append(preferred)
        seen.add(preferred)
    for col in columns:
        if role_by_name.get(col) in roles and col not in seen:
            candidates.append(col)
            seen.add(col)
    for col in columns:
        if col in seen:
            continue
        name = lower_names[col]
        if any(pattern in name for pattern in patterns):
            candidates.append(col)
            seen.add(col)
    return candidates


def _parse_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _sample_series(series: pd.Series, max_rows: int = MAX_SAMPLE_ROWS) -> pd.Series:
    if len(series) > max_rows:
        return series.head(max_rows)
    return series


def _normalize_values(series: pd.Series) -> pd.Series:
    sample = _sample_series(series)
    values = sample.dropna().astype(str).str.strip()
    if values.empty:
        return values
    values = values[~values.str.lower().isin(INVALID_STRINGS)]
    return values


def _score_process_column(series: pd.Series) -> tuple[float, float]:
    values = _normalize_values(series)
    if values.empty:
        return -1.0, 0.0
    alpha_ratio = float(values.str.contains(r"[A-Za-z]", regex=True).mean())
    digit_ratio = float(values.str.match(r"^\\d+(\\.0+)?$").mean())
    unique_ratio = float(values.nunique() / max(len(values), 1))
    score = alpha_ratio * 3.0 - digit_ratio * 2.0 + min(unique_ratio, 1.0) * 0.2
    return score, alpha_ratio


def _best_process_column(candidates: list[str], df: pd.DataFrame) -> str | None:
    best_col = None
    best_score = -1.0
    best_alpha = 0.0
    for col in candidates:
        score, alpha = _score_process_column(df[col])
        if score > best_score:
            best_col = col
            best_score = score
            best_alpha = alpha
    if best_col is None or best_alpha < MIN_PROCESS_ALPHA_RATIO:
        return None
    return best_col


def _select_dependency_columns(
    candidates: list[str], id_values: set[str], df: pd.DataFrame
) -> list[str]:
    row_count = len(df)
    min_overlap_count = 1 if row_count < 500 else MIN_DEP_OVERLAP_COUNT
    if not id_values:
        if row_count > 1000:
            return []
        selected: list[str] = []
        for col in candidates:
            values = _normalize_values(df[col])
            if values.empty:
                continue
            if values.shape[0] < 1:
                continue
            selected.append(col)
        return selected
    selected: list[str] = []
    for col in candidates:
        values = _normalize_values(df[col])
        if values.empty:
            continue
        dep_set = set(values.tolist())
        if len(dep_set) < min_overlap_count:
            continue
        overlap_count = len(dep_set & id_values)
        if overlap_count < min_overlap_count:
            continue
        ratio = overlap_count / max(len(dep_set), 1)
        if ratio >= MIN_DEP_OVERLAP_RATIO:
            selected.append(col)
    return selected


class Plugin:
    def run(self, ctx) -> PluginResult:
        df = ctx.dataset_loader()
        if df.empty:
            return PluginResult("skipped", "Empty dataset", {}, [], [], None)

        columns_meta = []
        role_by_name: dict[str, str] = {}
        if ctx.dataset_version_id:
            dataset_template = ctx.storage.fetch_dataset_template(ctx.dataset_version_id)
            if dataset_template and dataset_template.get("status") == "ready":
                fields = ctx.storage.fetch_template_fields(
                    int(dataset_template["template_id"])
                )
                columns_meta = fields
                role_by_name = {
                    field["name"]: (field.get("role") or "") for field in fields
                }
            else:
                columns_meta = ctx.storage.fetch_dataset_columns(ctx.dataset_version_id)
                role_by_name = {
                    col["original_name"]: (col.get("role") or "")
                    for col in columns_meta
                }

        columns = list(df.columns)
        lower_names = {col: str(col).lower() for col in columns}

        process_candidates = _candidate_columns(
            ctx.settings.get("process_column"),
            columns,
            role_by_name,
            {"process", "activity", "event", "step", "task", "action"},
            ["process", "activity", "event", "step", "task", "action", "job"],
            lower_names,
        )
        process_col = _best_process_column(process_candidates, df)

        dep_cols = _parse_list(ctx.settings.get("dependency_columns"))
        dep_candidates: list[str]
        if dep_cols:
            dep_candidates = [col for col in dep_cols if col in columns]
        else:
            dep_tokens = ("dep", "dependency", "parent", "master", "preced")
            dep_candidates = [
                col
                for col in columns
                if any(tok in lower_names[col] for tok in dep_tokens)
            ]

        id_candidates = _candidate_columns(
            None,
            columns,
            role_by_name,
            {"process_id", "id"},
            ["process_id", "proc_id", "queue_id", "job_id", "run_id", "id"],
            lower_names,
        )
        id_values: set[str] = set()
        for col in id_candidates:
            id_values |= set(_normalize_values(df[col]).tolist())

        dep_cols = _select_dependency_columns(dep_candidates, id_values, df)

        if not dep_cols:
            return PluginResult(
                "ok",
                "No dependency columns detected",
                {
                    "standalone_runs": int(df.shape[0]),
                    "sequence_runs": 0,
                },
                [
                    {
                        "kind": "sequence_classification",
                        "measurement_type": "not_applicable",
                        "reason": "No dependency pointer columns were detected.",
                        "columns": [col for col in [process_col] if col],
                    }
                ],
                [],
                None,
            )

        work = df.loc[:, [col for col in [process_col, *dep_cols] if col]].copy()
        if process_col and process_col in work.columns:
            work["__process"] = work[process_col].astype(str).str.strip()
            work["__process_norm"] = work["__process"].str.lower()
        else:
            work["__process"] = "unknown"
            work["__process_norm"] = "unknown"

        sequence_mask = pd.Series(False, index=work.index)
        for col in dep_cols:
            series = work[col]
            mask = series.notna()
            if mask.any():
                text = series.astype(str).str.strip().str.lower()
                mask = mask & (~text.isin(INVALID_STRINGS))
            sequence_mask |= mask

        work["__sequence"] = sequence_mask
        work["__standalone"] = ~sequence_mask

        total_runs = int(work.shape[0])
        sequence_runs = int(sequence_mask.sum())
        standalone_runs = total_runs - sequence_runs

        summaries = (
            work.groupby("__process_norm")
            .agg(
                runs=("__process_norm", "size"),
                sequence_runs=("__sequence", "sum"),
            )
            .reset_index()
        )
        summaries["sequence_ratio"] = summaries["sequence_runs"] / summaries["runs"].replace(0, 1)
        summaries = summaries.sort_values(
            ["sequence_ratio", "__process_norm"], ascending=[False, True]
        )

        max_processes = int(ctx.settings.get("max_processes", 10))
        min_ratio = float(ctx.settings.get("min_sequence_ratio", 0.0))
        max_examples = int(ctx.settings.get("max_examples", 25))
        columns_used = [col for col in [process_col, *dep_cols] if col]

        findings = []
        for _, row in summaries.iterrows():
            ratio = float(row["sequence_ratio"])
            if ratio < min_ratio:
                continue
            proc = row["__process_norm"]
            label_series = work.loc[work["__process_norm"] == proc, "__process"]
            label = (
                label_series.value_counts().index[0]
                if not label_series.empty
                else proc
            )
            row_ids = work.loc[
                (work["__process_norm"] == proc) & work["__sequence"]
            ].index.tolist()[:max_examples]
            findings.append(
                {
                    "kind": "sequence_classification",
                    "process": label,
                    "process_norm": proc,
                    "runs_total": int(row["runs"]),
                    "sequence_runs": int(row["sequence_runs"]),
                    "standalone_runs": int(row["runs"] - row["sequence_runs"]),
                    "sequence_ratio": ratio,
                    "measurement_type": "measured",
                    "row_ids": [int(i) for i in row_ids],
                    "dependency_columns": dep_cols,
                    "columns": columns_used,
                }
            )
            if len(findings) >= max_processes:
                break

        artifacts_dir = ctx.artifacts_dir("analysis_sequence_classification")
        out_path = artifacts_dir / "classification.json"
        write_json(
            out_path,
            {
                "summary": {
                    "process_column": process_col,
                    "dependency_columns": dep_cols,
                },
                "totals": {
                    "total_runs": total_runs,
                    "sequence_runs": sequence_runs,
                    "standalone_runs": standalone_runs,
                },
                "by_process": [
                    {
                        "process_norm": row["__process_norm"],
                        "runs": int(row["runs"]),
                        "sequence_runs": int(row["sequence_runs"]),
                        "sequence_ratio": float(row["sequence_ratio"]),
                    }
                    for _, row in summaries.iterrows()
                ],
            },
        )
        artifacts = [
            PluginArtifact(
                path=str(out_path.relative_to(ctx.run_dir)),
                type="json",
                description="Standalone vs sequence classification",
            )
        ]

        return PluginResult(
            "ok",
            "Classified standalone vs sequence-linked runs",
            {
                "standalone_runs": standalone_runs,
                "sequence_runs": sequence_runs,
            },
            findings,
            artifacts,
            None,
        )
