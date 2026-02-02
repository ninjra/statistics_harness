from __future__ import annotations

import re
from typing import Any

import pandas as pd

from statistic_harness.core.types import PluginResult
from statistic_harness.core.utils import quote_identifier

ROLE_KEYS = [
    "queue_time",
    "start_time",
    "end_time",
    "process_id",
    "process_name",
    "module_code",
    "user_id",
    "dependency_id",
    "master_id",
    "host_id",
    "status",
]

ROLE_TOKENS = {
    "queue_time": ["queue", "queued", "enqueue"],
    "start_time": ["start", "begin"],
    "end_time": ["end", "finish", "complete", "stop"],
    "process_id": ["process_id", "proc_id", "processqueue", "queue_id", "processqueueid"],
    "process_name": ["process", "job", "task", "activity", "step", "action", "proc"],
    "module_code": ["module", "module_cd", "module_code", "mod"],
    "user_id": ["user", "userid", "user_id", "operator", "owner"],
    "dependency_id": ["dep", "dependency", "parent", "prereq", "precede"],
    "master_id": ["master", "sequence", "chain", "workflow", "batch", "group", "case"],
    "host_id": ["host", "server", "node", "qpec", "worker", "machine"],
    "status": ["status", "state", "result", "outcome"],
}

TOKEN_SPLIT = re.compile(r"[^a-zA-Z0-9]+")


def _tokenize(name: str) -> list[str]:
    raw = TOKEN_SPLIT.split(name.lower())
    return [token for token in raw if token]


def _timestamp_ratio(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    return float(parsed.notna().mean())


def _numeric_ratio(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.notna().mean())


def _unique_ratio(values: pd.Series) -> float:
    total = len(values)
    if total == 0:
        return 0.0
    return float(values.nunique(dropna=True) / total)


def _score_role(role: str, name: str, tokens: list[str], stats: dict[str, float]) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    lower_name = name.lower()
    token_set = set(tokens)
    role_tokens = ROLE_TOKENS.get(role, [])
    if any(token in token_set for token in role_tokens) or any(t in lower_name for t in role_tokens):
        score += 2.0
        reasons.append("name_match")

    if role in {"queue_time", "start_time", "end_time"}:
        if any(token in token_set for token in ["time", "date", "dt"]):
            score += 1.0
            reasons.append("time_token")
        if stats["timestamp_ratio"] >= 0.8:
            score += 2.0
            reasons.append("timestamp_high")
        elif stats["timestamp_ratio"] >= 0.5:
            score += 1.0
            reasons.append("timestamp_medium")
        else:
            score -= 0.5
    else:
        if stats["timestamp_ratio"] >= 0.8:
            score -= 0.5

    if role in {"process_id", "dependency_id", "master_id", "user_id"}:
        if "id" in token_set or lower_name.endswith("id"):
            score += 1.0
            reasons.append("id_token")
        if stats["unique_ratio"] >= 0.8:
            score += 1.5
            reasons.append("unique_high")
        elif stats["unique_ratio"] >= 0.5:
            score += 1.0
            reasons.append("unique_medium")
        if stats["numeric_ratio"] >= 0.8:
            score += 0.5
            reasons.append("numeric_high")

    if role in {"process_name", "module_code", "status", "host_id"}:
        if stats["unique_ratio"] <= 0.2:
            score += 0.5
            reasons.append("unique_low")
        if stats["numeric_ratio"] >= 0.7:
            score -= 0.5

    if role == "process_name" and stats["unique_ratio"] >= 0.6:
        score -= 0.5
    if role == "status" and stats["unique_ratio"] <= 0.1:
        score += 0.5

    return max(score, 0.0), reasons


class Plugin:
    def run(self, ctx) -> PluginResult:
        if not ctx.dataset_version_id:
            return PluginResult("error", "Missing dataset version", {}, [], [], None)

        sample_rows = int(ctx.settings.get("sample_rows", 500))
        min_confidence = float(ctx.settings.get("min_confidence", 2.0))

        columns = ctx.storage.fetch_dataset_columns(ctx.dataset_version_id)
        if not columns:
            return PluginResult("skipped", "No columns available", {}, [], [], None)

        with ctx.storage.connection() as conn:
            version = ctx.storage.get_dataset_version(ctx.dataset_version_id, conn)
            if not version:
                return PluginResult("error", "Dataset version not found", {}, [], [], None)
            safe_cols = [col["safe_name"] for col in columns]
            if not safe_cols:
                return PluginResult("skipped", "No dataset columns", {}, [], [], None)
            quoted = ", ".join(quote_identifier(col) for col in safe_cols)
            sql = (
                f"SELECT {quoted} FROM {quote_identifier(version['table_name'])} "
                "ORDER BY row_index LIMIT ?"
            )
            df = pd.read_sql_query(sql, conn, params=(sample_rows,))

        df = df.rename(columns={col["safe_name"]: col["original_name"] for col in columns})
        candidates: list[dict[str, Any]] = []
        best_by_column: dict[int, tuple[str, float]] = {}
        low_confidence_roles: set[str] = set()

        for col in columns:
            name = col["original_name"]
            if name not in df.columns:
                continue
            series = df[name].dropna()
            stats = {
                "timestamp_ratio": _timestamp_ratio(series),
                "numeric_ratio": _numeric_ratio(series),
                "unique_ratio": _unique_ratio(series),
            }
            tokens = _tokenize(name)
            best_role = ""
            best_score = 0.0
            for role in ROLE_KEYS:
                score, reasons = _score_role(role, name, tokens, stats)
                if score <= 0:
                    continue
                candidates.append(
                    {
                        "column_id": col["column_id"],
                        "role": role,
                        "score": score,
                        "reasons": reasons,
                    }
                )
                if score > best_score or (score == best_score and role < best_role):
                    best_role = role
                    best_score = score
            if best_role:
                best_by_column[col["column_id"]] = (best_role, best_score)

        ctx.storage.replace_dataset_role_candidates(ctx.dataset_version_id, candidates)

        role_by_name: dict[str, str] = {}
        for col in columns:
            entry = best_by_column.get(col["column_id"])
            if not entry:
                continue
            role, score = entry
            if score >= min_confidence:
                role_by_name[col["original_name"]] = role
            else:
                low_confidence_roles.add(role)

        if role_by_name:
            ctx.storage.update_dataset_column_roles(ctx.dataset_version_id, role_by_name)

        findings = []
        for col in columns:
            entry = best_by_column.get(col["column_id"])
            if not entry:
                continue
            role, score = entry
            measurement_type = "measured" if score >= min_confidence else "not_applicable"
            findings.append(
                {
                    "kind": "role_inference",
                    "role": role,
                    "column": col["original_name"],
                    "score": score,
                    "measurement_type": measurement_type,
                }
            )

        for role in sorted(low_confidence_roles):
            findings.append(
                {
                    "kind": "role_confidence_low",
                    "role": role,
                    "measurement_type": "not_applicable",
                    "detail": "Best candidate below confidence threshold",
                }
            )

        metrics = {
            "columns_scanned": len(columns),
            "candidates": len(candidates),
            "roles_assigned": len(role_by_name),
            "low_confidence_roles": len(low_confidence_roles),
        }
        summary = f"Inferred roles for {len(role_by_name)} columns"
        return PluginResult("ok", summary, metrics, findings, [], None)
