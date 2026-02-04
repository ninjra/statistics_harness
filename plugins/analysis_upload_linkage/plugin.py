from __future__ import annotations

import csv
import re
from collections import Counter
from datetime import timedelta
from typing import Any

import pandas as pd

from statistic_harness.core.column_inference import infer_timestamp_series
from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json

PARAM_PATTERN = re.compile(r"\s*([^;=]+?)\s*\([^\)]*\)\s*=\s*([^;]+)")
FILENAME_PATTERN = re.compile(r"([A-Za-z0-9_\-\. ]+\.(?:xlsx|xls))", re.IGNORECASE)


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
    if preferred and preferred in columns:
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


def _best_datetime_column(candidates: list[str], df: pd.DataFrame) -> str | None:
    best_col = None
    best_score = 0.0
    for col in candidates:
        info = infer_timestamp_series(df[col], name_hint=col, sample_size=2000)
        if not info.valid:
            continue
        if info.score > best_score:
            best_score = info.score
            best_col = col
    return best_col


def _pick_process_column(candidates: list[str]) -> str | None:
    for col in candidates:
        name = str(col).lower()
        if name in {"process_id", "process"} or name.endswith("process_id"):
            return col
    for col in candidates:
        name = str(col).lower()
        if "process" in name and "queue" not in name and "parent" not in name:
            return col
    return candidates[0] if candidates else None


def _pick_user_column(candidates: list[str]) -> str | None:
    for col in candidates:
        name = str(col).lower()
        if "user_id" in name or name.endswith("_user"):
            return col
    return candidates[0] if candidates else None


def _parse_params(text: str) -> dict[str, str]:
    if not text:
        return {}
    out: dict[str, str] = {}
    for match in PARAM_PATTERN.finditer(text):
        key = match.group(1).strip().lower()
        value = match.group(2).strip()
        if key:
            out[key] = value
    return out


def _extract_filename(text: str, params: dict[str, str]) -> str | None:
    for key in ("override imp/exp filename", "imp/exp filename", "filename"):
        value = params.get(key)
        if value and ".xls" in value.lower():
            return value.strip()
    match = FILENAME_PATTERN.search(text or "")
    if match:
        return match.group(1).strip()
    return None


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

        process_col = None
        process_candidates = _candidate_columns(
            ctx.settings.get("process_column"),
            columns,
            role_by_name,
            {"process", "activity", "event", "step", "task"},
            ["process", "activity", "event", "step", "task", "job"],
            lower_names,
        )
        if "PROCESS_ID" in columns:
            process_col = "PROCESS_ID"
        elif process_candidates:
            process_col = _pick_process_column(process_candidates)

        params_candidates = _candidate_columns(
            ctx.settings.get("params_column"),
            columns,
            role_by_name,
            {"params", "attributes"},
            ["param", "descr", "argument", "input"],
            lower_names,
        )
        if "PARAM_DESCR_LIST" in columns:
            params_col = "PARAM_DESCR_LIST"
        else:
            params_col = params_candidates[0] if params_candidates else None

        user_candidates = _candidate_columns(
            ctx.settings.get("user_column"),
            columns,
            role_by_name,
            {"user"},
            ["user"],
            lower_names,
        )
        if "USER_ID" in columns:
            user_col = "USER_ID"
        else:
            user_col = _pick_user_column(user_candidates) if user_candidates else None

        start_candidates = _candidate_columns(
            ctx.settings.get("start_column"),
            columns,
            role_by_name,
            {"start_time", "start"},
            ["start", "begin"],
            lower_names,
        )
        queue_candidates = _candidate_columns(
            ctx.settings.get("queue_column"),
            columns,
            role_by_name,
            {"queue_time", "queue"},
            ["queue", "enqueue", "submitted"],
            lower_names,
        )
        if "START_DT" in columns:
            start_col = "START_DT"
        else:
            start_col = _best_datetime_column(start_candidates, df) if start_candidates else None
        if "QUEUE_DT" in columns:
            queue_col = "QUEUE_DT"
        else:
            queue_col = _best_datetime_column(queue_candidates, df) if queue_candidates else None

        timestamp_col = start_col or queue_col
        if not process_col or not params_col or not timestamp_col:
            return PluginResult(
                "ok",
                "Upload linkage not applicable",
                {"upload_rows": 0, "bkrvnu_rows": 0},
                [],
                [],
                None,
            )

        process_series = df[process_col].astype(str)
        params_series = df[params_col].fillna("").astype(str)
        user_series = (
            df[user_col]
            if user_col and user_col in df.columns
            else pd.Series([None] * len(df), index=df.index)
        )
        timestamp_series = pd.to_datetime(df[timestamp_col], errors="coerce")

        upload_mask = params_series.str.contains(r"\.xls", case=False, regex=True)
        bkrvnu_mask = process_series.str.lower() == "bkrvnu"

        upload_rows = []
        upload_processes: list[str] = []
        for idx in df.index[upload_mask & timestamp_series.notna()]:
            process = str(process_series.iloc[idx]).strip()
            params_text = params_series.iloc[idx]
            params = _parse_params(params_text)
            filename = _extract_filename(params_text, params)
            upload_rows.append(
                {
                    "process": process,
                    "params": params,
                    "params_text": params_text,
                    "filename": filename,
                    "batch_number": params.get("batch number"),
                    "business_unit": params.get("business unit"),
                    "business_segment": params.get("business segment"),
                    "accounting_month": params.get("accounting month"),
                    "user": user_series.iloc[idx],
                    "dt": timestamp_series.iloc[idx].to_pydatetime(),
                }
            )
            upload_processes.append(process)

        bkrvnu_rows = []
        for idx in df.index[bkrvnu_mask & timestamp_series.notna()]:
            params_text = params_series.iloc[idx]
            params = _parse_params(params_text)
            bkrvnu_rows.append(
                {
                    "params": params,
                    "params_text": params_text,
                    "rvnu_run_id": params.get("rvnu run id"),
                    "revenue_source_code": params.get("revenue source code"),
                    "user": user_series.iloc[idx],
                    "dt": timestamp_series.iloc[idx].to_pydatetime(),
                }
            )

        if not upload_rows or not bkrvnu_rows:
            summary = "No XLSX uploads or BKRVNU rows found for linkage analysis."
            metrics = {
                "upload_rows": len(upload_rows),
                "bkrvnu_rows": len(bkrvnu_rows),
            }
            return PluginResult("ok", summary, metrics, [], [], None)

        upload_rows.sort(key=lambda row: row["dt"])
        window_hours = ctx.settings.get("upload_window_hours") or 48
        window = timedelta(hours=float(window_hours))

        matched_any = 0
        matched_user = 0
        examples: list[dict[str, Any]] = []

        for row in bkrvnu_rows:
            dt = row["dt"]
            candidates = [
                upload
                for upload in upload_rows
                if timedelta(0) <= dt - upload["dt"] <= window
            ]
            if candidates:
                matched_any += 1
            user = row.get("user")
            user_matches = [
                upload for upload in candidates if user and upload.get("user") == user
            ]
            if user_matches:
                matched_user += 1
            if user_matches and len(examples) < int(ctx.settings.get("max_examples") or 10):
                closest = min(user_matches, key=lambda item: abs((dt - item["dt"]).total_seconds()))
                examples.append(
                    {
                        "bkrvnu_time": dt.isoformat(),
                        "bkrvnu_user": user,
                        "upload_time": closest["dt"].isoformat(),
                        "upload_user": closest.get("user"),
                        "upload_filename": closest.get("filename"),
                        "lag_hours": (dt - closest["dt"]).total_seconds() / 3600.0,
                    }
                )

        bkrvnu_count = len(bkrvnu_rows)
        upload_count = len(upload_rows)
        matched_any_pct = matched_any / bkrvnu_count if bkrvnu_count else 0.0
        matched_user_pct = matched_user / bkrvnu_count if bkrvnu_count else 0.0

        primary_upload = Counter(upload_processes).most_common(1)
        upload_process = primary_upload[0][0] if primary_upload else "upload"

        metrics = {
            "upload_rows": upload_count,
            "bkrvnu_rows": bkrvnu_count,
            "matched_any_count": matched_any,
            "matched_any_pct": matched_any_pct,
            "matched_user_count": matched_user,
            "matched_user_pct": matched_user_pct,
            "window_hours": window_hours,
        }

        findings = [
            {
                "kind": "upload_bkrvnu_linkage",
                "upload_process": upload_process,
                "upload_processes": sorted(set(upload_processes)),
                "upload_rows": upload_count,
                "bkrvnu_rows": bkrvnu_count,
                "matched_any_count": matched_any,
                "matched_any_pct": matched_any_pct,
                "matched_user_count": matched_user,
                "matched_user_pct": matched_user_pct,
                "window_hours": window_hours,
                "examples": examples,
                "measurement_type": "measured",
                "confidence": 0.85,
                "evidence": {
                    "dataset_id": ctx.dataset_id or "unknown",
                    "dataset_version_id": ctx.dataset_version_id or "unknown",
                    "row_ids": [],
                    "column_ids": [],
                    "query": None,
                },
            }
        ]

        summary = (
            f"Matched {matched_user} of {bkrvnu_count} BKRVNU rows to an XLSX upload by user +"
            f"{int(window_hours)}h window ({matched_user_pct:.1%} coverage)."
        )

        artifacts_dir = ctx.artifacts_dir("analysis_upload_linkage")
        results_path = artifacts_dir / "results.json"
        write_json(results_path, {"summary": summary, "metrics": metrics, "findings": findings})

        csv_path = artifacts_dir / "examples.csv"
        if examples:
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "bkrvnu_time",
                        "bkrvnu_user",
                        "upload_time",
                        "upload_user",
                        "upload_filename",
                        "lag_hours",
                    ],
                )
                writer.writeheader()
                for row in examples:
                    writer.writerow(row)

        artifacts = [
            PluginArtifact(
                path=str(results_path.relative_to(ctx.run_dir)),
                type="json",
                description="Upload-to-BKRVNU linkage summary",
            )
        ]
        if examples:
            artifacts.append(
                PluginArtifact(
                    path=str(csv_path.relative_to(ctx.run_dir)),
                    type="csv",
                    description="Sample BKRVNU to upload matches",
                )
            )

        return PluginResult("ok", summary, metrics, findings, artifacts, None)
