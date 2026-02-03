from __future__ import annotations

import re
from typing import Any


_PROCESS_HINT_RE = re.compile(
    r"(?:process|job|task|step|called)\s+([A-Za-z0-9_\-]+)", re.IGNORECASE
)
_UPPER_TOKEN_RE = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")
_STOPWORDS = {
    "CLOSE",
    "CYCLE",
    "SERVER",
    "SERVERS",
    "QPEC",
    "QRA",
    "ERP",
    "HOST",
    "HOSTS",
    "QUEUE",
    "WAIT",
    "ELIGIBLE",
    "CAPACITY",
}


def _extract_process(text: str, hint: str | None = None) -> str | None:
    if hint:
        cleaned = str(hint).strip()
        if cleaned:
            return cleaned
    match = _PROCESS_HINT_RE.search(text)
    if match:
        return match.group(1)
    tokens = [tok for tok in _UPPER_TOKEN_RE.findall(text) if tok not in _STOPWORDS]
    if tokens:
        return tokens[0]
    return None


def _flag(text: str, tokens: list[str]) -> bool:
    return any(token in text for token in tokens)


def _compact_title(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip())
    if not cleaned:
        return "Known issue"
    if len(cleaned) <= 80:
        return cleaned
    return cleaned[:77] + "..."


def compile_known_issues(
    natural_language: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    if not natural_language:
        return [], []
    compiled: list[dict[str, Any]] = []
    warnings: list[str] = []

    for entry in natural_language:
        if not isinstance(entry, dict):
            continue
        raw_text = str(entry.get("text") or "").strip()
        if not raw_text:
            continue
        text = raw_text.lower()
        process = _extract_process(raw_text, entry.get("process_hint"))
        mentions_wait = _flag(text, ["wait", "queue", "delay", "latency", "eligible"])
        mentions_close = _flag(text, ["close cycle", "close window", "close period"])
        mentions_disable = _flag(text, ["remove", "disable", "turn off", "eliminate"])
        mentions_third = _flag(
            text,
            [
                "3rd",
                "third",
                "third server",
                "3rd server",
                "add server",
                "add host",
                "third qpec",
                "3rd qpec",
            ],
        )
        mentions_contention = _flag(text, ["contention", "bottleneck", "congest"])

        issues: list[dict[str, Any]] = []
        title = _compact_title(raw_text)

        if process:
            process_norm = process.lower()
            if mentions_wait or mentions_close or mentions_contention:
                issues.append(
                    {
                        "title": title,
                        "description": raw_text,
                        "plugin_id": "analysis_queue_delay_decomposition",
                        "kind": "eligible_wait_process_stats",
                        "where": {"process_norm": process_norm},
                    }
                )
            if mentions_disable:
                issues.append(
                    {
                        "title": title,
                        "description": raw_text,
                        "plugin_id": "analysis_queue_delay_decomposition",
                        "kind": "eligible_wait_impact",
                        "where": {"process_norm": process_norm},
                    }
                )
            if mentions_third:
                issues.append(
                    {
                        "title": title,
                        "description": raw_text,
                        "plugin_id": "analysis_queue_delay_decomposition",
                        "kind": "capacity_scale_model",
                        "where": {"process_norm": process_norm},
                    }
                )

        if mentions_third:
            issues.append(
                {
                    "title": title,
                    "description": raw_text,
                    "plugin_id": "analysis_close_cycle_capacity_model",
                    "kind": "close_cycle_capacity_model",
                }
            )
            issues.append(
                {
                    "title": title,
                    "description": raw_text,
                    "plugin_id": "analysis_close_cycle_capacity_impact",
                    "kind": "close_cycle_capacity_impact",
                }
            )

        if not issues:
            warnings.append(
                f"Could not compile known issue: '{_compact_title(raw_text)}'"
            )
            continue

        for issue in issues:
            issue["source_text"] = raw_text
        compiled.extend(issues)

    return compiled, warnings
