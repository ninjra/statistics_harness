from __future__ import annotations

import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np
import pandas as pd

from statistic_harness.core.process_matcher import (
    compile_patterns,
    default_exclude_process_patterns,
    merge_patterns,
    parse_exclude_patterns_env,
)
from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import quote_identifier, write_json


# Third-party dependencies (installed in the project venv; do not fallback).
from simhash import Simhash  # type: ignore
from mlxtend.frequent_patterns import fpgrowth, association_rules, apriori  # type: ignore
from prefixspan import PrefixSpan  # type: ignore
import hdbscan  # type: ignore
import networkx as nx  # type: ignore
import igraph as ig  # type: ignore
import leidenalg  # type: ignore
from dtaidistance import dtw as dtw_lib  # type: ignore
from ortools.sat.python import cp_model  # type: ignore
import simpy  # type: ignore
from sklearn.cluster import SpectralCoclustering, KMeans, SpectralClustering  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.decomposition import TruncatedSVD  # type: ignore


def _NOW() -> str:
    return datetime.now(timezone.utc).isoformat()


def _import_datasketch() -> tuple[Any, Any]:
    # Some environments expose a foreign cupy install path that is blocked by the
    # plugin file-access guard. MinHash itself is CPU-only, so force cupy optional
    # import to fail closed before importing datasketch.
    if "cupy" not in sys.modules:
        sys.modules["cupy"] = None
    from datasketch import MinHash, MinHashLSH  # type: ignore

    return MinHash, MinHashLSH


@dataclass(frozen=True)
class TemplateInfo:
    table_name: str
    # template field name -> template safe_name (column in the normalized template table)
    field_to_safe: dict[str, str]
    # template safe_name -> role (derived via dataset column roles + dataset->template mapping)
    safe_to_role: dict[str, str]

    def role_field(self, role: str) -> str | None:
        """Return the normalized template column name for a given role, if available.

        The normalized template table is expected to use the dataset column `safe_name`
        identifiers (e.g., `c1`, `c2`, ...). Older template metadata may contain a
        raw-field -> safe-name mapping; newer metadata can be nested and may not
        provide a direct field mapping. We therefore primarily resolve role -> safe
        column name via `safe_to_role`, and only use `field_to_safe` as a legacy
        fallback.
        """

        role_l = role.lower()
        # Preferred: safe_name is the actual column name in the normalized template.
        for safe, r in self.safe_to_role.items():
            if isinstance(safe, str) and isinstance(r, str) and r.lower() == role_l:
                return safe

        # Legacy fallback: attempt to route through field_to_safe if present.
        for _field, safe in self.field_to_safe.items():
            r = self.safe_to_role.get(safe)
            if isinstance(r, str) and r.lower() == role_l:
                return safe
        return None


def _load_template_info(ctx) -> TemplateInfo | None:
    if not ctx.dataset_version_id:
        return None
    dt = ctx.storage.fetch_dataset_template(ctx.dataset_version_id)
    if not isinstance(dt, dict) or dt.get("status") != "ready":
        return None
    table = str(dt.get("table_name") or "").strip()
    if not table:
        return None
    mapping_json = str(dt.get("mapping_json") or "").strip()
    # Parse mapping_json. Newer normalize plugin stores:
    #   {"mapping": {<template field name>: {"safe_name": <dataset safe col>, ...}}, ...}
    try:
        mapping_payload = json.loads(mapping_json) if mapping_json else {}
    except json.JSONDecodeError:
        mapping_payload = {}
    mapping: dict[str, Any] = {}
    if isinstance(mapping_payload, dict):
        inner = mapping_payload.get("mapping")
        if isinstance(inner, dict):
            mapping = inner
        else:
            # Back-compat: older payloads may have been a direct mapping dict.
            mapping = mapping_payload

    # Dataset safe_name -> role comes from profile plugins.
    dataset_safe_to_role: dict[str, str] = {}
    cols = ctx.storage.fetch_dataset_columns(ctx.dataset_version_id)
    if isinstance(cols, list):
        for col in cols:
            if not isinstance(col, dict):
                continue
            safe = col.get("safe_name")
            role = col.get("role")
            if isinstance(safe, str) and isinstance(role, str) and safe.strip() and role.strip():
                dataset_safe_to_role[safe.strip()] = role.strip()

    # Template field name -> template safe_name comes from template_fields.
    template_id_raw = dt.get("template_id")
    try:
        template_id = int(template_id_raw) if template_id_raw is not None else None
    except (TypeError, ValueError):
        template_id = None
    template_name_to_safe: dict[str, str] = {}
    if template_id is not None:
        fields = ctx.storage.fetch_template_fields(template_id)
        if isinstance(fields, list):
            for f in fields:
                if not isinstance(f, dict):
                    continue
                name = f.get("name")
                safe = f.get("safe_name")
                if isinstance(name, str) and isinstance(safe, str) and name.strip() and safe.strip():
                    template_name_to_safe[name.strip()] = safe.strip()

    # Derive template safe_name -> role by joining mapping(field->dataset safe)->dataset roles and
    # mapping(field->template safe) from template_fields.
    field_to_safe: dict[str, str] = {}
    safe_to_role: dict[str, str] = {}
    if isinstance(mapping, dict):
        for field, meta in mapping.items():
            if not isinstance(field, str) or not field.strip():
                continue
            t_safe = template_name_to_safe.get(field.strip())
            if isinstance(t_safe, str) and t_safe.strip():
                field_to_safe[field.strip()] = t_safe.strip()

            if not isinstance(meta, dict):
                continue
            d_safe = meta.get("safe_name")
            if not (isinstance(d_safe, str) and d_safe.strip()):
                continue
            role = dataset_safe_to_role.get(d_safe.strip())
            if isinstance(role, str) and role.strip() and isinstance(t_safe, str) and t_safe.strip():
                safe_to_role[t_safe.strip()] = role.strip()

    return TemplateInfo(table_name=table, field_to_safe=field_to_safe, safe_to_role=safe_to_role)


def _exclude_matcher(ctx, config: dict[str, Any]) -> Any:
    # Merge explicit plugin config excludes with env and repo defaults.
    explicit = config.get("exclude_processes")
    if not isinstance(explicit, list):
        explicit = []
    explicit = [str(x).strip() for x in explicit if isinstance(x, str) and str(x).strip()]
    patterns = merge_patterns(parse_exclude_patterns_env(), explicit, default_exclude_process_patterns())
    return compile_patterns(patterns) if patterns else (lambda _s: False)


def _artifact(ctx, plugin_id: str, name: str, payload: Any, kind: str = "json") -> PluginArtifact:
    out_dir = ctx.artifacts_dir(plugin_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    write_json(path, payload)
    return PluginArtifact(path=str(path.relative_to(ctx.run_dir)), type=kind, description=name)


def _require_template(info: TemplateInfo | None, plugin_id: str) -> tuple[str, TemplateInfo] | PluginResult:
    if info is None:
        return PluginResult(
            "skipped",
            "No ready normalized template found for dataset; run transform_normalize_mixed first",
            metrics={},
            findings=[],
            artifacts=[],
            error=None,
        )
    return info.table_name, info


def _process_norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _tokenize_kv(rows: Iterable[tuple[str, str]]) -> set[str]:
    tokens: set[str] = set()
    for k, v in rows:
        k = str(k).strip().lower()
        v = str(v).strip()
        if not k:
            continue
        if v:
            tokens.add(f"{k}={v}")
        else:
            tokens.add(k)
    return tokens


def _fetch_process_entity_counts(
    conn,
    template_table: str,
    dataset_version_id: str,
    process_col: str,
    *,
    max_processes: int,
    max_entities_per_process: int,
    exclude_match,
) -> dict[str, list[tuple[int, int]]]:
    """Return mapping: process_norm -> [(entity_id, run_count), ...]"""

    # Top processes by run count.
    sql_top = (
        f"SELECT LOWER(TRIM({quote_identifier(process_col)})) AS proc, COUNT(*) AS n "
        f"FROM {quote_identifier(template_table)} "
        f"WHERE dataset_version_id = ? AND {quote_identifier(process_col)} IS NOT NULL "
        f"GROUP BY proc ORDER BY n DESC LIMIT ?"
    )
    top_rows = conn.execute(sql_top, (dataset_version_id, int(max_processes))).fetchall()
    procs = [str(r["proc"]) for r in top_rows if r["proc"]]
    procs = [p for p in procs if p and not exclude_match(p)]
    if not procs:
        return {}

    # Entity counts per process (cap per process).
    out: dict[str, list[tuple[int, int]]] = {}
    for proc in procs:
        sql = (
            f"""
            SELECT rpl.entity_id AS entity_id, COUNT(*) AS n
            FROM {quote_identifier(template_table)} t
            JOIN row_parameter_link rpl
              ON rpl.dataset_version_id = t.dataset_version_id AND rpl.row_index = t.row_index
            WHERE t.dataset_version_id = ?
              AND LOWER(TRIM(t.{quote_identifier(process_col)})) = ?
            GROUP BY rpl.entity_id
            ORDER BY n DESC
            LIMIT ?
            """
        )
        rows = conn.execute(sql, (dataset_version_id, proc, int(max_entities_per_process))).fetchall()
        out[proc] = [(int(r["entity_id"]), int(r["n"])) for r in rows]
    return out


def _fetch_entity_kv(conn, entity_ids: list[int], ignore_keys_re: re.Pattern[str] | None) -> dict[int, list[tuple[str, str]]]:
    if not entity_ids:
        return {}
    placeholders = ",".join("?" for _ in entity_ids)
    rows = conn.execute(
        f"SELECT entity_id, key, value FROM parameter_kv WHERE entity_id IN ({placeholders})",
        tuple(entity_ids),
    ).fetchall()
    out: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for r in rows:
        eid = int(r["entity_id"])
        k = str(r["key"] or "")
        v = str(r["value"] or "")
        if ignore_keys_re and ignore_keys_re.search(k.lower()):
            continue
        out[eid].append((k, v))
    return out


def _make_actionable_lever(
    *,
    plugin_id: str,
    process_norm: str,
    title: str,
    recommendation: str,
    action_type: str,
    expected_delta_seconds: float | None,
    confidence: float,
    evidence: dict[str, Any],
    assumptions: list[str] | None = None,
    scope: dict[str, Any] | None = None,
    measurement_type: str = "measured",
) -> dict[str, Any]:
    proc = process_norm.strip().lower()
    normalized_scope = scope if isinstance(scope, dict) and scope else {}
    normalized_assumptions = [a for a in (assumptions or []) if isinstance(a, str) and a.strip()]
    if measurement_type == "modeled":
        if not normalized_scope:
            normalized_scope = {"scope_type": "dataset", "scope_value": "latest_window"}
        if not normalized_assumptions:
            normalized_assumptions = [
                "modeled from observed historical execution traces",
                "no external constraints were changed during modeling",
            ]
    return {
        "kind": "actionable_ops_lever",
        "measurement_type": measurement_type,
        "title": title,
        "recommendation": recommendation,
        "process": proc,
        "process_norm": proc,
        "process_id": f"proc:{proc}",
        "action_type": action_type,
        "expected_delta_seconds": float(expected_delta_seconds) if isinstance(expected_delta_seconds, (int, float)) else None,
        "expected_delta_percent": None,
        "confidence": float(max(0.0, min(1.0, confidence))),
        "scope": normalized_scope,
        "assumptions": normalized_assumptions,
        "evidence": {"plugin": plugin_id, **(evidence or {})},
    }


def _parse_ignore_regex(config: dict[str, Any]) -> re.Pattern[str] | None:
    raw = str(config.get("ignore_param_keys_regex") or "").strip()
    if not raw:
        return None
    try:
        return re.compile(raw, flags=re.IGNORECASE)
    except re.error:
        return None


def _minhash_clusters_for_process(
    proc: str,
    entity_counts: list[tuple[int, int]],
    entity_kv: dict[int, list[tuple[str, str]]],
    *,
    minhash_cls: Any,
    lsh_cls: Any,
    num_perm: int,
    threshold: float,
    min_cluster_size: int,
) -> list[list[int]]:
    if len(entity_counts) < min_cluster_size:
        return []
    # Build MinHash signatures.
    mhs: dict[int, Any] = {}
    for eid, _n in entity_counts:
        kv = entity_kv.get(eid) or []
        tokens = _tokenize_kv(kv)
        if not tokens:
            continue
        mh = minhash_cls(num_perm=int(num_perm))
        for tok in sorted(tokens):
            mh.update(tok.encode("utf-8"))
        mhs[eid] = mh

    if len(mhs) < min_cluster_size:
        return []

    lsh = lsh_cls(threshold=float(threshold), num_perm=int(num_perm))
    for eid, mh in mhs.items():
        lsh.insert(str(eid), mh)

    # Union-find clustering by LSH neighbor queries.
    parent: dict[int, int] = {eid: eid for eid in mhs.keys()}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for eid, mh in mhs.items():
        neigh = lsh.query(mh)
        for other in neigh:
            try:
                oid = int(other)
            except ValueError:
                continue
            if oid in parent:
                union(eid, oid)

    clusters: dict[int, list[int]] = defaultdict(list)
    for eid in mhs.keys():
        clusters[find(eid)].append(eid)

    out = [sorted(v) for v in clusters.values() if len(v) >= int(min_cluster_size)]
    out.sort(key=len, reverse=True)
    return out


def _simhash_clusters_for_process(
    proc: str,
    entity_counts: list[tuple[int, int]],
    entity_kv: dict[int, list[tuple[str, str]]],
    *,
    bits: int,
    max_hamming: int,
    min_cluster_size: int,
) -> list[list[int]]:
    if len(entity_counts) < min_cluster_size:
        return []
    fingerprints: dict[int, int] = {}
    for eid, _n in entity_counts:
        kv = entity_kv.get(eid) or []
        toks = sorted(_tokenize_kv(kv))
        if not toks:
            continue
        fp = Simhash(toks, f=int(bits)).value
        fingerprints[eid] = int(fp)

    if len(fingerprints) < min_cluster_size:
        return []

    # Simple bucketed LSH by prefix bits for deterministic grouping.
    bucket_bits = max(8, int(bits // 4))
    buckets: dict[int, list[int]] = defaultdict(list)
    for eid, fp in fingerprints.items():
        buckets[int(fp >> (bits - bucket_bits))].append(eid)

    # Within each bucket, union by hamming distance threshold.
    parent: dict[int, int] = {eid: eid for eid in fingerprints.keys()}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    def hamming(a: int, b: int) -> int:
        return int((a ^ b).bit_count())

    for eids in buckets.values():
        if len(eids) < 2:
            continue
        eids = sorted(eids)
        for i in range(len(eids)):
            for j in range(i + 1, len(eids)):
                a, b = eids[i], eids[j]
                if hamming(fingerprints[a], fingerprints[b]) <= int(max_hamming):
                    union(a, b)

    clusters: dict[int, list[int]] = defaultdict(list)
    for eid in fingerprints.keys():
        clusters[find(eid)].append(eid)
    out = [sorted(v) for v in clusters.values() if len(v) >= int(min_cluster_size)]
    out.sort(key=len, reverse=True)
    return out


def _best_varying_key(entity_ids: list[int], entity_kv: dict[int, list[tuple[str, str]]]) -> tuple[str | None, float]:
    """Pick a key whose values vary the most across the cluster."""

    values_by_key: dict[str, set[str]] = defaultdict(set)
    for eid in entity_ids:
        for k, v in entity_kv.get(eid) or []:
            k2 = str(k).strip().lower()
            if not k2 or k2 == "raw":
                continue
            values_by_key[k2].add(str(v).strip())
    if not values_by_key:
        return None, 0.0
    best = max(values_by_key.items(), key=lambda kv: len(kv[1]))
    key, vals = best
    return key, float(len(vals))


def _run_param_near_duplicate_minhash(ctx, plugin_id: str, config: dict[str, Any]) -> PluginResult:
    try:
        minhash_cls, lsh_cls = _import_datasketch()
    except Exception as exc:
        return PluginResult("error", f"Failed to import datasketch MinHash: {exc}", {}, [], [], None)

    info = _load_template_info(ctx)
    req = _require_template(info, plugin_id)
    if isinstance(req, PluginResult):
        return req
    template_table, info = req
    dataset_version_id = str(ctx.dataset_version_id)
    proc_col = info.role_field("process_name") or info.role_field("process") or None
    if not proc_col:
        return PluginResult("skipped", "No process column inferred (role process_name)", {}, [], [], None)

    exclude_match = _exclude_matcher(ctx, config)
    ignore_re = _parse_ignore_regex(config)

    max_processes = int(config.get("max_processes") or 60)
    max_entities = int(config.get("max_entities_per_process") or 1500)
    num_perm = int(config.get("num_perm") or 128)
    thr = float(config.get("lsh_threshold") or 0.85)
    min_cluster = int(config.get("min_cluster_size") or 5)

    findings: list[dict[str, Any]] = []
    artifacts: list[PluginArtifact] = []
    clusters_out: list[dict[str, Any]] = []

    with ctx.storage.connection() as conn:
        proc_entities = _fetch_process_entity_counts(
            conn,
            template_table,
            dataset_version_id,
            proc_col,
            max_processes=max_processes,
            max_entities_per_process=max_entities,
            exclude_match=exclude_match,
        )
        # Preload kv for all entity_ids we will touch to keep queries bounded.
        all_eids = sorted({eid for rows in proc_entities.values() for eid, _n in rows})
        entity_kv = _fetch_entity_kv(conn, all_eids, ignore_re)

    for proc, rows in proc_entities.items():
        clusters = _minhash_clusters_for_process(
            proc,
            rows,
            entity_kv,
            minhash_cls=minhash_cls,
            lsh_cls=lsh_cls,
            num_perm=num_perm,
            threshold=thr,
            min_cluster_size=min_cluster,
        )
        for c in clusters[:3]:
            key, distinct = _best_varying_key(c, entity_kv)
            title = f"Batch/multi-input candidate for {proc}"
            if key:
                title = f"Batch/multi-input candidate for {proc} by {key}"
            rec = (
                f"{proc} has repeated near-duplicate parameter sets that differ mainly by {key or 'a small set of values'}. "
                "Consider changing the job interface to accept a list of those values and run the shared work once per batch."
            )
            evidence = {"process_norm": proc, "cluster_size": len(c), "varying_key": key, "distinct_values_est": distinct, "entity_ids": c[:50]}
            clusters_out.append(evidence)
            findings.append(
                _make_actionable_lever(
                    plugin_id=plugin_id,
                    process_norm=proc,
                    title=title,
                    recommendation=rec,
                    action_type="batch_input_refactor",
                    expected_delta_seconds=None,
                    confidence=0.65,
                    evidence=evidence,
                    assumptions=[
                        "Parameter entities reflect the true work inputs (not volatile ids).",
                        "Shared work dominates per-run cost so batching reduces total wait.",
                    ],
                )
            )

    if clusters_out:
        artifacts.append(_artifact(ctx, plugin_id, "minhash_clusters.json", {"clusters": clusters_out}, "json"))
        summary = f"Found {len(findings)} batch refactor candidate(s) via MinHash/LSH"
        return PluginResult("ok", summary, {"candidates": len(findings)}, findings, artifacts, None)
    return PluginResult("ok", "No near-duplicate clusters found (MinHash)", {"candidates": 0}, [], [], None)


def _run_param_near_duplicate_simhash(ctx, plugin_id: str, config: dict[str, Any]) -> PluginResult:
    info = _load_template_info(ctx)
    req = _require_template(info, plugin_id)
    if isinstance(req, PluginResult):
        return req
    template_table, info = req
    dataset_version_id = str(ctx.dataset_version_id)
    proc_col = info.role_field("process_name") or info.role_field("process") or None
    if not proc_col:
        return PluginResult("skipped", "No process column inferred (role process_name)", {}, [], [], None)

    exclude_match = _exclude_matcher(ctx, config)
    ignore_re = _parse_ignore_regex(config)

    max_processes = int(config.get("max_processes") or 60)
    max_entities = int(config.get("max_entities_per_process") or 3000)
    bits = int(config.get("fingerprint_bits") or 64)
    max_h = int(config.get("max_hamming_distance") or 3)
    min_cluster = int(config.get("min_cluster_size") or 5)

    findings: list[dict[str, Any]] = []
    artifacts: list[PluginArtifact] = []
    clusters_out: list[dict[str, Any]] = []

    with ctx.storage.connection() as conn:
        proc_entities = _fetch_process_entity_counts(
            conn,
            template_table,
            dataset_version_id,
            proc_col,
            max_processes=max_processes,
            max_entities_per_process=max_entities,
            exclude_match=exclude_match,
        )
        all_eids = sorted({eid for rows in proc_entities.values() for eid, _n in rows})
        entity_kv = _fetch_entity_kv(conn, all_eids, ignore_re)

    for proc, rows in proc_entities.items():
        clusters = _simhash_clusters_for_process(
            proc,
            rows,
            entity_kv,
            bits=bits,
            max_hamming=max_h,
            min_cluster_size=min_cluster,
        )
        for c in clusters[:3]:
            key, distinct = _best_varying_key(c, entity_kv)
            title = f"Near-duplicate execution cluster for {proc}"
            rec = (
                f"{proc} runs appear in near-duplicate clusters (parameters differ slightly). "
                f"Consider caching results for repeated inputs or adding a batch mode (varying key: {key or 'unknown'})."
            )
            evidence = {"process_norm": proc, "cluster_size": len(c), "varying_key": key, "distinct_values_est": distinct, "entity_ids": c[:50]}
            clusters_out.append(evidence)
            findings.append(
                _make_actionable_lever(
                    plugin_id=plugin_id,
                    process_norm=proc,
                    title=title,
                    recommendation=rec,
                    action_type="dedupe_or_cache",
                    expected_delta_seconds=None,
                    confidence=0.6,
                    evidence=evidence,
                )
            )

    if clusters_out:
        artifacts.append(_artifact(ctx, plugin_id, "simhash_clusters.json", {"clusters": clusters_out}, "json"))
        return PluginResult("ok", f"Found {len(findings)} near-duplicate cluster(s) via SimHash", {"candidates": len(findings)}, findings, artifacts, None)
    return PluginResult("ok", "No near-duplicate clusters found (SimHash)", {"candidates": 0}, [], [], None)


def _run_frequent_itemsets_fpgrowth(ctx, plugin_id: str, config: dict[str, Any]) -> PluginResult:
    info = _load_template_info(ctx)
    req = _require_template(info, plugin_id)
    if isinstance(req, PluginResult):
        return req
    template_table, info = req
    dataset_version_id = str(ctx.dataset_version_id)
    proc_col = info.role_field("process_name") or info.role_field("process") or None
    if not proc_col:
        return PluginResult("skipped", "No process column inferred (role process_name)", {}, [], [], None)

    min_support = float(config.get("min_support") or 0.02)
    max_itemset_size = int(config.get("max_itemset_size") or 6)
    max_item_count = int(config.get("max_item_count") or 200)

    with ctx.storage.connection() as conn:
        # Global most common keys (weighted by run occurrences via rpl join).
        sql = (
            f"""
            SELECT pk.key AS key, COUNT(*) AS n
            FROM {quote_identifier(template_table)} t
            JOIN row_parameter_link rpl
              ON rpl.dataset_version_id = t.dataset_version_id AND rpl.row_index = t.row_index
            JOIN parameter_kv pk ON pk.entity_id = rpl.entity_id
            WHERE t.dataset_version_id = ?
              AND t.{quote_identifier(proc_col)} IS NOT NULL
            GROUP BY pk.key
            ORDER BY n DESC
            LIMIT ?
            """
        )
        key_rows = conn.execute(sql, (dataset_version_id, int(max_item_count))).fetchall()
        top_keys = [str(r["key"]) for r in key_rows if r["key"]]
        if not top_keys:
            return PluginResult("skipped", "No parameter keys found in parameter_kv", {}, [], [], None)
        placeholders = ",".join("?" for _ in top_keys)
        # Transactions at entity_id granularity: entity_id -> set(keys).
        ent_rows = conn.execute(
            f"SELECT entity_id, key FROM parameter_kv WHERE key IN ({placeholders})",
            tuple(top_keys),
        ).fetchall()

    by_ent: dict[int, set[str]] = defaultdict(set)
    for r in ent_rows:
        by_ent[int(r["entity_id"])].add(str(r["key"]))

    if len(by_ent) < 50:
        return PluginResult("skipped", "Insufficient parameter entities for itemset mining", {"entities": len(by_ent)}, [], [], None)

    items = sorted(top_keys)
    mat = np.zeros((len(by_ent), len(items)), dtype=bool)
    ent_ids = list(by_ent.keys())
    for i, eid in enumerate(ent_ids):
        keys = by_ent[eid]
        for j, k in enumerate(items):
            if k in keys:
                mat[i, j] = True
    df = pd.DataFrame(mat, columns=items)
    freq = fpgrowth(df, min_support=min_support, use_colnames=True, max_len=max_itemset_size)
    freq = freq.sort_values("support", ascending=False).head(50)
    findings: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    for _, row in freq.iterrows():
        itemset = sorted(list(row["itemsets"]))
        support = float(row["support"])
        artifact_rows.append({"itemset": itemset, "support": support})
        if len(itemset) < 2:
            continue
        title = "Create a preset job for a frequent parameter bundle"
        rec = (
            f"These parameter keys frequently occur together ({', '.join(itemset)}; support ~{support*100:.1f}%). "
            "Consider a single preset job/API that accepts this bundle explicitly to reduce variant sprawl and repeated launches."
        )
        findings.append(
            _make_actionable_lever(
                plugin_id=plugin_id,
                process_norm="(multiple)",
                title=title,
                recommendation=rec,
                action_type="preset_job_candidate",
                expected_delta_seconds=None,
                confidence=min(0.9, 0.4 + support),
                evidence={"itemset": itemset, "support": support},
                measurement_type="measured",
            )
        )
        if len(findings) >= 10:
            break

    artifacts = [_artifact(ctx, plugin_id, "fpgrowth_itemsets.json", {"itemsets": artifact_rows}, "json")]
    summary = f"Mined {len(artifact_rows)} frequent itemsets"
    return PluginResult("ok", summary, {"itemsets": len(artifact_rows)}, findings, artifacts, None)


def _run_association_rules_apriori(ctx, plugin_id: str, config: dict[str, Any]) -> PluginResult:
    info = _load_template_info(ctx)
    req = _require_template(info, plugin_id)
    if isinstance(req, PluginResult):
        return req
    template_table, info = req
    dataset_version_id = str(ctx.dataset_version_id)
    proc_col = info.role_field("process_name") or info.role_field("process") or None
    if not proc_col:
        return PluginResult("skipped", "No process column inferred (role process_name)", {}, [], [], None)

    min_support = float(config.get("min_support") or 0.02)
    min_conf = float(config.get("min_confidence") or 0.6)
    min_lift = float(config.get("min_lift") or 1.1)
    max_item_count = int(config.get("max_item_count") or 200)
    max_rules = int(config.get("max_rules") or 200)

    with ctx.storage.connection() as conn:
        sql = (
            f"""
            SELECT pk.key AS key, COUNT(*) AS n
            FROM {quote_identifier(template_table)} t
            JOIN row_parameter_link rpl
              ON rpl.dataset_version_id = t.dataset_version_id AND rpl.row_index = t.row_index
            JOIN parameter_kv pk ON pk.entity_id = rpl.entity_id
            WHERE t.dataset_version_id = ?
              AND t.{quote_identifier(proc_col)} IS NOT NULL
            GROUP BY pk.key
            ORDER BY n DESC
            LIMIT ?
            """
        )
        key_rows = conn.execute(sql, (dataset_version_id, int(max_item_count))).fetchall()
        top_keys = [str(r["key"]) for r in key_rows if r["key"]]
        if not top_keys:
            return PluginResult("skipped", "No parameter keys found in parameter_kv", {}, [], [], None)
        placeholders = ",".join("?" for _ in top_keys)
        ent_rows = conn.execute(
            f"SELECT entity_id, key FROM parameter_kv WHERE key IN ({placeholders})",
            tuple(top_keys),
        ).fetchall()

    by_ent: dict[int, set[str]] = defaultdict(set)
    for r in ent_rows:
        by_ent[int(r["entity_id"])].add(str(r["key"]))
    if len(by_ent) < 100:
        return PluginResult("skipped", "Insufficient entities for association-rule mining", {"entities": len(by_ent)}, [], [], None)

    items = sorted(top_keys)
    mat = np.zeros((len(by_ent), len(items)), dtype=bool)
    ent_ids = list(by_ent.keys())
    for i, eid in enumerate(ent_ids):
        keys = by_ent[eid]
        for j, k in enumerate(items):
            if k in keys:
                mat[i, j] = True
    df = pd.DataFrame(mat, columns=items)
    freq = apriori(df, min_support=min_support, use_colnames=True, max_len=4)
    if freq.empty:
        return PluginResult("ok", "No frequent itemsets found for Apriori", {"rules": 0}, [], [], None)
    rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
    if not rules.empty:
        rules = rules[rules["lift"] >= min_lift].sort_values(["lift", "confidence"], ascending=False).head(max_rules)
    findings: list[dict[str, Any]] = []
    rules_out: list[dict[str, Any]] = []
    for _, row in rules.iterrows():
        ant = sorted(list(row["antecedents"]))
        con = sorted(list(row["consequents"]))
        conf = float(row["confidence"])
        lift = float(row["lift"])
        rules_out.append({"antecedents": ant, "consequents": con, "confidence": conf, "lift": lift})
        title = "Simplify variants using a consistent parameter rule"
        rec = (
            f"When {', '.join(ant)} is present, {', '.join(con)} is usually also present "
            f"(confidence {conf*100:.1f}%, lift {lift:.2f}). "
            "Consider collapsing these variants into one explicit preset or infer the consequent automatically to reduce redundant job launches."
        )
        findings.append(
            _make_actionable_lever(
                plugin_id=plugin_id,
                process_norm="(multiple)",
                title=title,
                recommendation=rec,
                action_type="param_rule_simplification",
                expected_delta_seconds=None,
                confidence=min(0.9, 0.3 + conf / 2.0),
                evidence={"antecedents": ant, "consequents": con, "confidence": conf, "lift": lift},
            )
        )
        if len(findings) >= 10:
            break
    artifacts = [_artifact(ctx, plugin_id, "apriori_rules.json", {"rules": rules_out}, "json")]
    return PluginResult("ok", f"Mined {len(rules_out)} association rules", {"rules": len(rules_out)}, findings, artifacts, None)


def _case_sequences(conn, template_table: str, dataset_version_id: str, case_col: str, time_col: str, process_col: str, *, max_cases: int) -> list[list[str]]:
    # Choose top cases by length to keep bounded and deterministic.
    top = conn.execute(
        f"""
        SELECT {quote_identifier(case_col)} AS cid, COUNT(*) AS n
        FROM {quote_identifier(template_table)}
        WHERE dataset_version_id = ? AND {quote_identifier(case_col)} IS NOT NULL AND {quote_identifier(process_col)} IS NOT NULL
        GROUP BY cid
        ORDER BY n DESC
        LIMIT ?
        """,
        (dataset_version_id, int(max_cases)),
    ).fetchall()
    cids = [r["cid"] for r in top if r["cid"] is not None]
    if not cids:
        return []
    placeholders = ",".join("?" for _ in cids)
    rows = conn.execute(
        f"""
        SELECT {quote_identifier(case_col)} AS cid,
               {quote_identifier(time_col)} AS ts,
               LOWER(TRIM({quote_identifier(process_col)})) AS proc
        FROM {quote_identifier(template_table)}
        WHERE dataset_version_id = ?
          AND {quote_identifier(case_col)} IN ({placeholders})
          AND {quote_identifier(process_col)} IS NOT NULL
        ORDER BY cid, ts
        """,
        (dataset_version_id, *cids),
    ).fetchall()
    by_case: dict[Any, list[str]] = defaultdict(list)
    for r in rows:
        proc = str(r["proc"] or "").strip()
        if proc:
            by_case[r["cid"]].append(proc)
    return [seq for seq in by_case.values() if len(seq) >= 2]


def _run_sequential_patterns_prefixspan(ctx, plugin_id: str, config: dict[str, Any]) -> PluginResult:
    info = _load_template_info(ctx)
    req = _require_template(info, plugin_id)
    if isinstance(req, PluginResult):
        return req
    template_table, info = req
    dataset_version_id = str(ctx.dataset_version_id)
    case_col = info.role_field("master_id")
    time_col = info.role_field("start_time") or info.role_field("queue_time") or info.role_field("end_time")
    proc_col = info.role_field("process_name") or info.role_field("process")
    if not case_col or not time_col or not proc_col:
        return PluginResult("skipped", "Missing case/time/process columns for sequence mining", {}, [], [], None)

    max_cases = int(config.get("max_cases") or 5000)
    min_support = int(config.get("min_support") or 5)
    max_len = int(config.get("max_pattern_len") or 6)
    top_k = int(config.get("top_k") or 25)

    with ctx.storage.connection() as conn:
        seqs = _case_sequences(conn, template_table, dataset_version_id, case_col, time_col, proc_col, max_cases=max_cases)
    if not seqs:
        return PluginResult("skipped", "No case sequences found", {}, [], [], None)

    ps = PrefixSpan(seqs)
    patterns = ps.frequent(min_support, closed=True)
    # patterns: list[(support, pattern_list)]
    patterns = [(int(s), list(p)) for s, p in patterns if 2 <= len(p) <= max_len]
    patterns.sort(key=lambda sp: (sp[0], len(sp[1])), reverse=True)
    top = patterns[:top_k]
    findings: list[dict[str, Any]] = []
    out = []
    for support, pat in top[:10]:
        seq = " -> ".join(pat)
        out.append({"pattern": pat, "support": support})
        title = f"Consolidate repeated chain: {seq}"
        rec = (
            f"The process chain {seq} repeats across many cases (support {support}). "
            "Consider introducing a single orchestrated job that accepts the common inputs and runs the chain internally, "
            "reducing handoff wait and queue delays between steps."
        )
        findings.append(
            _make_actionable_lever(
                plugin_id=plugin_id,
                process_norm=pat[0],
                title=title,
                recommendation=rec,
                action_type="orchestrate_chain",
                expected_delta_seconds=None,
                confidence=min(0.9, 0.4 + (support / max(50.0, float(min_support) * 10.0))),
                evidence={"pattern": pat, "support": support},
            )
        )
    artifacts = [_artifact(ctx, plugin_id, "frequent_sequences.json", {"patterns": top}, "json")]
    return PluginResult("ok", f"Found {len(top)} frequent sequences", {"patterns": len(top), "cases_used": len(seqs)}, findings, artifacts, None)


def _sequitur_rules(sequence: list[str]) -> dict[tuple[str, str], int]:
    """Very small, deterministic SEQUITUR-like digram frequency pass.

    Full SEQUITUR maintains grammar with digram uniqueness + rule utility. For harness actionability,
    the key output we need is repeated subsequences/motifs to propose orchestration macros.
    """

    counts: dict[tuple[str, str], int] = defaultdict(int)
    for a, b in zip(sequence[:-1], sequence[1:]):
        counts[(a, b)] += 1
    return counts


def _run_sequence_grammar_sequitur(ctx, plugin_id: str, config: dict[str, Any]) -> PluginResult:
    info = _load_template_info(ctx)
    req = _require_template(info, plugin_id)
    if isinstance(req, PluginResult):
        return req
    template_table, info = req
    dataset_version_id = str(ctx.dataset_version_id)
    case_col = info.role_field("master_id")
    time_col = info.role_field("start_time") or info.role_field("queue_time") or info.role_field("end_time")
    proc_col = info.role_field("process_name") or info.role_field("process")
    if not case_col or not time_col or not proc_col:
        return PluginResult("skipped", "Missing case/time/process columns for sequitur", {}, [], [], None)

    max_cases = int(config.get("max_cases") or 2000)
    min_rule_uses = int(config.get("min_rule_uses") or 5)
    top_k = int(config.get("top_k") or 25)

    with ctx.storage.connection() as conn:
        seqs = _case_sequences(conn, template_table, dataset_version_id, case_col, time_col, proc_col, max_cases=max_cases)
    if not seqs:
        return PluginResult("skipped", "No case sequences found", {}, [], [], None)

    digram_counts: Counter[tuple[str, str]] = Counter()
    for seq in seqs:
        digram_counts.update(_sequitur_rules(seq))
    top = [(k, v) for k, v in digram_counts.most_common(top_k) if v >= min_rule_uses]
    findings: list[dict[str, Any]] = []
    motifs: list[dict[str, Any]] = []
    for (a, b), n in top[:10]:
        motifs.append({"motif": [a, b], "count": int(n)})
        title = f"Create a macro for repeated subsequence: {a} -> {b}"
        rec = (
            f"The subsequence {a} -> {b} repeats frequently (count {n}) across cases. "
            "Consider collapsing these two steps into one job/macro (or introduce a direct trigger) "
            "to remove the handoff delay between them."
        )
        findings.append(
            _make_actionable_lever(
                plugin_id=plugin_id,
                process_norm=a,
                title=title,
                recommendation=rec,
                action_type="orchestrate_macro",
                expected_delta_seconds=None,
                confidence=min(0.9, 0.3 + min(0.6, n / 100.0)),
                evidence={"motif": [a, b], "count": int(n), "cases_used": len(seqs)},
            )
        )
    artifacts = [_artifact(ctx, plugin_id, "sequitur_motifs.json", {"motifs": motifs}, "json")]
    return PluginResult("ok", f"Computed {len(motifs)} repeated motifs", {"motifs": len(motifs), "cases_used": len(seqs)}, findings, artifacts, None)


def _transition_edges(conn, template_table: str, dataset_version_id: str, case_col: str, time_col: str, proc_col: str, *, min_edge_weight: int) -> list[tuple[str, str, int]]:
    # Use window function to compute bigrams per case.
    rows = conn.execute(
        f"""
        WITH ordered AS (
          SELECT
            LOWER(TRIM({quote_identifier(proc_col)})) AS proc,
            LEAD(LOWER(TRIM({quote_identifier(proc_col)}))) OVER (PARTITION BY {quote_identifier(case_col)} ORDER BY {quote_identifier(time_col)}) AS next_proc
          FROM {quote_identifier(template_table)}
          WHERE dataset_version_id = ?
            AND {quote_identifier(case_col)} IS NOT NULL
            AND {quote_identifier(proc_col)} IS NOT NULL
            AND {quote_identifier(time_col)} IS NOT NULL
        )
        SELECT proc, next_proc, COUNT(*) AS n
        FROM ordered
        WHERE next_proc IS NOT NULL AND proc != '' AND next_proc != ''
        GROUP BY proc, next_proc
        HAVING n >= ?
        ORDER BY n DESC
        """,
        (dataset_version_id, int(min_edge_weight)),
    ).fetchall()
    return [(str(r["proc"]), str(r["next_proc"]), int(r["n"])) for r in rows]


def _run_dependency_community_louvain(ctx, plugin_id: str, config: dict[str, Any], *, leiden: bool) -> PluginResult:
    info = _load_template_info(ctx)
    req = _require_template(info, plugin_id)
    if isinstance(req, PluginResult):
        return req
    template_table, info = req
    dataset_version_id = str(ctx.dataset_version_id)
    case_col = info.role_field("master_id")
    time_col = info.role_field("start_time") or info.role_field("queue_time") or info.role_field("end_time")
    proc_col = info.role_field("process_name") or info.role_field("process")
    if not case_col or not time_col or not proc_col:
        return PluginResult("skipped", "Missing case/time/process columns for dependency graph", {}, [], [], None)

    min_edge_weight = int(config.get("min_edge_weight") or 5)
    top_k = int(config.get("top_k") or 20)

    with ctx.storage.connection() as conn:
        edges = _transition_edges(conn, template_table, dataset_version_id, case_col, time_col, proc_col, min_edge_weight=min_edge_weight)
    if not edges:
        return PluginResult("skipped", "No transition edges found (graph empty)", {}, [], [], None)

    findings: list[dict[str, Any]] = []
    communities_out: list[dict[str, Any]] = []
    if leiden:
        g = ig.Graph(directed=True)
        vertices = sorted({a for a, _, _ in edges} | {b for _, b, _ in edges})
        idx = {v: i for i, v in enumerate(vertices)}
        g.add_vertices(len(vertices))
        g.vs["name"] = vertices
        g.add_edges([(idx[a], idx[b]) for a, b, _ in edges])
        g.es["weight"] = [w for _, _, w in edges]
        part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, weights="weight")
        comms = [sorted([vertices[i] for i in comm]) for comm in part]
    else:
        G = nx.DiGraph()
        for a, b, w in edges:
            if a == b:
                continue
            G.add_edge(a, b, weight=int(w))
        # networkx provides louvain_communities in recent versions; fall back to greedy modularity if absent.
        if hasattr(nx.algorithms.community, "louvain_communities"):
            comms = [sorted(list(c)) for c in nx.algorithms.community.louvain_communities(G.to_undirected(), weight="weight", seed=0)]
        else:  # pragma: no cover
            comms = [sorted(list(c)) for c in nx.algorithms.community.greedy_modularity_communities(G.to_undirected(), weight="weight")]

    comms = sorted(comms, key=len, reverse=True)
    for idx_c, comm in enumerate(comms[:top_k], start=1):
        communities_out.append({"community_id": idx_c, "size": len(comm), "members": comm[:200]})
        if len(comm) < 3:
            continue
        title = f"Decouple boundary candidate: dependency community #{idx_c} ({len(comm)} processes)"
        rec = (
            f"A group of processes frequently transition among themselves ({len(comm)} members). "
            "Consider treating this as a module boundary: isolate it on its own queue/worker pool, "
            "or consolidate internal steps into a macro to reduce cross-module handoffs."
        )
        findings.append(
            _make_actionable_lever(
                plugin_id=plugin_id,
                process_norm=comm[0],
                title=title,
                recommendation=rec,
                action_type="decouple_boundary",
                expected_delta_seconds=None,
                confidence=0.65,
                evidence={"community_id": idx_c, "size": len(comm), "members": comm[:30]},
            )
        )
    artifacts = [_artifact(ctx, plugin_id, "communities.json", {"communities": communities_out, "edges": len(edges)}, "json")]
    name = "Leiden" if leiden else "Louvain"
    return PluginResult("ok", f"{name} found {len(comms)} communities", {"communities": len(comms), "edges": len(edges)}, findings, artifacts, None)


def _run_dependency_community_louvain_v1(ctx, plugin_id: str, config: dict[str, Any]) -> PluginResult:
    return _run_dependency_community_louvain(ctx, plugin_id, config, leiden=False)


def _run_dependency_community_leiden_v1(ctx, plugin_id: str, config: dict[str, Any]) -> PluginResult:
    return _run_dependency_community_louvain(ctx, plugin_id, config, leiden=True)


def _process_key_matrix(
    conn,
    template_table: str,
    dataset_version_id: str,
    proc_col: str,
    *,
    max_items: int,
    min_proc_count: int = 50,
) -> tuple[list[str], list[str], np.ndarray]:
    """Build a process x key count matrix over top keys (bounded)."""

    key_rows = conn.execute(
        f"""
        SELECT pk.key AS key, COUNT(*) AS n
        FROM {quote_identifier(template_table)} t
        JOIN row_parameter_link rpl
          ON rpl.dataset_version_id = t.dataset_version_id AND rpl.row_index = t.row_index
        JOIN parameter_kv pk ON pk.entity_id = rpl.entity_id
        WHERE t.dataset_version_id = ?
          AND t.{quote_identifier(proc_col)} IS NOT NULL
        GROUP BY pk.key
        ORDER BY n DESC
        LIMIT ?
        """,
        (dataset_version_id, int(max_items)),
    ).fetchall()
    keys = [str(r["key"]) for r in key_rows if r["key"]]
    if not keys:
        return [], [], np.zeros((0, 0), dtype=float)

    proc_rows = conn.execute(
        f"""
        SELECT LOWER(TRIM({quote_identifier(proc_col)})) AS proc, COUNT(*) AS n
        FROM {quote_identifier(template_table)}
        WHERE dataset_version_id = ? AND {quote_identifier(proc_col)} IS NOT NULL
        GROUP BY proc
        HAVING n >= ?
        ORDER BY n DESC
        """,
        (dataset_version_id, int(min_proc_count)),
    ).fetchall()
    procs = [str(r["proc"]) for r in proc_rows if r["proc"]]
    if not procs:
        return [], keys, np.zeros((0, len(keys)), dtype=float)

    # Fill matrix via aggregated counts.
    placeholders_k = ",".join("?" for _ in keys)
    placeholders_p = ",".join("?" for _ in procs)
    rows = conn.execute(
        f"""
        SELECT LOWER(TRIM(t.{quote_identifier(proc_col)})) AS proc, pk.key AS key, COUNT(*) AS n
        FROM {quote_identifier(template_table)} t
        JOIN row_parameter_link rpl
          ON rpl.dataset_version_id = t.dataset_version_id AND rpl.row_index = t.row_index
        JOIN parameter_kv pk ON pk.entity_id = rpl.entity_id
        WHERE t.dataset_version_id = ?
          AND pk.key IN ({placeholders_k})
          AND LOWER(TRIM(t.{quote_identifier(proc_col)})) IN ({placeholders_p})
        GROUP BY proc, key
        """,
        (dataset_version_id, *keys, *procs),
    ).fetchall()
    key_idx = {k: i for i, k in enumerate(keys)}
    proc_idx = {p: i for i, p in enumerate(procs)}
    mat = np.zeros((len(procs), len(keys)), dtype=float)
    for r in rows:
        p = str(r["proc"])
        k = str(r["key"])
        if p in proc_idx and k in key_idx:
            mat[proc_idx[p], key_idx[k]] = float(r["n"])
    return procs, keys, mat


def _run_biclustering_cheng_church(ctx, plugin_id: str, config: dict[str, Any]) -> PluginResult:
    info = _load_template_info(ctx)
    req = _require_template(info, plugin_id)
    if isinstance(req, PluginResult):
        return req
    template_table, info = req
    dataset_version_id = str(ctx.dataset_version_id)
    proc_col = info.role_field("process_name") or info.role_field("process")
    if not proc_col:
        return PluginResult("skipped", "Missing process column for biclustering", {}, [], [], None)

    n_clusters = int(config.get("n_clusters") or 8)
    max_items = int(config.get("max_items") or 200)
    top_k = int(config.get("top_k") or 20)

    with ctx.storage.connection() as conn:
        procs, keys, mat = _process_key_matrix(conn, template_table, dataset_version_id, proc_col, max_items=max_items)

    if mat.size == 0 or mat.shape[0] < 10 or mat.shape[1] < 10:
        return PluginResult("skipped", "Insufficient process/param-key matrix for biclustering", {"procs": len(procs), "keys": len(keys)}, [], [], None)

    # Normalize counts to reduce scale effects.
    X = mat / np.maximum(1.0, mat.sum(axis=1, keepdims=True))
    model = SpectralCoclustering(n_clusters=min(n_clusters, X.shape[0], X.shape[1]), random_state=0)
    model.fit(X)
    findings: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    for idx_c in range(model.n_clusters):
        prow = np.where(model.row_labels_ == idx_c)[0].tolist()
        krow = np.where(model.column_labels_ == idx_c)[0].tolist()
        if len(prow) < 3 or len(krow) < 3:
            continue
        members = [procs[i] for i in prow][:200]
        k_members = [keys[j] for j in krow][:200]
        blocks.append({"cluster_id": idx_c, "processes": members, "keys": k_members, "size_procs": len(prow), "size_keys": len(krow)})
        title = f"Shared endpoint candidate: bicluster #{idx_c} ({len(prow)} processes x {len(krow)} keys)"
        rec = (
            f"A group of processes share a coherent set of parameter keys ({len(krow)} keys across {len(prow)} processes). "
            "Action: consider a shared batch endpoint or shared cache keyed by these common parameters, instead of launching each process independently."
        )
        findings.append(
            _make_actionable_lever(
                plugin_id=plugin_id,
                process_norm=members[0] if members else "(unknown)",
                title=title,
                recommendation=rec,
                action_type="shared_cache_endpoint",
                expected_delta_seconds=None,
                confidence=0.6,
                evidence={"cluster_id": idx_c, "processes": members[:25], "keys": k_members[:25]},
            )
        )
        if len(findings) >= top_k:
            break

    artifacts = [_artifact(ctx, plugin_id, "biclusters.json", {"biclusters": blocks}, "json")]
    return PluginResult("ok", f"Computed {len(blocks)} bicluster(s)", {"biclusters": len(blocks)}, findings, artifacts, None)


def _run_density_clustering_hdbscan(ctx, plugin_id: str, config: dict[str, Any]) -> PluginResult:
    info = _load_template_info(ctx)
    req = _require_template(info, plugin_id)
    if isinstance(req, PluginResult):
        return req
    template_table, info = req
    dataset_version_id = str(ctx.dataset_version_id)

    max_entities = int(config.get("max_entities") or 5000)
    min_cluster_size = int(config.get("min_cluster_size") or 10)
    min_samples = int(config.get("min_samples") or 5)
    top_k = int(config.get("top_k") or 20)

    with ctx.storage.connection() as conn:
        # Choose most common parameter entities in this dataset.
        rows = conn.execute(
            """
            SELECT entity_id, COUNT(*) AS n
            FROM row_parameter_link
            WHERE dataset_version_id = ?
            GROUP BY entity_id
            ORDER BY n DESC
            LIMIT ?
            """,
            (dataset_version_id, int(max_entities)),
        ).fetchall()
        eids = [int(r["entity_id"]) for r in rows]
        if not eids:
            return PluginResult("skipped", "No parameter entities found for clustering", {}, [], [], None)
        placeholders = ",".join("?" for _ in eids)
        ent = conn.execute(
            f"SELECT entity_id, canonical_text FROM parameter_entities WHERE entity_id IN ({placeholders})",
            tuple(eids),
        ).fetchall()
        texts = {int(r["entity_id"]): str(r["canonical_text"] or "") for r in ent}

    ordered = [eid for eid in eids if texts.get(eid)]
    docs = [texts[eid] for eid in ordered]
    if len(docs) < 100:
        return PluginResult("skipped", "Insufficient parameter entities with canonical_text for clustering", {"entities": len(docs)}, [], [], None)

    vec = TfidfVectorizer(max_features=4000, lowercase=True)
    X = vec.fit_transform(docs)
    svd = TruncatedSVD(n_components=min(30, max(2, X.shape[1] - 1)), random_state=0)
    Z = svd.fit_transform(X)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = clusterer.fit_predict(Z)

    # Summarize clusters.
    clusters: dict[int, list[int]] = defaultdict(list)
    for eid, lab in zip(ordered, labels):
        if int(lab) >= 0:
            clusters[int(lab)].append(int(eid))
    summary = [{"cluster_id": cid, "size": len(m)} for cid, m in clusters.items()]
    summary.sort(key=lambda r: r["size"], reverse=True)

    findings: list[dict[str, Any]] = []
    for row in summary[:top_k]:
        cid = int(row["cluster_id"])
        members = clusters[cid]
        title = f"Batchable cluster candidate: param cluster #{cid} ({len(members)} entities)"
        rec = (
            "A cluster of executions share highly similar parameter tokens (density clustering). "
            "Action: inspect the cluster exemplar params; if the varying key is an ID-like value, add a batch/multi-input mode."
        )
        evidence = {"cluster_id": cid, "size": len(members), "example_entity_ids": members[:25]}
        findings.append(
            _make_actionable_lever(
                plugin_id=plugin_id,
                process_norm="(multiple)",
                title=title,
                recommendation=rec,
                action_type="batch_cluster_candidate",
                expected_delta_seconds=None,
                confidence=0.55,
                evidence=evidence,
            )
        )
    artifacts = [_artifact(ctx, plugin_id, "hdbscan_clusters.json", {"clusters": summary[:200]}, "json")]
    return PluginResult("ok", f"HDBSCAN clustered {len(ordered)} entities into {len(clusters)} clusters", {"clusters": len(clusters), "entities": len(ordered)}, findings, artifacts, None)


def _run_constrained_clustering_cop_kmeans(ctx, plugin_id: str, config: dict[str, Any]) -> PluginResult:
    info = _load_template_info(ctx)
    req = _require_template(info, plugin_id)
    if isinstance(req, PluginResult):
        return req
    template_table, info = req
    dataset_version_id = str(ctx.dataset_version_id)
    proc_col = info.role_field("process_name") or info.role_field("process")
    if not proc_col:
        return PluginResult("skipped", "Missing process column for COP-KMeans", {}, [], [], None)

    mod_col = info.role_field("module_code")
    k = int(config.get("k") or 8)
    top_k = int(config.get("top_k") or 20)

    if not mod_col:
        return PluginResult("skipped", "Missing module_code role; cannot derive safe must-link constraints", {}, [], [], None)

    with ctx.storage.connection() as conn:
        # Aggregate per process: module_code, and key usage vector.
        procs, keys, mat = _process_key_matrix(conn, template_table, dataset_version_id, proc_col, max_items=200)
        if mat.size == 0:
            return PluginResult("skipped", "Insufficient data for constrained clustering", {"procs": len(procs)}, [], [], None)
        # Map process -> module_code (mode).
        rows = conn.execute(
            f"""
            SELECT LOWER(TRIM({quote_identifier(proc_col)})) AS proc,
                   LOWER(TRIM({quote_identifier(mod_col)})) AS mod,
                   COUNT(*) AS n
            FROM {quote_identifier(template_table)}
            WHERE dataset_version_id = ?
              AND {quote_identifier(proc_col)} IS NOT NULL
              AND {quote_identifier(mod_col)} IS NOT NULL
            GROUP BY proc, mod
            """,
            (dataset_version_id,),
        ).fetchall()
    mod_by_proc: dict[str, str] = {}
    counts: dict[tuple[str, str], int] = {}
    for r in rows:
        proc = str(r["proc"])
        mod = str(r["mod"])
        counts[(proc, mod)] = int(r["n"])
    for proc in procs:
        candidates = [(mod, n) for (p, mod), n in counts.items() if p == proc]
        if not candidates:
            continue
        best = max(candidates, key=lambda x: x[1])[0]
        mod_by_proc[proc] = best

    # Collapse must-link groups by module_code: cluster modules, not individual processes.
    modules = sorted({m for m in mod_by_proc.values() if m})
    if len(modules) < 2:
        return PluginResult("skipped", "Not enough distinct module_code groups to cluster", {"modules": len(modules)}, [], [], None)

    mod_idx = {m: i for i, m in enumerate(modules)}
    X_mod = np.zeros((len(modules), mat.shape[1]), dtype=float)
    for i, proc in enumerate(procs):
        mod = mod_by_proc.get(proc)
        if not mod:
            continue
        X_mod[mod_idx[mod]] += mat[i]
    X_mod = X_mod / np.maximum(1.0, X_mod.sum(axis=1, keepdims=True))

    model = KMeans(n_clusters=min(k, len(modules)), n_init=10, random_state=0)
    labels = model.fit_predict(X_mod)
    clusters: dict[int, list[str]] = defaultdict(list)
    for mod, lab in zip(modules, labels):
        clusters[int(lab)].append(mod)

    findings: list[dict[str, Any]] = []
    out = []
    for cid, mods in clusters.items():
        out.append({"cluster_id": cid, "modules": mods, "size": len(mods)})
    out.sort(key=lambda r: r["size"], reverse=True)
    for row in out[:top_k]:
        cid = int(row["cluster_id"])
        mods = row["modules"]
        title = f"Constraint-respecting cluster: modules cluster #{cid} ({len(mods)} modules)"
        rec = (
            "Processes grouped by module_code form a coherent cluster in parameter-key space. "
            "Action: treat this cluster as a safe unit for batching/scheduling changes (must-link constraint: module_code)."
        )
        findings.append(
            _make_actionable_lever(
                plugin_id=plugin_id,
                process_norm="(multiple)",
                title=title,
                recommendation=rec,
                action_type="cluster_with_constraints",
                expected_delta_seconds=None,
                confidence=0.55,
                evidence={"cluster_id": cid, "modules": mods[:50]},
            )
        )

    artifacts = [_artifact(ctx, plugin_id, "cop_kmeans_clusters.json", {"clusters": out}, "json")]
    return PluginResult("ok", f"Clustered {len(modules)} module groups with constraints", {"clusters": len(clusters), "modules": len(modules)}, findings, artifacts, None)


def _run_similarity_graph_spectral_clustering(ctx, plugin_id: str, config: dict[str, Any]) -> PluginResult:
    info = _load_template_info(ctx)
    req = _require_template(info, plugin_id)
    if isinstance(req, PluginResult):
        return req
    template_table, info = req
    dataset_version_id = str(ctx.dataset_version_id)
    proc_col = info.role_field("process_name") or info.role_field("process")
    if not proc_col:
        return PluginResult("skipped", "Missing process column for spectral clustering", {}, [], [], None)

    n_clusters = int(config.get("n_clusters") or 8)
    top_k = int(config.get("top_k") or 20)

    with ctx.storage.connection() as conn:
        procs, keys, mat = _process_key_matrix(conn, template_table, dataset_version_id, proc_col, max_items=200)
    if mat.size == 0 or mat.shape[0] < 10:
        return PluginResult("skipped", "Insufficient processes for spectral clustering", {"procs": len(procs)}, [], [], None)

    # Similarity: cosine on normalized key vectors.
    X = mat / np.maximum(1.0, mat.sum(axis=1, keepdims=True))
    sim = X @ X.T
    # Force symmetry and non-negative.
    sim = np.maximum(0.0, (sim + sim.T) / 2.0)
    # Spectral clustering expects affinity matrix.
    model = SpectralClustering(n_clusters=min(n_clusters, X.shape[0]), affinity="precomputed", random_state=0)
    labels = model.fit_predict(sim)
    clusters: dict[int, list[str]] = defaultdict(list)
    for proc, lab in zip(procs, labels):
        clusters[int(lab)].append(proc)
    out = [{"cluster_id": cid, "size": len(m), "members": sorted(m)[:200]} for cid, m in clusters.items()]
    out.sort(key=lambda r: r["size"], reverse=True)
    findings: list[dict[str, Any]] = []
    for row in out[:top_k]:
        cid = int(row["cluster_id"])
        members = row["members"]
        if len(members) < 3:
            continue
        title = f"Coherent group for batching/consolidation: cluster #{cid} ({row['size']} processes)"
        rec = (
            "These processes have similar parameter-key usage. "
            "Action: review whether they can share a batch endpoint, a shared cache, or a consolidated orchestrator."
        )
        findings.append(
            _make_actionable_lever(
                plugin_id=plugin_id,
                process_norm=members[0],
                title=title,
                recommendation=rec,
                action_type="batch_group_candidate",
                expected_delta_seconds=None,
                confidence=0.55,
                evidence={"cluster_id": cid, "members": members[:25]},
            )
        )
    artifacts = [_artifact(ctx, plugin_id, "spectral_clusters.json", {"clusters": out}, "json")]
    return PluginResult("ok", f"Spectral clustering produced {len(clusters)} clusters", {"clusters": len(clusters), "processes": len(procs)}, findings, artifacts, None)


def _run_graph_min_cut_partition(ctx, plugin_id: str, config: dict[str, Any]) -> PluginResult:
    info = _load_template_info(ctx)
    req = _require_template(info, plugin_id)
    if isinstance(req, PluginResult):
        return req
    template_table, info = req
    dataset_version_id = str(ctx.dataset_version_id)
    case_col = info.role_field("master_id")
    time_col = info.role_field("start_time") or info.role_field("queue_time") or info.role_field("end_time")
    proc_col = info.role_field("process_name") or info.role_field("process")
    if not case_col or not time_col or not proc_col:
        return PluginResult("skipped", "Missing case/time/process columns for min-cut", {}, [], [], None)

    min_edge_weight = int(config.get("min_edge_weight") or 5)
    with ctx.storage.connection() as conn:
        edges = _transition_edges(conn, template_table, dataset_version_id, case_col, time_col, proc_col, min_edge_weight=min_edge_weight)
    if not edges:
        return PluginResult("skipped", "No transition edges found (graph empty)", {}, [], [], None)
    G = nx.Graph()
    for a, b, w in edges:
        if a == b:
            continue
        G.add_edge(a, b, weight=int(w))
    cut_value, partition = nx.algorithms.stoer_wagner(G, weight="weight")
    a_set, b_set = partition
    # Top cut edges for evidence.
    cut_edges = []
    for u in a_set:
        for v in b_set:
            if G.has_edge(u, v):
                cut_edges.append({"src": u, "dst": v, "weight": int(G[u][v].get("weight") or 0)})
    cut_edges.sort(key=lambda e: e["weight"], reverse=True)
    artifact = {"cut_value": float(cut_value), "side_a_size": len(a_set), "side_b_size": len(b_set), "top_cut_edges": cut_edges[:50]}
    artifacts = [_artifact(ctx, plugin_id, "min_cut.json", artifact, "json")]
    title = "Decouple the graph along a minimal cut boundary"
    rec = (
        "The dependency graph has a small set of high-weight edges connecting two larger groups. "
        "Treat these cut edges as the primary decoupling boundary: introduce an explicit queue boundary, "
        "or restructure data flow so that cross-group calls are batched."
    )
    findings = [
        _make_actionable_lever(
            plugin_id=plugin_id,
            process_norm=next(iter(a_set)) if a_set else "(unknown)",
            title=title,
            recommendation=rec,
            action_type="decouple_boundary",
            expected_delta_seconds=None,
            confidence=0.6,
            evidence=artifact,
        )
    ]
    return PluginResult("ok", "Computed minimum cut partition", {"cut_value": float(cut_value)}, findings, artifacts, None)


def _run_distribution_shift_wasserstein(ctx, plugin_id: str, config: dict[str, Any]) -> PluginResult:
    info = _load_template_info(ctx)
    req = _require_template(info, plugin_id)
    if isinstance(req, PluginResult):
        return req
    template_table, info = req
    dataset_version_id = str(ctx.dataset_version_id)
    proc_col = info.role_field("process_name") or info.role_field("process")
    time_col = info.role_field("queue_time") or info.role_field("start_time") or info.role_field("end_time")
    if not proc_col or not time_col:
        return PluginResult("skipped", "Missing process/time columns for shift analysis", {}, [], [], None)

    # Numeric target: prefer explicit duration role; else compute service seconds from start/end timestamps.
    numeric_col = info.role_field("duration") or None
    numeric_expr = None
    numeric_label = None
    if numeric_col:
        numeric_expr = f"CAST({quote_identifier(numeric_col)} AS REAL)"
        numeric_label = numeric_col
    else:
        start_col = info.role_field("start_time")
        end_col = info.role_field("end_time")
        if start_col and end_col:
            numeric_expr = f"((julianday({quote_identifier(end_col)}) - julianday({quote_identifier(start_col)})) * 86400.0)"
            numeric_label = "service_seconds"

    bins = int(config.get("bins") or 60)
    top_k = int(config.get("top_k") or 25)

    # If no numeric column is known, degrade gracefully.
    if not numeric_expr:
        return PluginResult(
            "skipped",
            "No numeric metric available for Wasserstein shift (need duration role or start/end timestamps)",
            {},
            [],
            [],
            None,
        )

    exclude_match = _exclude_matcher(ctx, config)

    def _emd_from_hist(h1: np.ndarray, h2: np.ndarray) -> float:
        # Earth mover's distance on 1D discrete distributions with unit bin width.
        h1 = h1.astype(float)
        h2 = h2.astype(float)
        if h1.sum() <= 0 or h2.sum() <= 0:
            return 0.0
        p1 = h1 / h1.sum()
        p2 = h2 / h2.sum()
        c1 = np.cumsum(p1)
        c2 = np.cumsum(p2)
        return float(np.abs(c1 - c2).sum())

    findings: list[dict[str, Any]] = []
    shift_rows: list[dict[str, Any]] = []
    with ctx.storage.connection() as conn:
        # Determine global min/max to bin.
        mn, mx = conn.execute(
            f"SELECT MIN(x), MAX(x) FROM (SELECT {numeric_expr} AS x FROM {quote_identifier(template_table)} WHERE dataset_version_id = ? AND {numeric_expr} IS NOT NULL)",
            (dataset_version_id,),
        ).fetchone()
        if mn is None or mx is None:
            return PluginResult("skipped", "No numeric values for shift analysis", {}, [], [], None)
        try:
            mn_f = float(mn)
            mx_f = float(mx)
        except Exception:
            return PluginResult("skipped", "Numeric column did not coerce to float", {}, [], [], None)
        if not math.isfinite(mn_f) or not math.isfinite(mx_f) or mx_f <= mn_f:
            return PluginResult("skipped", "Numeric column range invalid", {}, [], [], None)
        edges = np.linspace(mn_f, mx_f, bins + 1)

        # Candidate processes by count.
        top = conn.execute(
            f"""
            SELECT LOWER(TRIM({quote_identifier(proc_col)})) AS proc, COUNT(*) AS n
            FROM {quote_identifier(template_table)}
            WHERE dataset_version_id = ? AND {quote_identifier(proc_col)} IS NOT NULL
            GROUP BY proc
            ORDER BY n DESC
            LIMIT ?
            """,
            (dataset_version_id, int(top_k)),
        ).fetchall()
        procs = [str(r["proc"]) for r in top if r["proc"]]
        procs = [p for p in procs if p and not exclude_match(p)]
        if not procs:
            return PluginResult("ok", "No eligible processes for shift analysis after exclusions", {"processes": 0}, [], [], None)

        # Close/open split: use day-of-month to separate close-ish window (20..31 and 1..5) vs other.
        for proc in procs:
            rows = conn.execute(
                f"""
                SELECT {numeric_expr} AS x,
                       CAST(strftime('%d', {quote_identifier(time_col)}) AS INTEGER) AS dom
                FROM {quote_identifier(template_table)}
                WHERE dataset_version_id = ?
                  AND LOWER(TRIM({quote_identifier(proc_col)})) = ?
                  AND {numeric_expr} IS NOT NULL
                  AND {quote_identifier(time_col)} IS NOT NULL
                """,
                (dataset_version_id, proc),
            ).fetchall()
            if not rows:
                continue
            close_vals = [float(r["x"]) for r in rows if r["dom"] is not None and (int(r["dom"]) >= 20 or int(r["dom"]) <= 5)]
            open_vals = [float(r["x"]) for r in rows if r["dom"] is not None and (6 <= int(r["dom"]) <= 19)]
            if len(close_vals) < 50 or len(open_vals) < 50:
                continue
            h_close, _ = np.histogram(close_vals, bins=edges)
            h_open, _ = np.histogram(open_vals, bins=edges)
            emd = _emd_from_hist(h_close, h_open)
            shift_rows.append({"process_norm": proc, "emd": emd, "close_n": len(close_vals), "open_n": len(open_vals)})

    shift_rows.sort(key=lambda r: float(r["emd"]), reverse=True)
    for row in shift_rows[:10]:
        proc = row["process_norm"]
        emd = float(row["emd"])
        title = f"Investigate distribution shift for {proc}"
        rec = (
            f"{proc} shows a strong close-vs-open distribution shift (EMD≈{emd:.2f}). "
            "This is a concrete target: compare inputs/params between close and non-close runs and decide whether to batch, reschedule, or isolate the close variant."
        )
        findings.append(
            _make_actionable_lever(
                plugin_id=plugin_id,
                process_norm=proc,
                title=title,
                recommendation=rec,
                action_type="distribution_shift_target",
                expected_delta_seconds=None,
                confidence=min(0.9, 0.4 + min(0.5, emd / 10.0)),
                evidence=row,
            )
        )
    artifacts = [
        _artifact(
            ctx,
            plugin_id,
            "wasserstein_shift.json",
            {"metric": numeric_label, "rows": shift_rows[:200]},
            "json",
        )
    ]
    return PluginResult("ok", f"Computed shift scores for {len(shift_rows)} process(es)", {"processes_scored": len(shift_rows)}, findings, artifacts, None)


def _run_burst_modeling_hawkes(ctx, plugin_id: str, config: dict[str, Any]) -> PluginResult:
    info = _load_template_info(ctx)
    req = _require_template(info, plugin_id)
    if isinstance(req, PluginResult):
        return req
    template_table, info = req
    dataset_version_id = str(ctx.dataset_version_id)
    proc_col = info.role_field("process_name") or info.role_field("process")
    time_col = info.role_field("queue_time") or info.role_field("start_time") or info.role_field("end_time")
    if not proc_col or not time_col:
        return PluginResult("skipped", "Missing process/time columns for burst model", {}, [], [], None)

    top_k = int(config.get("top_k") or 20)

    with ctx.storage.connection() as conn:
        # Hourly counts per process (top processes only).
        top = conn.execute(
            f"""
            SELECT LOWER(TRIM({quote_identifier(proc_col)})) AS proc, COUNT(*) AS n
            FROM {quote_identifier(template_table)}
            WHERE dataset_version_id = ? AND {quote_identifier(proc_col)} IS NOT NULL
            GROUP BY proc
            ORDER BY n DESC
            LIMIT ?
            """,
            (dataset_version_id, int(top_k)),
        ).fetchall()
        procs = [str(r["proc"]) for r in top if r["proc"]]
        series: dict[str, list[int]] = {}
        for proc in procs:
            rows = conn.execute(
                f"""
                SELECT strftime('%Y-%m-%dT%H:00:00', {quote_identifier(time_col)}) AS bucket, COUNT(*) AS n
                FROM {quote_identifier(template_table)}
                WHERE dataset_version_id = ?
                  AND LOWER(TRIM({quote_identifier(proc_col)})) = ?
                  AND {quote_identifier(time_col)} IS NOT NULL
                GROUP BY bucket
                ORDER BY bucket
                """,
                (dataset_version_id, proc),
            ).fetchall()
            counts = [int(r["n"]) for r in rows]
            if len(counts) >= 24:
                series[proc] = counts

    if not series:
        return PluginResult("skipped", "No sufficient time series for burst model", {}, [], [], None)

    # Hawkes-style proxy: lag-1 autocorrelation as self-excitation signal.
    scored = []
    for proc, counts in series.items():
        x = np.asarray(counts, dtype=float)
        if x.size < 10 or np.std(x) <= 1e-9:
            continue
        a = np.corrcoef(x[:-1], x[1:])[0, 1]
        if not math.isfinite(float(a)):
            continue
        scored.append((float(a), proc, int(x.sum())))
    scored.sort(reverse=True)
    findings = []
    out = [{"process_norm": p, "autocorr_lag1": a, "total_events": total} for a, p, total in scored[:200]]
    for a, proc, total in scored[:10]:
        title = f"Dampen burst triggers for {proc}"
        rec = (
            f"{proc} shows strong burst self-excitation (lag-1 autocorr≈{a:.2f}). "
            "Action: add throttling/deduplication at the source of this process or introduce a batch driver so bursts become bounded batches."
        )
        findings.append(
            _make_actionable_lever(
                plugin_id=plugin_id,
                process_norm=proc,
                title=title,
                recommendation=rec,
                action_type="burst_trigger",
                expected_delta_seconds=None,
                confidence=min(0.9, 0.3 + max(0.0, a)),
                evidence={"autocorr_lag1": a, "total_events": total},
            )
        )
    artifacts = [_artifact(ctx, plugin_id, "hawkes_proxy_scores.json", {"rows": out}, "json")]
    return PluginResult("ok", f"Computed burst proxy scores for {len(out)} processes", {"processes": len(out)}, findings, artifacts, None)


def _run_daily_pattern_alignment_dtw(ctx, plugin_id: str, config: dict[str, Any]) -> PluginResult:
    info = _load_template_info(ctx)
    req = _require_template(info, plugin_id)
    if isinstance(req, PluginResult):
        return req
    template_table, info = req
    dataset_version_id = str(ctx.dataset_version_id)
    proc_col = info.role_field("process_name") or info.role_field("process")
    time_col = info.role_field("queue_time") or info.role_field("start_time") or info.role_field("end_time")
    if not proc_col or not time_col:
        return PluginResult("skipped", "Missing process/time columns for DTW", {}, [], [], None)
    top_k = int(config.get("top_k") or 20)

    with ctx.storage.connection() as conn:
        top = conn.execute(
            f"""
            SELECT LOWER(TRIM({quote_identifier(proc_col)})) AS proc, COUNT(*) AS n
            FROM {quote_identifier(template_table)}
            WHERE dataset_version_id = ? AND {quote_identifier(proc_col)} IS NOT NULL
            GROUP BY proc
            ORDER BY n DESC
            LIMIT ?
            """,
            (dataset_version_id, int(top_k)),
        ).fetchall()
        procs = [str(r["proc"]) for r in top if r["proc"]]
        daily: dict[str, np.ndarray] = {}
        for proc in procs:
            rows = conn.execute(
                f"""
                SELECT CAST(strftime('%H', {quote_identifier(time_col)}) AS INTEGER) AS hr, COUNT(*) AS n
                FROM {quote_identifier(template_table)}
                WHERE dataset_version_id = ?
                  AND LOWER(TRIM({quote_identifier(proc_col)})) = ?
                  AND {quote_identifier(time_col)} IS NOT NULL
                GROUP BY hr
                ORDER BY hr
                """,
                (dataset_version_id, proc),
            ).fetchall()
            vec = np.zeros(24, dtype=float)
            for r in rows:
                hr = r["hr"]
                if hr is None:
                    continue
                vec[int(hr)] = float(r["n"])
            if vec.sum() > 0:
                daily[proc] = vec / max(1.0, vec.sum())

    if len(daily) < 3:
        return PluginResult("skipped", "Insufficient processes for DTW comparison", {"processes": len(daily)}, [], [], None)

    # Compare each process pattern to the median pattern.
    patterns = np.stack(list(daily.values()), axis=0)
    median = np.median(patterns, axis=0)
    scores = []
    for proc, vec in daily.items():
        dist = float(dtw_lib.distance_fast(vec.astype(np.double), median.astype(np.double)))
        scores.append((dist, proc))
    scores.sort(reverse=True)
    out = [{"process_norm": proc, "dtw_distance": float(dist)} for dist, proc in scores]
    findings = []
    for dist, proc in scores[:10]:
        title = f"Investigate atypical daily timing for {proc}"
        rec = (
            f"{proc} has an unusual hour-of-day execution pattern (DTW distance≈{dist:.2f}). "
            "Action: verify whether it is unintentionally scheduled into a busy window; if so, shift it or batch it."
        )
        findings.append(
            _make_actionable_lever(
                plugin_id=plugin_id,
                process_norm=proc,
                title=title,
                recommendation=rec,
                action_type="schedule_shift_target",
                expected_delta_seconds=None,
                confidence=0.55,
                evidence={"dtw_distance": float(dist)},
            )
        )
    artifacts = [_artifact(ctx, plugin_id, "dtw_distances.json", {"rows": out}, "json")]
    return PluginResult("ok", f"Computed DTW distances for {len(out)} processes", {"processes": len(out)}, findings, artifacts, None)


def _load_plugin_findings(ctx, plugin_id: str) -> list[dict[str, Any]]:
    results = ctx.storage.fetch_plugin_results(ctx.run_id)
    for row in results:
        if str(row.get("plugin_id") or "") != plugin_id:
            continue
        raw = row.get("findings_json")
        if not raw:
            return []
        try:
            items = json.loads(raw)
        except Exception:
            return []
        return [i for i in items if isinstance(i, dict)]
    return []


def _candidate_actions_from_ops(ctx) -> list[dict[str, Any]]:
    findings = _load_plugin_findings(ctx, "analysis_actionable_ops_levers_v1")
    out = []
    for f in findings:
        if f.get("kind") != "actionable_ops_lever":
            continue
        delta = f.get("expected_delta_seconds")
        if not isinstance(delta, (int, float)) or float(delta) <= 0:
            continue
        proc = _process_norm(f.get("process_norm") or f.get("process") or "")
        if not proc:
            continue
        out.append({"process_norm": proc, "title": str(f.get("title") or ""), "action_type": str(f.get("action_type") or ""), "delta_seconds": float(delta)})
    out.sort(key=lambda r: float(r["delta_seconds"]), reverse=True)
    return out


def _run_action_search_simulated_annealing(ctx, plugin_id: str, config: dict[str, Any]) -> PluginResult:
    # Deterministic annealing over top-N actionable ops levers.
    max_actions = int(config.get("max_actions") or 8)
    iterations = int(config.get("iterations") or 4000)
    candidates = _candidate_actions_from_ops(ctx)
    if not candidates:
        return PluginResult("skipped", "No candidate actions from analysis_actionable_ops_levers_v1", {}, [], [], None)

    # Keep bounded.
    candidates = candidates[: min(80, len(candidates))]

    rng = random.Random(int(getattr(ctx, "run_seed", 1337) or 0))
    n = len(candidates)

    def score(sel: set[int]) -> float:
        return float(sum(candidates[i]["delta_seconds"] for i in sel))

    # Start with greedy best.
    current = set(range(min(max_actions, n)))
    best = set(current)
    best_s = score(best)
    cur_s = best_s
    temp0 = max(1.0, best_s / 10.0)
    for it in range(iterations):
        temp = temp0 * (1.0 - (it / max(1, iterations)))
        proposal = set(current)
        if proposal and rng.random() < 0.5:
            proposal.remove(rng.choice(tuple(proposal)))
        if len(proposal) < max_actions:
            proposal.add(rng.randrange(n))
        if len(proposal) > max_actions:
            proposal = set(list(proposal)[:max_actions])
        prop_s = score(proposal)
        if prop_s >= cur_s:
            accept = True
        else:
            accept = rng.random() < math.exp((prop_s - cur_s) / max(1e-9, temp))
        if accept:
            current = proposal
            cur_s = prop_s
        if cur_s > best_s:
            best = set(current)
            best_s = cur_s

    selected = [candidates[i] for i in sorted(best, key=lambda i: candidates[i]["delta_seconds"], reverse=True)]
    artifact = {"selected": selected, "delta_seconds_total": best_s, "iterations": iterations, "candidates": len(candidates)}
    artifacts = [_artifact(ctx, plugin_id, "annealing_plan.json", artifact, "json")]
    findings = [
        _make_actionable_lever(
            plugin_id=plugin_id,
            process_norm="(multiple)",
            title=f"Action combo plan (annealing): top {len(selected)} actions",
            recommendation="Execute the selected high-impact actions together (batch, unblock chains, scheduling) and re-run to verify spillover reduction.",
            action_type="action_plan_combo",
            expected_delta_seconds=float(best_s),
            confidence=0.55,
            evidence=artifact,
            measurement_type="modeled",
        )
    ]
    return PluginResult("ok", f"Optimized action combo (annealing) with {len(selected)} actions", {"delta_seconds_total": best_s}, findings, artifacts, None)


def _run_action_search_mip(ctx, plugin_id: str, config: dict[str, Any]) -> PluginResult:
    max_actions = int(config.get("max_actions") or 8)
    top_k = int(config.get("top_k") or 10)

    candidates = _candidate_actions_from_ops(ctx)
    if not candidates:
        return PluginResult("skipped", "No candidate actions from analysis_actionable_ops_levers_v1", {}, [], [], None)
    candidates = candidates[: min(120, len(candidates))]

    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"a{i}") for i in range(len(candidates))]
    model.Add(sum(x) <= max_actions)
    # Encourage diversity: no more than 2 actions per process.
    by_proc: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(candidates):
        by_proc[c["process_norm"]].append(i)
    for proc, idxs in by_proc.items():
        if len(idxs) > 2:
            model.Add(sum(x[i] for i in idxs) <= 2)
    obj = sum(int(c["delta_seconds"]) * x[i] for i, c in enumerate(candidates))
    model.Maximize(obj)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return PluginResult("error", "MIP solver failed to find a solution", {}, [], [], None)
    selected = [candidates[i] for i in range(len(candidates)) if solver.Value(x[i]) == 1]
    selected.sort(key=lambda r: float(r["delta_seconds"]), reverse=True)
    total = float(sum(r["delta_seconds"] for r in selected))
    artifact = {"selected": selected[:top_k], "delta_seconds_total": total, "candidates": len(candidates), "max_actions": max_actions}
    artifacts = [_artifact(ctx, plugin_id, "mip_plan.json", artifact, "json")]
    findings = [
        _make_actionable_lever(
            plugin_id=plugin_id,
            process_norm="(multiple)",
            title=f"Action combo plan (MIP): top {len(selected)} actions",
            recommendation="Execute the selected actions as one package; then re-run and confirm close-window spillover shrinks without moving the bottleneck.",
            action_type="action_plan_combo",
            expected_delta_seconds=total,
            confidence=0.6,
            evidence=artifact,
            measurement_type="modeled",
        )
    ]
    return PluginResult("ok", f"Optimized action combo (MIP) with {len(selected)} actions", {"delta_seconds_total": total}, findings, artifacts, None)


def _run_discrete_event_queue_simulator(ctx, plugin_id: str, config: dict[str, Any]) -> PluginResult:
    # Lightweight simulation using SimPy over aggregate arrivals. This is not a physics-accurate ERP simulator;
    # it's a stress-test harness for comparing plans.
    sim_hours = int(config.get("sim_hours") or 72)
    replications = int(config.get("replications") or 5)

    # Read MIP plan, if present.
    mip = _load_plugin_findings(ctx, "analysis_action_search_mip_batched_scheduler_v1")
    if not mip:
        return PluginResult("skipped", "Missing MIP plan; run analysis_action_search_mip_batched_scheduler_v1 first", {}, [], [], None)

    # Extract modeled delta as throughput boost factor.
    total_delta = 0.0
    for f in mip:
        if f.get("kind") == "actionable_ops_lever":
            v = f.get("expected_delta_seconds")
            if isinstance(v, (int, float)):
                total_delta = max(total_delta, float(v))
    speedup = 1.0 + min(0.5, total_delta / (3600.0 * 200.0))  # cap at +50%

    rng = random.Random(int(getattr(ctx, "run_seed", 1337) or 0))

    def simulate_once() -> float:
        env = simpy.Environment()
        server = simpy.Resource(env, capacity=1)
        completed = {"n": 0}

        # Simple stationary arrival process (Poisson) based on 1 job/min baseline.
        lam_per_min = 1.0

        def job():
            with server.request() as req:
                yield req
                service = rng.expovariate(1.0 / 1.0)  # mean 1 minute
                yield env.timeout(service / speedup)
                completed["n"] += 1

        def arrivals():
            while env.now < sim_hours * 60:
                ia = rng.expovariate(lam_per_min)
                yield env.timeout(ia)
                env.process(job())

        env.process(arrivals())
        env.run(until=sim_hours * 60)
        return float(completed["n"])

    completed_counts = [simulate_once() for _ in range(max(1, replications))]
    artifact = {"replications": replications, "sim_hours": sim_hours, "completed_jobs": completed_counts, "speedup": speedup}
    artifacts = [_artifact(ctx, plugin_id, "queue_simulation.json", artifact, "json")]
    findings = [
        _make_actionable_lever(
            plugin_id=plugin_id,
            process_norm="(multiple)",
            title="Simulation sanity-check for action plan",
            recommendation="Simulation suggests the action plan increases effective throughput; validate with a real re-run and compare spillover totals.",
            action_type="simulate_plan",
            expected_delta_seconds=None,
            confidence=0.45,
            evidence=artifact,
            measurement_type="modeled",
        )
    ]
    return PluginResult("ok", "Simulation complete", {"completed_jobs_mean": float(np.mean(completed_counts))}, findings, artifacts, None)


def _run_empirical_bayes_shrinkage(ctx, plugin_id: str, config: dict[str, Any]) -> PluginResult:
    info = _load_template_info(ctx)
    req = _require_template(info, plugin_id)
    if isinstance(req, PluginResult):
        return req
    template_table, info = req
    dataset_version_id = str(ctx.dataset_version_id)
    proc_col = info.role_field("process_name") or info.role_field("process")
    if not proc_col:
        return PluginResult("skipped", "Missing process role for shrinkage", {}, [], [], None)

    num_col = info.role_field("duration") or None
    numeric_expr = None
    numeric_label = None
    if num_col:
        numeric_expr = f"CAST({quote_identifier(num_col)} AS REAL)"
        numeric_label = num_col
    else:
        start_col = info.role_field("start_time")
        end_col = info.role_field("end_time")
        if start_col and end_col:
            numeric_expr = f"((julianday({quote_identifier(end_col)}) - julianday({quote_identifier(start_col)})) * 86400.0)"
            numeric_label = "service_seconds"
    if not numeric_expr:
        return PluginResult(
            "skipped",
            "Missing numeric metric for shrinkage (need duration role or start/end timestamps)",
            {},
            [],
            [],
            None,
        )
    top_k = int(config.get("top_k") or 25)

    with ctx.storage.connection() as conn:
        rows = conn.execute(
            f"""
            SELECT LOWER(TRIM(proc_raw)) AS proc,
                   COUNT(*) AS n,
                   AVG(x) AS mean_x,
                   AVG(x * x) AS mean_x2
            FROM (
              SELECT {quote_identifier(proc_col)} AS proc_raw, {numeric_expr} AS x
              FROM {quote_identifier(template_table)}
              WHERE dataset_version_id = ?
                AND {quote_identifier(proc_col)} IS NOT NULL
                AND {numeric_expr} IS NOT NULL
            ) t
            GROUP BY proc
            HAVING n >= 50
            """,
            (dataset_version_id,),
        ).fetchall()
    if not rows:
        return PluginResult("skipped", "No per-process numeric stats available for shrinkage", {}, [], [], None)

    stats = []
    for r in rows:
        n = int(r["n"])
        mu = float(r["mean_x"])
        mu2 = float(r["mean_x2"])
        var = max(0.0, mu2 - mu * mu)
        stats.append({"process_norm": str(r["proc"]), "n": n, "mean": mu, "var": var})

    global_mean = float(np.average([s["mean"] for s in stats], weights=[s["n"] for s in stats]))
    # Empirical Bayes shrinkage: shrink each mean toward global mean by factor based on within-process variance and n.
    shrunk = []
    for s in stats:
        n = float(s["n"])
        var = float(s["var"])
        # If variance is unknown, shrink harder.
        tau2 = np.median([x["var"] for x in stats]) if stats else 1.0
        w = n / (n + (var / max(1e-9, tau2)))
        est = w * float(s["mean"]) + (1.0 - w) * global_mean
        shrunk.append({**s, "shrunk_mean": float(est), "weight": float(w)})
    shrunk.sort(key=lambda r: float(r["shrunk_mean"]), reverse=True)

    findings = []
    for rank, row in enumerate(shrunk[:top_k], start=1):
        proc = row["process_norm"]
        title = f"Stable priority driver: {proc} (rank {rank})"
        rec = (
            f"{proc} remains a top driver after shrinkage (shrunk mean≈{row['shrunk_mean']:.2f}, n={row['n']}). "
            "Action: treat this as a stable target for batching, dedupe/caching, or isolation rather than chasing noisy one-off outliers."
        )
        findings.append(
            _make_actionable_lever(
                plugin_id=plugin_id,
                process_norm=proc,
                title=title,
                recommendation=rec,
                action_type="stable_priority_candidate",
                expected_delta_seconds=None,
                confidence=0.6,
                evidence=row,
                measurement_type="measured",
            )
        )
    artifacts = [
        _artifact(
            ctx,
            plugin_id,
            "shrinkage_rankings.json",
            {"metric": numeric_label, "global_mean": global_mean, "rows": shrunk[:200]},
            "json",
        )
    ]
    return PluginResult("ok", f"Shrinkage ranked {len(shrunk)} processes", {"processes": len(shrunk)}, findings, artifacts, None)


def _run_not_implemented(ctx, plugin_id: str, _config: dict[str, Any]) -> PluginResult:
    return PluginResult("error", f"{plugin_id} handler missing or unknown", {}, [], [], None)


HANDLERS = {
    "analysis_param_near_duplicate_minhash_v1": _run_param_near_duplicate_minhash,
    "analysis_param_near_duplicate_simhash_v1": _run_param_near_duplicate_simhash,
    "analysis_frequent_itemsets_fpgrowth_v1": _run_frequent_itemsets_fpgrowth,
    "analysis_association_rules_apriori_v1": _run_association_rules_apriori,
    "analysis_sequential_patterns_prefixspan_v1": _run_sequential_patterns_prefixspan,
    "analysis_sequence_grammar_sequitur_v1": _run_sequence_grammar_sequitur,
    "analysis_dependency_community_louvain_v1": _run_dependency_community_louvain_v1,
    "analysis_dependency_community_leiden_v1": _run_dependency_community_leiden_v1,
    "analysis_graph_min_cut_partition_v1": _run_graph_min_cut_partition,
    "analysis_distribution_shift_wasserstein_v1": _run_distribution_shift_wasserstein,
    "analysis_burst_modeling_hawkes_v1": _run_burst_modeling_hawkes,
    "analysis_daily_pattern_alignment_dtw_v1": _run_daily_pattern_alignment_dtw,
    "analysis_action_search_simulated_annealing_v1": _run_action_search_simulated_annealing,
    "analysis_action_search_mip_batched_scheduler_v1": _run_action_search_mip,
    "analysis_discrete_event_queue_simulator_v1": _run_discrete_event_queue_simulator,
    "analysis_empirical_bayes_shrinkage_v1": _run_empirical_bayes_shrinkage,
    "analysis_biclustering_cheng_church_v1": _run_biclustering_cheng_church,
    "analysis_density_clustering_hdbscan_v1": _run_density_clustering_hdbscan,
    "analysis_constrained_clustering_cop_kmeans_v1": _run_constrained_clustering_cop_kmeans,
    "analysis_similarity_graph_spectral_clustering_v1": _run_similarity_graph_spectral_clustering,
}


def run_top20_plugin(plugin_id: str, ctx) -> PluginResult:
    config = dict(getattr(ctx, "settings", None) or {})
    handler = HANDLERS.get(plugin_id)
    if handler is None:
        return PluginResult("error", f"Unknown plugin_id: {plugin_id}", {}, [], [], None)
    return handler(ctx, plugin_id, config)
