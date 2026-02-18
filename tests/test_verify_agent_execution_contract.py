from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path


def _init_state(root: Path) -> sqlite3.Connection:
    appdata = root / "appdata"
    appdata.mkdir(parents=True, exist_ok=True)
    (appdata / "runs").mkdir(parents=True, exist_ok=True)
    db_path = appdata / "state.sqlite"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE runs (
            run_id TEXT PRIMARY KEY,
            status TEXT,
            dataset_version_id TEXT,
            run_seed INTEGER,
            requested_run_seed INTEGER,
            created_at TEXT
        );
        CREATE TABLE plugin_executions (
            run_id TEXT,
            plugin_id TEXT
        );
        CREATE TABLE plugin_results_v2 (
            run_id TEXT,
            plugin_id TEXT,
            status TEXT
        );
        """
    )
    conn.commit()
    return conn


def _write_run_artifacts(
    root: Path,
    run_id: str,
    *,
    known_status: str,
    known_items: list[dict[str, object]],
    plugins: dict[str, object] | None = None,
    recommendation_items: list[dict[str, object]] | None = None,
    explanation_items: list[dict[str, object]] | None = None,
) -> None:
    run_dir = root / "appdata" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    normalized_known_items: list[dict[str, object]] = []
    for item in known_items:
        if not isinstance(item, dict):
            continue
        row = dict(item)
        row.setdefault("evidence_source", "plugin_findings")
        normalized_known_items.append(row)
    report = {
        "plugins": plugins or {},
        "recommendations": {
            "status": "ok",
            "summary": "",
            "items": recommendation_items or [],
            "discovery": {"status": "ok", "summary": "", "items": []},
            "known": {"status": known_status, "summary": "", "items": normalized_known_items},
            "explanations": {
                "status": "ok",
                "summary": "",
                "items": explanation_items or [],
            },
        }
    }
    answers_summary = {
        "run_id": run_id,
        "known_issue_checks": {
            "status": known_status,
            "summary": "",
            "items": normalized_known_items,
            "totals": {},
        },
        "recommendations": {"status": "ok", "items": []},
    }
    (run_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
    (run_dir / "answers_summary.json").write_text(json.dumps(answers_summary), encoding="utf-8")


def _write_taxonomy(root: Path) -> None:
    docs = root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    payload = {
        "plugin_overrides": {
            "analysis_close_cycle_uplift": "direct_action_generators",
        }
    }
    (docs / "plugin_class_taxonomy.yaml").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _insert_run(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    dataset_version_id: str,
    run_seed: int,
    statuses: list[str],
) -> None:
    conn.execute(
        """
        INSERT INTO runs(run_id, status, dataset_version_id, run_seed, requested_run_seed, created_at)
        VALUES(?, 'completed', ?, ?, ?, '2026-02-17T00:00:00+00:00')
        """,
        (run_id, dataset_version_id, run_seed, run_seed),
    )
    for idx, status in enumerate(statuses, start=1):
        plugin_id = f"plugin_{idx}"
        conn.execute(
            "INSERT INTO plugin_executions(run_id, plugin_id) VALUES(?, ?)",
            (run_id, plugin_id),
        )
        conn.execute(
            "INSERT INTO plugin_results_v2(run_id, plugin_id, status) VALUES(?, ?, ?)",
            (run_id, plugin_id, status),
        )
    conn.commit()


def _run_verifier(root: Path, args: list[str]) -> tuple[int, dict[str, object]]:
    out_path = root / "out_contract.json"
    cmd = [
        sys.executable,
        "scripts/verify_agent_execution_contract.py",
        "--root",
        str(root),
        "--run-id",
        "run_a",
        "--out",
        str(out_path),
    ] + args
    proc = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    return proc.returncode, payload


def test_verify_agent_execution_contract_passes_for_valid_run(tmp_path: Path) -> None:
    conn = _init_state(tmp_path)
    try:
        _insert_run(
            conn,
            "run_a",
            dataset_version_id="dataset_x",
            run_seed=1337,
            statuses=["ok", "ok", "na"],
        )
    finally:
        conn.close()
    _write_run_artifacts(
        tmp_path,
        "run_a",
        known_status="ok",
        known_items=[
            {
                "plugin_id": "analysis_close_cycle_contention",
                "kind": "close_cycle_contention",
                "status": "confirmed",
            }
        ],
    )
    rc, payload = _run_verifier(
        tmp_path,
        [
            "--expected-known-issues-mode",
            "on",
            "--require-known-signature",
            "analysis_close_cycle_contention:close_cycle_contention",
        ],
    )
    assert rc == 0
    assert payload["ok"] is True


def test_verify_agent_execution_contract_fails_on_mixed_known_modes(tmp_path: Path) -> None:
    conn = _init_state(tmp_path)
    try:
        _insert_run(conn, "run_a", dataset_version_id="dataset_x", run_seed=1337, statuses=["ok", "ok"])
        _insert_run(conn, "run_b", dataset_version_id="dataset_x", run_seed=1337, statuses=["ok", "ok"])
    finally:
        conn.close()
    _write_run_artifacts(
        tmp_path,
        "run_a",
        known_status="ok",
        known_items=[
            {
                "plugin_id": "analysis_close_cycle_contention",
                "kind": "close_cycle_contention",
                "status": "confirmed",
            }
        ],
    )
    _write_run_artifacts(tmp_path, "run_b", known_status="suppressed", known_items=[])
    rc, payload = _run_verifier(
        tmp_path,
        [
            "--compare-run-id",
            "run_b",
            "--expected-known-issues-mode",
            "any",
            "--require-known-signature",
            "analysis_close_cycle_contention:close_cycle_contention",
        ],
    )
    assert rc == 1
    checks = {str(item.get("id")): bool(item.get("ok")) for item in payload.get("checks", [])}
    assert checks["compare.same_known_issues_mode"] is False


def test_verify_agent_execution_contract_fails_when_required_known_missing(tmp_path: Path) -> None:
    conn = _init_state(tmp_path)
    try:
        _insert_run(conn, "run_a", dataset_version_id="dataset_x", run_seed=1337, statuses=["ok", "ok"])
    finally:
        conn.close()
    _write_run_artifacts(tmp_path, "run_a", known_status="ok", known_items=[])
    rc, payload = _run_verifier(
        tmp_path,
        [
            "--expected-known-issues-mode",
            "on",
            "--require-known-signature",
            "analysis_close_cycle_contention:close_cycle_contention",
        ],
    )
    assert rc == 1
    checks = {str(item.get("id")): bool(item.get("ok")) for item in payload.get("checks", [])}
    assert checks["run.required_known_signatures"] is False


def test_verify_agent_execution_contract_fails_unrouted_direct_action_signal(tmp_path: Path) -> None:
    _write_taxonomy(tmp_path)
    conn = _init_state(tmp_path)
    try:
        _insert_run(conn, "run_a", dataset_version_id="dataset_x", run_seed=1337, statuses=["ok"])
    finally:
        conn.close()
    _write_run_artifacts(
        tmp_path,
        "run_a",
        known_status="ok",
        known_items=[],
        plugins={
            "analysis_close_cycle_uplift": {
                "status": "ok",
                "findings": [
                    {
                        "kind": "close_cycle_share_shift",
                        "process_norm": "qemail",
                        "share_delta": 0.12,
                    }
                ],
            }
        },
        recommendation_items=[],
        explanation_items=[
            {
                "plugin_id": "analysis_close_cycle_uplift",
                "reason_code": "NOT_ROUTED_TO_ACTION",
            }
        ],
    )
    rc, payload = _run_verifier(tmp_path, ["--expected-known-issues-mode", "any"])
    assert rc == 1
    checks = {str(item.get("id")): bool(item.get("ok")) for item in payload.get("checks", [])}
    assert checks["run.direct_action_signals_routed"] is False


def test_verify_agent_execution_contract_fails_non_independent_known_matches(
    tmp_path: Path,
) -> None:
    conn = _init_state(tmp_path)
    try:
        _insert_run(
            conn,
            "run_a",
            dataset_version_id="dataset_x",
            run_seed=1337,
            statuses=["ok"],
        )
    finally:
        conn.close()
    _write_run_artifacts(
        tmp_path,
        "run_a",
        known_status="ok",
        known_items=[
            {
                "plugin_id": "analysis_close_cycle_contention",
                "kind": "close_cycle_contention",
                "status": "confirmed",
                "evidence_source": "synthetic_fallback",
            }
        ],
    )
    rc, payload = _run_verifier(
        tmp_path,
        [
            "--expected-known-issues-mode",
            "on",
            "--require-known-signature",
            "analysis_close_cycle_contention:close_cycle_contention",
        ],
    )
    assert rc == 1
    checks = {str(item.get("id")): bool(item.get("ok")) for item in payload.get("checks", [])}
    assert checks["run.known_issues_independent"] is False
