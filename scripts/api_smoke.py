#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib import error, parse, request


DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def _http_request(
    method: str,
    url: str,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
) -> tuple[int, dict[str, str], bytes]:
    req = request.Request(url, data=data, method=method)
    if headers:
        for key, value in headers.items():
            req.add_header(key, value)
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            return resp.status, dict(resp.headers), body
    except error.HTTPError as exc:
        body = exc.read()
        detail = body.decode("utf-8", errors="ignore")
        raise RuntimeError(f"{method} {url} -> {exc.code} {detail}") from exc


def _get_json(url: str, timeout: int = 30) -> dict[str, Any]:
    status, _, body = _http_request("GET", url, timeout=timeout)
    if status >= 400:
        raise RuntimeError(f"GET {url} failed with {status}")
    return json.loads(body.decode("utf-8"))


def _post_json(url: str, payload: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    status, _, body = _http_request(
        "POST",
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    if status >= 400:
        raise RuntimeError(f"POST {url} failed with {status}")
    return json.loads(body.decode("utf-8"))


def _post_form(url: str, payload: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
    encoded = parse.urlencode(payload).encode("utf-8")
    status, _, body = _http_request(
        "POST",
        url,
        data=encoded,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=timeout,
    )
    if status >= 400:
        raise RuntimeError(f"POST {url} failed with {status}")
    return json.loads(body.decode("utf-8"))


def _post_raw_file(url: str, file_path: Path, timeout: int = 60) -> dict[str, Any]:
    data = file_path.read_bytes()
    status, _, body = _http_request(
        "POST",
        url,
        data=data,
        headers={"Content-Type": "application/octet-stream"},
        timeout=timeout,
    )
    if status >= 400:
        raise RuntimeError(f"POST {url} failed with {status}")
    return json.loads(body.decode("utf-8"))


def _wait_for_server(base_url: str, timeout: int) -> None:
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            _get_json(f"{base_url}/api/uploads", timeout=5)
            return
        except Exception as exc:  # pragma: no cover - waiting loop
            last_err = exc
            time.sleep(0.5)
    raise RuntimeError(f"Server not ready: {last_err}")


def _start_server(base_url: str) -> subprocess.Popen:
    parsed = parse.urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8000
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-m", "statistic_harness.cli", "serve", "--host", host, "--port", str(port)]
    return subprocess.Popen(cmd, cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _poll_run(base_url: str, run_id: str, timeout: int) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = _get_json(f"{base_url}/api/runs/{run_id}")
        status = (payload.get("run") or {}).get("status")
        if status in {"completed", "failed", "error"}:
            return payload
        time.sleep(1)
    raise RuntimeError(f"Run {run_id} did not finish before timeout")


def main() -> int:
    parser = argparse.ArgumentParser(description="API smoke test for Statistic Harness")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--fixture", default="scripts/fixtures/sample.csv")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--start-server", action="store_true")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    fixture_path = Path(args.fixture)
    if not fixture_path.exists():
        raise SystemExit(f"Fixture not found: {fixture_path}")

    server_proc = None
    try:
        if args.start_server:
            server_proc = _start_server(base_url)
        _wait_for_server(base_url, args.timeout)

        print("Uploading fixture (raw upload)...")
        upload = _post_raw_file(
            f"{base_url}/api/upload/raw?filename={parse.quote(fixture_path.name)}",
            fixture_path,
        )
        print("Upload response:", upload)
        if not upload.get("upload_id"):
            raise RuntimeError("Upload did not return upload_id")

        print("Uploading same fixture to test dedupe...")
        upload2 = _post_raw_file(
            f"{base_url}/api/upload/raw?filename={parse.quote(fixture_path.name)}",
            fixture_path,
        )
        print("Dedupe response:", upload2)
        if not upload2.get("deduplicated"):
            raise RuntimeError("Expected deduplicated=true on second upload")

        print("Listing uploads...")
        uploads = _get_json(f"{base_url}/api/uploads")
        if not isinstance(uploads.get("uploads"), list):
            raise RuntimeError("Uploads payload missing list")

        upload_id = upload.get("upload_id")
        print("Saving known issues...")
        known_payload = {
            "strict": False,
            "notes": "smoke test",
            "expected_findings": [],
        }
        saved = _post_json(
            f"{base_url}/api/known-issues",
            {"upload_id": upload_id, "known_issues": known_payload},
        )
        if saved.get("status") != "ok":
            raise RuntimeError("Known issues save did not return status=ok")

        print("Starting auto-evaluate run...")
        run_resp = _post_form(
            f"{base_url}/api/runs/auto-evaluate",
            {"upload_id": upload_id},
        )
        run_id = run_resp.get("run_id")
        if not run_id:
            raise RuntimeError("auto-evaluate did not return run_id")

        print("Waiting for run completion...")
        _poll_run(base_url, run_id, args.timeout)

        print("Evaluating run (template mode)...")
        eval_payload = _post_json(
            f"{base_url}/api/runs/{run_id}/evaluate",
            {"mode": "template"},
        )
        if "result" not in eval_payload:
            raise RuntimeError("Evaluation payload missing result")

        print("Fetching evaluation JSON...")
        eval_saved = _get_json(f"{base_url}/api/runs/{run_id}/evaluation")
        if "result" not in eval_saved:
            raise RuntimeError("Saved evaluation missing result")

        print("Fetching report JSON...")
        _http_request("GET", f"{base_url}/api/runs/{run_id}/report.json")

        print("API smoke test completed successfully.")
        return 0
    finally:
        if server_proc:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
