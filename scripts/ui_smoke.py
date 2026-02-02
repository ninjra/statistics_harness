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
from urllib import parse, request

DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def _wait_for_server(base_url: str, timeout: int) -> None:
    deadline = time.time() + timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            with request.urlopen(f"{base_url}/api/uploads", timeout=5) as resp:
                if resp.status < 500:
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


def _parse_json(text: str) -> dict[str, Any]:
    return json.loads(text)


def main() -> int:
    parser = argparse.ArgumentParser(description="UI smoke test (Playwright)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--fixture", default="scripts/fixtures/sample.csv")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--start-server", action="store_true")
    parser.add_argument("--headed", action="store_true")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    fixture_path = Path(args.fixture)
    if not fixture_path.exists():
        raise SystemExit(f"Fixture not found: {fixture_path}")

    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # pragma: no cover - missing dependency
        raise SystemExit(
            "Playwright is not installed. Run: pip install -e .[dev] && python -m playwright install"
        ) from exc

    server_proc = None
    try:
        if args.start_server:
            server_proc = _start_server(base_url)
        _wait_for_server(base_url, args.timeout)

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=not args.headed)
            page = browser.new_page()

            page.goto(f"{base_url}/wizard", wait_until="networkidle")
            page.set_input_files("#upload-form input[type=file]", str(fixture_path))
            page.click("#upload-form button[type=submit]")
            page.wait_for_function(
                "document.getElementById('upload-result').textContent.includes('upload_id')",
                timeout=args.timeout * 1000,
            )
            upload_text = page.inner_text("#upload-result")
            upload_payload = _parse_json(upload_text)
            upload_id = upload_payload.get("upload_id")
            if not upload_id:
                raise RuntimeError("Upload did not return upload_id")

            # Ensure strict is unchecked to avoid false-positive enforcement
            if page.is_checked("#strict-flag"):
                page.click("#strict-flag")

            # Exercise quick-add and remove
            page.click("#quick-qemail")
            page.locator(".issue-row").last.locator("button").click()

            page.click("#save-issues")
            page.wait_for_function(
                "document.getElementById('save-result').textContent.includes('status')",
                timeout=args.timeout * 1000,
            )
            save_text = page.inner_text("#save-result")
            if "status" not in save_text:
                raise RuntimeError("Known issues save did not return status")

            page.click("#run-auto")
            page.wait_for_function(
                "document.getElementById('run-status').textContent.startsWith('completed')",
                timeout=args.timeout * 1000,
            )
            eval_text = page.inner_text("#eval-result")
            eval_payload = _parse_json(eval_text)
            if "result" not in eval_payload:
                raise RuntimeError("Evaluation payload missing result")

            # Basic navigation checks
            page.goto(f"{base_url}/", wait_until="networkidle")
            if "Statistic Harness" not in page.title():
                raise RuntimeError("Home page title missing")

            browser.close()

        print("UI smoke test completed successfully.")
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
