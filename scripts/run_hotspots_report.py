#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def _connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def _rss_bytes_from_kb(kb: int | None) -> int | None:
    if kb is None:
        return None
    if kb < 0:
        return None
    return int(kb) * 1024


def _fmt_int(v: int | None) -> str:
    return str(v) if v is not None else "null"


def _fmt_float(v: float | None) -> str:
    if v is None:
        return "null"
    return f"{v:.3f}"


def _render_md(rows: list[dict], *, top_n: int) -> str:
    lines: list[str] = []
    lines.append("# Run Hotspots")
    lines.append("")
    lines.append(f"- Plugin executions: {len(rows)}")

    def key_dur(r: dict) -> tuple[int, str]:
        d = r.get("duration_ms")
        return (-(int(d) if isinstance(d, int) else 0), str(r.get("plugin_id") or ""))

    def key_rss(r: dict) -> tuple[int, str]:
        kb = r.get("max_rss_kb")
        return (-(int(kb) if isinstance(kb, int) else 0), str(r.get("plugin_id") or ""))

    top_dur = sorted(rows, key=key_dur)[:top_n]
    top_rss = sorted(rows, key=key_rss)[:top_n]
    failures = [
        r
        for r in sorted(rows, key=lambda x: (str(x.get("status") or ""), str(x.get("plugin_id") or "")))
        if str(r.get("status") or "") not in {"ok", "skipped"} or (r.get("exit_code") not in {None, 0})
    ][:top_n]

    def table(title: str, items: list[dict]) -> None:
        lines.append("")
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| plugin_id | status | duration_ms | max_rss_kb | max_rss_bytes | exit_code |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for r in items:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(r.get("plugin_id") or ""),
                        str(r.get("status") or ""),
                        _fmt_int(r.get("duration_ms")),
                        _fmt_int(r.get("max_rss_kb")),
                        _fmt_int(r.get("max_rss_bytes")),
                        _fmt_int(r.get("exit_code")),
                    ]
                )
                + " |"
            )

    table(f"Top {top_n} By Duration", top_dur)
    table(f"Top {top_n} By RSS", top_rss)
    if failures:
        table(f"Top {top_n} Failures", failures)
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=Path("appdata/state.sqlite"))
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--top-n", type=int, default=15)
    ap.add_argument("--out-md", type=Path, default=None)
    args = ap.parse_args()

    db_path = args.db
    if not db_path.exists():
        raise SystemExit(f"Missing DB: {db_path}")

    con = _connect(db_path)
    try:
        cur = con.execute(
            """
            SELECT plugin_id, status, duration_ms, max_rss, exit_code, started_at, completed_at
            FROM plugin_executions
            WHERE run_id = ?
            """,
            (str(args.run_id),),
        )
        rows = []
        for r in cur.fetchall():
            kb = r["max_rss"] if r["max_rss"] is not None else None
            kb_int = int(kb) if isinstance(kb, (int, float)) else None
            rows.append(
                {
                    "plugin_id": str(r["plugin_id"] or ""),
                    "status": str(r["status"] or ""),
                    "duration_ms": int(r["duration_ms"]) if r["duration_ms"] is not None else None,
                    "max_rss_kb": kb_int,
                    "max_rss_bytes": _rss_bytes_from_kb(kb_int),
                    "exit_code": int(r["exit_code"]) if r["exit_code"] is not None else None,
                    "started_at": str(r["started_at"] or ""),
                    "completed_at": str(r["completed_at"] or ""),
                }
            )
    finally:
        con.close()

    top_n = max(1, int(args.top_n))
    md = _render_md(rows, top_n=top_n)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md, encoding="utf-8")
    else:
        print(md, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

