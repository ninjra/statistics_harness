from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _resolve_route_plan(path_like: str) -> Path:
    base = Path(path_like).expanduser().resolve()
    if base.is_file():
        return base
    if base.is_dir():
        return base / "artifacts" / "analysis_ebm_action_verifier_v1" / "route_plan.json"
    return base


def _route_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    totals = payload.get("totals") if isinstance(payload.get("totals"), dict) else {}
    steps = payload.get("steps") if isinstance(payload.get("steps"), list) else []
    modeled = str(payload.get("decision") or "").strip().lower() == "modeled"
    return {
        "decision": str(payload.get("decision") or ""),
        "modeled": bool(modeled),
        "step_count": int(len([s for s in steps if isinstance(s, dict)])),
        "total_delta_energy": float(totals.get("total_delta_energy") or 0.0),
        "energy_before": float(totals.get("energy_before") or 0.0),
        "energy_after": float(totals.get("energy_after") or 0.0),
        "route_confidence": float(totals.get("route_confidence") or 0.0),
        "stop_reason": str(totals.get("stop_reason") or ""),
        "expanded_states": int(totals.get("expanded_states") or 0),
    }


def _pct_improvement(energy_before: float, delta: float) -> float:
    if energy_before <= 0.0:
        return 0.0
    return max(0.0, min(100.0, (delta / energy_before) * 100.0))


def _build_comparison(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_metrics = _route_metrics(before)
    after_metrics = _route_metrics(after)
    before_delta = float(before_metrics.get("total_delta_energy") or 0.0)
    after_delta = float(after_metrics.get("total_delta_energy") or 0.0)
    before_conf = float(before_metrics.get("route_confidence") or 0.0)
    after_conf = float(after_metrics.get("route_confidence") or 0.0)
    before_steps = int(before_metrics.get("step_count") or 0)
    after_steps = int(after_metrics.get("step_count") or 0)
    before_before = float(before_metrics.get("energy_before") or 0.0)
    after_before = float(after_metrics.get("energy_before") or 0.0)
    return {
        "schema_version": "kona_route_compare.v1",
        "before": before_metrics,
        "after": after_metrics,
        "delta": {
            "total_delta_energy_diff": float(after_delta - before_delta),
            "route_confidence_diff": float(after_conf - before_conf),
            "step_count_diff": int(after_steps - before_steps),
            "modeled_percent_before": float(_pct_improvement(before_before, before_delta)),
            "modeled_percent_after": float(_pct_improvement(after_before, after_delta)),
        },
    }


def _to_markdown(summary: dict[str, Any]) -> str:
    before = summary.get("before") if isinstance(summary.get("before"), dict) else {}
    after = summary.get("after") if isinstance(summary.get("after"), dict) else {}
    delta = summary.get("delta") if isinstance(summary.get("delta"), dict) else {}
    lines = [
        "# Kona Route Quality Comparison",
        "",
        "| Metric | Before | After | Diff |",
        "|---|---:|---:|---:|",
        f"| Modeled | {before.get('modeled')} | {after.get('modeled')} | - |",
        f"| Step Count | {before.get('step_count', 0)} | {after.get('step_count', 0)} | {delta.get('step_count_diff', 0)} |",
        (
            f"| Total Delta Energy | {float(before.get('total_delta_energy', 0.0)):.6f} | "
            f"{float(after.get('total_delta_energy', 0.0)):.6f} | {float(delta.get('total_delta_energy_diff', 0.0)):.6f} |"
        ),
        (
            f"| Route Confidence | {float(before.get('route_confidence', 0.0)):.4f} | "
            f"{float(after.get('route_confidence', 0.0)):.4f} | {float(delta.get('route_confidence_diff', 0.0)):.4f} |"
        ),
        (
            f"| Modeled Improvement % | {float(delta.get('modeled_percent_before', 0.0)):.2f}% | "
            f"{float(delta.get('modeled_percent_after', 0.0)):.2f}% | "
            f"{(float(delta.get('modeled_percent_after', 0.0)) - float(delta.get('modeled_percent_before', 0.0))):.2f}% |"
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare before/after Kona route_plan quality.")
    parser.add_argument("--before", required=True, help="Before run dir or route_plan.json path.")
    parser.add_argument("--after", required=True, help="After run dir or route_plan.json path.")
    parser.add_argument("--out-json", required=True, help="Output JSON summary path.")
    parser.add_argument("--out-md", default="", help="Optional markdown summary output path.")
    args = parser.parse_args()

    before_path = _resolve_route_plan(args.before)
    after_path = _resolve_route_plan(args.after)
    before = _load_json(before_path) if before_path.exists() else {}
    after = _load_json(after_path) if after_path.exists() else {}
    summary = _build_comparison(before, after)

    out_json = Path(args.out_json).expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    out_md_raw = str(args.out_md or "").strip()
    if out_md_raw:
        out_md = Path(out_md_raw).expanduser().resolve()
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(_to_markdown(summary) + "\n", encoding="utf-8")

    print(f"before={before_path}")
    print(f"after={after_path}")
    print(f"out_json={out_json}")
    if out_md_raw:
        print(f"out_md={Path(out_md_raw).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
