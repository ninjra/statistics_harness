from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from statistic_harness.core.sql_pack import validate_sql_pack
from statistic_harness.core.types import PluginArtifact, PluginResult


MODEL_DIRS = {
    # Organized under /mnt/d/autocapture/models/sql/text2sql/<...>
    "SQLCoder-7B-2": "sql/text2sql/defog/sqlcoder-7b-2",
    # Back-compat alias (earlier plan drafts used this name).
    "SQLCoder2-7B-2": "sql/text2sql/defog/sqlcoder-7b-2",
    "Snowflake-Arctic-Text2SQL-R1-7B": "sql/text2sql/snowflake/arctic-text2sql-r1-7b",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _build_prompt(schema_snapshot: dict[str, Any], intents: list[dict[str, Any]]) -> str:
    # Keep the prompt stable and strict: output MUST be JSON matching docs/sql_pack.schema.json.
    return (
        "You are generating a SQL pack for SQLite.\n"
        "Output MUST be a single JSON object that validates against the schema in docs/sql_pack.schema.json.\n"
        "Rules:\n"
        "- Single-statement SQL only (no semicolons).\n"
        "- Use deterministic ordering (ORDER BY) for any top-N queries.\n"
        "- mode MUST be 'ro' for every query. Do not emit DDL/DML.\n"
        "- DO NOT use PRAGMA/ATTACH/DETACH/VACUUM.\n"
        "\n"
        "Schema snapshot (JSON):\n"
        + _stable_json(schema_snapshot)
        + "\n"
        "Query intents (JSON):\n"
        + _stable_json({"intents": intents})
    )


class Plugin:
    def run(self, ctx) -> PluginResult:
        enabled = bool(ctx.settings.get("enabled", False))
        if not enabled:
            return PluginResult(
                "skipped",
                "Local text2sql generation disabled (set enabled=true for this plugin)",
                metrics={},
                findings=[],
                artifacts=[],
                error=None,
            )

        # Load schema + intents from prior transform artifacts.
        intents_path = ctx.run_dir / "artifacts" / "transform_sql_intents_pack_v1" / "sql_intents.json"
        schema_path = ctx.run_dir / "artifacts" / "transform_sql_intents_pack_v1" / "schema_snapshot.json"
        if not intents_path.exists() or not schema_path.exists():
            return PluginResult(
                "error",
                "Missing sql intent pack artifacts",
                metrics={},
                findings=[],
                artifacts=[],
                error=None,
            )

        schema = _read_json(schema_path)
        intents_payload = _read_json(intents_path)
        intents = intents_payload.get("intents") if isinstance(intents_payload, dict) else None
        if not isinstance(schema, dict) or not isinstance(intents, list):
            return PluginResult(
                "error",
                "Invalid sql intent pack artifacts",
                metrics={},
                findings=[],
                artifacts=[],
                error=None,
            )

        primary_model = str(ctx.settings.get("model_name") or "SQLCoder2-7B-2")
        council = ctx.settings.get("council_models")
        council_models: list[str] = []
        if isinstance(council, list):
            council_models = [str(x) for x in council if str(x).strip()]
        # Always include the primary model first.
        model_names = [primary_model] + [m for m in council_models if m != primary_model]
        models_root = Path(str(ctx.settings.get("models_root") or "/mnt/d/autocapture/models"))
        model_dirs: dict[str, Path] = {}
        for model_name in model_names:
            rel = MODEL_DIRS.get(model_name)
            if not rel:
                return PluginResult(
                    "error",
                    f"Unknown model_name={model_name}; supported={sorted(MODEL_DIRS.keys())}",
                    metrics={},
                    findings=[],
                    artifacts=[],
                    error=None,
                )
            model_dir = (models_root / rel).resolve()
            model_dirs[model_name] = model_dir

        missing = [f"{n}={p}" for n, p in model_dirs.items() if not p.exists()]
        if missing:
            # Bypassable: if models aren't present, skip cleanly.
            return PluginResult(
                "skipped",
                "Model dir(s) missing: " + ", ".join(missing),
                metrics={"models_root": str(models_root)},
                findings=[],
                artifacts=[],
                error=None,
            )

        # Import vLLM lazily; keep plugin runnable when deps are absent.
        try:
            from vllm import LLM, SamplingParams  # type: ignore
        except Exception as exc:
            return PluginResult(
                "skipped",
                f"vLLM not available in this environment ({type(exc).__name__}: {exc})",
                metrics={"model_dir": str(model_dir)},
                findings=[],
                artifacts=[],
                error=None,
            )

        temperature = float(ctx.settings.get("temperature") or 0.0)
        top_p = float(ctx.settings.get("top_p") or 1.0)
        max_tokens = int(ctx.settings.get("max_tokens") or 4096)

        prompt = _build_prompt(schema, intents)
        artifacts_dir = ctx.artifacts_dir("llm_text2sql_local_generate_v1")
        prompt_path = artifacts_dir / "prompt.md"
        prompt_path.write_text(prompt + "\n", encoding="utf-8")

        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        out_dir = Path(str(ctx.settings.get("out_dir") or "appdata/sqlpacks")).resolve()

        candidates: list[tuple[str, dict[str, Any], str]] = []
        errors: list[str] = []
        for model_name in model_names:
            mdir = model_dirs[model_name]
            try:
                llm = LLM(model=str(mdir), trust_remote_code=False)
                outputs = llm.generate([prompt], params)
                text = ""
                if outputs and outputs[0].outputs:
                    text = str(outputs[0].outputs[0].text or "").strip()
                if not text:
                    raise ValueError("empty output")
                raw_path = artifacts_dir / f"raw_{model_name}.txt"
                raw_path.write_text(text + "\n", encoding="utf-8")

                # Parse + validate.
                try:
                    pack = json.loads(text)
                except Exception:
                    cleaned = text
                    if cleaned.startswith("```"):
                        cleaned = cleaned.strip("`")
                        cleaned = cleaned.replace("json", "", 1).strip()
                    pack = json.loads(cleaned)

                if not isinstance(pack, dict):
                    raise ValueError("output is not a JSON object")

                pack.setdefault("schema_hash", str(schema.get("schema_hash") or ""))
                pack.setdefault("dialect", "sqlite")
                pack.setdefault("model", {"name": model_name, "source": "local"})
                pack.setdefault(
                    "decode",
                    {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens},
                )

                schema_path = Path("docs/sql_pack.schema.json")
                validate_sql_pack(pack, schema_path)

                candidates.append((model_name, pack, text))
            except Exception as exc:
                errors.append(f"{model_name}: {type(exc).__name__}: {exc}")

        if not candidates:
            return PluginResult(
                "error",
                "No model produced a valid SQL pack",
                metrics={},
                findings=[],
                artifacts=[
                    PluginArtifact(path=str(prompt_path.relative_to(ctx.run_dir)), type="markdown", description="Prompt")
                ],
                error=None,
            )

        # Council selection:
        # Prefer the pack with the most "ro" queries that validate against current schema (EXPLAIN).
        # This stays deterministic without running expensive queries.
        best_model = candidates[0][0]
        best_pack = candidates[0][1]
        best_valid = -1
        validation_notes: dict[str, Any] = {}
        for model_name, pack, _raw in candidates:
            valid = 0
            bad: list[str] = []
            for q in (pack.get("queries") or []):
                if not isinstance(q, dict):
                    continue
                qid = str(q.get("id") or "").strip()
                mode = str(q.get("mode") or "").strip()
                sql = str(q.get("sql") or "").strip()
                if mode != "ro" or not qid or not sql:
                    bad.append(qid or "<missing_id>")
                    continue
                try:
                    if ctx.sql is not None:
                        ctx.sql.validate_ro_sql(sql, query_id=f"validate_{model_name}_{qid}")
                    valid += 1
                except Exception:
                    bad.append(qid)
            validation_notes[model_name] = {"valid_queries": valid, "invalid_query_ids": bad[:25]}
            if valid > best_valid:
                best_valid = valid
                best_model = model_name
                best_pack = pack
            elif valid == best_valid and model_name == primary_model:
                # Tie-breaker: prefer primary model deterministically.
                best_model = model_name
                best_pack = pack

        schema_hash = str(best_pack.get("schema_hash") or "").strip() or "unknown_schema"
        sql_pack_path = artifacts_dir / "sql_pack.json"
        sql_pack_path.write_text(_stable_json(best_pack), encoding="utf-8")

        # Also persist in appdata/sqlpacks/<schema_hash>/sql_pack.json for replay.
        out_pack_dir = out_dir / schema_hash
        out_pack_dir.mkdir(parents=True, exist_ok=True)
        out_pack_path = out_pack_dir / "sql_pack.json"
        out_pack_path.write_text(_stable_json(best_pack), encoding="utf-8")
        out_manifest = out_pack_dir / "generation_manifest.json"
        out_manifest.write_text(
            _stable_json(
                {
                    "schema_hash": schema_hash,
                    "chosen_model": best_model,
                    "candidate_models": model_names,
                    "validation": validation_notes,
                    "errors": errors,
                }
            ),
            encoding="utf-8",
        )

        out_dir_display = str(ctx.settings.get("out_dir") or "appdata/sqlpacks").rstrip("/").strip() or "appdata/sqlpacks"
        artifacts = [
            PluginArtifact(path=str(prompt_path.relative_to(ctx.run_dir)), type="markdown", description="Prompt"),
            PluginArtifact(path=str(sql_pack_path.relative_to(ctx.run_dir)), type="json", description="SQL pack (run artifact)"),
            PluginArtifact(
                path=f"{out_dir_display}/{schema_hash}/sql_pack.json",
                type="json",
                description="SQL pack (replayable; stable path)",
            ),
            PluginArtifact(
                path=f"{out_dir_display}/{schema_hash}/generation_manifest.json",
                type="json",
                description="Generation manifest (chosen model, validations, errors)",
            ),
        ]
        return PluginResult(
            "ok",
            f"Generated SQL pack (chosen={best_model}) with {len(best_pack.get('queries') or [])} queries",
            metrics={
                "chosen_model": best_model,
                "models": ",".join(model_names),
                "schema_hash": schema_hash,
                "out_pack_path": str(out_pack_path),
            },
            findings=[
                {
                    "kind": "sql_pack_generated",
                    "schema_hash": schema_hash,
                    "queries": int(len(best_pack.get("queries") or [])),
                    "measurement_type": "measured",
                }
            ],
            artifacts=artifacts,
            error=None,
        )
