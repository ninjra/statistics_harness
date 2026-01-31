from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from statistic_harness.core.types import PluginArtifact, PluginResult
from statistic_harness.core.utils import write_json


class Plugin:
    def run(self, ctx) -> PluginResult:
        input_file = ctx.settings.get("input_file")
        if not input_file:
            raise ValueError("Input file not found")
        path = Path(input_file)
        if not path.exists():
            raise ValueError("Input file not found")

        encoding = ctx.settings.get("encoding", "utf-8")
        max_rows = int(ctx.settings.get("max_rows", 10000))
        delimiter = ctx.settings.get("delimiter")
        sheet_name = ctx.settings.get("sheet_name")

        if path.suffix.lower() in {".xls", ".xlsx"}:
            df = pd.read_excel(path, sheet_name=sheet_name)
        elif path.suffix.lower() == ".json":
            df = pd.read_json(path)
        else:
            df = pd.read_csv(path, delimiter=delimiter, encoding=encoding)

        df = df.head(max_rows)
        canonical_path = ctx.run_dir / "dataset" / "canonical.csv"
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(canonical_path, index=False)

        schema = {
            "columns": [
                {
                    "name": col,
                    "dtype": str(dtype),
                    "missing": float(df[col].isna().mean()),
                }
                for col, dtype in df.dtypes.items()
            ]
        }
        schema_path = ctx.run_dir / "dataset" / "schema.json"
        write_json(schema_path, schema)

        return PluginResult(
            status="ok",
            summary=f"Ingested {df.shape[0]} rows",
            metrics={"rows": int(df.shape[0]), "cols": int(df.shape[1])},
            findings=[],
            artifacts=[
                PluginArtifact(path=str(schema_path.relative_to(ctx.run_dir)), type="json", description="Schema")
            ],
            error=None,
        )
