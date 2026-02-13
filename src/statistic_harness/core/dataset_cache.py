from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .dataset_io import DatasetAccessor


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _json_dumps_stable(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_appdata_cache_root() -> Path:
    # All caches are local-only and must stay under ./appdata/.
    root = Path("appdata").resolve()
    (root / "cache" / "datasets").mkdir(parents=True, exist_ok=True)
    return root / "cache" / "datasets"


def _stable_schema_sig(columns: list[dict[str, Any]]) -> str:
    # Only include stable, non-PII-ish metadata in the cache keying material.
    sig = [
        {
            "original_name": str(c.get("original_name") or ""),
            "safe_name": str(c.get("safe_name") or ""),
            "dtype": str(c.get("dtype") or ""),
        }
        for c in columns
    ]
    return _json_dumps_stable(sig)


@dataclass(frozen=True)
class CachedColumnSpec:
    name: str
    kind: str  # "float64" | "int64_masked" | "bool_masked"
    data_file: str
    mask_file: str | None


@dataclass(frozen=True)
class DatasetCacheKey:
    dataset_version_id: str
    data_hash: str
    schema_hash: str
    key: str

    @staticmethod
    def from_dataset(
        *,
        dataset_version_id: str,
        data_hash: str | None,
        columns: list[dict[str, Any]],
    ) -> "DatasetCacheKey":
        dh = (data_hash or dataset_version_id or "").strip()
        schema_hash = _sha256_hex(_stable_schema_sig(columns))
        key_material = _json_dumps_stable(
            {
                "dataset_version_id": dataset_version_id,
                "data_hash": dh,
                "schema_hash": schema_hash,
                "v": 1,
            }
        )
        key = _sha256_hex(key_material)[:24]
        return DatasetCacheKey(
            dataset_version_id=dataset_version_id,
            data_hash=dh,
            schema_hash=schema_hash,
            key=key,
        )


class DatasetCache:
    """Optional columnar cache for large datasets.

    This cache is intentionally conservative:
    - It caches ONLY numeric-ish columns (float/int/bool) for now.
    - It uses .npy arrays (with optional boolean masks) to allow mmap reads.
    - It never reads/writes outside ./appdata/.
    """

    def __init__(self, cache_key: DatasetCacheKey) -> None:
        self.cache_key = cache_key
        self.root = _safe_appdata_cache_root() / cache_key.key
        self.manifest_path = self.root / "manifest.json"

    @staticmethod
    def enabled() -> bool:
        return _env_truthy("STAT_HARNESS_DATASET_CACHE", default=False)

    def exists(self) -> bool:
        return self.manifest_path.exists()

    def load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(self.manifest_path)

    def _specs_from_manifest(self, manifest: dict[str, Any]) -> dict[str, CachedColumnSpec]:
        specs: dict[str, CachedColumnSpec] = {}
        for item in (manifest.get("columns") or []):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "")
            kind = str(item.get("kind") or "")
            data_file = str(item.get("data_file") or "")
            mask_file = item.get("mask_file")
            if name and kind and data_file:
                specs[name] = CachedColumnSpec(
                    name=name,
                    kind=kind,
                    data_file=data_file,
                    mask_file=str(mask_file) if mask_file else None,
                )
        return specs

    def can_serve_columns(self, columns: list[str] | None) -> bool:
        if not self.exists():
            return False
        manifest = self.load_manifest()
        specs = self._specs_from_manifest(manifest)
        if columns is None:
            # "All columns" can't be served from the numeric-only cache.
            return False
        needed = set(columns)
        return needed.issubset(set(specs.keys()))

    def iter_batches(
        self,
        *,
        columns: list[str],
        batch_size: int,
        row_limit: int | None,
    ) -> Iterable[pd.DataFrame]:
        manifest = self.load_manifest()
        specs = self._specs_from_manifest(manifest)
        row_count = int(manifest.get("row_count") or 0)
        if row_limit is not None:
            row_count = min(row_count, int(row_limit))
        if row_count <= 0:
            return

        # Load row_index array (always present in a valid manifest).
        row_index_path = self.root / "row_index.npy"
        row_index = np.load(row_index_path, mmap_mode="r")

        data_mm: dict[str, np.ndarray] = {}
        mask_mm: dict[str, np.ndarray | None] = {}
        for name in columns:
            spec = specs.get(name)
            if spec is None:
                raise RuntimeError(f"column not cached: {name}")
            data_mm[name] = np.load(self.root / spec.data_file, mmap_mode="r")
            if spec.mask_file:
                mask_mm[name] = np.load(self.root / spec.mask_file, mmap_mode="r")
            else:
                mask_mm[name] = None

        bs = max(1, int(batch_size))
        start = 0
        while start < row_count:
            end = min(row_count, start + bs)
            batch: dict[str, Any] = {}
            for name in columns:
                spec = specs[name]
                data_slice = data_mm[name][start:end]
                m = mask_mm[name]
                mask_slice = m[start:end] if m is not None else None
                if spec.kind == "float64":
                    batch[name] = np.asarray(data_slice, dtype=np.float64)
                elif spec.kind == "int64_masked":
                    if mask_slice is None:
                        raise RuntimeError(f"cache manifest missing mask for {name}")
                    batch[name] = pd.array(
                        np.asarray(data_slice, dtype=np.int64),
                        mask=np.asarray(mask_slice, dtype=np.bool_),
                        dtype="Int64",
                    )
                elif spec.kind == "bool_masked":
                    if mask_slice is None:
                        raise RuntimeError(f"cache manifest missing mask for {name}")
                    batch[name] = pd.array(
                        np.asarray(data_slice, dtype=np.uint8).astype(bool, copy=False),
                        mask=np.asarray(mask_slice, dtype=np.bool_),
                        dtype="boolean",
                    )
                else:
                    raise RuntimeError(f"unknown cached column kind: {spec.kind}")

            df = pd.DataFrame(batch)
            df.index = pd.Index(np.asarray(row_index[start:end], dtype=np.int64), name=None)
            yield df
            start = end

    def materialize_numeric_cache(
        self,
        *,
        accessor: "DatasetAccessor",
        columns: list[str] | None = None,
        batch_size: int = 100_000,
        force: bool = False,
    ) -> dict[str, Any]:
        """Create or update a numeric-only cache for the given dataset accessor.

        Returns the manifest dict. This is deterministic for a given dataset and config.
        """

        if self.exists() and not force:
            return self.load_manifest()

        info = accessor.info()
        total_rows = int(info.get("rows") or 0)
        if total_rows <= 0:
            raise RuntimeError("dataset has zero rows; refusing to build cache")

        # Decide columns to cache without reading "all columns" into memory.
        inferred = info.get("inferred_types") or {}
        requested = list(columns) if columns is not None else sorted([str(k) for k in inferred.keys()])

        # Only cache numeric-ish columns. Strings/objects are intentionally skipped.
        cache_candidates: list[str] = []
        for name in requested:
            dt = str(inferred.get(name) or "").lower()
            if any(tok in dt for tok in ("int", "float", "double", "number", "bool")):
                cache_candidates.append(name)

        if not cache_candidates:
            raise RuntimeError("no numeric-ish columns to cache")

        # Use one small batch to finalize per-column cache encoding decisions.
        first_batch = None
        for b in accessor.iter_batches(columns=cache_candidates, batch_size=min(int(batch_size), total_rows)):
            first_batch = b
            break
        if first_batch is None:
            raise RuntimeError("failed to read first batch")

        cache_cols = [c for c in cache_candidates if c in first_batch.columns]

        self.root.mkdir(parents=True, exist_ok=True)
        row_index_mm = np.lib.format.open_memmap(
            self.root / "row_index.npy", mode="w+", dtype=np.int64, shape=(total_rows,)
        )

        # Pre-create memmaps for each cached column.
        specs: dict[str, CachedColumnSpec] = {}
        data_mms: dict[str, np.memmap] = {}
        mask_mms: dict[str, np.memmap | None] = {}

        for name in cache_cols:
            s = first_batch[name]
            if pd.api.types.is_bool_dtype(s) or str(s.dtype).lower() in {"boolean"}:
                data_file = f"{name}.u1.npy"
                mask_file = f"{name}.mask.npy"
                data_mms[name] = np.lib.format.open_memmap(
                    self.root / data_file, mode="w+", dtype=np.uint8, shape=(total_rows,)
                )
                mask_mms[name] = np.lib.format.open_memmap(
                    self.root / mask_file, mode="w+", dtype=np.bool_, shape=(total_rows,)
                )
                specs[name] = CachedColumnSpec(name=name, kind="bool_masked", data_file=data_file, mask_file=mask_file)
            elif pd.api.types.is_integer_dtype(s) or str(s.dtype).lower().startswith("int"):
                data_file = f"{name}.i8.npy"
                mask_file = f"{name}.mask.npy"
                data_mms[name] = np.lib.format.open_memmap(
                    self.root / data_file, mode="w+", dtype=np.int64, shape=(total_rows,)
                )
                mask_mms[name] = np.lib.format.open_memmap(
                    self.root / mask_file, mode="w+", dtype=np.bool_, shape=(total_rows,)
                )
                specs[name] = CachedColumnSpec(name=name, kind="int64_masked", data_file=data_file, mask_file=mask_file)
            else:
                data_file = f"{name}.f8.npy"
                data_mms[name] = np.lib.format.open_memmap(
                    self.root / data_file, mode="w+", dtype=np.float64, shape=(total_rows,)
                )
                mask_mms[name] = None
                specs[name] = CachedColumnSpec(name=name, kind="float64", data_file=data_file, mask_file=None)

        # Write batches.
        pos = 0
        for df in accessor.iter_batches(columns=cache_cols, batch_size=int(batch_size)):
            n = int(len(df))
            if n <= 0:
                continue
            if pos + n > total_rows:
                raise RuntimeError("dataset iter_batches yielded more rows than expected")
            idx = df.index.to_numpy(dtype=np.int64, copy=False)
            row_index_mm[pos : pos + n] = idx

            for name in cache_cols:
                s = df[name]
                spec = specs[name]
                if spec.kind == "float64":
                    vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64, copy=False)
                    data_mms[name][pos : pos + n] = vals
                elif spec.kind == "int64_masked":
                    assert mask_mms[name] is not None
                    arr = pd.array(s, dtype="Int64")
                    data_mms[name][pos : pos + n] = arr.to_numpy(dtype=np.int64, na_value=0)
                    mask_mms[name][pos : pos + n] = pd.isna(arr).to_numpy(dtype=np.bool_)
                elif spec.kind == "bool_masked":
                    assert mask_mms[name] is not None
                    arr = pd.array(s, dtype="boolean")
                    # Note: for boolean extension arrays, to_numpy(bool) would coerce missing
                    # to False, so we keep a mask explicitly.
                    data_mms[name][pos : pos + n] = arr.to_numpy(dtype=np.uint8, na_value=0)
                    mask_mms[name][pos : pos + n] = pd.isna(arr).to_numpy(dtype=np.bool_)
                else:
                    raise RuntimeError(f"unsupported cache kind: {spec.kind}")

            pos += n

        if pos != total_rows:
            raise RuntimeError(f"cache build incomplete: expected {total_rows} rows, wrote {pos}")

        manifest = {
            "cache_key": self.cache_key.key,
            "dataset_version_id": self.cache_key.dataset_version_id,
            "data_hash": self.cache_key.data_hash,
            "schema_hash": self.cache_key.schema_hash,
            "row_count": total_rows,
            "column_count_cached": len(cache_cols),
            "columns": [
                {
                    "name": specs[name].name,
                    "kind": specs[name].kind,
                    "data_file": specs[name].data_file,
                    "mask_file": specs[name].mask_file,
                }
                for name in sorted(specs.keys())
            ],
            "format": "npy-v1",
        }
        self._write_manifest(manifest)
        return manifest
