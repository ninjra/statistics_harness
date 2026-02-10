from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .utils import ensure_dir, file_sha256


@dataclass(frozen=True)
class CasPaths:
    blobs_dir: Path
    quarantine_dir: Path


def cas_paths(appdata_dir: Path) -> CasPaths:
    base = appdata_dir / "uploads"
    return CasPaths(
        blobs_dir=base / "blobs",
        quarantine_dir=base / "_quarantine",
    )


def blob_path(appdata_dir: Path, sha256_hex: str) -> Path:
    return cas_paths(appdata_dir).blobs_dir / sha256_hex


def quarantine_dir(appdata_dir: Path, upload_id: str) -> Path:
    return cas_paths(appdata_dir).quarantine_dir / upload_id


def verify_sha256_on_disk(path: Path, expected_sha256_hex: str) -> None:
    actual = file_sha256(path)
    if actual != expected_sha256_hex:
        raise ValueError(
            f"SHA256 mismatch for {path.name}: expected={expected_sha256_hex} actual={actual}"
        )


def promote_quarantine_file(
    appdata_dir: Path,
    upload_id: str,
    filename: str,
    sha256_hex: str,
    *,
    verify_on_write: bool = True,
) -> Path:
    """Move a file from quarantine to CAS blob storage atomically (within a filesystem).

    Returns the final blob path.
    """

    qdir = quarantine_dir(appdata_dir, upload_id)
    src = qdir / filename
    if not src.exists():
        raise FileNotFoundError(str(src))
    if verify_on_write:
        verify_sha256_on_disk(src, sha256_hex)

    paths = cas_paths(appdata_dir)
    ensure_dir(paths.blobs_dir)
    dest = paths.blobs_dir / sha256_hex
    if dest.exists():
        # If it already exists, verify it matches, then discard quarantine.
        verify_sha256_on_disk(dest, sha256_hex)
        try:
            src.unlink()
        finally:
            _rmtree_best_effort(qdir)
        return dest

    tmp_dest = dest.with_suffix(dest.suffix + ".tmp")
    if tmp_dest.exists():
        tmp_dest.unlink()
    # Stage into a temp name, then atomic replace to final CAS path.
    os.replace(src, tmp_dest)
    os.replace(tmp_dest, dest)
    _rmtree_best_effort(qdir)
    return dest


def _rmtree_best_effort(path: Path) -> None:
    # Minimal local helper to avoid importing shutil in hot paths.
    try:
        if not path.exists():
            return
        for child in path.iterdir():
            try:
                if child.is_dir():
                    _rmtree_best_effort(child)
                else:
                    child.unlink()
            except Exception:
                pass
        try:
            path.rmdir()
        except Exception:
            pass
    except Exception:
        pass

