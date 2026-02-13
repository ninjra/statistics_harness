from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from statistic_harness.core.storage import Storage
from statistic_harness.core.upload_cas import (
    blob_path,
    promote_quarantine_file,
    quarantine_dir,
)
from statistic_harness.core.utils import now_iso


def test_promote_quarantine_file_verifies_and_moves(tmp_path: Path) -> None:
    appdata = tmp_path / "appdata"
    upload_id = "u1"
    filename = "data.csv"
    payload = b"a,b\n1,2\n"
    sha = hashlib.sha256(payload).hexdigest()

    qdir = quarantine_dir(appdata, upload_id)
    qdir.mkdir(parents=True, exist_ok=True)
    (qdir / filename).write_bytes(payload)

    dest = promote_quarantine_file(appdata, upload_id, filename, sha, verify_on_write=True)
    assert dest == blob_path(appdata, sha)
    assert dest.exists()
    assert not qdir.exists()


def test_promote_quarantine_file_detects_corruption(tmp_path: Path) -> None:
    appdata = tmp_path / "appdata"
    upload_id = "u1"
    filename = "data.csv"
    payload = b"a,b\n1,2\n"
    sha = hashlib.sha256(payload).hexdigest()

    qdir = quarantine_dir(appdata, upload_id)
    qdir.mkdir(parents=True, exist_ok=True)
    path = qdir / filename
    path.write_bytes(payload)
    # Corrupt after hash is computed.
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 0xFF
    path.write_bytes(bytes(raw))

    with pytest.raises(ValueError):
        promote_quarantine_file(appdata, upload_id, filename, sha, verify_on_write=True)


def test_upload_blob_refcount_increments(tmp_path: Path) -> None:
    db_path = tmp_path / "appdata" / "state.sqlite"
    storage = Storage(db_path)

    sha = "abc123"
    created_at = now_iso()
    storage.create_upload("u1", "file.csv", 10, sha, created_at, verified_at=created_at)
    storage.create_upload("u2", "file.csv", 10, sha, created_at, verified_at=created_at)

    blob = storage.fetch_upload_blob(sha)
    assert blob is not None
    assert int(blob.get("refcount") or 0) == 2
    assert blob.get("verified_at") == created_at

