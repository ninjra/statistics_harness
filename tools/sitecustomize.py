from __future__ import annotations

import errno
import os
import shutil


if os.environ.get("STAT_HARNESS_SAFE_RENAME") == "1":
    _ORIG_RENAME = os.rename
    _ORIG_REPLACE = os.replace

    def _copy_fallback(src: str, dst: str) -> None:
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            shutil.rmtree(src)
        else:
            shutil.copy2(src, dst)
            os.unlink(src)

    def _safe_rename(src: str, dst: str) -> None:
        try:
            _ORIG_RENAME(src, dst)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            _copy_fallback(src, dst)

    def _safe_replace(src: str, dst: str) -> None:
        try:
            _ORIG_REPLACE(src, dst)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            _copy_fallback(src, dst)

    os.rename = _safe_rename
    os.replace = _safe_replace
