from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_repo_tests_conftest() -> object:
    path = Path(__file__).resolve().parent / "tests" / "conftest.py"
    spec = importlib.util.spec_from_file_location("statistics_harness_tests_conftest", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load local test conftest at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_mod = _load_repo_tests_conftest()
for _name in dir(_mod):
    if _name.startswith("_"):
        continue
    globals()[_name] = getattr(_mod, _name)

