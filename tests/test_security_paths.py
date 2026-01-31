from pathlib import Path

import pytest

from statistic_harness.core.utils import safe_join


def test_safe_join_blocks_traversal(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    with pytest.raises(ValueError):
        safe_join(base, "..", "secret.txt")
