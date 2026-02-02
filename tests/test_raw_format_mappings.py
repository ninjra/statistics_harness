import hashlib

from statistic_harness.core.storage import Storage
from statistic_harness.core.utils import json_dumps, now_iso


def test_raw_format_mapping_storage(tmp_path):
    storage = Storage(tmp_path / "state.sqlite")
    format_id = storage.ensure_raw_format(
        fingerprint="format-fp",
        name="Format A",
        created_at=now_iso(),
    )
    template_id = storage.create_template(
        name="Template A",
        fields=[{"name": "x", "dtype": "float"}],
        description=None,
        version=None,
        created_at=now_iso(),
    )
    mapping = {"x": "col_x"}
    mapping_hash = hashlib.sha256(
        json_dumps(mapping).encode("utf-8")
    ).hexdigest()
    storage.add_raw_format_mapping(
        format_id,
        template_id,
        json_dumps(mapping),
        mapping_hash,
        "note",
        now_iso(),
    )
    mappings = storage.list_raw_format_mappings(format_id)
    assert any(m["mapping_hash"] == mapping_hash for m in mappings)
