from statistic_harness.core.utils import file_size_limit, max_upload_bytes


def test_max_upload_bytes_env(monkeypatch):
    monkeypatch.delenv("STAT_HARNESS_MAX_UPLOAD_BYTES", raising=False)
    assert max_upload_bytes() is None

    monkeypatch.setenv("STAT_HARNESS_MAX_UPLOAD_BYTES", "0")
    assert max_upload_bytes() is None

    monkeypatch.setenv("STAT_HARNESS_MAX_UPLOAD_BYTES", "not-a-number")
    assert max_upload_bytes() is None

    monkeypatch.setenv("STAT_HARNESS_MAX_UPLOAD_BYTES", "100")
    assert max_upload_bytes() == 100


def test_file_size_limit(tmp_path):
    path = tmp_path / "sample.bin"
    path.write_bytes(b"x" * 10)
    file_size_limit(path, 10)
    try:
        file_size_limit(path, 9)
    except ValueError:
        return
    raise AssertionError("Expected size limit to raise")
