from statistic_harness.ui.server import _apply_security_headers


def test_apply_security_headers_sets_expected_keys():
    headers: dict[str, str] = {}
    _apply_security_headers(headers)
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert "default-src" in headers["Content-Security-Policy"]

