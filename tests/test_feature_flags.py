from pathlib import Path

from statistic_harness.core.tenancy import get_tenant_context, tenancy_enabled
from statistic_harness.core.utils import auth_enabled, vector_store_enabled


def test_default_flags_disabled(monkeypatch):
    monkeypatch.delenv("STAT_HARNESS_ENABLE_AUTH", raising=False)
    monkeypatch.delenv("STAT_HARNESS_ENABLE_VECTOR_STORE", raising=False)
    monkeypatch.delenv("STAT_HARNESS_ENABLE_TENANCY", raising=False)
    assert not auth_enabled()
    assert vector_store_enabled()
    assert not tenancy_enabled()


def test_auth_flag_enabled(monkeypatch):
    monkeypatch.setenv("STAT_HARNESS_ENABLE_AUTH", "1")
    assert auth_enabled()


def test_vector_store_flag_enabled(monkeypatch):
    monkeypatch.setenv("STAT_HARNESS_ENABLE_VECTOR_STORE", "true")
    assert vector_store_enabled()


def test_tenant_context_paths(monkeypatch, tmp_path):
    monkeypatch.delenv("STAT_HARNESS_ENABLE_TENANCY", raising=False)
    ctx = get_tenant_context("alpha", tmp_path)
    assert ctx.tenant_root == tmp_path

    monkeypatch.setenv("STAT_HARNESS_ENABLE_TENANCY", "1")
    ctx = get_tenant_context("alpha", tmp_path)
    assert ctx.tenant_root == Path(tmp_path) / "tenants" / "alpha"
