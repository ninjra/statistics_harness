import pytest

from statistic_harness.core.vector_store import VectorStore


def _make_store(tmp_path, monkeypatch):
    monkeypatch.setenv("STAT_HARNESS_ENABLE_VECTOR_STORE", "1")
    try:
        return VectorStore(tmp_path / "state.sqlite", tenant_id="tenant_a")
    except RuntimeError as exc:
        pytest.skip(str(exc))


def test_vector_store_add_query_delete(tmp_path, monkeypatch):
    store = _make_store(tmp_path, monkeypatch)
    store.add(
        "demo",
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        item_ids=["a", "b"],
        payloads=[{"label": "a"}, {"label": "b"}],
    )
    results = store.query("demo", [1.0, 0.0, 0.0], k=2)
    assert results
    assert results[0]["item_id"] == "a"

    store.delete("demo", ["a"], dimensions=3)
    results = store.query("demo", [1.0, 0.0, 0.0], k=2)
    assert all(row["item_id"] != "a" for row in results)


def test_vector_store_deterministic_order(tmp_path, monkeypatch):
    store = _make_store(tmp_path, monkeypatch)
    store.add(
        "ties",
        [[1.0, 0.0], [1.0, 0.0]],
        item_ids=["b", "a"],
        payloads=[None, None],
    )
    results = store.query("ties", [1.0, 0.0], k=2)
    assert [row["item_id"] for row in results] == ["a", "b"]


def test_vector_store_as_of_filter(tmp_path, monkeypatch):
    store = _make_store(tmp_path, monkeypatch)
    import statistic_harness.core.vector_store as vector_store

    monkeypatch.setattr(
        vector_store, "now_iso", lambda: "2026-02-02T00:00:00+00:00"
    )
    store.add("asof", [[1.0, 0.0]], item_ids=["a"], payloads=[None])
    past = store.query(
        "asof", [1.0, 0.0], k=1, as_of="2026-02-01T00:00:00+00:00"
    )
    future = store.query(
        "asof", [1.0, 0.0], k=1, as_of="2026-02-03T00:00:00+00:00"
    )
    assert past == []
    assert [row["item_id"] for row in future] == ["a"]


def test_vector_store_tenant_isolation(tmp_path, monkeypatch):
    monkeypatch.setenv("STAT_HARNESS_ENABLE_VECTOR_STORE", "1")
    try:
        store_a = VectorStore(tmp_path / "state.sqlite", tenant_id="tenant_a")
        store_b = VectorStore(tmp_path / "state.sqlite", tenant_id="tenant_b")
    except RuntimeError as exc:
        pytest.skip(str(exc))

    store_a.add("demo", [[1.0, 0.0]], item_ids=["a"], payloads=[None])
    store_b.add("demo", [[1.0, 0.0]], item_ids=["b"], payloads=[None])

    results_a = store_a.query("demo", [1.0, 0.0], k=2)
    results_b = store_b.query("demo", [1.0, 0.0], k=2)
    assert [row["item_id"] for row in results_a] == ["a"]
    assert [row["item_id"] for row in results_b] == ["b"]
