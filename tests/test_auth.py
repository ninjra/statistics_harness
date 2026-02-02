from statistic_harness.core.auth import (
    generate_api_key,
    generate_session_token,
    hash_password,
    hash_token,
    verify_password,
)
from statistic_harness.core.storage import Storage
from statistic_harness.core.utils import now_iso


def test_password_hash_roundtrip():
    hashed = hash_password("secret")
    assert verify_password("secret", hashed)
    assert not verify_password("nope", hashed)


def test_auth_storage_records(tmp_path):
    storage = Storage(tmp_path / "state.sqlite")
    user_id = storage.create_user("user@example.com", hash_password("pw"), None, True, now_iso())
    storage.ensure_membership(user_id, "admin", now_iso())

    session_token = generate_session_token()
    storage.create_session(user_id, hash_token(session_token), now_iso(), now_iso())
    session = storage.fetch_session_by_hash(hash_token(session_token))
    assert session is not None
    assert int(session["user_id"]) == user_id

    api_key = generate_api_key()
    key_id = storage.create_api_key(user_id, hash_token(api_key), "cli", now_iso())
    key_row = storage.fetch_api_key_by_hash(hash_token(api_key))
    assert key_row is not None
    assert int(key_row["key_id"]) == key_id
