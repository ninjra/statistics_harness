from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from dataclasses import dataclass
from typing import Any

PBKDF2_ITERATIONS = 200_000
SALT_BYTES = 16
SESSION_TOKEN_PREFIX = "shs_"
API_KEY_PREFIX = "shk_"


@dataclass(frozen=True)
class AuthUser:
    user_id: int
    email: str
    name: str | None
    is_admin: bool
    tenant_id: str


def normalize_email(email: str) -> str:
    return email.strip().lower()


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(SALT_BYTES)
    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    )
    return "pbkdf2_sha256${}${}${}".format(
        PBKDF2_ITERATIONS,
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(dk).decode("ascii"),
    )


def verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, iter_s, salt_b64, hash_b64 = encoded.split("$", 3)
    except ValueError:
        return False
    if scheme != "pbkdf2_sha256":
        return False
    try:
        iterations = int(iter_s)
        salt = base64.b64decode(salt_b64.encode("ascii"))
        expected = base64.b64decode(hash_b64.encode("ascii"))
    except (ValueError, base64.binascii.Error):
        return False
    computed = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, iterations
    )
    return hmac.compare_digest(computed, expected)


def generate_session_token() -> str:
    return SESSION_TOKEN_PREFIX + secrets.token_urlsafe(32)


def generate_api_key() -> str:
    return API_KEY_PREFIX + secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def is_api_key(token: str) -> bool:
    return token.startswith(API_KEY_PREFIX)


def is_session_token(token: str) -> bool:
    return token.startswith(SESSION_TOKEN_PREFIX)


def auth_user_from_row(row: dict[str, Any]) -> AuthUser:
    return AuthUser(
        user_id=int(row["user_id"]),
        email=str(row["email"]),
        name=row.get("name"),
        is_admin=bool(row.get("is_admin")),
        tenant_id=str(row.get("tenant_id") or ""),
    )
