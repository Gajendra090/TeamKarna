"""Official Google OAuth 2.0 (Authorization Code + PKCE + ID Token) authentication for KNOWLEDGE X."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "knowledge_x_history.db"
_AUTH_WAL_LOCK = threading.Lock()
_AUTH_WAL_CONFIGURED_PATHS: set[str] = set()
GOOGLE_AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
DEFAULT_REDIRECT_URI = "http://localhost:8501"
SESSION_COOKIE_NAME = "kx_auth_session"
SESSION_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days
OAUTH_STATE_TTL_SECONDS = 10 * 60  # 10 minutes
_PLACEHOLDER_VALUES = {
    "",
    "your_google_client_id_here",
    "your_google_client_secret_here",
    "your_random_cookie_secret_here",
}


class AuthenticationError(Exception):
    """Raised when Google OAuth configuration, callback validation, or token checks fail."""


@contextmanager
def _connect(db_path: Path | str = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_auth_db(db_path: Path | str = DEFAULT_DB_PATH) -> None:
    """Ensure SQLite tables for single-use OAuth states, sessions, and signing key exist."""
    resolved = str(Path(db_path).resolve())
    with _connect(db_path) as conn:
        with _AUTH_WAL_LOCK:
            if resolved not in _AUTH_WAL_CONFIGURED_PATHS:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                _AUTH_WAL_CONFIGURED_PATHS.add(resolved)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS oauth_states (
                state_hash TEXT PRIMARY KEY,
                code_verifier TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS auth_sessions (
                token_hash TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                email TEXT NOT NULL,
                name TEXT NOT NULL,
                picture TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS app_settings (
                setting_key TEXT PRIMARY KEY,
                setting_value TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        conn.commit()


def _sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_streamlit_secrets() -> dict[str, Any]:
    """Safely read Streamlit secrets if .streamlit/secrets.toml is present."""
    try:
        import streamlit as st

        secrets_obj = getattr(st, "secrets", None)
        if secrets_obj is None:
            return {}
        for section in ("google_oauth", "auth"):
            if section in secrets_obj:
                return dict(secrets_obj[section])
    except Exception:
        return {}
    return {}


def load_google_oauth_config(
    env_path: Path | str = ".env",
    db_path: Path | str = DEFAULT_DB_PATH,
) -> dict[str, str]:
    """Load Google OAuth credentials from environment/.env or Streamlit secrets."""
    path = Path(env_path)
    if path.exists():
        load_dotenv(dotenv_path=path, override=True)
    else:
        load_dotenv(override=False)

    st_sec = _read_streamlit_secrets()

    def _clean(val: Any) -> str:
        s = str(val or "").strip().strip('"').strip("'")
        return "" if s in _PLACEHOLDER_VALUES else s

    client_id = _clean(os.getenv("GOOGLE_CLIENT_ID") or st_sec.get("client_id"))
    client_secret = _clean(os.getenv("GOOGLE_CLIENT_SECRET") or st_sec.get("client_secret"))
    redirect_uri = (
        _clean(os.getenv("GOOGLE_REDIRECT_URI") or st_sec.get("redirect_uri"))
        or DEFAULT_REDIRECT_URI
    )
    cookie_secret = _clean(os.getenv("AUTH_COOKIE_SECRET") or st_sec.get("cookie_secret"))

    if not cookie_secret:
        cookie_secret = _get_or_create_signing_secret(db_path=db_path)

    return {
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "cookie_secret": cookie_secret,
    }


def _get_or_create_signing_secret(db_path: Path | str = DEFAULT_DB_PATH) -> str:
    """Persist a generated random cookie-signing key locally if not provided in env/secrets."""
    init_auth_db(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT setting_value FROM app_settings WHERE setting_key = 'auth_cookie_secret'"
        ).fetchone()
        if row and row["setting_value"]:
            return str(row["setting_value"])
        generated = secrets.token_urlsafe(48)
        conn.execute(
            "INSERT OR REPLACE INTO app_settings (setting_key, setting_value) VALUES (?, ?)",
            ("auth_cookie_secret", generated),
        )
        conn.commit()
        return generated


def is_google_oauth_configured(env_path: Path | str = ".env") -> tuple[bool, str]:
    """Return whether Google OAuth Client ID and Client Secret are configured."""
    cfg = load_google_oauth_config(env_path=env_path)
    if cfg["client_id"] and cfg["client_secret"]:
        return True, "Google OAuth configured"
    missing = []
    if not cfg["client_id"]:
        missing.append("GOOGLE_CLIENT_ID")
    if not cfg["client_secret"]:
        missing.append("GOOGLE_CLIENT_SECRET")
    return False, f"Missing {', '.join(missing)} in .env or Streamlit secrets"


def _pkce_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def create_google_authorization_url(
    env_path: Path | str = ".env",
    db_path: Path | str = DEFAULT_DB_PATH,
) -> tuple[str, str]:
    """Generate a Google OAuth 2.0 authorization URL with PKCE S256 and single-use CSRF state."""
    cfg = load_google_oauth_config(env_path=env_path, db_path=db_path)
    if not cfg["client_id"] or not cfg["client_secret"]:
        raise AuthenticationError(
            "Google OAuth credentials are not configured. Set GOOGLE_CLIENT_ID and "
            "GOOGLE_CLIENT_SECRET in .env or .streamlit/secrets.toml."
        )

    init_auth_db(db_path)
    state = secrets.token_urlsafe(32)
    code_verifier = secrets.token_urlsafe(64)
    now = time.time()

    with _connect(db_path) as conn:
        conn.execute("DELETE FROM oauth_states WHERE expires_at < ?", (now,))
        conn.execute(
            """
            INSERT INTO oauth_states (state_hash, code_verifier, created_at, expires_at)
            VALUES (?, ?, ?, ?)
            """,
            (_sha256_hex(state), code_verifier, now, now + OAUTH_STATE_TTL_SECONDS),
        )
        conn.commit()

    params = {
        "client_id": cfg["client_id"],
        "redirect_uri": cfg["redirect_uri"],
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "code_challenge": _pkce_challenge(code_verifier),
        "code_challenge_method": "S256",
        "access_type": "online",
        "prompt": "select_account",
    }
    return f"{GOOGLE_AUTH_ENDPOINT}?{urlencode(params)}", state


def _consume_oauth_state(
    state: str,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> str:
    """Validate and consume a single-use OAuth state token, returning its PKCE code_verifier."""
    if not state or not state.strip():
        raise AuthenticationError("Missing OAuth state parameter.")

    init_auth_db(db_path)
    now = time.time()
    state_hash = _sha256_hex(state.strip())

    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT code_verifier, expires_at FROM oauth_states WHERE state_hash = ?",
            (state_hash,),
        ).fetchone()
        conn.execute("DELETE FROM oauth_states WHERE state_hash = ? OR expires_at < ?", (state_hash, now))
        conn.commit()

    if not row:
        raise AuthenticationError(
            "Authentication session expired or invalid (CSRF state mismatch). Please sign in again."
        )
    if float(row["expires_at"]) < now:
        raise AuthenticationError("Google sign-in request timed out. Please try again.")
    return str(row["code_verifier"])


def verify_google_id_token_claims(
    id_token_str: str,
    client_id: str,
) -> dict[str, Any]:
    """Cryptographically verify a Google OIDC ID token and return normalized user claims."""
    if not id_token_str:
        raise AuthenticationError("Google response did not include an ID token.")

    try:
        claims = google_id_token.verify_oauth2_token(
            id_token_str,
            google_requests.Request(),
            audience=client_id,
        )
    except Exception as exc:
        raise AuthenticationError(f"Failed to verify Google identity token: {exc}") from exc

    issuer = str(claims.get("iss", ""))
    if issuer not in ("accounts.google.com", "https://accounts.google.com"):
        raise AuthenticationError("Untrusted identity token issuer.")

    sub = str(claims.get("sub", "")).strip()
    email = str(claims.get("email", "")).strip()
    if not sub or not email:
        raise AuthenticationError("Google identity token is missing required user subject or email.")

    if claims.get("email_verified") is False:
        raise AuthenticationError("Your Google email address is not verified.")

    return {
        "user_id": f"google:{sub}",
        "google_sub": sub,
        "email": email,
        "name": str(claims.get("name") or email.split("@")[0]).strip(),
        "picture": str(claims.get("picture") or "").strip(),
    }


def complete_google_oauth_callback(
    code: str,
    state: str,
    error: str = "",
    error_description: str = "",
    env_path: Path | str = ".env",
    db_path: Path | str = DEFAULT_DB_PATH,
) -> tuple[dict[str, Any], str]:
    """Handle OAuth callback parameters, exchange code, verify ID token, and issue signed session cookie."""
    if error:
        clean_err = error.strip()
        if clean_err == "access_denied":
            raise AuthenticationError("Google sign-in was cancelled.")
        desc = f": {error_description.strip()}" if error_description else ""
        raise AuthenticationError(f"Google sign-in failed ({clean_err}{desc}).")

    if not code or not code.strip():
        raise AuthenticationError("Missing authorization code from Google callback.")

    cfg = load_google_oauth_config(env_path=env_path, db_path=db_path)
    if not cfg["client_id"] or not cfg["client_secret"]:
        raise AuthenticationError("Google OAuth credentials are missing.")

    code_verifier = _consume_oauth_state(state, db_path=db_path)

    payload = {
        "code": code.strip(),
        "client_id": cfg["client_id"],
        "client_secret": cfg["client_secret"],
        "redirect_uri": cfg["redirect_uri"],
        "grant_type": "authorization_code",
        "code_verifier": code_verifier,
    }

    try:
        resp = requests.post(GOOGLE_TOKEN_ENDPOINT, data=payload, timeout=15)
    except requests.exceptions.RequestException as exc:
        raise AuthenticationError(
            f"Network error contacting Google OAuth token endpoint: {exc}"
        ) from exc

    if resp.status_code != 200:
        err_msg = f"HTTP {resp.status_code}"
        try:
            err_json = resp.json()
            err_code = err_json.get("error", "")
            err_desc = err_json.get("error_description", "")
            if err_code or err_desc:
                err_msg = f"{err_code}: {err_desc}".strip(": ")
        except Exception:
            pass
        raise AuthenticationError(f"Google token exchange failed ({err_msg}).")

    token_data = resp.json()
    id_token_str = str(token_data.get("id_token") or "").strip()
    user_info = verify_google_id_token_claims(id_token_str, cfg["client_id"])

    signed_cookie = create_auth_session(
        user_info=user_info,
        cookie_secret=cfg["cookie_secret"],
        db_path=db_path,
    )
    return user_info, signed_cookie


def create_auth_session(
    user_info: dict[str, Any],
    cookie_secret: str,
    ttl_seconds: int = SESSION_TTL_SECONDS,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> str:
    """Create a persistent server-side session record and return a signed cookie token."""
    init_auth_db(db_path)
    raw_token = secrets.token_urlsafe(32)
    token_hash = _sha256_hex(raw_token)
    now = time.time()
    expires_at = now + max(60, ttl_seconds)

    with _connect(db_path) as conn:
        conn.execute("DELETE FROM auth_sessions WHERE expires_at < ?", (now,))
        conn.execute(
            """
            INSERT INTO auth_sessions (
                token_hash, user_id, email, name, picture, created_at, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                token_hash,
                user_info["user_id"],
                user_info["email"],
                user_info["name"],
                user_info.get("picture", ""),
                now,
                expires_at,
            ),
        )
        conn.commit()

    serializer = URLSafeTimedSerializer(cookie_secret, salt="kx-google-auth-v1")
    return serializer.dumps({"tok": raw_token, "uid": user_info["user_id"]})


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _hash_password(password: str) -> str:
    """Hash a password with a per-user random salt using scrypt."""
    salt = secrets.token_bytes(16)
    derived = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=16384,
        r=8,
        p=1,
        dklen=32,
    )
    return f"scrypt${base64.b64encode(salt).decode('ascii')}${base64.b64encode(derived).decode('ascii')}"


def _verify_password(password: str, stored_hash: str) -> bool:
    """Verify a plaintext password against a stored scrypt hash in constant time."""
    try:
        algo, salt_b64, key_b64 = stored_hash.split("$", 2)
        if algo != "scrypt":
            return False
        salt = base64.b64decode(salt_b64.encode("ascii"))
        expected = base64.b64decode(key_b64.encode("ascii"))
        actual = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=16384,
            r=8,
            p=1,
            dklen=len(expected),
        )
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


def register_local_user(
    name: str,
    email: str,
    password: str,
    confirm_password: str,
    env_path: Path | str = ".env",
    db_path: Path | str = DEFAULT_DB_PATH,
) -> tuple[dict[str, Any], str]:
    """Validate inputs, create a new local user account, and issue a signed session cookie."""
    clean_name = (name or "").strip()
    clean_email = (email or "").strip().lower()
    if len(clean_name) < 2:
        raise AuthenticationError("Please enter your full name (at least 2 characters).")
    if not clean_email or not _EMAIL_RE.match(clean_email):
        raise AuthenticationError("Please enter a valid email address.")
    if not password or len(password) < 8:
        raise AuthenticationError("Password must be at least 8 characters long.")
    if password != (confirm_password or ""):
        raise AuthenticationError("Passwords do not match.")

    init_auth_db(db_path)
    now = time.time()
    user_id = f"local:{_sha256_hex(clean_email)[:24]}"
    pw_hash = _hash_password(password)

    with _connect(db_path) as conn:
        existing = conn.execute(
            "SELECT user_id FROM users WHERE LOWER(email) = ?",
            (clean_email,),
        ).fetchone()
        if existing:
            raise AuthenticationError("An account with this email already exists. Please sign in instead.")
        conn.execute(
            """
            INSERT INTO users (user_id, email, name, password_hash, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, clean_email, clean_name, pw_hash, now),
        )
        conn.commit()

    cfg = load_google_oauth_config(env_path=env_path, db_path=db_path)
    user_info = {
        "user_id": user_id,
        "email": clean_email,
        "name": clean_name,
        "picture": "",
    }
    signed_cookie = create_auth_session(
        user_info=user_info,
        cookie_secret=cfg["cookie_secret"],
        db_path=db_path,
    )
    return user_info, signed_cookie


def authenticate_local_user(
    email: str,
    password: str,
    env_path: Path | str = ".env",
    db_path: Path | str = DEFAULT_DB_PATH,
) -> tuple[dict[str, Any], str]:
    """Authenticate an existing local user by email and password, returning user info and cookie."""
    clean_email = (email or "").strip().lower()
    if not clean_email or not _EMAIL_RE.match(clean_email):
        raise AuthenticationError("Please enter a valid email address.")
    if not password:
        raise AuthenticationError("Please enter your password.")

    init_auth_db(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT user_id, email, name, password_hash FROM users WHERE LOWER(email) = ?",
            (clean_email,),
        ).fetchone()

    if not row or not _verify_password(password, str(row["password_hash"])):
        raise AuthenticationError("Invalid email or password.")

    cfg = load_google_oauth_config(env_path=env_path, db_path=db_path)
    user_info = {
        "user_id": str(row["user_id"]),
        "email": str(row["email"]),
        "name": str(row["name"]),
        "picture": "",
    }
    signed_cookie = create_auth_session(
        user_info=user_info,
        cookie_secret=cfg["cookie_secret"],
        db_path=db_path,
    )
    return user_info, signed_cookie


def verify_auth_session_cookie(
    signed_cookie: str,
    env_path: Path | str = ".env",
    db_path: Path | str = DEFAULT_DB_PATH,
) -> dict[str, Any] | None:
    """Validate signed session cookie against SQLite auth_sessions and return user dict or None."""
    if not signed_cookie or not signed_cookie.strip():
        return None

    cfg = load_google_oauth_config(env_path=env_path, db_path=db_path)
    serializer = URLSafeTimedSerializer(cfg["cookie_secret"], salt="kx-google-auth-v1")
    try:
        payload = serializer.loads(signed_cookie.strip(), max_age=SESSION_TTL_SECONDS)
    except (BadSignature, SignatureExpired, Exception):
        return None

    raw_token = str(payload.get("tok") or "")
    uid = str(payload.get("uid") or "")
    if not raw_token or not uid:
        return None

    init_auth_db(db_path)
    now = time.time()
    token_hash = _sha256_hex(raw_token)

    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT user_id, email, name, picture, expires_at FROM auth_sessions WHERE token_hash = ?",
            (token_hash,),
        ).fetchone()
        if not row:
            return None
        if float(row["expires_at"]) < now or str(row["user_id"]) != uid:
            conn.execute("DELETE FROM auth_sessions WHERE token_hash = ?", (token_hash,))
            conn.commit()
            return None

    return {
        "user_id": str(row["user_id"]),
        "email": str(row["email"]),
        "name": str(row["name"]),
        "picture": str(row["picture"]),
    }


def revoke_auth_session_cookie(
    signed_cookie: str,
    env_path: Path | str = ".env",
    db_path: Path | str = DEFAULT_DB_PATH,
) -> None:
    """Revoke a session in SQLite on logout so the token cannot be reused."""
    if not signed_cookie:
        return
    cfg = load_google_oauth_config(env_path=env_path, db_path=db_path)
    serializer = URLSafeTimedSerializer(cfg["cookie_secret"], salt="kx-google-auth-v1")
    try:
        payload = serializer.loads(signed_cookie.strip(), max_age=SESSION_TTL_SECONDS * 2)
    except (BadSignature, SignatureExpired, Exception):
        return

    raw_token = str(payload.get("tok") or "")
    if not raw_token:
        return

    init_auth_db(db_path)
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM auth_sessions WHERE token_hash = ?", (_sha256_hex(raw_token),))
        conn.commit()


def get_user_uploads_dir(base_uploads_dir: Path | str, user_info: dict[str, Any] | None) -> Path:
    """Return an isolated upload directory per authenticated user (or shared guest dir if signed out)."""
    base = Path(base_uploads_dir)
    if user_info and user_info.get("user_id"):
        short_hash = hashlib.sha256(str(user_info["user_id"]).encode("utf-8")).hexdigest()[:16]
        target = base / f"user_{short_hash}"
    else:
        target = base
    target.mkdir(parents=True, exist_ok=True)
    return target


def cookie_sync_html(action: str, value: str = "") -> str:
    """Return minimal same-origin script to set or clear the browser auth session cookie."""
    if action == "set" and value:
        safe_val = json.dumps(value)
        return (
            "<script>"
            "try {"
            f"  window.parent.document.cookie = '{SESSION_COOKIE_NAME}=' + encodeURIComponent({safe_val}) + '; Path=/; Max-Age={SESSION_TTL_SECONDS}; SameSite=Lax';"
            "} catch (e) {}"
            "</script>"
        )
    if action == "clear":
        return (
            "<script>"
            "try {"
            f"  window.parent.document.cookie = '{SESSION_COOKIE_NAME}=; Path=/; Max-Age=0; SameSite=Lax';"
            "} catch (e) {}"
            "</script>"
        )
    return ""
