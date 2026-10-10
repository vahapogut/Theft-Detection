"""Local user accounts and revocable, opaque sessions.

Only public account fields leave the store. Passwords are salted PBKDF2 hashes;
session bearer tokens are never persisted in plaintext.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import hmac
import os
from pathlib import Path
import re
import secrets
import sqlite3
import time


PASSWORD_ITERATIONS = 600_000
SESSION_TTL = 8 * 60 * 60
ROLES = frozenset({"admin", "operator", "viewer"})
DEFAULT_AUTH_DB = "data/auth.sqlite3"
_USERNAME = re.compile(r"[a-z0-9_.-]{3,64}\Z")


def _password_hash(password):
    salt = secrets.token_bytes(32)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS)
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${salt.hex()}${digest.hex()}"


def _password_matches(password, encoded):
    try:
        algorithm, rounds, salt, expected = encoded.split("$")
        if algorithm != "pbkdf2_sha256" or int(rounds) < PASSWORD_ITERATIONS:
            return False
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), int(rounds))
        return hmac.compare_digest(digest, bytes.fromhex(expected))
    except (ValueError, TypeError, AttributeError, OverflowError):
        return False


def _validate_password(password):
    if not isinstance(password, str) or not 12 <= len(password) <= 1024:
        raise ValueError("Password must contain between 12 and 1024 characters.")


def _public_user(row):
    return {
        "id": row["id"],
        "username": row["username"],
        "role": row["role"],
        "active": bool(row["active"]),
        "created_at": row["created_at"],
    }


class AuthStore:
    def __init__(self, path=DEFAULT_AUTH_DB, session_ttl=SESSION_TTL):
        self.path = str(path)
        self.session_ttl = session_ttl
        if session_ttl <= 0:
            raise ValueError("Session lifetime must be positive.")
        target = Path(self.path)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(fd)
        with self._connection() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('admin', 'operator', 'viewer')),
                    active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    expires_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS sessions_user ON sessions(user_id);
                CREATE INDEX IF NOT EXISTS sessions_expiry ON sessions(expires_at);
            """)

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialized(self):
        with self._connection() as connection:
            return bool(connection.execute("SELECT 1 FROM users WHERE active = 1 AND role = 'admin' LIMIT 1").fetchone())

    def create_user(self, username, password, role="viewer"):
        username = username.strip().lower() if isinstance(username, str) else ""
        if not _USERNAME.fullmatch(username):
            raise ValueError("Username must be 3–64 characters: letters, numbers, dot, dash or underscore.")
        if role not in ROLES:
            raise ValueError("Role must be admin, operator or viewer.")
        _validate_password(password)
        encoded = _password_hash(password)
        try:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                if role != "admin" and not connection.execute("SELECT 1 FROM users LIMIT 1").fetchone():
                    raise ValueError("The first account must be an administrator.")
                cursor = connection.execute(
                    "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
                    (username, encoded, role, datetime.now(timezone.utc).isoformat()),
                )
                return _public_user(connection.execute("SELECT * FROM users WHERE id = ?", (cursor.lastrowid,)).fetchone())
        except sqlite3.IntegrityError as exc:
            raise ValueError("Username already exists.") from exc

    def list_users(self):
        with self._connection() as connection:
            return [_public_user(row) for row in connection.execute("SELECT * FROM users ORDER BY username")]

    def update_user(self, user_id, *, password=None, role=None, active=None):
        if role is not None and role not in ROLES:
            raise ValueError("Role must be admin, operator or viewer.")
        if active is not None and not isinstance(active, bool):
            raise ValueError("Active must be a boolean.")
        encoded = None
        if password is not None:
            _validate_password(password)
            encoded = _password_hash(password)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            if current is None:
                raise KeyError(user_id)
            next_role = current["role"] if role is None else role
            next_active = bool(current["active"]) if active is None else active
            if current["role"] == "admin" and current["active"] and (next_role != "admin" or not next_active):
                admins = connection.execute("SELECT COUNT(*) FROM users WHERE role = 'admin' AND active = 1").fetchone()[0]
                if admins <= 1:
                    raise ValueError("The last active administrator cannot be disabled or demoted.")
            changed = encoded is not None or next_role != current["role"] or next_active != bool(current["active"])
            connection.execute(
                "UPDATE users SET password_hash = ?, role = ?, active = ? WHERE id = ?",
                (encoded or current["password_hash"], next_role, int(next_active), user_id),
            )
            if changed:
                connection.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            return _public_user(connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone())

    def authenticate(self, username, password):
        if not isinstance(username, str) or not isinstance(password, str) or len(password) > 1024:
            return None
        with self._connection() as connection:
            user = connection.execute("SELECT * FROM users WHERE username = ?", (username.strip().lower(),)).fetchone()
        # Equal-cost work also for an unknown account; do not reveal account existence.
        encoded = user["password_hash"] if user else f"pbkdf2_sha256${PASSWORD_ITERATIONS}${'00' * 32}${'00' * 32}"
        matches = _password_matches(password, encoded)
        if not user or not matches or not user["active"]:
            return None
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode("ascii")).hexdigest()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            # An administrator may have changed the account while PBKDF2 ran.
            current = connection.execute("SELECT * FROM users WHERE id = ?", (user["id"],)).fetchone()
            if not current or not current["active"] or current["password_hash"] != encoded or current["role"] != user["role"]:
                return None
            now = time.time()
            connection.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
            connection.execute("INSERT INTO sessions VALUES (?, ?, ?)", (token_hash, user["id"], now + self.session_ttl))
        return token

    def get_session(self, token):
        if not isinstance(token, str) or not token or len(token) > 256:
            return None
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with self._connection() as connection:
            row = connection.execute(
                "SELECT users.* FROM sessions JOIN users ON users.id = sessions.user_id "
                "WHERE token_hash = ? AND expires_at > ? AND users.active = 1",
                (token_hash, time.time()),
            ).fetchone()
            return _public_user(row) if row else None

    def revoke_session(self, token):
        if isinstance(token, str) and token:
            with self._connection() as connection:
                connection.execute("DELETE FROM sessions WHERE token_hash = ?", (hashlib.sha256(token.encode("utf-8")).hexdigest(),))
