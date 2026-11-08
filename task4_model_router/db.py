"""On-disk SQLite storage for tenants and token usage."""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS tenants (
    key_hash          TEXT PRIMARY KEY,
    tenant_id         TEXT NOT NULL UNIQUE,
    tokens_per_minute INTEGER NOT NULL CHECK (tokens_per_minute > 0),
    created_at        INTEGER NOT NULL
);

-- One row per tenant per second. The sliding window is the sum of the rows
-- inside it, and rows that fall out of the window are deleted, so a tenant
-- never has more than window_seconds rows no matter how busy it is.
CREATE TABLE IF NOT EXISTS token_usage (
    tenant_id TEXT    NOT NULL,
    bucket    INTEGER NOT NULL,
    tokens    INTEGER NOT NULL CHECK (tokens >= 0),
    PRIMARY KEY (tenant_id, bucket)
) WITHOUT ROWID;
"""


def connect(path: str | Path) -> sqlite3.Connection:
    """Open the database with settings suited to many small concurrent writes."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10.0, isolation_level=None, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")  # readers never block the writer
    conn.execute("PRAGMA synchronous=NORMAL")  # durable at checkpoints, fast commits
    conn.execute("PRAGMA busy_timeout=10000")  # wait for the write lock instead of failing
    conn.executescript(SCHEMA)
    return conn


class Database:
    """One SQLite connection plus the lock that serializes its use.

    A ``sqlite3.Connection`` must not be used by two threads at once, and the
    async handlers call into SQLite from a thread pool. Every component that
    touches the database shares this object and holds ``lock`` while it does.
    Cross-process safety comes from SQLite's own locking (``BEGIN IMMEDIATE``).
    """

    def __init__(self, path: str | Path) -> None:
        self.conn = connect(path)
        self.lock = threading.Lock()

    def close(self) -> None:
        with self.lock:
            self.conn.close()


def hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Tenant:
    tenant_id: str
    tokens_per_minute: int


class TenantStore:
    def __init__(self, db: Database) -> None:
        self._db = db

    def create(self, tenant_id: str, tokens_per_minute: int, api_key: str | None = None) -> str:
        """Register a tenant and return its API key. Only the key's hash is stored."""
        api_key = api_key or f"sk-tenant-{secrets.token_urlsafe(24)}"
        with self._db.lock:
            self._db.conn.execute(
                "INSERT INTO tenants (key_hash, tenant_id, tokens_per_minute, created_at) "
                "VALUES (?, ?, ?, ?)",
                (hash_api_key(api_key), tenant_id, tokens_per_minute, int(time.time())),
            )
        return api_key

    def find_by_key(self, api_key: str) -> Tenant | None:
        with self._db.lock:
            row = self._db.conn.execute(
                "SELECT tenant_id, tokens_per_minute FROM tenants WHERE key_hash = ?",
                (hash_api_key(api_key),),
            ).fetchone()
        return Tenant(tenant_id=row[0], tokens_per_minute=row[1]) if row else None
