"""Token-aware sliding-window rate limiter backed by on-disk SQLite.

Algorithm (sliding log, aggregated per second)
-----------------------------------------------
Usage is recorded in one row per tenant per second. A request is admitted if
the tokens used in the last ``window`` seconds plus the tokens it needs fit
under the tenant's limit. Admission and recording happen in one
``BEGIN IMMEDIATE`` transaction:

    evict rows older than the window -> sum the window -> compare -> record

``BEGIN IMMEDIATE`` takes SQLite's write lock up front, so the
check-then-write is atomic across threads *and* processes: two gateway
workers sharing the database file can never both spend the last tokens.

Compared with a fixed window this has no boundary burst (a tenant cannot use
2x the limit across a minute boundary), and compared with the two-bucket
approximation it is exact to one second. Storage is bounded at
``window`` rows per tenant.

Tokens are reserved before the call using an estimate, then reconciled with
the provider's reported usage, or released if no provider answered.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

from .db import Database
from .errors import RateLimitExceededError, RequestTooLargeError

Clock = Callable[[], float]


@dataclass(frozen=True)
class Reservation:
    tenant_id: str
    bucket: int
    tokens: int
    limit: int
    remaining: int


class SlidingWindowRateLimiter:
    def __init__(self, db: Database, *, window_seconds: int = 60, clock: Clock = time.time) -> None:
        # The shared lock serializes this process; BEGIN IMMEDIATE serializes
        # writers across processes using the same database file.
        self._conn = db.conn
        self._lock = db.lock
        self._window = window_seconds
        self._clock = clock

    def reserve(self, tenant_id: str, tokens: int, limit: int) -> Reservation:
        if tokens > limit:
            raise RequestTooLargeError(requested=tokens, limit=limit)

        now = self._clock()
        bucket = int(now)
        oldest_live = bucket - self._window + 1

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "DELETE FROM token_usage WHERE tenant_id = ? AND bucket < ?", (tenant_id, oldest_live)
                )
                rows = self._conn.execute(
                    "SELECT bucket, tokens FROM token_usage WHERE tenant_id = ? ORDER BY bucket",
                    (tenant_id,),
                ).fetchall()
                used = sum(t for _, t in rows)

                if used + tokens > limit:
                    self._conn.execute("COMMIT")  # keep the eviction
                    raise RateLimitExceededError(
                        retry_after=self._retry_after(rows, used, tokens, limit, now),
                        limit=limit,
                        remaining=limit - used,
                    )

                self._conn.execute(
                    "INSERT INTO token_usage (tenant_id, bucket, tokens) VALUES (?, ?, ?) "
                    "ON CONFLICT (tenant_id, bucket) DO UPDATE SET tokens = tokens + excluded.tokens",
                    (tenant_id, bucket, tokens),
                )
                self._conn.execute("COMMIT")
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.execute("ROLLBACK")
                raise

        return Reservation(tenant_id, bucket, tokens, limit, remaining=limit - used - tokens)

    def reconcile(self, reservation: Reservation, actual_tokens: int) -> None:
        """Replace the estimate with real usage. Releasing is ``actual_tokens=0``."""
        delta = max(actual_tokens, 0) - reservation.tokens
        if delta == 0:
            return
        with self._lock:
            # The bucket may already have left the window and been evicted;
            # then there is nothing left to correct.
            self._conn.execute(
                "UPDATE token_usage SET tokens = MAX(tokens + ?, 0) WHERE tenant_id = ? AND bucket = ?",
                (delta, reservation.tenant_id, reservation.bucket),
            )

    def release(self, reservation: Reservation) -> None:
        self.reconcile(reservation, 0)

    def usage(self, tenant_id: str) -> int:
        oldest_live = int(self._clock()) - self._window + 1
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(tokens), 0) FROM token_usage WHERE tenant_id = ? AND bucket >= ?",
                (tenant_id, oldest_live),
            ).fetchone()
        return int(row[0])

    def _retry_after(
        self, rows: list[tuple[int, int]], used: int, tokens: int, limit: int, now: float
    ) -> int:
        """Seconds until enough old usage leaves the window for this request to fit."""
        must_free = used + tokens - limit
        freed = 0
        for bucket, bucket_tokens in rows:
            freed += bucket_tokens
            if freed >= must_free:
                # The bucket stops counting once it is a full window old.
                return max(1, math.ceil(bucket + self._window - now))
        return self._window
