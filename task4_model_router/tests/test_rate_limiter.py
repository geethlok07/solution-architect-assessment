from __future__ import annotations

import multiprocessing
import threading
from pathlib import Path
from typing import Any

import pytest

from task4_model_router.db import Database, TenantStore, connect, hash_api_key
from task4_model_router.errors import RateLimitExceededError, RequestTooLargeError
from task4_model_router.rate_limiter import SlidingWindowRateLimiter

from .conftest import FakeClock

LIMIT = 50_000


@pytest.fixture
def limiter(database: Database, clock: FakeClock) -> SlidingWindowRateLimiter:
    return SlidingWindowRateLimiter(database, window_seconds=60, clock=clock)


def _rows(db: Database, tenant: str) -> int:
    return db.conn.execute("SELECT COUNT(*) FROM token_usage WHERE tenant_id = ?", (tenant,)).fetchone()[0]


def test_admits_up_to_the_limit_then_rejects(limiter: SlidingWindowRateLimiter) -> None:
    for _ in range(5):
        limiter.reserve("acme", 10_000, LIMIT)

    with pytest.raises(RateLimitExceededError) as info:
        limiter.reserve("acme", 1, LIMIT)

    assert info.value.remaining == 0
    assert info.value.headers["Retry-After"] == "60"
    assert limiter.usage("acme") == LIMIT


def test_reservation_reports_remaining_tokens(limiter: SlidingWindowRateLimiter) -> None:
    first = limiter.reserve("acme", 12_000, LIMIT)
    second = limiter.reserve("acme", 8_000, LIMIT)
    assert (first.remaining, second.remaining) == (38_000, 30_000)


def test_window_slides_rather_than_resetting(limiter: SlidingWindowRateLimiter, clock: FakeClock) -> None:
    limiter.reserve("acme", 30_000, LIMIT)  # t = 0
    clock.advance(30)
    limiter.reserve("acme", 20_000, LIMIT)  # t = 30, window now full

    clock.advance(29)  # t = 59: the t=0 usage still counts
    with pytest.raises(RateLimitExceededError) as info:
        limiter.reserve("acme", 1, LIMIT)
    assert info.value.retry_after == 1

    clock.advance(1)  # t = 60: the t=0 bucket has left the window
    limiter.reserve("acme", 30_000, LIMIT)
    with pytest.raises(RateLimitExceededError) as info:
        limiter.reserve("acme", 1, LIMIT)
    assert info.value.retry_after == 30  # waits for the t=30 bucket


def test_no_double_spend_across_a_minute_boundary(
    limiter: SlidingWindowRateLimiter, clock: FakeClock
) -> None:
    """A fixed window would allow 100k tokens in two seconds across a boundary."""
    clock.now = 1_700_000_059.5
    limiter.reserve("acme", LIMIT, LIMIT)
    clock.advance(1)
    with pytest.raises(RateLimitExceededError):
        limiter.reserve("acme", 1, LIMIT)


def test_retry_after_is_when_enough_tokens_expire(
    limiter: SlidingWindowRateLimiter, clock: FakeClock
) -> None:
    limiter.reserve("acme", 10_000, LIMIT)  # t = 0
    clock.advance(10)
    limiter.reserve("acme", 10_000, LIMIT)  # t = 10
    clock.advance(10)
    limiter.reserve("acme", 30_000, LIMIT)  # t = 20

    clock.advance(5)  # t = 25, request for 15k needs both older buckets gone
    with pytest.raises(RateLimitExceededError) as info:
        limiter.reserve("acme", 15_000, LIMIT)
    assert info.value.retry_after == 45  # second bucket (t=10) expires at t=70


def test_expired_usage_is_evicted_from_disk(
    limiter: SlidingWindowRateLimiter, database: Database, clock: FakeClock
) -> None:
    for _ in range(120):
        limiter.reserve("acme", 100, LIMIT)
        clock.advance(1)
    assert _rows(database, "acme") == 60  # bounded by window length, not traffic

    clock.advance(61)
    limiter.reserve("acme", 1, LIMIT)
    assert _rows(database, "acme") == 1


def test_tenants_are_isolated(limiter: SlidingWindowRateLimiter) -> None:
    limiter.reserve("acme", LIMIT, LIMIT)
    limiter.reserve("globex", LIMIT, LIMIT)
    with pytest.raises(RateLimitExceededError):
        limiter.reserve("acme", 1, LIMIT)


def test_request_larger_than_limit_is_rejected_outright(limiter: SlidingWindowRateLimiter) -> None:
    with pytest.raises(RequestTooLargeError):
        limiter.reserve("acme", LIMIT + 1, LIMIT)
    assert limiter.usage("acme") == 0


def test_reconcile_replaces_estimate_with_actual_usage(limiter: SlidingWindowRateLimiter) -> None:
    reservation = limiter.reserve("acme", 4_000, LIMIT)
    limiter.reconcile(reservation, 1_250)
    assert limiter.usage("acme") == 1_250

    over = limiter.reserve("acme", 2_000, LIMIT)
    limiter.reconcile(over, 2_600)  # model used more than estimated
    assert limiter.usage("acme") == 3_850


def test_release_returns_tokens(limiter: SlidingWindowRateLimiter) -> None:
    reservation = limiter.reserve("acme", LIMIT, LIMIT)
    limiter.release(reservation)
    limiter.reserve("acme", LIMIT, LIMIT)


def test_reconcile_after_bucket_expired_is_harmless(
    limiter: SlidingWindowRateLimiter, clock: FakeClock
) -> None:
    reservation = limiter.reserve("acme", 1_000, LIMIT)
    clock.advance(120)
    limiter.reserve("acme", 1, LIMIT)  # evicts the old bucket
    limiter.reconcile(reservation, 5_000)
    assert limiter.usage("acme") == 1


def test_state_survives_restart(db_path: Path, clock: FakeClock) -> None:
    first = Database(db_path)
    SlidingWindowRateLimiter(first, clock=clock).reserve("acme", 45_000, LIMIT)
    first.close()

    second = Database(db_path)
    try:
        with pytest.raises(RateLimitExceededError):
            SlidingWindowRateLimiter(second, clock=clock).reserve("acme", 10_000, LIMIT)
    finally:
        second.close()


def test_concurrent_threads_never_overspend(limiter: SlidingWindowRateLimiter) -> None:
    admitted, rejected = [], []
    barrier = threading.Barrier(100)

    def worker() -> None:
        barrier.wait()
        try:
            limiter.reserve("acme", 1_000, LIMIT)
            admitted.append(1)
        except RateLimitExceededError:
            rejected.append(1)

    threads = [threading.Thread(target=worker) for _ in range(100)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert (len(admitted), len(rejected)) == (50, 50)
    assert limiter.usage("acme") == LIMIT


def _process_worker(db_path: str, attempts: int, start: Any, results: Any) -> None:
    db = Database(db_path)
    limiter = SlidingWindowRateLimiter(db, clock=lambda: 1_700_000_000.0)
    start.wait()
    admitted = 0
    for _ in range(attempts):
        try:
            limiter.reserve("acme", 1_000, LIMIT)
            admitted += 1
        except RateLimitExceededError:
            pass
    db.close()
    results.put(admitted)


def test_concurrent_processes_sharing_the_file_never_overspend(db_path: Path) -> None:
    connect(db_path).close()
    ctx = multiprocessing.get_context("spawn")
    start, results = ctx.Event(), ctx.Queue()
    processes = [
        ctx.Process(target=_process_worker, args=(str(db_path), 30, start, results)) for _ in range(4)
    ]
    for process in processes:
        process.start()
    start.set()
    admitted = sum(results.get(timeout=60) for _ in processes)
    for process in processes:
        process.join(timeout=60)

    assert admitted == 50


def test_api_keys_are_stored_hashed(database: Database) -> None:
    store = TenantStore(database)
    key = store.create("acme", LIMIT)

    stored = database.conn.execute("SELECT key_hash FROM tenants").fetchone()[0]
    assert key not in stored and stored == hash_api_key(key)
    tenant = store.find_by_key(key)
    assert tenant is not None and tenant.tenant_id == "acme"
    assert store.find_by_key("sk-wrong") is None
