from __future__ import annotations

import time

import httpx
import pytest
from fastapi import FastAPI

from task4_model_router.config import ProviderSettings
from task4_model_router.mock_providers import Mode, create_mock_provider
from task4_model_router.providers import ProviderClient, ProviderRejectedRequest
from task4_model_router.router import AllProvidersFailed, CircuitBreaker, ModelRouter

from .conftest import FakeClock, HostRoutingTransport

BODY = {"model": "gpt-test", "messages": [{"role": "user", "content": "hello"}]}


def build(
    primary_mode: Mode = "ok",
    secondary_mode: Mode = "ok",
    *,
    primary_timeout: float = 0.3,
    breaker: CircuitBreaker | None = None,
) -> tuple[ModelRouter, FastAPI, FastAPI]:
    primary_app = create_mock_provider("primary", primary_mode)
    secondary_app = create_mock_provider("secondary", secondary_mode)
    http = httpx.AsyncClient(
        transport=HostRoutingTransport({"primary": primary_app, "secondary": secondary_app})
    )
    router = ModelRouter(
        ProviderClient(
            ProviderSettings("primary", "http://primary/v1/chat/completions", None, primary_timeout), http
        ),
        ProviderClient(
            ProviderSettings("secondary", "http://secondary/v1/chat/completions", None, 5.0), http
        ),
        breaker=breaker,
    )
    return router, primary_app, secondary_app


async def test_primary_answers_when_healthy() -> None:
    router, primary, secondary = build()
    result = await router.complete(BODY, "r1")

    assert result.response.provider == "primary"
    assert not result.fallback_used
    assert (primary.state.calls, secondary.state.calls) == (1, 0)


async def test_429_from_primary_fails_over_to_secondary() -> None:
    router, _, _ = build(primary_mode="rate_limited")
    result = await router.complete(BODY, "r1")

    assert result.response.provider == "secondary"
    assert result.fallback_used
    assert [(a.provider, a.outcome) for a in result.attempts] == [
        ("primary", "rate_limited"),
        ("secondary", "ok"),
    ]


async def test_slow_primary_is_abandoned_at_the_deadline_and_cancelled() -> None:
    router, primary, _ = build(primary_mode="slow", primary_timeout=0.3)

    started = time.perf_counter()
    result = await router.complete(BODY, "r1")
    elapsed = time.perf_counter() - started

    assert result.response.provider == "secondary"
    assert result.attempts[0].outcome == "timeout"
    assert 0.3 <= elapsed < 1.0
    # The in-flight primary call was really cancelled, so it can never
    # produce a late second answer or keep consuming resources.
    assert primary.state.cancelled == 1


async def test_default_three_second_deadline() -> None:
    router, _, _ = build(primary_mode="slow", primary_timeout=3.0)

    started = time.perf_counter()
    result = await router.complete(BODY, "r1")
    elapsed = time.perf_counter() - started

    assert result.response.provider == "secondary"
    assert 3.0 <= elapsed < 3.8


async def test_5xx_from_primary_fails_over() -> None:
    router, _, _ = build(primary_mode="error")
    assert (await router.complete(BODY, "r1")).response.provider == "secondary"


async def test_connection_failure_fails_over() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        if request.url.host == "primary":
            raise httpx.ConnectError("refused")
        return httpx.Response(200, json={"choices": [], "usage": {"total_tokens": 3}})

    http = httpx.AsyncClient(transport=httpx.MockTransport(refuse))
    router = ModelRouter(
        ProviderClient(ProviderSettings("primary", "http://primary/x"), http),
        ProviderClient(ProviderSettings("secondary", "http://secondary/x"), http),
    )
    assert (await router.complete(BODY, "r1")).response.provider == "secondary"


async def test_client_errors_do_not_fail_over() -> None:
    def reject(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "bad request"})

    http = httpx.AsyncClient(transport=httpx.MockTransport(reject))
    router = ModelRouter(
        ProviderClient(ProviderSettings("primary", "http://primary/x"), http),
        ProviderClient(ProviderSettings("secondary", "http://secondary/x"), http),
    )
    with pytest.raises(ProviderRejectedRequest):
        await router.complete(BODY, "r1")


async def test_both_providers_failing_raises_with_attempt_history() -> None:
    router, _, _ = build(primary_mode="rate_limited", secondary_mode="error")
    with pytest.raises(AllProvidersFailed) as info:
        await router.complete(BODY, "r1")
    assert [(a.provider, a.outcome) for a in info.value.attempts] == [
        ("primary", "rate_limited"),
        ("secondary", "unavailable"),
    ]


async def test_circuit_opens_after_repeated_failures_and_recovers() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=3, cooldown_seconds=30, clock=clock)
    router, primary, _ = build(primary_mode="error", breaker=breaker)

    for _ in range(3):
        await router.complete(BODY, "r")
    assert breaker.is_open and primary.state.calls == 3

    result = await router.complete(BODY, "r")  # skipped: no wasted call to a known-bad provider
    assert primary.state.calls == 3
    assert result.attempts[0].outcome == "skipped_circuit_open"

    primary.state.mode = "ok"
    clock.advance(31)  # half-open: one probe goes to the primary and succeeds
    assert (await router.complete(BODY, "r")).response.provider == "primary"
    assert not breaker.is_open


def test_breaker_honours_provider_retry_after() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=5, clock=clock)
    breaker.record_failure(retry_after=20)
    clock.advance(10)
    assert not breaker.allow()
    clock.advance(11)
    assert breaker.allow()


def test_half_open_admits_a_single_probe() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, cooldown_seconds=5, clock=clock)
    breaker.record_failure()
    clock.advance(6)
    assert breaker.allow() is True
    assert breaker.allow() is False  # second request waits for the probe's outcome
    breaker.abandon_trial()
    assert breaker.allow() is True
