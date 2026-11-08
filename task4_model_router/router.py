"""Primary/secondary model routing with failover and a circuit breaker."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .providers import ProviderClient, ProviderError, ProviderRateLimited, ProviderResponse

logger = logging.getLogger("llm_router")


@dataclass
class CircuitBreaker:
    """Skips a provider that keeps failing, so requests do not wait 3 s for a known outage.

    closed    -> calls flow; consecutive failures are counted
    open      -> calls are skipped until the cooldown (or the provider's
                 Retry-After, if longer) has passed
    half-open -> one trial call; success closes the breaker, failure reopens it
    """

    failure_threshold: int = 3
    cooldown_seconds: float = 30.0
    clock: Callable[[], float] = time.monotonic
    _failures: int = 0
    _open_until: float = 0.0
    _trial_in_flight: bool = False

    def allow(self) -> bool:
        if self._failures < self.failure_threshold:
            return True
        if self.clock() < self._open_until or self._trial_in_flight:
            return False
        self._trial_in_flight = True  # half-open: let exactly one request probe
        return True

    def record_success(self) -> None:
        self._failures = 0
        self._open_until = 0.0
        self._trial_in_flight = False

    def abandon_trial(self) -> None:
        """The call was cancelled before an outcome; let the next request probe instead."""
        self._trial_in_flight = False

    def record_failure(self, retry_after: float | None = None) -> None:
        self._failures += 1
        self._trial_in_flight = False
        if self._failures >= self.failure_threshold:
            self._open_until = self.clock() + max(self.cooldown_seconds, retry_after or 0.0)

    @property
    def is_open(self) -> bool:
        return self._failures >= self.failure_threshold and self.clock() < self._open_until


@dataclass(frozen=True)
class Attempt:
    provider: str
    outcome: str
    latency_ms: float


@dataclass
class RouteResult:
    response: ProviderResponse
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def fallback_used(self) -> bool:
        return self.response.provider != self.attempts[0].provider if self.attempts else False


class AllProvidersFailed(Exception):
    def __init__(self, attempts: list[Attempt]) -> None:
        super().__init__("all providers failed")
        self.attempts = attempts


class ModelRouter:
    def __init__(
        self,
        primary: ProviderClient,
        secondary: ProviderClient,
        *,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        self.primary = primary
        self.secondary = secondary
        self.breaker = breaker or CircuitBreaker()

    async def complete(self, body: dict[str, Any], request_id: str) -> RouteResult:
        attempts: list[Attempt] = []

        if self.breaker.allow():
            started = time.perf_counter()
            try:
                response = await self.primary.complete(body, request_id)
            except ProviderError as exc:
                attempts.append(Attempt(self.primary.name, exc.reason, _elapsed_ms(started)))
                if not exc.retryable:
                    raise
                retry_after = exc.retry_after if isinstance(exc, ProviderRateLimited) else None
                self.breaker.record_failure(retry_after)
                logger.warning(
                    "primary failed, failing over request_id=%s reason=%s breaker_open=%s",
                    request_id,
                    exc.reason,
                    self.breaker.is_open,
                )
            except BaseException:
                self.breaker.abandon_trial()  # cancelled (e.g. client went away)
                raise
            else:
                self.breaker.record_success()
                attempts.append(Attempt(self.primary.name, "ok", response.latency_ms))
                return RouteResult(response, attempts)
        else:
            attempts.append(Attempt(self.primary.name, "skipped_circuit_open", 0.0))

        started = time.perf_counter()
        try:
            response = await self.secondary.complete(body, request_id)
        except ProviderError as exc:
            attempts.append(Attempt(self.secondary.name, exc.reason, _elapsed_ms(started)))
            if not exc.retryable:
                raise
            raise AllProvidersFailed(attempts) from None
        attempts.append(Attempt(self.secondary.name, "ok", response.latency_ms))
        return RouteResult(response, attempts)


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)
