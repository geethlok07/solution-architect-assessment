"""Model provider clients.

Each call runs under a hard deadline (``asyncio.timeout``) that covers the
whole exchange: connecting, sending, waiting for and reading the response.
httpx's own timeouts are per network operation, so a provider that trickles
bytes could otherwise exceed the budget without ever tripping them.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .config import ProviderSettings


class ProviderError(Exception):
    """Base class. ``retryable`` failures are eligible for failover."""

    retryable = True
    reason = "error"


class ProviderRateLimited(ProviderError):
    reason = "rate_limited"

    def __init__(self, retry_after: float | None) -> None:
        super().__init__("provider returned 429")
        self.retry_after = retry_after


class ProviderTimeout(ProviderError):
    reason = "timeout"


class ProviderUnavailable(ProviderError):
    reason = "unavailable"


class ProviderRejectedRequest(ProviderError):
    """4xx other than 429: the request itself is bad, so another provider would reject it too."""

    retryable = False
    reason = "rejected"

    def __init__(self, status_code: int) -> None:
        super().__init__(f"provider returned {status_code}")
        self.status_code = status_code


@dataclass(frozen=True)
class ProviderResponse:
    provider: str
    payload: dict[str, Any]
    latency_ms: float

    @property
    def total_tokens(self) -> int | None:
        usage = self.payload.get("usage")
        if isinstance(usage, dict) and isinstance(usage.get("total_tokens"), int):
            return int(usage["total_tokens"])
        return None


def _parse_retry_after(value: str | None) -> float | None:
    try:
        return max(float(value), 0.0) if value is not None else None
    except ValueError:
        return None


class ProviderClient:
    def __init__(self, settings: ProviderSettings, http: httpx.AsyncClient) -> None:
        self.settings = settings
        self.name = settings.name
        self._http = http

    async def complete(self, body: dict[str, Any], request_id: str) -> ProviderResponse:
        headers = {"x-request-id": request_id}
        if self.settings.api_key:
            headers["authorization"] = f"Bearer {self.settings.api_key}"

        started = time.perf_counter()
        try:
            async with asyncio.timeout(self.settings.timeout_seconds):
                response = await self._http.post(self.settings.url, json=body, headers=headers)
                payload = response.json() if response.status_code < 400 else None
        except TimeoutError:
            # asyncio.timeout cancelled the request; httpx closes the connection
            # on cancellation, so a late answer can never be delivered.
            raise ProviderTimeout(f"no response within {self.settings.timeout_seconds:.1f}s") from None
        except httpx.TimeoutException:
            raise ProviderTimeout("network timeout") from None
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(type(exc).__name__) from None
        except ValueError:
            raise ProviderUnavailable("invalid JSON from provider") from None

        if response.status_code == 429:
            raise ProviderRateLimited(_parse_retry_after(response.headers.get("retry-after")))
        if response.status_code >= 500:
            raise ProviderUnavailable(f"provider returned {response.status_code}")
        if response.status_code >= 400:
            raise ProviderRejectedRequest(response.status_code)
        if not isinstance(payload, dict):
            raise ProviderUnavailable("unexpected response shape")

        return ProviderResponse(self.name, payload, (time.perf_counter() - started) * 1000)
