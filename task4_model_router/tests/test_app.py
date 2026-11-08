from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from task4_model_router.app import create_app
from task4_model_router.config import ProviderSettings, RouterSettings
from task4_model_router.mock_providers import LEAKY_TRACEBACK, create_mock_provider
from task4_model_router.router import ModelRouter
from task4_model_router.tokens import estimate_prompt_tokens

from .conftest import FakeClock, HostRoutingTransport

MESSAGES = [{"role": "user", "content": "Summarise the refund policy in one line."}]
ERROR_KEYS = {"type", "message", "request_id"}


class Stack:
    def __init__(self, tmp_path: Path, clock: FakeClock, **overrides: object) -> None:
        self.primary = create_mock_provider("primary")
        self.secondary = create_mock_provider("secondary")
        self.upstream = httpx.AsyncClient(
            transport=HostRoutingTransport({"primary": self.primary, "secondary": self.secondary})
        )
        settings = RouterSettings(
            database_path=str(tmp_path / "gateway.db"),
            primary=ProviderSettings("primary", "http://primary/v1/chat/completions", None, 0.3),
            secondary=ProviderSettings("secondary", "http://secondary/v1/chat/completions", None, 2.0),
        )
        self.app = create_app(replace(settings, **overrides), http_client=self.upstream, clock=clock)
        self.key = self.app.state.tenants.create("acme", 50_000)
        # raise_app_exceptions=False mirrors a real server: the client sees the
        # 500 response instead of the exception being re-raised into the test.
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app, raise_app_exceptions=False), base_url="http://gateway"
        )

    async def chat(self, *, key: str | None = None, **body: object) -> httpx.Response:
        payload = {"model": "gpt-test", "messages": MESSAGES, "max_tokens": 100, **body}
        return await self.client.post(
            "/v1/chat/completions", json=payload, headers={"Authorization": f"Bearer {key or self.key}"}
        )


@pytest.fixture
async def stack(tmp_path: Path, clock: FakeClock) -> AsyncIterator[Stack]:
    s = Stack(tmp_path, clock)
    yield s
    await s.client.aclose()
    await s.upstream.aclose()


def assert_standard_error(response: httpx.Response, status: int, error_type: str) -> dict:
    assert response.status_code == status
    body = response.json()
    assert set(body) == {"error"} and set(body["error"]) == ERROR_KEYS
    assert body["error"]["type"] == error_type
    assert body["error"]["request_id"] == response.headers["x-request-id"]
    return body["error"]


async def test_successful_completion_reports_rate_limit_headers(stack: Stack) -> None:
    response = await stack.chat()

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Answer from primary."
    assert response.headers["x-gateway-route"] == "primary"
    assert response.headers["x-ratelimit-limit-tokens"] == "50000"
    used = response.json()["usage"]["total_tokens"]
    assert int(response.headers["x-ratelimit-remaining-tokens"]) == 50_000 - used


async def test_usage_is_reconciled_to_actual_tokens(stack: Stack) -> None:
    response = await stack.chat(max_tokens=5_000)  # reserves ~5k up front
    actual = response.json()["usage"]["total_tokens"]
    assert actual < 5_000
    assert stack.app.state.limiter.usage("acme") == actual


async def test_fallback_on_primary_429(stack: Stack) -> None:
    stack.primary.state.mode = "rate_limited"
    response = await stack.chat()

    assert response.status_code == 200
    assert response.headers["x-gateway-route"] == "fallback"
    assert response.json()["choices"][0]["message"]["content"] == "Answer from secondary."
    assert "org-8812" not in response.text


async def test_fallback_on_primary_timeout(stack: Stack) -> None:
    stack.primary.state.mode = "slow"
    started = time.perf_counter()
    response = await stack.chat()

    assert response.headers["x-gateway-route"] == "fallback"
    assert time.perf_counter() - started < 1.0


async def test_all_providers_down_returns_sanitized_503(stack: Stack) -> None:
    stack.primary.state.mode = "error"
    stack.secondary.state.mode = "error"
    response = await stack.chat()

    error = assert_standard_error(response, 503, "upstream_unavailable")
    for leak in ["Traceback", "worker.py", "10.0.4.17", "gpu-node", "CUDA", "primary", "secondary", "httpx"]:
        assert leak not in response.text
    assert LEAKY_TRACEBACK not in json.dumps(error)
    assert stack.app.state.limiter.usage("acme") == 0  # reservation released


async def test_rate_limit_returns_429_with_retry_after(stack: Stack, clock: FakeClock) -> None:
    max_tokens = 25_000 - estimate_prompt_tokens(MESSAGES)
    assert (await stack.chat(max_tokens=max_tokens)).status_code == 200
    # Usage was reconciled down to the real count, so fill the window directly.
    stack.app.state.limiter.reserve("acme", 50_000 - stack.app.state.limiter.usage("acme"), 50_000)
    response = await stack.chat()

    assert_standard_error(response, 429, "rate_limit_exceeded")
    assert response.headers["retry-after"] == "60"
    assert response.headers["x-ratelimit-remaining-tokens"] == "0"
    assert stack.primary.state.calls == 1  # rejected before any provider call

    clock.advance(60)
    assert (await stack.chat()).status_code == 200


async def test_concurrent_requests_cannot_exceed_the_limit(stack: Stack) -> None:
    """100 simultaneous ~1,000-token requests against a 50,000 budget.

    Provider calls are held until the limiter has decided on all 100, so no
    request can finish early and hand back unused estimate. That makes the
    expected split exactly 50/50 regardless of scheduling.
    """
    loop = asyncio.get_running_loop()
    all_decided = asyncio.Event()
    decisions = 0
    limiter = stack.app.state.limiter
    reserve, complete = limiter.reserve, stack.app.state.router.complete

    def counting_reserve(*args: object, **kwargs: object) -> object:
        nonlocal decisions
        try:
            return reserve(*args, **kwargs)
        finally:
            decisions += 1
            if decisions == 100:
                loop.call_soon_threadsafe(all_decided.set)

    async def gated_complete(body: dict, request_id: str) -> object:
        await all_decided.wait()
        return await complete(body, request_id)

    limiter.reserve = counting_reserve
    stack.app.state.router.complete = gated_complete
    max_tokens = 1_000 - estimate_prompt_tokens(MESSAGES)

    responses = await asyncio.gather(*(stack.chat(max_tokens=max_tokens) for _ in range(100)))

    statuses = [r.status_code for r in responses]
    assert statuses.count(200) == 50
    assert statuses.count(429) == 50


async def test_request_bigger_than_the_whole_budget(stack: Stack) -> None:
    response = await stack.chat(max_tokens=60_000)
    assert_standard_error(response, 400, "request_too_large")


@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer sk-unknown"}, {"Authorization": "Basic abc"}],
)
async def test_authentication_failures(stack: Stack, headers: dict[str, str]) -> None:
    response = await stack.client.post("/v1/chat/completions", json={"bad": "body"}, headers=headers)
    assert_standard_error(response, 401, "authentication_error")


@pytest.mark.parametrize(
    "body",
    [
        {"model": "gpt-test", "messages": []},
        {"model": "gpt-test"},
        {"model": "", "messages": MESSAGES},
        {"model": "gpt-test", "messages": MESSAGES, "max_tokens": -5},
        {"model": "gpt-test", "messages": [{"role": "wizard", "content": "hi"}]},
    ],
)
async def test_validation_errors_are_standardized(stack: Stack, body: dict) -> None:
    response = await stack.client.post(
        "/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {stack.key}"}
    )
    error = assert_standard_error(response, 400, "invalid_request")
    assert "pydantic" not in error["message"].lower()


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"{not json", "The request body must be valid JSON."),
        (b"[1, 2]", "The request body must be a JSON object."),
        (b"", "A JSON request body is required."),
        (b"{}", "Missing or invalid fields: messages, model."),
    ],
)
async def test_unusable_bodies_get_clear_messages(stack: Stack, content: bytes, message: str) -> None:
    response = await stack.client.post(
        "/v1/chat/completions",
        content=content,
        headers={"Authorization": f"Bearer {stack.key}", "Content-Type": "application/json"},
    )
    assert assert_standard_error(response, 400, "invalid_request")["message"] == message


async def test_streaming_requests_are_rejected_clearly(stack: Stack) -> None:
    assert_standard_error(await stack.chat(stream=True), 400, "invalid_request")


async def test_provider_rejecting_request_is_not_retried(stack: Stack) -> None:
    async def reject(*args: object, **kwargs: object) -> None:
        from task4_model_router.providers import ProviderRejectedRequest

        raise ProviderRejectedRequest(422)

    stack.app.state.router.primary.complete = reject
    response = await stack.chat()

    assert_standard_error(response, 400, "invalid_request")
    assert stack.secondary.state.calls == 0
    assert stack.app.state.limiter.usage("acme") == 0


async def test_unexpected_exceptions_are_sanitized(stack: Stack) -> None:
    async def explode(self: ModelRouter, body: dict, request_id: str) -> None:
        raise RuntimeError("sqlite file /var/lib/gateway/secret.db is corrupt")

    stack.app.state.router.complete = explode.__get__(stack.app.state.router)
    response = await stack.chat()

    error = assert_standard_error(response, 500, "internal_error")
    assert "secret.db" not in response.text and "RuntimeError" not in response.text
    assert error["message"] == "An internal error occurred. Please retry later."
    assert stack.app.state.limiter.usage("acme") == 0


async def test_unknown_route_uses_the_standard_shape(stack: Stack) -> None:
    assert_standard_error(await stack.client.get("/v1/models/secret"), 404, "not_found")


async def test_client_supplied_request_id_is_echoed(stack: Stack) -> None:
    response = await stack.client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": MESSAGES},
        headers={"Authorization": f"Bearer {stack.key}", "X-Request-ID": "trace-123"},
    )
    assert response.headers["x-request-id"] == "trace-123"
