"""LLM gateway: per-tenant token rate limiting + resilient model routing.

``POST /v1/chat/completions`` (OpenAI-compatible, non-streaming):

    authenticate tenant key -> validate -> estimate tokens -> reserve in the
    sliding window (429 if over) -> primary model, 3 s deadline -> on 429 /
    timeout / 5xx fail over to secondary -> reconcile real token usage

Run::

    python -m task4_model_router.app            # listens on :8082
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, Literal

import httpx
from fastapi import Depends, FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .config import RouterSettings
from .db import Database, Tenant, TenantStore
from .errors import (
    AuthenticationError,
    InvalidRequestError,
    UpstreamUnavailableError,
    install_error_handlers,
)
from .providers import ProviderClient, ProviderRejectedRequest
from .rate_limiter import SlidingWindowRateLimiter
from .router import AllProvidersFailed, CircuitBreaker, ModelRouter
from .tokens import estimate_request_tokens

logger = logging.getLogger("llm_router")


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: Literal["system", "developer", "user", "assistant", "tool", "function"]
    content: str | list[Any] | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str = Field(min_length=1, max_length=200)
    messages: list[ChatMessage] = Field(min_length=1, max_length=2_000)
    max_tokens: int | None = Field(default=None, gt=0, le=200_000)
    stream: bool = False


def create_app(
    settings: RouterSettings | None = None,
    *,
    http_client: httpx.AsyncClient | None = None,
    clock: Callable[[], float] = time.time,
) -> FastAPI:
    settings = settings or RouterSettings.from_env()
    db = Database(settings.database_path)
    tenants = TenantStore(db)
    limiter = SlidingWindowRateLimiter(db, window_seconds=settings.window_seconds, clock=clock)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        if http_client is None:
            await shared_http.aclose()
        db.close()

    shared_http = http_client or httpx.AsyncClient(
        limits=httpx.Limits(max_connections=500, max_keepalive_connections=100),
        timeout=httpx.Timeout(30.0, connect=2.0),
    )
    router = ModelRouter(
        ProviderClient(settings.primary, shared_http),
        ProviderClient(settings.secondary, shared_http),
        breaker=CircuitBreaker(
            failure_threshold=settings.breaker_failure_threshold,
            cooldown_seconds=settings.breaker_cooldown_seconds,
        ),
    )

    app = FastAPI(title="LLM Gateway - Rate Limiter & Model Router", version="1.0.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.tenants = tenants
    app.state.limiter = limiter
    app.state.router = router
    install_error_handlers(app)

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request.state.request_id = request.headers.get("x-request-id") or f"req_{uuid.uuid4().hex[:20]}"
        response = await call_next(request)
        response.headers["x-request-id"] = request.state.request_id
        return response

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"status": "ok", "primary_circuit_open": router.breaker.is_open}

    async def authenticated_tenant(request: Request) -> Tenant:
        # Declared as a dependency so it runs before body validation:
        # unauthenticated callers get 401, never details about the payload.
        tenant = await asyncio.to_thread(tenants.find_by_key, _bearer_token(request))
        if tenant is None:
            raise AuthenticationError("The API key is not recognised.")
        return tenant

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: Request,
        tenant: Tenant = Depends(authenticated_tenant),  # noqa: B008 - FastAPI dependency
        payload: ChatCompletionRequest | None = None,
    ) -> JSONResponse:
        request_id: str = request.state.request_id
        if payload is None:
            raise InvalidRequestError("A JSON request body is required.")
        if payload.stream:
            raise InvalidRequestError("Streaming is not supported on this endpoint; set stream to false.")

        body = payload.model_dump(exclude_none=True)
        estimated = estimate_request_tokens(body, settings.default_max_completion_tokens)
        reservation = await asyncio.to_thread(
            limiter.reserve, tenant.tenant_id, estimated, tenant.tokens_per_minute
        )

        try:
            result = await router.complete(body, request_id)
        except AllProvidersFailed as exc:
            await asyncio.to_thread(limiter.release, reservation)
            logger.error("all providers failed request_id=%s attempts=%s", request_id, exc.attempts)
            raise UpstreamUnavailableError() from None
        except ProviderRejectedRequest:
            await asyncio.to_thread(limiter.release, reservation)
            raise InvalidRequestError("The model provider rejected the request as invalid.") from None
        except BaseException:
            await asyncio.shield(asyncio.to_thread(limiter.release, reservation))
            raise

        actual = result.response.total_tokens
        if actual is not None:
            await asyncio.to_thread(limiter.reconcile, reservation, actual)
        remaining = tenant.tokens_per_minute - await asyncio.to_thread(limiter.usage, tenant.tenant_id)

        logger.info(
            "completed request_id=%s tenant=%s provider=%s fallback=%s estimated=%d actual=%s attempts=%s",
            request_id,
            tenant.tenant_id,
            result.response.provider,
            result.fallback_used,
            estimated,
            actual,
            [(a.provider, a.outcome, a.latency_ms) for a in result.attempts],
        )
        return JSONResponse(
            result.response.payload,
            headers={
                "X-RateLimit-Limit-Tokens": str(tenant.tokens_per_minute),
                "X-RateLimit-Remaining-Tokens": str(max(remaining, 0)),
                "X-Gateway-Route": "fallback" if result.fallback_used else "primary",
            },
        )

    return app


def _bearer_token(request: Request) -> str:
    scheme, _, token = (request.headers.get("authorization") or "").partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise AuthenticationError("Send the API key as 'Authorization: Bearer <key>'.")
    return token.strip()


def main() -> None:
    import uvicorn

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    uvicorn.run(
        create_app(),
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8082")),
        log_level="info",
    )


if __name__ == "__main__":
    main()
