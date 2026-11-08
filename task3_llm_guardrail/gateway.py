"""LLM gateway endpoint with a streaming PII-redaction guardrail.

``POST /v1/chat/completions`` is OpenAI-compatible. The request is forwarded
to the configured provider; the response is redacted on the way back.

Streaming path, per network read::

    bytes -> SSEParser (incremental UTF-8) -> JSON chunk
          -> StreamingRedactor per choice (delta.content only)
          -> re-encoded SSE event -> client

Nothing is accumulated beyond the redactor's small hold-back window, so
memory is constant per stream and the first safe token is released as soon
as it arrives.

Run::

    python -m task3_llm_guardrail.gateway       # listens on :8081
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .redactor import StreamingRedactor, redact
from .sse import SSEEvent, SSEParser

logger = logging.getLogger("llm_gateway")

STREAM_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",  # stop nginx-style proxies from buffering the stream
}


@dataclass(frozen=True)
class GuardrailSettings:
    upstream_url: str = "http://127.0.0.1:9200/v1/chat/completions"
    upstream_api_key: str | None = None
    connect_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 60.0

    @classmethod
    def from_env(cls) -> GuardrailSettings:
        defaults = cls()
        return cls(
            upstream_url=os.getenv("LLM_UPSTREAM_URL", defaults.upstream_url),
            upstream_api_key=os.getenv("LLM_UPSTREAM_API_KEY") or None,
            read_timeout_seconds=float(os.getenv("LLM_READ_TIMEOUT_SECONDS", defaults.read_timeout_seconds)),
        )


def _error(status: int, error_type: str, message: str, request_id: str) -> JSONResponse:
    return JSONResponse(
        {"error": {"type": error_type, "message": message, "request_id": request_id}},
        status_code=status,
        headers={"x-request-id": request_id},
    )


class StreamRedactionSession:
    """Applies redaction to a sequence of chat-completion SSE events."""

    def __init__(self) -> None:
        self._redactors: dict[int, StreamingRedactor] = {}
        self._template: dict[str, Any] = {}

    def _redactor(self, index: int) -> StreamingRedactor:
        if index not in self._redactors:
            self._redactors[index] = StreamingRedactor()
        return self._redactors[index]

    @property
    def redactions(self) -> int:
        return sum(r.stats.redactions for r in self._redactors.values())

    @property
    def max_held(self) -> int:
        return max((r.stats.max_held for r in self._redactors.values()), default=0)

    def process(self, event: SSEEvent) -> list[bytes]:
        if not event.data:
            return [event.encode()]  # keep-alive comments, retry hints
        if event.data.strip() == "[DONE]":
            return [*self.finish(), event.encode()]

        try:
            chunk = json.loads(event.data)
        except ValueError:
            logger.warning("dropping undecodable stream event (%d bytes)", len(event.data))
            return []
        if not isinstance(chunk, dict) or not isinstance(chunk.get("choices"), list):
            # Error objects or usage-only events: no deltas, but still scrub them.
            event.data = redact(event.data)
            return [event.encode()]

        self._template = {k: chunk[k] for k in ("id", "object", "created", "model") if k in chunk}
        emit = bool(chunk.get("usage"))
        for choice in chunk["choices"]:
            if isinstance(choice, dict) and self._redact_choice(choice):
                emit = True
        if not emit:
            return []  # everything in this chunk is being held back
        event.data = json.dumps(chunk, ensure_ascii=False, separators=(",", ":"))
        return [event.encode()]

    def _redact_choice(self, choice: dict[str, Any]) -> bool:
        """Redact one choice in place. Returns True if it still carries anything."""
        index = choice.get("index")
        redactor = self._redactor(index if isinstance(index, int) else 0)
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            delta = {}
        content = delta.get("content")

        text = redactor.feed(content) if isinstance(content, str) else ""
        if choice.get("finish_reason"):
            text += redactor.flush()
        if isinstance(content, str) or text:
            delta["content"] = text
            choice["delta"] = delta

        other_fields = {k for k, v in delta.items() if k != "content" and v is not None}
        return bool(text) or bool(other_fields) or bool(choice.get("finish_reason"))

    def finish(self) -> list[bytes]:
        """Release whatever is still held (stream ended or ``[DONE]`` arrived)."""
        frames = []
        for index, redactor in sorted(self._redactors.items()):
            tail = redactor.flush()
            if tail:
                chunk = {
                    **self._template,
                    "choices": [{"index": index, "delta": {"content": tail}, "finish_reason": None}],
                }
                frames.append(
                    SSEEvent(data=json.dumps(chunk, ensure_ascii=False, separators=(",", ":"))).encode()
                )
        return frames


async def _relay(upstream: httpx.Response, request_id: str, started: float) -> AsyncIterator[bytes]:
    parser = SSEParser()
    session = StreamRedactionSession()
    first_byte_ms: float | None = None
    try:
        async for raw in upstream.aiter_bytes():
            for event in parser.feed(raw):
                for frame in session.process(event):
                    if first_byte_ms is None:
                        first_byte_ms = (time.perf_counter() - started) * 1000
                    yield frame
        for event in parser.close():
            for frame in session.process(event):
                yield frame
        for frame in session.finish():
            yield frame
    except UnicodeDecodeError:
        logger.error("upstream sent invalid UTF-8; stream terminated request_id=%s", request_id)
    except httpx.HTTPError as exc:
        logger.warning("upstream stream interrupted request_id=%s error=%s", request_id, type(exc).__name__)
    finally:
        await upstream.aclose()
        logger.info(
            "stream complete request_id=%s ttfb_ms=%s redactions=%d max_held_chars=%d total_ms=%.1f",
            request_id,
            f"{first_byte_ms:.1f}" if first_byte_ms is not None else "n/a",
            session.redactions,
            session.max_held,
            (time.perf_counter() - started) * 1000,
        )


def _redact_completion(payload: dict[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices")
    for choice in choices if isinstance(choices, list) else []:
        message = choice.get("message") if isinstance(choice, dict) else None
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            message["content"] = redact(message["content"])
    return payload


def create_app(
    settings: GuardrailSettings | None = None,
    *,
    http_client: httpx.AsyncClient | None = None,
) -> FastAPI:
    settings = settings or GuardrailSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owns_client = http_client is None
        app.state.http = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(settings.read_timeout_seconds, connect=settings.connect_timeout_seconds),
            limits=httpx.Limits(max_connections=500, max_keepalive_connections=100),
        )
        try:
            yield
        finally:
            if owns_client:
                await app.state.http.aclose()

    app = FastAPI(title="LLM Gateway - PII Guardrail", version="1.0.0", lifespan=lifespan)
    app.state.http = http_client

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        started = time.perf_counter()
        try:
            body = await request.json()
        except ValueError:
            return _error(400, "invalid_request", "Request body must be valid JSON", request_id)
        if not isinstance(body, dict):
            return _error(400, "invalid_request", "Request body must be a JSON object", request_id)

        headers = {"x-request-id": request_id}
        if settings.upstream_api_key:
            headers["authorization"] = f"Bearer {settings.upstream_api_key}"
        client: httpx.AsyncClient = request.app.state.http
        upstream_request = client.build_request("POST", settings.upstream_url, json=body, headers=headers)

        try:
            upstream = await client.send(upstream_request, stream=True)
        except httpx.TimeoutException:
            return _error(504, "upstream_timeout", "The model provider did not respond in time", request_id)
        except httpx.HTTPError:
            return _error(502, "upstream_unavailable", "The model provider is unavailable", request_id)

        if upstream.status_code >= 400:
            await upstream.aclose()
            logger.warning("upstream error status=%d request_id=%s", upstream.status_code, request_id)
            status = upstream.status_code if upstream.status_code in {400, 404, 413, 422, 429} else 502
            return _error(
                status, "upstream_error", f"Model provider returned HTTP {upstream.status_code}", request_id
            )

        if body.get("stream") is True:
            return StreamingResponse(
                _relay(upstream, request_id, started),
                media_type="text/event-stream",
                headers={**STREAM_HEADERS, "x-request-id": request_id},
            )

        try:
            payload = json.loads(await upstream.aread())
        except httpx.TimeoutException:
            return _error(504, "upstream_timeout", "The model provider did not respond in time", request_id)
        except (ValueError, httpx.HTTPError):
            payload = None
        finally:
            await upstream.aclose()
        if not isinstance(payload, dict):
            return _error(502, "upstream_error", "Model provider returned an invalid response", request_id)
        return JSONResponse(_redact_completion(payload), headers={"x-request-id": request_id})

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    uvicorn.run(
        create_app(),
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8081")),
        log_level="info",
    )


if __name__ == "__main__":
    main()
