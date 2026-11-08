"""MCP security gateway: an authenticating, authorizing reverse proxy.

Request pipeline for ``POST /mcp``::

    authenticate (Bearer JWT) -> size limit -> parse JSON-RPC -> authorize
        -> deny: answer directly, downstream is never contacted
        -> allow: forward body unchanged, stream the downstream response back

Run::

    python -m task2_mcp_gateway.gateway          # listens on :8080
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from .auth import AuthenticationError, Principal, authenticate
from .config import GatewaySettings
from .jsonrpc import (
    INTERNAL_ERROR,
    UNAUTHORIZED_TOOL_CALL,
    JsonRpcError,
    JsonRpcMessage,
    error_response,
    parse_message,
)
from .policy import authorize, filter_tools_for

logger = logging.getLogger("mcp_gateway")
audit_logger = logging.getLogger("mcp_gateway.audit")

# Headers that describe the MCP session and payload; everything else, notably
# the caller's Authorization header, stays at the gateway.
FORWARDED_REQUEST_HEADERS = (
    "content-type",
    "accept",
    "mcp-session-id",
    "mcp-protocol-version",
    "last-event-id",
)
FORWARDED_RESPONSE_HEADERS = ("content-type", "mcp-session-id", "mcp-protocol-version", "cache-control")


def _audit(request_id: str, principal: Principal | None, outcome: str, **fields: Any) -> None:
    record = {
        "request_id": request_id,
        "subject": principal.subject if principal else None,
        "role": principal.role.value if principal else None,
        "outcome": outcome,
        **fields,
    }
    audit_logger.info(json.dumps(record, separators=(",", ":"), default=str))


def _jsonrpc_error(error: JsonRpcError, *, status_code: int = 200) -> JSONResponse:
    return JSONResponse(error.to_response(), status_code=status_code)


def create_app(
    settings: GatewaySettings | None = None,
    *,
    http_client: httpx.AsyncClient | None = None,
) -> FastAPI:
    settings = settings or GatewaySettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owns_client = http_client is None
        app.state.http = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(settings.upstream_timeout_seconds, connect=5.0),
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
        )
        try:
            yield
        finally:
            if owns_client:
                await app.state.http.aclose()

    app = FastAPI(title="MCP Security Gateway", version="1.0.0", lifespan=lifespan)
    app.state.http = http_client

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.api_route("/mcp", methods=["GET", "DELETE"])
    async def unsupported_mcp_method() -> Response:
        # Streamable HTTP lets a server decline the standalone SSE stream (GET)
        # and client-initiated session termination (DELETE) with 405.
        return Response(status_code=405, headers={"Allow": "POST"})

    @app.post("/mcp")
    async def mcp_endpoint(request: Request) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        started = time.perf_counter()

        try:
            principal = authenticate(request.headers.get("authorization"), settings)
        except AuthenticationError as exc:
            _audit(request_id, None, "unauthenticated", reason=exc.reason)
            return JSONResponse(
                {"error": "unauthorized", "message": exc.reason},
                status_code=401,
                headers={"WWW-Authenticate": 'Bearer realm="mcp-gateway"'},
            )

        body = await _read_body(request, settings.max_body_bytes)
        if body is None:
            _audit(request_id, principal, "rejected", reason="body_too_large")
            return JSONResponse({"error": "payload_too_large"}, status_code=413)

        try:
            message = parse_message(body)
            decision = authorize(principal, message)
        except JsonRpcError as exc:
            _audit(
                request_id,
                principal,
                "denied" if exc.code == UNAUTHORIZED_TOOL_CALL else "rejected",
                code=exc.code,
                reason=exc.message,
                **_describe(body),
            )
            return _jsonrpc_error(exc)

        response = await _forward(request, body, message, principal, request_id, settings)
        _audit(
            request_id,
            principal,
            "forwarded",
            method=message.method,
            tool=decision.tool,
            status=response.status_code,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return response

    return app


async def _read_body(request: Request, limit: int) -> bytes | None:
    """Read the request body, giving up as soon as it exceeds ``limit`` bytes.

    The declared Content-Length is checked first, and the stream is counted
    as it arrives, so an oversized body is never buffered in full.
    """
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > limit:
        return None
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def _describe(body: bytes) -> dict[str, Any]:
    """Best-effort method/tool names for the audit trail of a denied request."""
    try:
        payload = json.loads(body)
    except ValueError:
        return {}
    if not isinstance(payload, dict):
        return {}
    params = payload.get("params")
    tool = params.get("name") if isinstance(params, dict) else None
    return {"method": payload.get("method"), "tool": tool if isinstance(tool, str) else None}


async def _forward(
    request: Request,
    body: bytes,
    message: JsonRpcMessage,
    principal: Principal,
    request_id: str,
    settings: GatewaySettings,
) -> Response:
    client: httpx.AsyncClient = request.app.state.http
    headers = {name: value for name in FORWARDED_REQUEST_HEADERS if (value := request.headers.get(name))}
    headers["x-request-id"] = request_id
    # Identity context for downstream audit; trust it only on a channel that
    # authenticates the gateway itself (mTLS or a private network).
    headers["x-authenticated-subject"] = principal.subject
    headers["x-authenticated-role"] = principal.role.value

    upstream_request = client.build_request("POST", settings.downstream_url, content=body, headers=headers)
    try:
        upstream = await client.send(upstream_request, stream=True)
    except httpx.TimeoutException:
        logger.warning("downstream timeout request_id=%s", request_id)
        return _upstream_failure(message, request_id, "Upstream MCP server timed out", 504)
    except httpx.HTTPError as exc:
        logger.warning("downstream unreachable request_id=%s error=%s", request_id, type(exc).__name__)
        return _upstream_failure(message, request_id, "Upstream MCP server unavailable", 502)

    response_headers = {
        name: value for name in FORWARDED_RESPONSE_HEADERS if (value := upstream.headers.get(name))
    }
    response_headers["x-request-id"] = request_id

    if settings.filter_tools_list and message.method == "tools/list" and upstream.status_code == 200:
        try:
            raw = await upstream.aread()
        finally:
            await upstream.aclose()
        filtered = _filter_tools_list_body(raw, upstream.headers.get("content-type", ""), principal)
        return Response(filtered, status_code=upstream.status_code, headers=response_headers)

    # aiter_bytes undoes any Content-Encoding the downstream applied; that
    # header is not forwarded, so the client always receives plain bytes.
    return StreamingResponse(
        upstream.aiter_bytes(),
        status_code=upstream.status_code,
        headers=response_headers,
        background=BackgroundTask(upstream.aclose),
    )


def _upstream_failure(message: JsonRpcMessage, request_id: str, text: str, status: int) -> Response:
    if message.is_notification:
        return Response(status_code=status)
    body = error_response(message.id, INTERNAL_ERROR, text, {"request_id": request_id})
    return JSONResponse(body, status_code=status)


def _filter_tools_list_body(raw: bytes, content_type: str, principal: Principal) -> bytes:
    def rewrite(message: Any) -> Any:
        result = message.get("result") if isinstance(message, dict) else None
        if isinstance(result, dict) and isinstance(result.get("tools"), list):
            result["tools"] = filter_tools_for(principal, result["tools"])
        return message

    if content_type.startswith("text/event-stream"):
        lines = []
        for line in raw.decode("utf-8").split("\n"):
            if line.startswith("data:"):
                try:
                    data = rewrite(json.loads(line[5:].strip()))
                    line = "data: " + json.dumps(data, separators=(",", ":"))
                except ValueError:
                    pass
            lines.append(line)
        return "\n".join(lines).encode("utf-8")

    try:
        return json.dumps(rewrite(json.loads(raw)), separators=(",", ":")).encode("utf-8")
    except ValueError:
        return raw


def main() -> None:
    import uvicorn

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    uvicorn.run(
        create_app(),
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8080")),
        log_level="info",
    )


if __name__ == "__main__":
    main()
