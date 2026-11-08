"""Standardized gateway errors.

Every failure leaves the gateway in one shape::

    {"error": {"type": "...", "message": "...", "request_id": "..."}}

Messages are written for the API consumer. Upstream bodies, exception
classes, stack traces, hostnames and file paths are logged server-side under
the same ``request_id`` and never returned.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("llm_router.errors")


class GatewayError(Exception):
    status_code = 500
    error_type = "internal_error"
    default_message = "An internal error occurred. Please retry later."

    def __init__(self, message: str | None = None, *, headers: dict[str, str] | None = None) -> None:
        super().__init__(message or self.default_message)
        self.message = message or self.default_message
        self.headers = headers or {}


class InvalidRequestError(GatewayError):
    status_code = 400
    error_type = "invalid_request"
    default_message = "The request is invalid."


class AuthenticationError(GatewayError):
    status_code = 401
    error_type = "authentication_error"
    default_message = "A valid API key is required."


class RateLimitExceededError(GatewayError):
    status_code = 429
    error_type = "rate_limit_exceeded"

    def __init__(self, *, retry_after: int, limit: int, remaining: int) -> None:
        super().__init__(
            f"Token rate limit of {limit} tokens per minute exceeded. Retry after {retry_after}s.",
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit-Tokens": str(limit),
                "X-RateLimit-Remaining-Tokens": str(max(remaining, 0)),
            },
        )
        self.retry_after = retry_after
        self.limit = limit
        self.remaining = remaining


class RequestTooLargeError(GatewayError):
    status_code = 400
    error_type = "request_too_large"

    def __init__(self, *, requested: int, limit: int) -> None:
        super().__init__(
            f"This request needs about {requested} tokens, more than the {limit} tokens per minute "
            "allowed for this API key. Reduce the prompt or max_tokens."
        )


class UpstreamUnavailableError(GatewayError):
    status_code = 503
    error_type = "upstream_unavailable"
    default_message = "All model providers are currently unavailable. Please retry shortly."


def error_response(
    request: Request, status_code: int, error_type: str, message: str, headers: dict[str, str] | None = None
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None) or f"req_{uuid.uuid4().hex[:20]}"
    body: dict[str, Any] = {"error": {"type": error_type, "message": message, "request_id": request_id}}
    return JSONResponse(
        body, status_code=status_code, headers={**(headers or {}), "x-request-id": request_id}
    )


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(GatewayError)
    async def gateway_error(request: Request, exc: GatewayError) -> JSONResponse:
        return error_response(request, exc.status_code, exc.error_type, exc.message, exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = exc.errors()
        if any(err["type"] == "json_invalid" for err in errors):
            message = "The request body must be valid JSON."
        else:
            fields = sorted({".".join(str(p) for p in err["loc"][1:]) for err in errors} - {""})
            message = (
                f"Missing or invalid fields: {', '.join(fields)}."
                if fields
                else "The request body must be a JSON object."
            )
        return error_response(request, 400, "invalid_request", message)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        error_type = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "invalid_request")
        message = {404: "Resource not found.", 405: "Method not allowed."}.get(
            exc.status_code, "Invalid request."
        )
        return error_response(request, exc.status_code, error_type, message)

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
        response = error_response(request, 500, GatewayError.error_type, GatewayError.default_message)
        logger.exception("unhandled error request_id=%s", response.headers["x-request-id"])
        return response
