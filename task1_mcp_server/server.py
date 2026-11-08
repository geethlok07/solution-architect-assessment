"""Customer-support MCP server over stdio.

Exposes two tools:

* ``get_customer_record`` - read-only lookup by customer id.
* ``trigger_refund``      - issues a refund against a customer account.

Error model (MCP specification, "Error Handling"):

* Protocol errors -> JSON-RPC ``error`` object.
  Unknown tool or arguments that violate the input schema -> ``-32602``.
  Anything unexpected inside the server -> ``-32603`` (details stay in the log).
* Tool execution errors -> a normal result with ``isError: true`` so the
  model can read the message and self-correct (for example, unknown customer).

Stdio discipline: stdout is the JSON-RPC wire and nothing else writes to it.
All diagnostics go to stderr through ``logging`` and ``print`` is banned by the
linter. While serving, the transport points file descriptor 1 at stderr and
``sys.stdout`` is redirected to stderr as well, so stray output from
third-party code cannot reach the wire, whether it is written straight away
or sits in Python's stdout buffer until the process exits.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import anyio
from mcp import types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import MCPError
from mcp.shared.message import SessionMessage
from pydantic import BaseModel, ValidationError

from .models import GetCustomerRecordInput, TriggerRefundInput, describe_validation_error
from .store import CustomerNotFoundError, CustomerStore, RefundRejectedError

SERVER_NAME = "customer-support"
SERVER_VERSION = "1.0.0"

logger = logging.getLogger("task1_mcp_server")


def configure_logging() -> None:
    """Route every log record to stderr. Never touch stdout."""
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())


ToolHandler = Callable[[Any], Awaitable[types.CallToolResult]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    title: str
    description: str
    input_model: type[BaseModel]
    handler: ToolHandler
    annotations: types.ToolAnnotations

    def to_mcp(self) -> types.Tool:
        return types.Tool(
            name=self.name,
            title=self.title,
            description=self.description,
            input_schema=self.input_model.model_json_schema(),
            annotations=self.annotations,
        )


def _text_result(payload: dict[str, Any], *, is_error: bool = False) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(payload, separators=(",", ":")))],
        structured_content=None if is_error else payload,
        is_error=is_error,
    )


class CustomerSupportTools:
    """Tool implementations. Pure business logic; no protocol concerns."""

    def __init__(self, store: CustomerStore) -> None:
        self._store = store

    async def get_customer_record(self, args: GetCustomerRecordInput) -> types.CallToolResult:
        try:
            record = self._store.get(args.customer_id)
        except CustomerNotFoundError:
            return _text_result(
                {"error": "customer_not_found", "customer_id": args.customer_id},
                is_error=True,
            )
        return _text_result(record)

    async def trigger_refund(self, args: TriggerRefundInput) -> types.CallToolResult:
        try:
            refund = self._store.refund(args.customer_id, args.amount, args.reason)
        except CustomerNotFoundError:
            return _text_result(
                {"error": "customer_not_found", "customer_id": args.customer_id},
                is_error=True,
            )
        except RefundRejectedError as exc:
            return _text_result(
                {"error": "refund_rejected", "customer_id": args.customer_id, "detail": str(exc)},
                is_error=True,
            )
        logger.info(
            "refund issued refund_id=%s customer_id=%s amount=%.2f",
            refund["refund_id"],
            args.customer_id,
            args.amount,
        )
        return _text_result(refund)


def build_tool_registry(tools: CustomerSupportTools) -> dict[str, ToolSpec]:
    specs = [
        ToolSpec(
            name="get_customer_record",
            title="Get customer record",
            description="Fetch a customer's profile and account balance by customer id.",
            input_model=GetCustomerRecordInput,
            handler=tools.get_customer_record,
            annotations=types.ToolAnnotations(read_only_hint=True, idempotent_hint=True),
        ),
        ToolSpec(
            name="trigger_refund",
            title="Trigger refund",
            description=(
                "Issue a refund to a customer. The amount must be positive with at most two "
                "decimal places, and a reason of at least 10 characters is required for audit."
            ),
            input_model=TriggerRefundInput,
            handler=tools.trigger_refund,
            annotations=types.ToolAnnotations(destructive_hint=True, idempotent_hint=False),
        ),
    ]
    return {spec.name: spec for spec in specs}


def create_server(store: CustomerStore | None = None) -> Server[Any]:
    registry = build_tool_registry(CustomerSupportTools(store or CustomerStore.with_sample_data()))

    async def list_tools(ctx: Any, params: types.PaginatedRequestParams | None) -> types.ListToolsResult:
        return types.ListToolsResult(tools=[spec.to_mcp() for spec in registry.values()])

    async def call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        spec = registry.get(params.name)
        if spec is None:
            raise MCPError(
                code=types.INVALID_PARAMS,
                message=f"Unknown tool: {params.name}",
                data={"available_tools": sorted(registry)},
            )

        try:
            args = spec.input_model.model_validate(params.arguments or {})
        except ValidationError as exc:
            errors = describe_validation_error(exc)
            logger.info(
                "rejected arguments tool=%s fields=%s",
                spec.name,
                sorted({e["field"] for e in errors}),
            )
            raise MCPError(
                code=types.INVALID_PARAMS,
                message=f"Invalid arguments for tool '{spec.name}'",
                data={"tool": spec.name, "errors": errors},
            ) from None

        try:
            return await spec.handler(args)
        except MCPError:
            raise
        except Exception:
            incident = uuid.uuid4().hex[:12]
            logger.exception("tool failure tool=%s incident=%s", spec.name, incident)
            raise MCPError(
                code=types.INTERNAL_ERROR,
                message="Internal error while executing tool",
                data={"incident_id": incident},
            ) from None

    return Server(
        SERVER_NAME,
        version=SERVER_VERSION,
        instructions="Customer-support tools: look up customer records and issue refunds.",
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )


class _MalformedMessageReporter:
    """Read-stream wrapper that answers unparseable lines on the wire.

    The stdio transport hands undecodable lines to the server as exception
    objects, which the dispatcher only logs. JSON-RPC 2.0 requires a reply:
    ``-32700 Parse error`` for invalid JSON and ``-32600 Invalid Request`` for
    JSON that is not a JSON-RPC message, both with ``id: null``. Writing the
    reply through the transport's own write stream keeps a single writer on
    stdout, so responses can never interleave.
    """

    def __init__(self, inner: Any, writer: Any) -> None:
        self._inner = inner
        self._writer = writer

    async def _report(self, exc: Exception) -> None:
        is_parse_error = isinstance(exc, ValidationError) and any(
            err["type"] == "json_invalid" for err in exc.errors()
        )
        code, message = (
            (types.PARSE_ERROR, "Parse error")
            if is_parse_error
            else (types.INVALID_REQUEST, "Invalid Request")
        )
        logger.warning("discarded malformed message from client: %s", message)
        error = types.JSONRPCError(jsonrpc="2.0", id=None, error=types.ErrorData(code=code, message=message))
        await self._writer.send(SessionMessage(error))

    async def receive(self) -> Any:
        while True:
            item = await self._inner.receive()
            if not isinstance(item, Exception):
                return item
            await self._report(item)

    def __aiter__(self) -> _MalformedMessageReporter:
        return self

    async def __anext__(self) -> Any:
        while True:
            item = await self._inner.__anext__()
            if not isinstance(item, Exception):
                return item
            await self._report(item)

    async def aclose(self) -> None:
        await self._inner.aclose()

    async def __aenter__(self) -> _MalformedMessageReporter:
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self._inner.__aexit__(*exc_info)


async def serve() -> None:
    server = create_server()
    options = server.create_initialization_options(notification_options=NotificationOptions())
    logger.info("starting %s v%s on stdio", SERVER_NAME, SERVER_VERSION)
    async with stdio_server() as (read_stream, write_stream):
        # The transport diverts file descriptor 1, but text written with
        # print() can sit in sys.stdout's buffer and be flushed at exit, after
        # the descriptor has been handed back. Redirecting sys.stdout for the
        # whole session sends such writes straight to stderr instead.
        original_stdout = sys.stdout
        try:
            with contextlib.redirect_stdout(sys.stderr):
                await server.run(_MalformedMessageReporter(read_stream, write_stream), write_stream, options)
        finally:
            # Anything already buffered drains while fd 1 still points at stderr.
            original_stdout.flush()
    logger.info("stdin closed, shutting down at %s", datetime.now(UTC).isoformat())


def main() -> None:
    configure_logging()
    try:
        anyio.run(serve)
    except KeyboardInterrupt:
        logger.info("interrupted")


if __name__ == "__main__":
    main()
