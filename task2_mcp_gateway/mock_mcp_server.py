"""A small downstream MCP server speaking JSON-RPC over HTTP.

It implements just enough of the Streamable HTTP transport for the gateway:
``initialize``, ``ping``, ``tools/list`` and ``tools/call`` on ``POST /mcp``.
Every request it receives is recorded so tests can prove that denied calls
never reach it.

Run::

    python -m task2_mcp_gateway.mock_mcp_server   # listens on :9100
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_customer_record",
        "description": "Fetch a customer record by id.",
        "inputSchema": {
            "type": "object",
            "properties": {"customer_id": {"type": "string", "pattern": "^CUST-[0-9]{5}$"}},
            "required": ["customer_id"],
        },
    },
    {
        "name": "trigger_refund",
        "description": "Issue a refund to a customer.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string"},
                "amount": {"type": "number", "exclusiveMinimum": 0},
                "reason": {"type": "string", "minLength": 10},
            },
            "required": ["customer_id", "amount", "reason"],
        },
    },
    {
        "name": "admin_reset_key",
        "description": "Rotate the API signing key for a tenant. Administrators only.",
        "inputSchema": {
            "type": "object",
            "properties": {"tenant_id": {"type": "string"}},
            "required": ["tenant_id"],
        },
    },
]


def create_mock_server(*, stream_responses: bool = False) -> FastAPI:
    """Build the mock. ``stream_responses`` answers with SSE instead of JSON."""
    app = FastAPI(title="Mock MCP Server")
    app.state.received = []

    def reply(payload: dict[str, Any], session_id: str | None = None) -> Response:
        headers = {"mcp-session-id": session_id} if session_id else {}
        if stream_responses:
            body = f"event: message\ndata: {json.dumps(payload)}\n\n"
            return Response(body, media_type="text/event-stream", headers=headers)
        return JSONResponse(payload, headers=headers)

    @app.post("/mcp")
    async def mcp(request: Request) -> Response:
        message = await request.json()
        app.state.received.append({"message": message, "headers": dict(request.headers)})
        method, request_id = message.get("method"), message.get("id")

        if "id" not in message:
            return Response(status_code=202)
        if method == "initialize":
            result = {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "mock-mcp-server", "version": "1.0.0"},
            }
            return reply({"jsonrpc": "2.0", "id": request_id, "result": result}, uuid.uuid4().hex)
        if method == "ping":
            return reply({"jsonrpc": "2.0", "id": request_id, "result": {}})
        if method == "tools/list":
            return reply({"jsonrpc": "2.0", "id": request_id, "result": {"tools": TOOLS}})
        if method == "tools/call":
            return reply(_call_tool(request_id, message.get("params") or {}))
        return reply(
            {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": "Method not found"}}
        )

    return app


def _call_tool(request_id: Any, params: dict[str, Any]) -> dict[str, Any]:
    name, args = params.get("name"), params.get("arguments") or {}
    if name == "get_customer_record":
        data = {"customer_id": args.get("customer_id"), "name": "Ava Thompson", "plan": "enterprise"}
    elif name == "trigger_refund":
        data = {"refund_id": f"rf_{uuid.uuid4().hex[:12]}", "status": "processed"}
    elif name == "admin_reset_key":
        data = {"tenant_id": args.get("tenant_id"), "key_rotated": True}
    else:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32602, "message": f"Unknown tool: {name}"},
        }
    result = {
        "content": [{"type": "text", "text": json.dumps(data)}],
        "structuredContent": data,
        "isError": False,
    }
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def main() -> None:
    import uvicorn

    uvicorn.run(
        create_mock_server(),
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "9100")),
        log_level="info",
    )


if __name__ == "__main__":
    main()
