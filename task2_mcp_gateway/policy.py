"""Method- and tool-level authorization.

Decisions are made on the parsed JSON-RPC message and the authenticated
principal only. The policy is deny-by-default: methods outside the MCP
surface the gateway knows about are rejected before reaching the downstream.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .auth import Principal, Role
from .jsonrpc import (
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    UNAUTHORIZED_TOOL_CALL,
    JsonRpcError,
    JsonRpcMessage,
)

PRIVILEGED_TOOL_PREFIX = "admin_"

# MCP tool names: 1-128 characters from this set (MCP spec, "Tool names").
# Rejecting everything else closes Unicode look-alike and whitespace tricks.
_TOOL_NAME = re.compile(r"[A-Za-z0-9_.\-]{1,128}")

FORWARDED_METHODS = frozenset(
    {
        "initialize",
        "ping",
        "tools/list",
        "tools/call",
        "resources/list",
        "resources/templates/list",
        "resources/read",
        "prompts/list",
        "prompts/get",
        "completion/complete",
        "logging/setLevel",
    }
)


@dataclass(frozen=True)
class Decision:
    allowed: bool
    tool: str | None = None
    reason: str = "allowed"


def is_privileged_tool(name: str) -> bool:
    # Case-insensitive so "Admin_reset_key" cannot slip past a downstream
    # that resolves tool names without regard to case.
    return name.lower().startswith(PRIVILEGED_TOOL_PREFIX)


def authorize(principal: Principal, message: JsonRpcMessage) -> Decision:
    """Return the decision, or raise ``JsonRpcError`` for a deny the client must see."""
    if message.method.startswith("notifications/"):
        return Decision(allowed=True, reason="notification")

    if message.method not in FORWARDED_METHODS:
        raise JsonRpcError(METHOD_NOT_FOUND, "Method not found", request_id=message.id)

    if message.method != "tools/call":
        return Decision(allowed=True)

    if message.is_notification:
        raise JsonRpcError(INVALID_REQUEST, "Invalid Request: tools/call requires an id")

    tool = _tool_name(message)
    if is_privileged_tool(tool) and principal.role is not Role.ADMIN:
        raise JsonRpcError(
            UNAUTHORIZED_TOOL_CALL,
            "Unauthorized Tool Call",
            request_id=message.id,
            data={"tool": tool, "required_role": Role.ADMIN.value},
        )
    return Decision(allowed=True, tool=tool)


def _tool_name(message: JsonRpcMessage) -> str:
    params: Any = message.params
    name = params.get("name") if isinstance(params, dict) else None
    if not isinstance(name, str) or not _TOOL_NAME.fullmatch(name):
        raise JsonRpcError(
            INVALID_PARAMS,
            "Invalid params: params.name must be a valid tool name",
            request_id=message.id,
        )
    return name


def filter_tools_for(principal: Principal, tools: list[Any]) -> list[Any]:
    """Hide privileged tools from non-admin callers (optional ``tools/list`` filtering)."""
    if principal.role is Role.ADMIN:
        return tools
    return [
        tool
        for tool in tools
        if not (isinstance(tool, dict) and is_privileged_tool(str(tool.get("name", ""))))
    ]
