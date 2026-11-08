"""JSON-RPC 2.0 wire-format parsing for the gateway.

Only the envelope is validated here; ``params`` contents are method specific
and are inspected by the policy layer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
UNAUTHORIZED_TOOL_CALL = -32001

RequestId = str | int | None


@dataclass(frozen=True)
class JsonRpcMessage:
    method: str
    id: RequestId
    params: dict[str, Any] | list[Any] | None
    is_notification: bool


class JsonRpcError(Exception):
    def __init__(self, code: int, message: str, *, request_id: RequestId = None, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.request_id = request_id
        self.data = data

    def to_response(self) -> dict[str, Any]:
        return error_response(self.request_id, self.code, self.message, self.data)


def error_response(request_id: RequestId, code: int, message: str, data: Any = None) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


def _valid_id(value: Any) -> bool:
    # bool is a subclass of int in Python but is not a legal JSON-RPC id.
    return value is None or (isinstance(value, (str, int)) and not isinstance(value, bool))


def parse_message(body: bytes) -> JsonRpcMessage:
    """Parse a single JSON-RPC request or notification.

    Raises ``JsonRpcError`` with the code mandated by JSON-RPC 2.0 for each
    failure. The request ``id`` is echoed whenever it could be recovered.
    """
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise JsonRpcError(PARSE_ERROR, "Parse error") from None

    if isinstance(payload, list):
        # JSON-RPC batching was removed from MCP in protocol revision 2025-06-18.
        raise JsonRpcError(INVALID_REQUEST, "Batch requests are not supported")
    if not isinstance(payload, dict):
        raise JsonRpcError(INVALID_REQUEST, "Invalid Request")

    raw_id = payload.get("id")
    request_id = raw_id if _valid_id(raw_id) else None

    if payload.get("jsonrpc") != "2.0":
        raise JsonRpcError(INVALID_REQUEST, "Invalid Request: jsonrpc must be '2.0'", request_id=request_id)
    if not _valid_id(raw_id):
        raise JsonRpcError(INVALID_REQUEST, "Invalid Request: id must be a string or integer")
    if "result" in payload or "error" in payload:
        raise JsonRpcError(
            INVALID_REQUEST, "Invalid Request: responses are not accepted", request_id=request_id
        )

    method = payload.get("method")
    if not isinstance(method, str) or not method:
        raise JsonRpcError(INVALID_REQUEST, "Invalid Request: method must be a string", request_id=request_id)

    params = payload.get("params")
    if params is not None and not isinstance(params, (dict, list)):
        raise JsonRpcError(
            INVALID_REQUEST, "Invalid Request: params must be an object or array", request_id=request_id
        )

    return JsonRpcMessage(
        method=method,
        id=request_id,
        params=params,
        is_notification="id" not in payload,
    )
