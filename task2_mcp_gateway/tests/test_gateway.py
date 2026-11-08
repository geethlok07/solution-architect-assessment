from __future__ import annotations

import gzip
import json
import time
from collections.abc import Callable

import httpx
import jwt
import pytest
from fastapi import FastAPI

from task2_mcp_gateway.config import GatewaySettings
from task2_mcp_gateway.mock_mcp_server import TOOLS, create_mock_server

from .conftest import bearer, rpc

UNAUTHORIZED_TOOL_CALL = -32001


def _downstream_methods(downstream: FastAPI) -> list[str]:
    return [entry["message"].get("method") for entry in downstream.state.received]


# --- authentication --------------------------------------------------------


async def test_missing_token_is_rejected_before_parsing(
    gateway: httpx.AsyncClient, downstream: FastAPI
) -> None:
    response = await gateway.post("/mcp", content=b"not even json")

    assert response.status_code == 401
    assert response.headers["www-authenticate"].startswith("Bearer")
    assert downstream.state.received == []


@pytest.mark.parametrize("header", ["Basic YWxpY2U6c2VjcmV0", "Bearer", "Bearer    ", "token-without-scheme"])
async def test_malformed_authorization_header_is_rejected(gateway: httpx.AsyncClient, header: str) -> None:
    response = await gateway.post("/mcp", json=rpc("tools/list"), headers={"Authorization": header})
    assert response.status_code == 401


async def test_tampered_token_is_rejected(gateway: httpx.AsyncClient, token: Callable[..., str]) -> None:
    header, payload, signature = token("viewer").split(".")
    claims = json.loads(jwt.utils.base64url_decode(payload))
    claims["role"] = "admin"
    forged_payload = jwt.utils.base64url_encode(json.dumps(claims).encode()).decode()

    response = await gateway.post(
        "/mcp", json=rpc("tools/list"), headers=bearer(f"{header}.{forged_payload}.{signature}")
    )

    assert response.status_code == 401


async def test_unsigned_alg_none_token_is_rejected(
    gateway: httpx.AsyncClient, settings: GatewaySettings
) -> None:
    claims = {
        "sub": "mallory",
        "role": "admin",
        "aud": settings.jwt_audience,
        "iss": settings.jwt_issuer,
        "exp": int(time.time()) + 60,
    }
    unsigned = jwt.encode(claims, key=None, algorithm="none")

    response = await gateway.post("/mcp", json=rpc("tools/list"), headers=bearer(unsigned))

    assert response.status_code == 401


@pytest.mark.parametrize(
    "overrides",
    [
        {"jwt_secret": "a-completely-different-signing-secret!!"},
        {"jwt_audience": "some-other-service"},
        {"jwt_issuer": "https://evil.example"},
    ],
)
async def test_token_for_wrong_key_audience_or_issuer_is_rejected(
    gateway: httpx.AsyncClient, token: Callable[..., str], overrides: dict[str, str]
) -> None:
    response = await gateway.post("/mcp", json=rpc("tools/list"), headers=bearer(token("admin", **overrides)))
    assert response.status_code == 401


async def test_expired_token_is_rejected(gateway: httpx.AsyncClient, settings: GatewaySettings) -> None:
    claims = {
        "sub": "alice",
        "role": "admin",
        "aud": settings.jwt_audience,
        "iss": settings.jwt_issuer,
        "exp": int(time.time()) - 5,
    }
    expired = jwt.encode(claims, settings.jwt_secret, algorithm="HS256")

    response = await gateway.post("/mcp", json=rpc("tools/list"), headers=bearer(expired))

    assert response.status_code == 401
    assert response.json()["message"] == "token has expired"


async def test_unknown_role_is_rejected(gateway: httpx.AsyncClient, token: Callable[..., str]) -> None:
    response = await gateway.post("/mcp", json=rpc("tools/list"), headers=bearer(token("superuser")))
    assert response.status_code == 401


# --- transparent forwarding ------------------------------------------------


async def test_tools_list_is_forwarded_transparently(
    gateway: httpx.AsyncClient, token: Callable[..., str], downstream: FastAPI
) -> None:
    response = await gateway.post(
        "/mcp", json=rpc("tools/list", request_id="list-1"), headers=bearer(token("viewer"))
    )

    assert response.status_code == 200
    assert response.json() == {"jsonrpc": "2.0", "id": "list-1", "result": {"tools": TOOLS}}
    assert _downstream_methods(downstream) == ["tools/list"]


async def test_request_body_reaches_downstream_byte_for_byte(
    gateway: httpx.AsyncClient, token: Callable[..., str], downstream: FastAPI
) -> None:
    raw = b'{"jsonrpc":"2.0","id":7,"method":"tools/list","params":{"cursor":"abc"}}'
    await gateway.post(
        "/mcp", content=raw, headers={**bearer(token("viewer")), "Content-Type": "application/json"}
    )

    assert downstream.state.received[0]["message"] == json.loads(raw)


async def test_credentials_stay_at_the_gateway_and_identity_is_propagated(
    gateway: httpx.AsyncClient, token: Callable[..., str], downstream: FastAPI
) -> None:
    await gateway.post(
        "/mcp",
        json=rpc("tools/list"),
        headers={
            **bearer(token("viewer", subject="bob")),
            "Mcp-Session-Id": "sess-42",
            "X-Request-ID": "req-1",
        },
    )

    forwarded = downstream.state.received[0]["headers"]
    assert "authorization" not in forwarded
    assert forwarded["mcp-session-id"] == "sess-42"
    assert forwarded["x-request-id"] == "req-1"
    assert forwarded["x-authenticated-subject"] == "bob"
    assert forwarded["x-authenticated-role"] == "viewer"


async def test_session_id_from_downstream_is_returned(
    gateway: httpx.AsyncClient, token: Callable[..., str]
) -> None:
    params = {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "t", "version": "1"},
    }
    response = await gateway.post("/mcp", json=rpc("initialize", params), headers=bearer(token("viewer")))

    assert response.status_code == 200
    assert response.headers["mcp-session-id"]


async def test_notifications_are_forwarded(
    gateway: httpx.AsyncClient, token: Callable[..., str], downstream: FastAPI
) -> None:
    response = await gateway.post(
        "/mcp", json=rpc("notifications/initialized", request_id=None), headers=bearer(token("viewer"))
    )

    assert response.status_code == 202
    assert _downstream_methods(downstream) == ["notifications/initialized"]


async def test_sse_responses_are_streamed_through(make_gateway, token: Callable[..., str]) -> None:
    streaming_downstream = create_mock_server(stream_responses=True)
    gateway = make_gateway(transport=httpx.ASGITransport(app=streaming_downstream))

    response = await gateway.post("/mcp", json=rpc("tools/list"), headers=bearer(token("viewer")))

    assert response.headers["content-type"].startswith("text/event-stream")
    data_line = next(line for line in response.text.splitlines() if line.startswith("data:"))
    assert json.loads(data_line[5:])["result"]["tools"] == TOOLS


# --- tool-level authorization ----------------------------------------------


async def test_viewer_can_call_regular_tools(gateway: httpx.AsyncClient, token: Callable[..., str]) -> None:
    response = await gateway.post(
        "/mcp",
        json=rpc("tools/call", {"name": "get_customer_record", "arguments": {"customer_id": "CUST-10001"}}),
        headers=bearer(token("viewer")),
    )
    assert response.json()["result"]["structuredContent"]["customer_id"] == "CUST-10001"


async def test_viewer_calling_admin_tool_is_intercepted(
    gateway: httpx.AsyncClient, token: Callable[..., str], downstream: FastAPI
) -> None:
    response = await gateway.post(
        "/mcp",
        json=rpc("tools/call", {"name": "admin_reset_key", "arguments": {"tenant_id": "t-1"}}, request_id=42),
        headers=bearer(token("viewer")),
    )

    assert response.status_code == 200
    assert response.json() == {
        "jsonrpc": "2.0",
        "id": 42,
        "error": {
            "code": UNAUTHORIZED_TOOL_CALL,
            "message": "Unauthorized Tool Call",
            "data": {"tool": "admin_reset_key", "required_role": "admin"},
        },
    }
    assert downstream.state.received == [], "a denied call must never reach the downstream server"


async def test_admin_can_call_admin_tools(
    gateway: httpx.AsyncClient, token: Callable[..., str], downstream: FastAPI
) -> None:
    response = await gateway.post(
        "/mcp",
        json=rpc("tools/call", {"name": "admin_reset_key", "arguments": {"tenant_id": "t-1"}}),
        headers=bearer(token("admin")),
    )

    assert response.json()["result"]["structuredContent"] == {"tenant_id": "t-1", "key_rotated": True}
    assert _downstream_methods(downstream) == ["tools/call"]


@pytest.mark.parametrize("name", ["Admin_reset_key", "ADMIN_RESET_KEY", "aDmIn_purge"])
async def test_case_variants_of_admin_prefix_are_denied(
    gateway: httpx.AsyncClient, token: Callable[..., str], downstream: FastAPI, name: str
) -> None:
    response = await gateway.post(
        "/mcp", json=rpc("tools/call", {"name": name}), headers=bearer(token("viewer"))
    )

    assert response.json()["error"]["code"] == UNAUTHORIZED_TOOL_CALL
    assert downstream.state.received == []


@pytest.mark.parametrize(
    "params",
    [
        {"name": "\u0430dmin_reset_key"},  # Cyrillic 'a' look-alike
        {"name": " admin_reset_key"},  # leading space
        {"name": "admin_reset_key\u200b"},  # zero-width space
        {"name": ""},
        {"name": 123},
        {"name": None},
        {"arguments": {}},
        {"name": "x" * 129},
    ],
)
async def test_invalid_tool_names_are_rejected_as_invalid_params(
    gateway: httpx.AsyncClient, token: Callable[..., str], downstream: FastAPI, params: dict
) -> None:
    response = await gateway.post("/mcp", json=rpc("tools/call", params), headers=bearer(token("viewer")))

    assert response.json()["error"]["code"] == -32602
    assert downstream.state.received == []


async def test_tools_call_without_params_is_invalid_params(
    gateway: httpx.AsyncClient, token: Callable[..., str]
) -> None:
    response = await gateway.post("/mcp", json=rpc("tools/call"), headers=bearer(token("admin")))
    assert response.json()["error"]["code"] == -32602


# --- JSON-RPC wire format --------------------------------------------------


@pytest.mark.parametrize(
    ("body", "code", "expected_id"),
    [
        (b"{not json", -32700, None),
        (b"\xff\xfe", -32700, None),
        (b"[]", -32600, None),
        (b'[{"jsonrpc":"2.0","id":1,"method":"tools/list"}]', -32600, None),
        (b'"tools/list"', -32600, None),
        (b'{"jsonrpc":"1.0","id":5,"method":"tools/list"}', -32600, 5),
        (b'{"jsonrpc":"2.0","id":5}', -32600, 5),
        (b'{"jsonrpc":"2.0","id":5,"method":42}', -32600, 5),
        (b'{"jsonrpc":"2.0","id":{"x":1},"method":"tools/list"}', -32600, None),
        (b'{"jsonrpc":"2.0","id":true,"method":"tools/list"}', -32600, None),
        (b'{"jsonrpc":"2.0","id":5,"method":"tools/list","params":"x"}', -32600, 5),
        (b'{"jsonrpc":"2.0","id":5,"result":{}}', -32600, 5),
    ],
)
async def test_malformed_jsonrpc_is_answered_with_standard_errors(
    gateway: httpx.AsyncClient,
    token: Callable[..., str],
    downstream: FastAPI,
    body: bytes,
    code: int,
    expected_id: int | None,
) -> None:
    response = await gateway.post("/mcp", content=body, headers=bearer(token("admin")))

    assert response.status_code == 200
    assert response.json()["jsonrpc"] == "2.0"
    assert response.json()["error"]["code"] == code
    assert response.json()["id"] == expected_id
    assert downstream.state.received == []


async def test_unknown_methods_are_not_forwarded(
    gateway: httpx.AsyncClient, token: Callable[..., str], downstream: FastAPI
) -> None:
    response = await gateway.post(
        "/mcp", json=rpc("sampling/createMessage", {}), headers=bearer(token("admin"))
    )

    assert response.json()["error"]["code"] == -32601
    assert downstream.state.received == []


async def test_oversized_body_is_rejected(make_gateway, token: Callable[..., str]) -> None:
    gateway = make_gateway(max_body_bytes=256)
    big = rpc("tools/call", {"name": "trigger_refund", "arguments": {"reason": "x" * 1024}})

    response = await gateway.post("/mcp", json=big, headers=bearer(token("admin")))

    assert response.status_code == 413


async def test_oversized_streamed_body_without_content_length_is_rejected(
    make_gateway, token: Callable[..., str], downstream: FastAPI
) -> None:
    gateway = make_gateway(max_body_bytes=256)

    async def chunks():
        for _ in range(64):
            yield b"x" * 64

    response = await gateway.post("/mcp", content=chunks(), headers=bearer(token("admin")))

    assert response.status_code == 413
    assert downstream.state.received == []


async def test_compressed_downstream_responses_reach_the_client_decoded(
    make_gateway, token: Callable[..., str]
) -> None:
    payload = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"tools": TOOLS}}).encode()

    def gzipped(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=gzip.compress(payload),
            headers={"content-type": "application/json", "content-encoding": "gzip"},
        )

    gateway = make_gateway(transport=httpx.MockTransport(gzipped))
    response = await gateway.post("/mcp", json=rpc("tools/list"), headers=bearer(token("viewer")))

    assert "content-encoding" not in response.headers
    assert response.json()["result"]["tools"] == TOOLS


async def test_get_and_delete_are_declined(gateway: httpx.AsyncClient) -> None:
    assert (await gateway.get("/mcp")).status_code == 405
    assert (await gateway.delete("/mcp")).status_code == 405


# --- downstream failures ---------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "status"),
    [(httpx.ConnectError("refused"), 502), (httpx.ReadTimeout("slow"), 504)],
)
async def test_downstream_failure_maps_to_jsonrpc_error(
    make_gateway, token: Callable[..., str], exc: Exception, status: int
) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise exc

    gateway = make_gateway(transport=httpx.MockTransport(fail))

    response = await gateway.post(
        "/mcp", json=rpc("tools/list", request_id=9), headers=bearer(token("viewer"))
    )

    assert response.status_code == status
    body = response.json()
    assert body["id"] == 9
    assert body["error"]["code"] == -32603
    assert "refused" not in response.text and "slow" not in response.text


# --- optional tools/list filtering ------------------------------------------


@pytest.mark.parametrize("stream_responses", [False, True])
async def test_filtering_hides_admin_tools_from_viewers_only(
    make_gateway, token: Callable[..., str], stream_responses: bool
) -> None:
    downstream = create_mock_server(stream_responses=stream_responses)
    gateway = make_gateway(transport=httpx.ASGITransport(app=downstream), filter_tools_list=True)

    def names(response: httpx.Response) -> set[str]:
        text = response.text
        if stream_responses:
            text = next(line for line in text.splitlines() if line.startswith("data:"))[5:]
        return {tool["name"] for tool in json.loads(text)["result"]["tools"]}

    viewer = await gateway.post("/mcp", json=rpc("tools/list"), headers=bearer(token("viewer")))
    admin = await gateway.post("/mcp", json=rpc("tools/list"), headers=bearer(token("admin")))

    assert names(viewer) == {"get_customer_record", "trigger_refund"}
    assert names(admin) == {"get_customer_record", "trigger_refund", "admin_reset_key"}
