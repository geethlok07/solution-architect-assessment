from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from task2_mcp_gateway.auth import Role, mint_token
from task2_mcp_gateway.config import GatewaySettings
from task2_mcp_gateway.gateway import create_app
from task2_mcp_gateway.mock_mcp_server import create_mock_server

DOWNSTREAM_URL = "http://downstream.internal/mcp"


@pytest.fixture
def settings() -> GatewaySettings:
    return GatewaySettings(downstream_url=DOWNSTREAM_URL, jwt_secret="test-secret-with-at-least-32-bytes!!")


@pytest.fixture
def downstream() -> FastAPI:
    return create_mock_server()


@pytest.fixture
def token(settings: GatewaySettings) -> Callable[..., str]:
    def _mint(role: Role | str, subject: str = "alice", **overrides: Any) -> str:
        return mint_token(replace(settings, **overrides), subject=subject, role=role)

    return _mint


@pytest.fixture
async def make_gateway(
    settings: GatewaySettings, downstream: FastAPI
) -> AsyncIterator[Callable[..., httpx.AsyncClient]]:
    clients: list[httpx.AsyncClient] = []

    def _make(
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        **setting_overrides: Any,
    ) -> httpx.AsyncClient:
        upstream = httpx.AsyncClient(transport=transport or httpx.ASGITransport(app=downstream))
        app = create_app(replace(settings, **setting_overrides), http_client=upstream)
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway")
        clients.extend([upstream, client])
        return client

    yield _make
    for client in clients:
        await client.aclose()


@pytest.fixture
def gateway(make_gateway: Callable[..., httpx.AsyncClient]) -> httpx.AsyncClient:
    return make_gateway()


def rpc(
    method: str, params: dict[str, Any] | None = None, request_id: int | str | None = 1
) -> dict[str, Any]:
    message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if request_id is not None:
        message["id"] = request_id
    if params is not None:
        message["params"] = params
    return message


def bearer(value: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {value}"}
