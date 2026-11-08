from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from task4_model_router.db import Database


class FakeClock:
    def __init__(self, start: float = 1_700_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class HostRoutingTransport(httpx.AsyncBaseTransport):
    """Sends each request to the in-process ASGI app registered for its host."""

    def __init__(self, apps: dict[str, FastAPI]) -> None:
        self._transports = {host: httpx.ASGITransport(app=app) for host, app in apps.items()}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self._transports[request.url.host].handle_async_request(request)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "gateway.db"


@pytest.fixture
def database(db_path: Path) -> Iterator[Database]:
    db = Database(db_path)
    yield db
    db.close()
