from __future__ import annotations

import itertools
import json
import os
import queue
import subprocess
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_VERSION = "2025-06-18"


class StdioClient:
    """Minimal JSON-RPC client that drives the server as a real subprocess.

    Every line the server writes to stdout is recorded in ``raw_stdout`` so the
    tests can assert that the wire carried JSON-RPC and nothing else.
    """

    def __init__(self, argv: list[str], env: dict[str, str] | None = None) -> None:
        self._proc = subprocess.Popen(
            argv,
            cwd=REPO_ROOT,
            env={**os.environ, **(env or {})},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._ids = itertools.count(1)
        self._lines: queue.Queue[str] = queue.Queue()
        self.raw_stdout: list[str] = []
        self._stderr: list[str] = []
        self._pumps = [
            threading.Thread(target=self._pump_stdout, daemon=True),
            threading.Thread(target=self._pump_stderr, daemon=True),
        ]
        for pump in self._pumps:
            pump.start()

    def _pump_stdout(self) -> None:
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            self.raw_stdout.append(line)
            self._lines.put(line)

    def _pump_stderr(self) -> None:
        assert self._proc.stderr is not None
        for line in self._proc.stderr:
            self._stderr.append(line)

    @property
    def stderr(self) -> str:
        return "".join(self._stderr)

    def send_raw(self, line: str) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write(line + "\n")
        self._proc.stdin.flush()

    def send(self, message: dict[str, Any]) -> None:
        self.send_raw(json.dumps(message))

    def read(self, timeout: float = 10.0) -> dict[str, Any]:
        return json.loads(self._lines.get(timeout=timeout))

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        request_id = next(self._ids)
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        self.send(message)
        response = self.read()
        assert response.get("id") == request_id, response
        return response

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"name": name}
        if arguments is not None:
            params["arguments"] = arguments
        return self.request("tools/call", params)

    def initialize(self) -> dict[str, Any]:
        response = self.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0"},
            },
        )
        self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return response

    def close(self) -> None:
        if self._proc.stdin and not self._proc.stdin.closed:
            self._proc.stdin.close()
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        # Wait until everything the process wrote has been collected, so
        # assertions on raw_stdout and stderr see the complete output.
        for pump in self._pumps:
            pump.join(timeout=5)


@pytest.fixture
def server() -> Iterator[StdioClient]:
    client = StdioClient([sys.executable, "-m", "task1_mcp_server.server"])
    client.initialize()
    yield client
    client.close()
