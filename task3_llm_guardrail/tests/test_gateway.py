from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest
import uvicorn

from task3_llm_guardrail.gateway import GuardrailSettings, StreamRedactionSession, create_app
from task3_llm_guardrail.mock_llm import SCRIPTED_REPLY, create_mock_llm
from task3_llm_guardrail.redactor import REDACTED, redact
from task3_llm_guardrail.sse import SSEEvent, SSEParser

SECRETS = ["jane.doe@example.com", "123-45-6789", "4111 1111 1111 1111", "billing-team+refunds@acme-corp.io"]
UPSTREAM = "http://llm.internal/v1/chat/completions"


def _chat(content: str = "hi", *, stream: bool = True, model: str = "mock-assistant", seed: int = 0) -> dict:
    return {
        "model": model,
        "stream": stream,
        "seed": seed,
        "messages": [{"role": "user", "content": content}],
    }


def _events(raw: bytes) -> list[SSEEvent]:
    parser = SSEParser()
    return parser.feed(raw) + parser.close()


def _content(events: list[SSEEvent]) -> str:
    text = ""
    for event in events:
        if event.data and event.data != "[DONE]":
            for choice in json.loads(event.data)["choices"]:
                if isinstance(choice, dict):
                    text += (choice.get("delta") or {}).get("content") or ""
    return text


@pytest.fixture
async def gateway() -> AsyncIterator[httpx.AsyncClient]:
    upstream = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_mock_llm(ttft_ms=0, chunk_delay_ms=0))
    )
    app = create_app(GuardrailSettings(upstream_url=UPSTREAM), http_client=upstream)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway") as client:
        yield client
    await upstream.aclose()


async def test_streamed_response_is_redacted(gateway: httpx.AsyncClient) -> None:
    response = await gateway.post("/v1/chat/completions", json=_chat())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-accel-buffering"] == "no"
    for secret in SECRETS:
        assert secret not in response.text
    assert _content(_events(response.content)) == redact(SCRIPTED_REPLY)


@pytest.mark.parametrize("seed", range(25))
async def test_redaction_holds_for_any_upstream_chunking(gateway: httpx.AsyncClient, seed: int) -> None:
    prompt = "Reach me at sam.smith@corp.example or 4111-1111-1111-1111; SSN 078-05-1120. Thanks!"
    response = await gateway.post("/v1/chat/completions", json=_chat(prompt, model="mock-echo", seed=seed))

    assert (
        _content(_events(response.content))
        == f"Reach me at {REDACTED} or {REDACTED}; SSN {REDACTED}. Thanks!"
    )


async def test_stream_framing_is_preserved(gateway: httpx.AsyncClient) -> None:
    events = _events((await gateway.post("/v1/chat/completions", json=_chat())).content)
    chunks = [json.loads(e.data) for e in events if e.data != "[DONE]"]

    assert events[-1].data == "[DONE]"
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert all(c["object"] == "chat.completion.chunk" and c["id"] == chunks[0]["id"] for c in chunks)


async def test_non_streaming_response_is_redacted(gateway: httpx.AsyncClient) -> None:
    response = await gateway.post("/v1/chat/completions", json=_chat(stream=False))

    body = response.json()
    assert body["choices"][0]["message"]["content"] == redact(SCRIPTED_REPLY)
    assert body["usage"]["total_tokens"] > 0


async def test_invalid_request_body(gateway: httpx.AsyncClient) -> None:
    response = await gateway.post("/v1/chat/completions", content=b"{oops")
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request"


@pytest.mark.parametrize(
    ("upstream_status", "expected"),
    [(429, 429), (400, 400), (500, 502), (503, 502)],
)
async def test_upstream_errors_are_sanitized(upstream_status: int, expected: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(upstream_status, json={"error": "Traceback ... internal host 10.0.0.12"})

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(GuardrailSettings(upstream_url=UPSTREAM), http_client=upstream)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://g") as client:
        response = await client.post("/v1/chat/completions", json=_chat())

    assert response.status_code == expected
    assert "10.0.0.12" not in response.text and "Traceback" not in response.text
    assert response.json()["error"]["request_id"]


async def test_upstream_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(GuardrailSettings(upstream_url=UPSTREAM), http_client=upstream)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://g") as client:
        response = await client.post("/v1/chat/completions", json=_chat())

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "upstream_unavailable"


# --- session-level behaviour ---------------------------------------------------


def _chunk(choices: list[dict[str, Any]], **extra: Any) -> SSEEvent:
    payload = {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": choices,
        **extra,
    }
    return SSEEvent(data=json.dumps(payload))


def _run(session: StreamRedactionSession, events: list[SSEEvent]) -> list[SSEEvent]:
    out = b"".join(frame for event in events for frame in session.process(event))
    return _events(out)


def test_each_choice_has_its_own_redactor() -> None:
    session = StreamRedactionSession()
    events = [
        _chunk(
            [{"index": 0, "delta": {"content": "a: john"}}, {"index": 1, "delta": {"content": "b: 123-45"}}]
        ),
        _chunk(
            [{"index": 0, "delta": {"content": "@x.io ok"}}, {"index": 1, "delta": {"content": "-6789 ok"}}]
        ),
        _chunk(
            [
                {"index": 0, "delta": {}, "finish_reason": "stop"},
                {"index": 1, "delta": {}, "finish_reason": "stop"},
            ]
        ),
        SSEEvent(data="[DONE]"),
    ]
    per_choice: dict[int, str] = {0: "", 1: ""}
    for event in _run(session, events):
        if event.data != "[DONE]":
            for choice in json.loads(event.data)["choices"]:
                per_choice[choice["index"]] += choice.get("delta", {}).get("content") or ""

    assert per_choice == {0: f"a: {REDACTED} ok", 1: f"b: {REDACTED} ok"}


def test_held_text_is_flushed_when_stream_ends_without_finish_reason() -> None:
    session = StreamRedactionSession()
    out = _run(
        session, [_chunk([{"index": 0, "delta": {"content": "call 123-45-6789"}}]), SSEEvent(data="[DONE]")]
    )

    assert _content(out) == f"call {REDACTED}"
    assert out[-1].data == "[DONE]"


def test_chunks_that_are_entirely_held_are_not_emitted() -> None:
    session = StreamRedactionSession()
    assert session.process(_chunk([{"index": 0, "delta": {"content": "john"}}])) == []


def test_keepalive_comments_pass_through_and_error_events_are_scrubbed() -> None:
    session = StreamRedactionSession()
    assert session.process(SSEEvent(comments=["ping"])) == [b":ping\n\n"]
    (frame,) = session.process(SSEEvent(data='{"error":{"message":"bad input from a@b.co"}}'))
    assert b"a@b.co" not in frame and REDACTED.encode() in frame


def test_malformed_choices_do_not_break_the_stream() -> None:
    session = StreamRedactionSession()
    events = [
        _chunk(["not-a-choice", {"index": "zero", "delta": {"content": "mail a@b.co now"}}]),
        _chunk([{"index": 0, "delta": None, "finish_reason": "stop"}]),
        SSEEvent(data="[DONE]"),
    ]
    assert _content(_run(session, events)) == f"mail {REDACTED} now"


async def test_non_object_upstream_body_is_rejected_cleanly() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["unexpected", "shape"])

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(GuardrailSettings(upstream_url=UPSTREAM), http_client=upstream)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://g") as client:
        response = await client.post("/v1/chat/completions", json=_chat(stream=False))

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "upstream_error"


# --- latency: the guardrail must not buffer the stream --------------------------


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def live_stack() -> Iterator[str]:
    """Mock provider and gateway on real sockets, as in production."""
    mock_port, gateway_port = _free_port(), _free_port()
    servers = [
        uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        for app, port in [
            (create_mock_llm(ttft_ms=0, chunk_delay_ms=25), mock_port),
            (
                create_app(
                    GuardrailSettings(upstream_url=f"http://127.0.0.1:{mock_port}/v1/chat/completions")
                ),
                gateway_port,
            ),
        ]
    ]
    threads = [threading.Thread(target=server.run, daemon=True) for server in servers]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + 10
    while not all(server.started for server in servers):
        assert time.monotonic() < deadline, "servers did not start"
        time.sleep(0.02)
    yield f"http://127.0.0.1:{gateway_port}"
    for server in servers:
        server.should_exit = True
    for thread in threads:
        thread.join(timeout=10)


async def test_first_tokens_arrive_long_before_the_stream_ends(live_stack: str) -> None:
    started = time.perf_counter()
    first_content_at: float | None = None
    async with (
        httpx.AsyncClient() as client,
        client.stream("POST", f"{live_stack}/v1/chat/completions", json=_chat()) as response,
    ):
        parser = SSEParser()
        async for raw in response.aiter_bytes():
            for event in parser.feed(raw):
                if first_content_at is None and event.data != "[DONE]" and _content([event]):
                    first_content_at = time.perf_counter() - started
    total = time.perf_counter() - started

    assert first_content_at is not None
    # The mock streams ~85 chunks 25 ms apart (about 2 s). Words must flow
    # through as they are produced, not after the whole response is buffered.
    assert total > 1.5
    assert first_content_at < 0.3, f"first content after {first_content_at:.3f}s of {total:.3f}s"
