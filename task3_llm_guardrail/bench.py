"""Benchmarks for the streaming guardrail.

1. Redactor throughput, per-chunk cost and memory on a multi-megabyte stream.
2. Added time-to-first-token and total latency, measured over real sockets:
   client -> mock provider directly vs client -> gateway -> mock provider.

Run::

    python -m task3_llm_guardrail.bench
"""

from __future__ import annotations

import asyncio
import json
import random
import socket
import statistics
import threading
import time
import tracemalloc

import httpx
import uvicorn

from .gateway import GuardrailSettings, create_app
from .mock_llm import create_mock_llm
from .redactor import StreamingRedactor
from .sse import SSEParser

WORDS = [
    "the",
    "account",
    "was",
    "reviewed",
    "and",
    "the",
    "refund",
    "issued",
    "after",
    "we",
    "confirmed",
    "details",
    "with",
    "support",
]
PII = ["jane.doe@example.com", "123-45-6789", "4111 1111 1111 1111", "ops+alerts@acme-corp.io"]


def _synthetic_text(size: int, seed: int = 7) -> str:
    rng = random.Random(seed)
    parts: list[str] = []
    length = 0
    while length < size:
        word = rng.choice(PII) if rng.random() < 0.01 else rng.choice(WORDS)
        parts.append(word)
        length += len(word) + 1
    return " ".join(parts)


def bench_redactor(size: int = 5_000_000, chunk: int = 4) -> None:
    text = _synthetic_text(size)
    chunks = [text[i : i + chunk] for i in range(0, len(text), chunk)]

    redactor = StreamingRedactor()
    started = time.perf_counter()
    for piece in chunks:
        redactor.feed(piece)
    redactor.flush()
    elapsed = time.perf_counter() - started

    sampled = []
    timer = StreamingRedactor()
    for index, piece in enumerate(chunks):
        if index % 50:
            timer.feed(piece)
            continue
        t0 = time.perf_counter_ns()
        timer.feed(piece)
        sampled.append(time.perf_counter_ns() - t0)
    sampled.sort()

    tracemalloc.start()
    probe = StreamingRedactor()
    baseline, _ = tracemalloc.get_traced_memory()
    for piece in chunks:
        probe.feed(piece)
    probe.flush()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("Redactor (pure CPU, single core)")
    print(f"  input                 {len(text) / 1e6:.1f} MB in {len(chunks):,} chunks of {chunk} chars")
    print(f"  throughput            {len(text) / elapsed / 1e6:.2f} MB/s")
    p50_us, p99_us = sampled[len(sampled) // 2] / 1000, sampled[int(len(sampled) * 0.99)] / 1000
    print(f"  per chunk p50 / p99   {p50_us:.1f} us / {p99_us:.1f} us")
    print(f"  max chars held        {redactor.stats.max_held}")
    print(
        f"  peak memory           {(peak - baseline) / 1024:.1f} KiB while streaming {len(text) / 1e6:.1f} MB"
    )
    print(f"  redactions            {redactor.stats.redactions:,}")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _has_content(data: str) -> bool:
    if not data or data == "[DONE]":
        return False
    choices = json.loads(data).get("choices") or []
    return any((choice.get("delta") or {}).get("content") for choice in choices)


async def _measure(url: str, runs: int) -> tuple[list[float], list[float]]:
    body = {"model": "mock-assistant", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
    ttft, total = [], []
    async with httpx.AsyncClient(timeout=30) as client:
        for _ in range(runs):
            started = time.perf_counter()
            first = None
            parser = SSEParser()
            async with client.stream("POST", url, json=body) as response:
                async for raw in response.aiter_bytes():
                    if first is None and any(_has_content(event.data) for event in parser.feed(raw)):
                        first = time.perf_counter() - started
            ttft.append((first or 0) * 1000)
            total.append((time.perf_counter() - started) * 1000)
    return ttft, total


def bench_end_to_end(runs: int = 40) -> None:
    mock_port, gateway_port = _free_port(), _free_port()
    mock_url = f"http://127.0.0.1:{mock_port}/v1/chat/completions"
    servers = [
        uvicorn.Server(
            uvicorn.Config(create_mock_llm(ttft_ms=20, chunk_delay_ms=2), port=mock_port, log_level="error")
        ),
        uvicorn.Server(
            uvicorn.Config(
                create_app(GuardrailSettings(upstream_url=mock_url)), port=gateway_port, log_level="error"
            )
        ),
    ]
    for server in servers:
        threading.Thread(target=server.run, daemon=True).start()
    while not all(server.started for server in servers):
        time.sleep(0.02)

    gateway_url = f"http://127.0.0.1:{gateway_port}/v1/chat/completions"
    asyncio.run(_measure(mock_url, 5))  # warm-up both paths
    asyncio.run(_measure(gateway_url, 5))
    direct_ttft, direct_total = asyncio.run(_measure(mock_url, runs))
    gw_ttft, gw_total = asyncio.run(_measure(gateway_url, runs))
    for server in servers:
        server.should_exit = True

    def p50(values: list[float]) -> float:
        return statistics.median(values)

    print(f"\nEnd to end over localhost ({runs} streamed completions each, mock TTFT 20 ms)")
    print("  path                   TTFT p50     total p50")
    print(f"  direct to provider     {p50(direct_ttft):7.1f} ms   {p50(direct_total):7.1f} ms")
    print(f"  through guardrail      {p50(gw_ttft):7.1f} ms   {p50(gw_total):7.1f} ms")
    added_ttft, added_total = p50(gw_ttft) - p50(direct_ttft), p50(gw_total) - p50(direct_total)
    print(f"  added by guardrail     {added_ttft:7.1f} ms   {added_total:7.1f} ms")


if __name__ == "__main__":
    bench_redactor()
    bench_end_to_end()
