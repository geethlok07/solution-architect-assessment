"""Controllable mock model providers for tests and demos.

Each provider is an OpenAI-compatible ``/v1/chat/completions`` endpoint whose
behaviour can be switched at runtime::

    curl -X POST localhost:9301/control -H 'content-type: application/json' -d '{"mode": "rate_limited"}'

Modes: ``ok``, ``slow`` (answers after ``slow_seconds``), ``rate_limited``
(429), ``error`` (500 with a stack trace in the body, to prove the gateway
never forwards it).

Run::

    python -m task4_model_router.mock_providers --name primary --port 9301
    python -m task4_model_router.mock_providers --name secondary --port 9302
"""

from __future__ import annotations

import argparse
import asyncio
import math
import time
import uuid
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

Mode = Literal["ok", "slow", "rate_limited", "error"]

LEAKY_TRACEBACK = (
    'Traceback (most recent call last):\n  File "/srv/inference/worker.py", line 214, in generate\n'
    "torch.cuda.OutOfMemoryError: CUDA out of memory on gpu-node-17.internal (10.0.4.17)"
)


class Control(BaseModel):
    mode: Mode
    slow_seconds: float | None = None


def create_mock_provider(name: str, mode: Mode = "ok", *, slow_seconds: float = 5.0) -> FastAPI:
    app = FastAPI(title=f"Mock provider ({name})")
    app.state.mode = mode
    app.state.slow_seconds = slow_seconds
    app.state.calls = 0
    app.state.cancelled = 0

    @app.post("/control")
    async def control(update: Control) -> dict[str, Any]:
        app.state.mode = update.mode
        if update.slow_seconds is not None:
            app.state.slow_seconds = update.slow_seconds
        return {"provider": name, "mode": app.state.mode}

    @app.post("/v1/chat/completions")
    async def completions(request: Request) -> JSONResponse:
        app.state.calls += 1
        body = await request.json()
        mode: Mode = app.state.mode

        if mode == "rate_limited":
            return JSONResponse(
                {"error": {"message": f"Rate limit reached for org-8812 on {name}", "type": "requests"}},
                status_code=429,
                headers={"retry-after": "20"},
            )
        if mode == "error":
            return JSONResponse({"error": LEAKY_TRACEBACK}, status_code=500)
        if mode == "slow":
            try:
                await asyncio.sleep(app.state.slow_seconds)
            except asyncio.CancelledError:
                app.state.cancelled += 1
                raise

        prompt_chars = sum(len(str(m.get("content", ""))) for m in body.get("messages", []))
        text = f"Answer from {name}."
        prompt_tokens = 3 + 4 * len(body.get("messages", [])) + math.ceil(prompt_chars / 4)
        # The mock "uses" half of the completion budget it was given, so real
        # usage differs from the gateway's up-front estimate.
        completion_tokens = max(6, int(body.get("max_tokens") or 12) // 2)
        return JSONResponse(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": f"{body.get('model')}@{name}",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
        )

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(
        prog="python -m task4_model_router.mock_providers", description="Run a mock model provider."
    )
    parser.add_argument("--name", default="primary")
    parser.add_argument("--port", type=int, default=9301)
    parser.add_argument("--mode", default="ok", choices=["ok", "slow", "rate_limited", "error"])
    args = parser.parse_args()
    uvicorn.run(
        create_mock_provider(args.name, args.mode), host="127.0.0.1", port=args.port, log_level="warning"
    )


if __name__ == "__main__":
    main()
