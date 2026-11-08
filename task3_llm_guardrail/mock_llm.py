"""OpenAI-compatible mock provider used for tests, demos and benchmarks.

``POST /v1/chat/completions`` supports ``stream: true`` (SSE) and plain JSON.
Content is cut into random 1-6 character chunks so sensitive values are
regularly split across chunk boundaries, as they are with real tokenizers.

Models:
  * ``mock-echo``      - replies with the last user message verbatim.
  * anything else      - replies with a scripted support answer containing PII.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

SCRIPTED_REPLY = (
    "Thanks for reaching out! I found the account. The primary contact is "
    "jane.doe@example.com and the SSN on file is 123-45-6789. The card ending in "
    "1111 (full number 4111 1111 1111 1111) was charged twice, so I have issued a "
    "refund. You can also reach billing at billing-team+refunds@acme-corp.io. "
    "Order 4111-2222 is unaffected."
)


def _reply_text(body: dict[str, Any]) -> str:
    if body.get("model") == "mock-echo":
        for message in reversed(body.get("messages") or []):
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                return str(message["content"])
        return ""
    return SCRIPTED_REPLY


def _chunks(text: str, seed: int) -> list[str]:
    rng = random.Random(seed)
    pieces, index = [], 0
    while index < len(text):
        size = rng.randint(1, 6)
        pieces.append(text[index : index + size])
        index += size
    return pieces


def create_mock_llm(*, ttft_ms: float | None = None, chunk_delay_ms: float | None = None) -> FastAPI:
    first_token_delay = (
        ttft_ms if ttft_ms is not None else float(os.getenv("MOCK_LLM_TTFT_MS", "50"))
    ) / 1000
    inter_chunk_delay = (
        chunk_delay_ms if chunk_delay_ms is not None else float(os.getenv("MOCK_LLM_CHUNK_DELAY_MS", "10"))
    ) / 1000
    app = FastAPI(title="Mock LLM Provider")

    @app.post("/v1/chat/completions")
    async def completions(request: Request) -> Response:
        body = await request.json()
        text = _reply_text(body)
        model = body.get("model", "mock-assistant")
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if not body.get("stream"):
            await asyncio.sleep(first_token_delay)
            return JSONResponse(
                {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": len(text) // 4,
                        "total_tokens": 12 + len(text) // 4,
                    },
                }
            )

        def frame(delta: dict[str, Any], finish_reason: str | None = None) -> bytes:
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }
            return f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n".encode()

        async def stream() -> AsyncIterator[bytes]:
            await asyncio.sleep(first_token_delay)
            yield frame({"role": "assistant", "content": ""})
            for piece in _chunks(text, int(body.get("seed") or 0)):
                yield frame({"content": piece})
                if inter_chunk_delay:
                    await asyncio.sleep(inter_chunk_delay)
            yield frame({}, finish_reason="stop")
            yield b"data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


def main() -> None:
    import uvicorn

    uvicorn.run(
        create_mock_llm(),
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "9200")),
        log_level="warning",
    )


if __name__ == "__main__":
    main()
