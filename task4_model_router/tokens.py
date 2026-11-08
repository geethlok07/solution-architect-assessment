"""Token estimation for admission control.

The limiter has to decide *before* calling the model, so it reserves an
upper-bound estimate: prompt tokens plus the completion budget
(``max_tokens``). The reservation is corrected to the provider's reported
``usage.total_tokens`` afterwards.

The prompt estimate uses the common ~4 characters per token heuristic plus
the per-message framing overhead of chat formats. A production deployment
would plug in the provider's tokenizer (for example ``tiktoken``) behind the
same function; the heuristic avoids a network download at startup.
"""

from __future__ import annotations

import json
import math
from typing import Any

CHARS_PER_TOKEN = 4
TOKENS_PER_MESSAGE = 4
TOKENS_PER_REQUEST = 3


def estimate_prompt_tokens(messages: list[dict[str, Any]]) -> int:
    total = TOKENS_PER_REQUEST
    for message in messages:
        content = message.get("content")
        text = content if isinstance(content, str) else json.dumps(content or "", separators=(",", ":"))
        total += TOKENS_PER_MESSAGE + math.ceil(len(text) / CHARS_PER_TOKEN)
    return total


def estimate_request_tokens(body: dict[str, Any], default_max_completion_tokens: int) -> int:
    completion_budget = body.get("max_tokens") or default_max_completion_tokens
    return estimate_prompt_tokens(body.get("messages") or []) + int(completion_budget)
