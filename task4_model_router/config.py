from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderSettings:
    name: str
    url: str
    api_key: str | None = None
    timeout_seconds: float = 3.0


@dataclass(frozen=True)
class RouterSettings:
    database_path: str = "data/llm_gateway.db"
    default_tokens_per_minute: int = 50_000
    window_seconds: int = 60
    default_max_completion_tokens: int = 512
    primary: ProviderSettings = ProviderSettings(
        name="primary", url="http://127.0.0.1:9301/v1/chat/completions", timeout_seconds=3.0
    )
    secondary: ProviderSettings = ProviderSettings(
        name="secondary", url="http://127.0.0.1:9302/v1/chat/completions", timeout_seconds=10.0
    )
    breaker_failure_threshold: int = 3
    breaker_cooldown_seconds: float = 30.0

    @classmethod
    def from_env(cls) -> RouterSettings:
        defaults = cls()
        return cls(
            database_path=os.getenv("ROUTER_DB_PATH", defaults.database_path),
            default_tokens_per_minute=int(
                os.getenv("ROUTER_TOKENS_PER_MINUTE", defaults.default_tokens_per_minute)
            ),
            primary=ProviderSettings(
                name="primary",
                url=os.getenv("PRIMARY_MODEL_URL", defaults.primary.url),
                api_key=os.getenv("PRIMARY_MODEL_API_KEY") or None,
                timeout_seconds=float(os.getenv("PRIMARY_TIMEOUT_SECONDS", defaults.primary.timeout_seconds)),
            ),
            secondary=ProviderSettings(
                name="secondary",
                url=os.getenv("SECONDARY_MODEL_URL", defaults.secondary.url),
                api_key=os.getenv("SECONDARY_MODEL_API_KEY") or None,
                timeout_seconds=float(
                    os.getenv("SECONDARY_TIMEOUT_SECONDS", defaults.secondary.timeout_seconds)
                ),
            ),
        )
