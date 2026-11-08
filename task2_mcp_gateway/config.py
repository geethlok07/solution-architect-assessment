from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class GatewaySettings:
    downstream_url: str = "http://127.0.0.1:9100/mcp"
    jwt_secret: str = "dev-only-change-me-0123456789abcdef"
    jwt_algorithm: str = "HS256"
    jwt_audience: str = "mcp-gateway"
    jwt_issuer: str = "auth.example.local"
    upstream_timeout_seconds: float = 30.0
    max_body_bytes: int = 1_048_576
    filter_tools_list: bool = False

    @classmethod
    def from_env(cls) -> GatewaySettings:
        defaults = cls()
        return cls(
            downstream_url=os.getenv("MCP_DOWNSTREAM_URL", defaults.downstream_url),
            jwt_secret=os.getenv("MCP_GATEWAY_JWT_SECRET", defaults.jwt_secret),
            jwt_audience=os.getenv("MCP_GATEWAY_JWT_AUDIENCE", defaults.jwt_audience),
            jwt_issuer=os.getenv("MCP_GATEWAY_JWT_ISSUER", defaults.jwt_issuer),
            upstream_timeout_seconds=float(
                os.getenv("MCP_UPSTREAM_TIMEOUT_SECONDS", defaults.upstream_timeout_seconds)
            ),
            max_body_bytes=int(os.getenv("MCP_MAX_BODY_BYTES", defaults.max_body_bytes)),
            filter_tools_list=_env_bool("MCP_FILTER_TOOLS_LIST", defaults.filter_tools_list),
        )
