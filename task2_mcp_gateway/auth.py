"""Bearer token authentication.

Tokens are signed JWTs. The gateway verifies signature, expiry, audience and
issuer, then reads the caller's role from the ``role`` claim. The role is
never taken from an unsigned source such as a request header or body field.

Mint a token for local testing::

    python -m task2_mcp_gateway.auth --role admin --subject alice
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from enum import StrEnum

import jwt

from .config import GatewaySettings


class Role(StrEnum):
    ADMIN = "admin"
    VIEWER = "viewer"


@dataclass(frozen=True)
class Principal:
    subject: str
    role: Role


class AuthenticationError(Exception):
    """Raised when the request carries no usable credentials."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def extract_bearer_token(authorization: str | None) -> str:
    if not authorization:
        raise AuthenticationError("missing Authorization header")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise AuthenticationError("Authorization header must use the Bearer scheme")
    return token.strip()


def authenticate(authorization: str | None, settings: GatewaySettings) -> Principal:
    token = extract_bearer_token(authorization)
    try:
        claims = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],  # pinned: blocks alg=none / alg-confusion
            audience=settings.jwt_audience,
            issuer=settings.jwt_issuer,
            options={"require": ["exp", "sub", "role"]},
        )
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("token has expired") from None
    except jwt.InvalidTokenError:
        raise AuthenticationError("token is invalid") from None

    try:
        role = Role(claims["role"])
    except (ValueError, TypeError):
        raise AuthenticationError("token carries an unknown role") from None
    subject = claims["sub"]
    if not isinstance(subject, str) or not subject:
        raise AuthenticationError("token subject is invalid")
    return Principal(subject=subject, role=role)


def mint_token(settings: GatewaySettings, *, subject: str, role: Role | str, ttl_seconds: int = 3600) -> str:
    now = int(time.time())
    claims = {
        "sub": subject,
        "role": str(role),
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
        "iat": now,
        "exp": now + ttl_seconds,
    }
    return jwt.encode(claims, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m task2_mcp_gateway.auth", description="Mint a gateway access token for local testing."
    )
    parser.add_argument("--role", choices=[r.value for r in Role], required=True)
    parser.add_argument("--subject", default="demo-user")
    parser.add_argument("--ttl", type=int, default=3600, help="lifetime in seconds")
    args = parser.parse_args()
    token = mint_token(GatewaySettings.from_env(), subject=args.subject, role=args.role, ttl_seconds=args.ttl)
    print(token)  # noqa: T201 - CLI output


if __name__ == "__main__":
    main()
