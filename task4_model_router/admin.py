"""Tenant administration.

python -m task4_model_router.admin create-tenant acme --tokens-per-minute 50000
"""

from __future__ import annotations

import argparse
import sqlite3

from .config import RouterSettings
from .db import Database, TenantStore


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m task4_model_router.admin", description="Manage gateway tenants."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create-tenant", help="register a tenant and print its API key")
    create.add_argument("tenant_id")
    create.add_argument("--tokens-per-minute", type=int, default=None)
    create.add_argument("--api-key", default=None, help="use this key instead of generating one")
    args = parser.parse_args()

    settings = RouterSettings.from_env()
    limit = settings.default_tokens_per_minute if args.tokens_per_minute is None else args.tokens_per_minute
    if limit <= 0:
        parser.error("--tokens-per-minute must be a positive integer")

    db = Database(settings.database_path)
    try:
        key = TenantStore(db).create(args.tenant_id, limit, api_key=args.api_key)
    except sqlite3.IntegrityError:
        parser.exit(1, f"error: tenant '{args.tenant_id}' (or that API key) already exists\n")
    finally:
        db.close()
    print(key)  # noqa: T201 - CLI output


if __name__ == "__main__":
    main()
