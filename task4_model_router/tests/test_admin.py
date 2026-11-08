from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from task4_model_router.db import Database, TenantStore

REPO_ROOT = Path(__file__).resolve().parents[2]


def _admin(db_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "task4_model_router.admin", *args],
        cwd=REPO_ROOT,
        env={**os.environ, "ROUTER_DB_PATH": str(db_path)},
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_create_tenant_prints_a_working_key(db_path: Path) -> None:
    result = _admin(db_path, "create-tenant", "acme", "--tokens-per-minute", "1500")

    assert result.returncode == 0
    key = result.stdout.strip()
    db = Database(db_path)
    try:
        tenant = TenantStore(db).find_by_key(key)
    finally:
        db.close()
    assert tenant is not None and (tenant.tenant_id, tenant.tokens_per_minute) == ("acme", 1500)


def test_duplicate_tenant_is_a_clean_error(db_path: Path) -> None:
    assert _admin(db_path, "create-tenant", "acme").returncode == 0
    result = _admin(db_path, "create-tenant", "acme")

    assert result.returncode == 1
    assert "already exists" in result.stderr
    assert "Traceback" not in result.stderr


def test_non_positive_limit_is_rejected(db_path: Path) -> None:
    result = _admin(db_path, "create-tenant", "acme", "--tokens-per-minute", "0")
    assert result.returncode == 2
    assert "positive integer" in result.stderr
