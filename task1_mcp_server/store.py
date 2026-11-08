"""In-memory customer store used by the MCP tools.

Stands in for a real billing/CRM backend. Balances are tracked in integer
cents so refunds never accumulate floating-point error.
"""

from __future__ import annotations

import threading
import uuid
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any


class CustomerNotFoundError(LookupError):
    pass


class RefundRejectedError(ValueError):
    pass


_SAMPLE_CUSTOMERS: dict[str, dict[str, Any]] = {
    "CUST-10001": {
        "name": "Ava Thompson",
        "email": "ava.thompson@example.com",
        "plan": "enterprise",
        "status": "active",
        "balance_cents": 125_000,
        "created_at": "2023-02-14",
    },
    "CUST-10002": {
        "name": "Rahul Mehta",
        "email": "rahul.mehta@example.com",
        "plan": "pro",
        "status": "active",
        "balance_cents": 4_999,
        "created_at": "2024-07-01",
    },
    "CUST-10003": {
        "name": "Lena Fischer",
        "email": "lena.fischer@example.com",
        "plan": "starter",
        "status": "suspended",
        "balance_cents": 0,
        "created_at": "2025-01-20",
    },
}


class CustomerStore:
    def __init__(self, customers: dict[str, dict[str, Any]]) -> None:
        self._customers = deepcopy(customers)
        self._refunds: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    @classmethod
    def with_sample_data(cls) -> CustomerStore:
        return cls(_SAMPLE_CUSTOMERS)

    def get(self, customer_id: str) -> dict[str, Any]:
        with self._lock:
            customer = self._customers.get(customer_id)
            if customer is None:
                raise CustomerNotFoundError(customer_id)
            return self._public_view(customer_id, customer)

    def refund(self, customer_id: str, amount: float, reason: str) -> dict[str, Any]:
        amount_cents = round(amount * 100)
        with self._lock:
            customer = self._customers.get(customer_id)
            if customer is None:
                raise CustomerNotFoundError(customer_id)
            if customer["status"] != "active":
                raise RefundRejectedError(f"account status is '{customer['status']}'")
            if amount_cents > customer["balance_cents"]:
                raise RefundRejectedError("amount exceeds the refundable balance")

            customer["balance_cents"] -= amount_cents
            refund = {
                "refund_id": f"rf_{uuid.uuid4().hex[:16]}",
                "customer_id": customer_id,
                "amount": amount_cents / 100,
                "reason": reason,
                "status": "processed",
                "remaining_balance": customer["balance_cents"] / 100,
                "processed_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            self._refunds.append(refund)
            return dict(refund)

    @staticmethod
    def _public_view(customer_id: str, customer: dict[str, Any]) -> dict[str, Any]:
        return {
            "customer_id": customer_id,
            "name": customer["name"],
            "email": customer["email"],
            "plan": customer["plan"],
            "status": customer["status"],
            "balance": customer["balance_cents"] / 100,
            "created_at": customer["created_at"],
        }
