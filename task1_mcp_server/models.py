"""Input contracts for the customer-support MCP tools.

The models are the single source of truth for validation: the JSON Schemas
advertised in ``tools/list`` are generated from them, and every ``tools/call``
is validated against them before any business logic runs.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
)

CUSTOMER_ID_PATTERN = r"^CUST-[0-9]{5}$"
MAX_REFUND_AMOUNT = 10_000.00
REASON_MIN_LENGTH = 10
REASON_MAX_LENGTH = 500

CustomerId = Annotated[
    str,
    StringConstraints(pattern=CUSTOMER_ID_PATTERN),
    Field(
        description="Customer identifier in the form CUST-XXXXX, where X is a digit.",
        examples=["CUST-10001"],
    ),
]


class _StrictInput(BaseModel):
    """Base for tool inputs: no coercion, no unknown keys, immutable."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class GetCustomerRecordInput(_StrictInput):
    customer_id: CustomerId


class TriggerRefundInput(_StrictInput):
    customer_id: CustomerId
    amount: float = Field(
        gt=0,
        le=MAX_REFUND_AMOUNT,
        allow_inf_nan=False,
        description=(
            "Refund amount in the account currency. Must be positive, finite, "
            f"at most {MAX_REFUND_AMOUNT:,.2f} and have no more than two decimal places."
        ),
        examples=[49.99],
    )
    reason: str = Field(
        min_length=REASON_MIN_LENGTH,
        max_length=REASON_MAX_LENGTH,
        description=(
            f"Why the refund is being issued ({REASON_MIN_LENGTH}-{REASON_MAX_LENGTH} "
            "characters, ignoring surrounding whitespace)."
        ),
        examples=["Duplicate charge on the March invoice"],
    )

    @field_validator("amount")
    @classmethod
    def _at_most_two_decimals(cls, value: float) -> float:
        # Decimal(str(x)) gives the shortest round-tripping representation,
        # so 19.99 stays 19.99 instead of exposing binary float noise.
        exponent = Decimal(str(value)).as_tuple().exponent
        if isinstance(exponent, int) and exponent < -2:
            raise ValueError("amount must have at most two decimal places")
        return value

    @field_validator("reason")
    @classmethod
    def _meaningful_reason(cls, value: str) -> str:
        stripped = value.strip()
        if len(stripped) < REASON_MIN_LENGTH:
            raise ValueError(
                f"reason must be at least {REASON_MIN_LENGTH} characters long, "
                "excluding surrounding whitespace"
            )
        return stripped


def describe_validation_error(exc: ValidationError) -> list[dict[str, Any]]:
    """Flatten a Pydantic error into a stable, client-friendly structure.

    The rejected value itself is deliberately left out: it is caller data and
    may contain sensitive content that should not be echoed back or logged.
    """
    details: list[dict[str, Any]] = []
    for err in exc.errors(include_url=False, include_input=False):
        field = ".".join(str(part) for part in err["loc"]) or "<root>"
        details.append({"field": field, "type": err["type"], "message": err["msg"]})
    return details
