from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from task1_mcp_server.models import (
    GetCustomerRecordInput,
    TriggerRefundInput,
    describe_validation_error,
)

VALID_REFUND = {
    "customer_id": "CUST-10001",
    "amount": 49.99,
    "reason": "Duplicate charge on the March invoice",
}


def _failed_fields(model: type, payload: dict[str, Any]) -> set[str]:
    with pytest.raises(ValidationError) as info:
        model.model_validate(payload)
    return {detail["field"] for detail in describe_validation_error(info.value)}


@pytest.mark.parametrize("customer_id", ["CUST-00000", "CUST-12345", "CUST-99999"])
def test_customer_id_accepts_canonical_format(customer_id: str) -> None:
    assert GetCustomerRecordInput(customer_id=customer_id).customer_id == customer_id


@pytest.mark.parametrize(
    "customer_id",
    [
        "cust-12345",  # wrong case
        "CUST-1234",  # too short
        "CUST-123456",  # too long
        "CUST12345",  # missing dash
        " CUST-12345",  # leading whitespace
        "CUST-12345\n",  # trailing newline
        "CUST-ABCDE",  # letters
        "CUST-１２３４５",  # full-width unicode digits
        "",
        12345,  # wrong type
        None,
        ["CUST-12345"],
    ],
)
def test_customer_id_rejects_malformed_values(customer_id: Any) -> None:
    assert _failed_fields(GetCustomerRecordInput, {"customer_id": customer_id}) == {"customer_id"}


def test_missing_and_unknown_fields_are_rejected() -> None:
    assert _failed_fields(GetCustomerRecordInput, {}) == {"customer_id"}
    assert _failed_fields(GetCustomerRecordInput, {"customer_id": "CUST-12345", "include_pii": True}) == {
        "include_pii"
    }


@pytest.mark.parametrize("amount", [0.01, 1, 49.99, 10_000])
def test_refund_accepts_valid_amounts(amount: float) -> None:
    refund = TriggerRefundInput.model_validate({**VALID_REFUND, "amount": amount})
    assert refund.amount == amount


@pytest.mark.parametrize(
    "amount",
    [
        0,
        -1,
        -0.01,
        10_000.01,
        12.345,  # sub-cent precision
        0.1 + 0.2,  # binary float noise
        float("nan"),
        float("inf"),
        float("-inf"),
        "49.99",  # numeric string is not coerced
        True,  # bool is not a number here
        None,
    ],
)
def test_refund_rejects_invalid_amounts(amount: Any) -> None:
    assert _failed_fields(TriggerRefundInput, {**VALID_REFUND, "amount": amount}) == {"amount"}


@pytest.mark.parametrize(
    "reason",
    [
        "too short",  # 9 characters
        " " * 20,
        "   short    ",  # padded to length, but only 5 meaningful characters
        "x" * 501,
        42,
    ],
)
def test_refund_rejects_invalid_reasons(reason: Any) -> None:
    assert _failed_fields(TriggerRefundInput, {**VALID_REFUND, "reason": reason}) == {"reason"}


def test_reason_is_trimmed_and_minimum_is_inclusive() -> None:
    refund = TriggerRefundInput.model_validate({**VALID_REFUND, "reason": "  0123456789  "})
    assert refund.reason == "0123456789"


def test_every_invalid_field_is_reported_at_once() -> None:
    fields = _failed_fields(
        TriggerRefundInput, {"customer_id": "bad", "amount": -5, "reason": "no", "extra": 1}
    )
    assert fields == {"customer_id", "amount", "reason", "extra"}


def test_error_details_never_echo_caller_input() -> None:
    secret = "CUST-SECRET-4111111111111111"
    with pytest.raises(ValidationError) as info:
        GetCustomerRecordInput.model_validate({"customer_id": secret})
    assert secret not in str(describe_validation_error(info.value))


def test_published_schema_matches_validation_rules() -> None:
    schema = TriggerRefundInput.model_json_schema()
    props = schema["properties"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"customer_id", "amount", "reason"}
    assert props["customer_id"]["pattern"] == "^CUST-[0-9]{5}$"
    assert props["amount"]["exclusiveMinimum"] == 0
    assert props["reason"]["minLength"] == 10
