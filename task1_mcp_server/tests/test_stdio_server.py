"""End-to-end tests: the server runs as a subprocess and is driven over stdio."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .conftest import REPO_ROOT, StdioClient

INVALID_PARAMS = -32602
METHOD_NOT_FOUND = -32601
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
INTERNAL_ERROR = -32603


def _tool_payload(response: dict) -> dict:
    return json.loads(response["result"]["content"][0]["text"])


def _error_fields(response: dict) -> set[str]:
    return {e["field"] for e in response["error"]["data"]["errors"]}


def test_initialize_advertises_tools_capability() -> None:
    client = StdioClient([sys.executable, "-m", "task1_mcp_server.server"])
    try:
        response = client.initialize()
        assert response["result"]["serverInfo"]["name"] == "customer-support"
        assert "tools" in response["result"]["capabilities"]
    finally:
        client.close()


def test_tools_list_publishes_strict_schemas(server: StdioClient) -> None:
    tools = {t["name"]: t for t in server.request("tools/list")["result"]["tools"]}

    assert set(tools) == {"get_customer_record", "trigger_refund"}
    refund_schema = tools["trigger_refund"]["inputSchema"]
    assert refund_schema["additionalProperties"] is False
    assert refund_schema["properties"]["reason"]["minLength"] == 10
    assert tools["get_customer_record"]["annotations"]["readOnlyHint"] is True
    assert tools["trigger_refund"]["annotations"]["destructiveHint"] is True


def test_get_customer_record_returns_structured_record(server: StdioClient) -> None:
    response = server.call_tool("get_customer_record", {"customer_id": "CUST-10001"})

    result = response["result"]
    assert result["isError"] is False
    assert result["structuredContent"]["customer_id"] == "CUST-10001"
    assert _tool_payload(response) == result["structuredContent"]


def test_trigger_refund_processes_and_updates_balance(server: StdioClient) -> None:
    before = server.call_tool("get_customer_record", {"customer_id": "CUST-10001"})
    response = server.call_tool(
        "trigger_refund",
        {"customer_id": "CUST-10001", "amount": 20.5, "reason": "Duplicate charge on invoice"},
    )
    after = server.call_tool("get_customer_record", {"customer_id": "CUST-10001"})

    refund = response["result"]["structuredContent"]
    assert refund["status"] == "processed"
    assert refund["refund_id"].startswith("rf_")
    assert _tool_payload(after)["balance"] == _tool_payload(before)["balance"] - 20.5


def test_malformed_customer_id_is_a_jsonrpc_invalid_params_error(server: StdioClient) -> None:
    response = server.call_tool("get_customer_record", {"customer_id": "cust-1"})

    assert "result" not in response
    assert response["error"]["code"] == INVALID_PARAMS
    assert response["error"]["data"]["tool"] == "get_customer_record"
    assert _error_fields(response) == {"customer_id"}


def test_all_refund_violations_are_reported_together(server: StdioClient) -> None:
    response = server.call_tool(
        "trigger_refund",
        {"customer_id": "CUST-1", "amount": -3, "reason": "short", "override": True},
    )

    assert response["error"]["code"] == INVALID_PARAMS
    assert _error_fields(response) == {"customer_id", "amount", "reason", "override"}


def test_wrongly_typed_amount_is_not_coerced(server: StdioClient) -> None:
    response = server.call_tool(
        "trigger_refund",
        {"customer_id": "CUST-10001", "amount": "20", "reason": "Customer requested a refund"},
    )
    assert response["error"]["code"] == INVALID_PARAMS
    assert _error_fields(response) == {"amount"}


def test_missing_arguments_are_rejected(server: StdioClient) -> None:
    response = server.call_tool("trigger_refund")
    assert response["error"]["code"] == INVALID_PARAMS
    assert _error_fields(response) == {"customer_id", "amount", "reason"}


def test_unknown_tool_is_invalid_params(server: StdioClient) -> None:
    response = server.call_tool("delete_customer", {"customer_id": "CUST-10001"})
    assert response["error"]["code"] == INVALID_PARAMS
    assert "delete_customer" in response["error"]["message"]


def test_business_failures_are_tool_errors_not_protocol_errors(server: StdioClient) -> None:
    missing = server.call_tool("get_customer_record", {"customer_id": "CUST-99999"})
    suspended = server.call_tool(
        "trigger_refund",
        {"customer_id": "CUST-10003", "amount": 5, "reason": "Goodwill credit for outage"},
    )
    too_large = server.call_tool(
        "trigger_refund",
        {"customer_id": "CUST-10002", "amount": 9_999, "reason": "Refund entire annual plan"},
    )

    assert missing["result"]["isError"] is True
    assert _tool_payload(missing)["error"] == "customer_not_found"
    assert _tool_payload(suspended)["error"] == "refund_rejected"
    assert _tool_payload(too_large)["error"] == "refund_rejected"


def test_unknown_method_is_method_not_found(server: StdioClient) -> None:
    assert server.request("customers/delete")["error"]["code"] == METHOD_NOT_FOUND


def test_invalid_json_gets_parse_error_and_session_survives(server: StdioClient) -> None:
    server.send_raw('{"jsonrpc": "2.0", "id": 99, "method": ')
    response = server.read()
    assert response == {
        "jsonrpc": "2.0",
        "id": None,
        "error": {"code": PARSE_ERROR, "message": "Parse error"},
    }

    assert server.request("ping")["result"] == {}


def test_non_jsonrpc_object_gets_invalid_request(server: StdioClient) -> None:
    server.send_raw('{"hello": "world"}')
    assert server.read()["error"]["code"] == INVALID_REQUEST


def test_stdout_carries_only_jsonrpc_messages(server: StdioClient) -> None:
    server.request("tools/list")
    server.call_tool("get_customer_record", {"customer_id": "CUST-10001"})
    server.call_tool("get_customer_record", {"customer_id": "bad"})
    server.call_tool(
        "trigger_refund", {"customer_id": "CUST-10002", "amount": 1, "reason": "Late delivery refund"}
    )
    server.send_raw("not json at all")
    server.read()
    server.close()

    assert server.raw_stdout, "server produced no output"
    for line in server.raw_stdout:
        message = json.loads(line)
        assert message["jsonrpc"] == "2.0"
        assert ("result" in message) ^ ("error" in message)
    assert "starting customer-support" in server.stderr
    assert "refund issued" in server.stderr


def test_stray_print_in_tool_code_cannot_corrupt_stdout(tmp_path: Path) -> None:
    """Simulates a careless dependency that prints from inside a tool call."""
    noisy_entrypoint = tmp_path / "noisy_server.py"
    noisy_entrypoint.write_text(
        "import builtins\n"
        "from task1_mcp_server import server, store\n"
        "original_get = store.CustomerStore.get\n"
        "def noisy_get(self, customer_id):\n"
        "    builtins.print('debug: looking up', customer_id)\n"
        "    return original_get(self, customer_id)\n"
        "store.CustomerStore.get = noisy_get\n"
        "server.main()\n"
    )
    client = StdioClient([sys.executable, str(noisy_entrypoint)], env={"PYTHONPATH": str(REPO_ROOT)})
    try:
        client.initialize()
        response = client.call_tool("get_customer_record", {"customer_id": "CUST-10001"})
        assert response["result"]["isError"] is False
    finally:
        client.close()

    assert all(json.loads(line)["jsonrpc"] == "2.0" for line in client.raw_stdout)
    assert "debug: looking up CUST-10001" in client.stderr


def test_unexpected_tool_failure_is_internal_error_without_details(tmp_path: Path) -> None:
    failing_entrypoint = tmp_path / "failing_server.py"
    failing_entrypoint.write_text(
        "from task1_mcp_server import server, store\n"
        "def broken_get(self, customer_id):\n"
        "    raise RuntimeError('connection to billing-db.internal:5432 refused')\n"
        "store.CustomerStore.get = broken_get\n"
        "server.main()\n"
    )
    client = StdioClient([sys.executable, str(failing_entrypoint)], env={"PYTHONPATH": str(REPO_ROOT)})
    try:
        client.initialize()
        response = client.call_tool("get_customer_record", {"customer_id": "CUST-10001"})
    finally:
        client.close()

    assert response["error"]["code"] == INTERNAL_ERROR
    assert response["error"]["message"] == "Internal error while executing tool"
    assert "billing-db" not in json.dumps(response)
    incident = response["error"]["data"]["incident_id"]
    assert f"incident={incident}" in client.stderr
    assert "billing-db.internal" in client.stderr
