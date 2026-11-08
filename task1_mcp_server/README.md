# Task 1 — Customer-Support MCP Server (stdio)

A Model Context Protocol server built on the official Python SDK (`mcp==2.3.0`) that exposes two tools over the stdio transport, with strict Pydantic validation and standard JSON-RPC error mapping.

| Tool | Inputs | Behaviour |
| --- | --- | --- |
| `get_customer_record` | `customer_id` (`CUST-XXXXX`) | Returns the customer profile and balance. Read-only, idempotent. |
| `trigger_refund` | `customer_id`, `amount` (positive float), `reason` (≥ 10 chars) | Debits the refundable balance and returns a refund receipt. Destructive. |

## Run

```bash
pip install -e ".[dev]"            # from the repository root
python -m task1_mcp_server.server  # speaks JSON-RPC on stdin/stdout, logs on stderr
```

Interactive check with the MCP Inspector:

```bash
npx @modelcontextprotocol/inspector python -m task1_mcp_server.server
```

Configuration for any stdio MCP host (point `command` at the Python interpreter of the virtual environment the project was installed into):

```json
{
  "mcpServers": {
    "customer-support": {
      "command": "/path/to/solution-architect-assessment/.venv/bin/python",
      "args": ["-m", "task1_mcp_server.server"],
      "env": { "LOG_LEVEL": "INFO" }
    }
  }
}
```

Sample data: `CUST-10001` (active, 1,250.00), `CUST-10002` (active, 49.99), `CUST-10003` (suspended).

## Design

### Two classes of error, deliberately kept apart

The MCP specification distinguishes protocol errors from tool execution errors, and clients treat them differently: a protocol error tells the host the request itself was wrong, while a tool error is returned to the model so it can adjust and retry.

| Situation | Response | Code |
| --- | --- | --- |
| Arguments violate the input schema | JSON-RPC `error` with per-field details in `data.errors` | `-32602` |
| Unknown tool name | JSON-RPC `error` listing available tools | `-32602` |
| Unknown method | JSON-RPC `error` | `-32601` |
| Line is not valid JSON | JSON-RPC `error`, `id: null` | `-32700` |
| Valid JSON but not a JSON-RPC message | JSON-RPC `error`, `id: null` | `-32600` |
| Unexpected exception inside a tool | JSON-RPC `error` with an incident id only; traceback goes to stderr | `-32603` |
| Customer not found / refund not allowed | Normal `result` with `isError: true` | — |

Example — this request breaks three rules at once:

```json
{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"trigger_refund",
 "arguments":{"customer_id":"CUST-10001","amount":-3,"reason":"short","override":true}}}
```

and gets one response listing every violation:

```json
{"jsonrpc":"2.0","id":5,"error":{"code":-32602,"message":"Invalid arguments for tool 'trigger_refund'",
 "data":{"tool":"trigger_refund","errors":[
   {"field":"amount","type":"greater_than","message":"Input should be greater than 0"},
   {"field":"reason","type":"string_too_short","message":"String should have at least 10 characters"},
   {"field":"override","type":"extra_forbidden","message":"Extra inputs are not permitted"}]}}}
```

The SDK's high-level `MCPServer` wrapper converts argument validation failures into `isError` tool results. Because this assignment asks for JSON-RPC error codes, the server uses the low-level `Server` API and owns the `tools/list` and `tools/call` handlers directly: arguments are validated against our own models and failures are raised as `MCPError(-32602)`.

### Validation rules

The Pydantic models in [`models.py`](models.py) are the single source of truth. The JSON Schemas published in `tools/list` are generated from them, so what clients see is exactly what the server enforces.

All models use `strict=True` (no type coercion), `extra="forbid"` (unknown keys rejected) and `frozen=True`.

| Field | Rule | Rejected examples |
| --- | --- | --- |
| `customer_id` | `^CUST-[0-9]{5}$` (ASCII digits only) | `cust-12345`, `CUST-1234`, `CUST-123456`, `" CUST-12345"`, `CUST-１２３４５`, `12345`, `null` |
| `amount` | number, `> 0`, `≤ 10,000`, finite, at most 2 decimal places | `0`, `-0.01`, `"49.99"`, `true`, `NaN`, `Infinity`, `12.345`, `0.1 + 0.2` |
| `reason` | string, 10–500 characters after trimming surrounding whitespace | `"too short"`, 20 spaces, `"   short    "`, `42` |

All violations are reported in one response, and error details never echo the rejected value back (it is caller data and may be sensitive).

### Keeping stdout clean

stdout is the JSON-RPC wire; a single stray byte breaks the client's framing. Protection is layered:

1. **Logging** — the root logger has exactly one handler, bound to `sys.stderr`. The server never writes to stdout itself.
2. **Lint** — Ruff rule `T20` fails the build on any `print` call.
3. **Transport** — while serving, the SDK's stdio transport points file descriptor 1 at stderr and writes protocol messages through its own private handle, so output from third-party code (including C extensions) cannot reach the wire.
4. **Python buffer** — `sys.stdout` is also redirected to stderr for the whole session, and flushed before the transport hands file descriptor 1 back. Without this, a `print()` held in Python's stdout buffer would be written to the real stdout when the process exits.
5. **Single writer** — the parse-error responses added for malformed lines go through the transport's write stream rather than a second handle, so responses can never interleave.
6. **Tests** — the end-to-end suite runs the server as a real subprocess, records every byte on stdout and asserts each line is a JSON-RPC 2.0 response. A dedicated test injects a `print()` inside a tool call and proves it lands on stderr, both with Python's default buffered stdout and with `PYTHONUNBUFFERED=1`.

### Other details

- Balances are tracked in integer cents to avoid floating-point drift.
- Tool annotations (`readOnlyHint`, `destructiveHint`, `idempotentHint`) let hosts ask for confirmation before a refund.
- Logs record field names for rejected input, never values.

## Tests

```bash
pytest task1_mcp_server
```

- `test_models.py` — boundary and edge cases for every field, published schema matches enforcement.
- `test_stdio_server.py` — subprocess-driven protocol tests: handshake, `tools/list`, successful calls, every error class above, session survives malformed input, stdout purity.

## Assumptions

- `CUST-XXXXX` means five ASCII digits. Allowing letters is a one-line change to `CUSTOMER_ID_PATTERN`.
- Refunds are capped at 10,000.00 and limited to cent precision, as a real payments API would be.
- The in-memory store stands in for a billing backend; it is process-local and resets on restart.
