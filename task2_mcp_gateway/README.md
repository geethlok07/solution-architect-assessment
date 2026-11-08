# Task 2 — MCP Security Gateway (tool-level authorization proxy)

An HTTP reverse proxy that sits between an AI agent and a downstream MCP server. It authenticates every request with a Bearer JWT, parses the JSON-RPC envelope, and enforces tool-level authorization: calls to `admin_*` tools by a non-admin are answered by the gateway with `-32001 Unauthorized Tool Call` and never reach the downstream.

```
 AI agent ──POST /mcp + Bearer JWT──▶ ┌──────────── MCP Gateway ────────────┐ ──▶ downstream MCP server
                                       │ 1 authenticate   (401 on failure)   │
                                       │ 2 size limit     (413)              │
                                       │ 3 parse JSON-RPC (-32700 / -32600)  │
                                       │ 4 authorize      (-32001 / -32602 / │
                                       │                   -32601)           │
                                       │ 5 forward body unchanged, stream    │
                                       │   the response back; audit log      │
                                       └─────────────────────────────────────┘
```

## Run

Start each server in its own terminal, from the repository root:

```bash
pip install -e ".[dev]"
python -m task2_mcp_gateway.mock_mcp_server   # terminal 1: downstream on :9100
python -m task2_mcp_gateway.gateway           # terminal 2: gateway on :8080

# terminal 3
TOKEN=$(python -m task2_mcp_gateway.auth --role viewer --subject bob)
curl -s localhost:8080/mcp -H "authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"admin_reset_key","arguments":{"tenant_id":"t-1"}}}'
# {"jsonrpc":"2.0","id":4,"error":{"code":-32001,"message":"Unauthorized Tool Call",
#  "data":{"tool":"admin_reset_key","required_role":"admin"}}}
```

`make demo-t2` (or `bash scripts/demo_task2.sh`) starts both servers in the background, runs the main scenarios (no token, viewer, admin, malformed JSON) with curl and prints the audit log.

Configuration (environment variables):

| Variable | Default | Purpose |
| --- | --- | --- |
| `MCP_DOWNSTREAM_URL` | `http://127.0.0.1:9100/mcp` | Downstream MCP endpoint |
| `MCP_GATEWAY_JWT_SECRET` | development value | HS256 signing key |
| `MCP_GATEWAY_JWT_AUDIENCE` / `_ISSUER` | `mcp-gateway` / `auth.example.local` | Required `aud` / `iss` claims |
| `MCP_UPSTREAM_TIMEOUT_SECONDS` | `30` | Downstream read timeout |
| `MCP_MAX_BODY_BYTES` | `1048576` | Request size limit |
| `MCP_FILTER_TOOLS_LIST` | `false` | Hide `admin_*` tools from non-admins in `tools/list` |

## Design

### Authentication: the role comes from a signed token

The Bearer token is a JWT. The gateway pins the algorithm to HS256 (which blocks `alg: none` and algorithm-confusion attacks), verifies the signature, and requires `exp`, `sub`, `role`, the expected audience and the expected issuer. The role must be one of `admin` or `viewer`. A role is never read from an unsigned header or the request body.

Authentication happens before the body is parsed, so unauthenticated callers learn nothing about the parser and cannot use it as an attack surface. Failures return HTTP 401 with `WWW-Authenticate: Bearer`.

### Authentication vs authorization responses

| Failure | Transport | Why |
| --- | --- | --- |
| No/invalid/expired token | HTTP 401 | The HTTP request itself is unauthenticated. |
| Valid token, forbidden tool | HTTP 200 + JSON-RPC `-32001` echoing the request `id` | The JSON-RPC request was understood and refused; the client's RPC layer can correlate it. |

### JSON-RPC parsing

The envelope is validated strictly against JSON-RPC 2.0, and every failure gets the standard code with the request `id` echoed whenever it can be recovered.

| Input | Response |
| --- | --- |
| Invalid JSON or invalid UTF-8 | `-32700 Parse error`, `id: null` |
| JSON array (batch) | `-32600` — MCP removed JSON-RPC batching in protocol revision 2025-06-18 |
| `jsonrpc` ≠ `"2.0"`, missing/non-string `method`, `params` that is not an object or array, `id` that is not a string/integer (booleans included), a response object | `-32600 Invalid Request` |
| Method outside the MCP surface | `-32601 Method not found` (not forwarded) |
| `tools/call` with missing or invalid `params.name` | `-32602 Invalid params` |

### Authorization policy

- `tools/list`, `initialize`, `ping`, notifications and the other standard MCP methods are forwarded transparently.
- `tools/call` with a name starting with `admin_` requires the `admin` role. The prefix check is **case-insensitive**, so `Admin_reset_key` cannot slip through to a downstream that resolves names without regard to case.
- Tool names must match the MCP naming rule (`[A-Za-z0-9_.-]{1,128}`). This rejects look-alike characters (Cyrillic `а`), zero-width spaces and padding that could be normalised differently downstream.
- Anything not explicitly allowed is denied (deny by default).

The decision is made on the parsed message only; the original bytes are forwarded unchanged, so the downstream sees exactly what was authorized.

### Proxying

- One shared `httpx.AsyncClient` with a bounded connection pool, created in the app lifespan.
- Request body forwarded byte-for-byte. Only MCP-relevant headers are passed on (`content-type`, `accept`, `mcp-session-id`, `mcp-protocol-version`, `last-event-id`). The caller's `Authorization` header is **not** forwarded.
- `x-request-id`, `x-authenticated-subject` and `x-authenticated-role` are added for downstream auditing.
- The downstream response is streamed back as it arrives, so Streamable HTTP SSE responses work without buffering. Any compression applied by the downstream is decoded first, so the client always receives plain bytes.
- Bodies larger than `MCP_MAX_BODY_BYTES` are rejected with 413. The declared `Content-Length` is checked first and the stream is counted as it arrives, so an oversized body is never held in memory.
- Downstream unreachable → HTTP 502, timeout → HTTP 504, both with a JSON-RPC `-32603` body carrying the original `id` and the request id; internal error text never leaks to the caller.
- `GET` and `DELETE` on `/mcp` return 405, which the Streamable HTTP spec allows for servers that do not offer a standalone SSE stream or client-initiated session termination.

### Audit log

One structured JSON line per request on the `mcp_gateway.audit` logger: request id, subject, role, outcome (`forwarded` / `denied` / `rejected` / `unauthenticated`), method, tool, HTTP status and latency.

### Optional: tool filtering

The assignment requires `tools/list` to be forwarded transparently, and that is the default. Setting `MCP_FILTER_TOOLS_LIST=true` additionally removes `admin_*` tools from the list returned to non-admins (JSON and SSE responses). Hiding tools a caller cannot use keeps them out of the model's context and reduces failed calls; enforcement still happens on `tools/call` regardless.

## Tests

```bash
pytest task2_mcp_gateway
```

The suite runs the gateway and the mock downstream in-process over ASGI transports. It covers forged, unsigned (`alg: none`), expired, wrong-audience and wrong-issuer tokens; transparent forwarding (byte-for-byte body, headers, session id, SSE); the `-32001` path with proof that the downstream received **zero** requests; case and Unicode bypass attempts; every JSON-RPC parsing rule; oversized and compressed bodies; downstream outages and timeouts; and the optional filter.

## Production hardening (next steps)

- Asymmetric tokens validated against the identity provider's JWKS (OIDC), with key rotation.
- mTLS between gateway and downstream so the identity headers can be trusted.
- Policy loaded from configuration or an engine such as OPA, with per-tool allow lists per role.
- Per-principal rate limits and argument-level checks for destructive tools.
