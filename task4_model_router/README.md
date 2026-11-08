# Task 4 — Token Rate Limiter & Model Fallback Router

An OpenAI-compatible `POST /v1/chat/completions` gateway that:

1. authenticates the tenant's API key,
2. admits the request through a **token-aware sliding-window limiter** (default 50,000 tokens/minute per tenant) stored in **on-disk SQLite**,
3. calls the **primary** model with a **3,000 ms deadline** and fails over to the **secondary** on `429`, timeout, 5xx or connection failure,
4. corrects the reservation to the tokens the provider actually used,
5. returns every failure in **one standardized error shape** with no upstream details.

```
request ─▶ tenant key ─▶ validate ─▶ estimate tokens ─▶ reserve (SQLite, BEGIN IMMEDIATE)
                                                          │ over limit → 429 + Retry-After
                                                          ▼
                       ┌── circuit closed ── primary (asyncio.timeout 3.0 s) ── ok ──▶ reconcile ─▶ 200
                       │                         │ 429 / timeout / 5xx / connect error
                       │                         ▼
                       └── circuit open ───▶ secondary ── ok ──▶ reconcile ─▶ 200 (X-Gateway-Route: fallback)
                                                 │ fails
                                                 ▼
                                    release reservation ─▶ 503 upstream_unavailable
```

## Run

Run from the repository root (the database path is relative to it):

```bash
pip install -e ".[dev]"
python -m task4_model_router.mock_providers --name primary --port 9301 &
python -m task4_model_router.mock_providers --name secondary --port 9302 &
python -m task4_model_router.app &                                   # :8082, DB at data/llm_gateway.db

KEY=$(python -m task4_model_router.admin create-tenant acme)          # prints the API key once
curl -s localhost:8082/v1/chat/completions -H "authorization: Bearer $KEY" -H 'content-type: application/json' \
  -d '{"model":"gpt-demo","max_tokens":400,"messages":[{"role":"user","content":"Hello"}]}'

curl -s -X POST localhost:9301/control -d '{"mode":"slow"}' -H 'content-type: application/json'   # ok | slow | rate_limited | error
```

`make demo-t4` (or `bash scripts/demo_task4.sh`) starts everything with a temporary database and runs every scenario end to end with curl. Results from a real run:

| Scenario | Result |
| --- | --- |
| Healthy primary | 200 in 9 ms, `X-Gateway-Route: primary` |
| Tenant at 1,500 tok/min, second ~1,000-token request | 429, `Retry-After: 60`, `X-RateLimit-Remaining-Tokens: 990` |
| Primary returns 429 | 200 in 7 ms from the secondary, `X-Gateway-Route: fallback` |
| Primary hangs for 5 s | 200 in **3.01 s** from the secondary |
| Both providers return 500 with a stack trace | 503 `upstream_unavailable`, no trace, hostnames or provider names |
| Primary recovered, circuit still open | 200 in 5 ms from the secondary (no 3 s wait) |
| Unknown API key | 401 `authentication_error` |

Configuration: `ROUTER_DB_PATH`, `ROUTER_TOKENS_PER_MINUTE` (limit given to tenants created without `--tokens-per-minute`; each tenant's limit is stored with it), `PRIMARY_MODEL_URL`, `PRIMARY_MODEL_API_KEY`, `PRIMARY_TIMEOUT_SECONDS` (default 3.0), `SECONDARY_MODEL_URL`, `SECONDARY_MODEL_API_KEY`, `SECONDARY_TIMEOUT_SECONDS` (default 10).

## Rate limiter

### Algorithm: sliding log aggregated per second

Usage is stored as one row per tenant per second (`token_usage(tenant_id, bucket, tokens)`, primary key on both, `WITHOUT ROWID`). A request is admitted when the tokens in the last 60 one-second buckets plus the tokens it needs fit under the tenant's limit.

| Option | Trade-off |
| --- | --- |
| Fixed window | Allows 2× the limit across a minute boundary (tested: `test_no_double_spend_across_a_minute_boundary`). |
| Two-bucket sliding-window approximation | Assumes uniform traffic in the previous window; can over- or under-admit by a large margin on bursty LLM traffic. |
| Per-request log | Exact, but rows grow with traffic. |
| **Per-second buckets (chosen)** | Exact to one second, and storage is capped at 60 rows per tenant however busy it is. |

### Atomicity and eviction

Admission is one `BEGIN IMMEDIATE` transaction:

```
DELETE rows older than the window   -- eviction happens on every check, so state never goes stale
SELECT the tenant's remaining rows  -- the window
compare used + requested with the limit
INSERT … ON CONFLICT DO UPDATE      -- record into the current second
COMMIT
```

`BEGIN IMMEDIATE` takes SQLite's write lock before reading, so check-and-record is atomic across threads **and processes**. Several gateway workers can share one database file without ever spending the same tokens twice. Inside a process, one lock-guarded connection is shared by the tenant store and the limiter, and calls run through `asyncio.to_thread` so the event loop never blocks on disk. The database runs in WAL mode with `busy_timeout`, so readers never block the writer and writers queue instead of failing.

### Token accounting

- **Estimate before the call:** prompt tokens (about 4 characters per token plus per-message overhead) + `max_tokens` (512 if not given). Because the whole completion budget is reserved up front, a burst of concurrent requests cannot push a tenant past its limit while the calls are still running.
- **Reconcile after the call** to `usage.total_tokens` reported by whichever provider answered.
- **Release** the reservation if no provider answered, the provider rejected the request, or the request was cancelled.
- A single request larger than the tenant's whole per-minute budget gets a clear 400 `request_too_large` rather than a 429 that could never succeed.

### Responses

`429` carries `Retry-After` computed from the window itself: the number of seconds until enough of the oldest buckets expire for this request to fit. Every success carries `X-RateLimit-Limit-Tokens` and `X-RateLimit-Remaining-Tokens`.

API keys are stored only as SHA-256 hashes.

## Failover

- **Hard deadline:** the primary call runs inside `asyncio.timeout(3.0)`. This bounds the whole exchange (connect, send, wait, read). httpx's own timeouts apply per network operation, so on their own a provider that trickles bytes could take far longer than 3 s.
- **Race safety:** when the deadline fires, the in-flight request is cancelled and its connection closed. It can never deliver a late second answer or be billed twice. A test asserts the primary's handler actually observed the cancellation.
- **What fails over:** `429`, deadline exceeded, network timeout, connection error, `5xx`, invalid JSON. A `4xx` other than 429 means the request itself is wrong, so it is **not** retried on the secondary (that would double cost and fail the same way).
- **Circuit breaker:** after 3 consecutive primary failures, the primary is skipped for 30 s, or for the provider's `Retry-After` if that is longer. Without this, every request during an outage would wait the full 3 s first. After the cooldown, one half-open probe is allowed; success closes the circuit. A probe cancelled by a client disconnect does not wedge the breaker.

## Error sanitization

Every error, from validation failures to unhandled exceptions to unknown routes, has exactly this shape, with `x-request-id` echoed as a header:

```json
{ "error": { "type": "upstream_unavailable", "message": "All model providers are currently unavailable. Please retry shortly.", "request_id": "req_5b3f38b88c60468a96db" } }
```

| Type | HTTP | When |
| --- | --- | --- |
| `invalid_request` | 400 | Body fails validation, `stream: true`, provider rejected the request |
| `request_too_large` | 400 | Estimate exceeds the tenant's whole per-minute budget |
| `authentication_error` | 401 | Missing, malformed or unknown API key (checked before the body is parsed) |
| `not_found` / `method_not_allowed` | 404 / 405 | Unknown route or method |
| `rate_limit_exceeded` | 429 | Sliding window full; includes `Retry-After` |
| `internal_error` | 500 | Any unexpected exception |
| `upstream_unavailable` | 503 | Primary and secondary both failed |

Provider bodies, exception classes, stack traces, hostnames and file paths are logged server-side under the `request_id` and never returned. The tests feed the gateway a provider error containing a CUDA traceback, an internal IP and a host name, and assert none of it reaches the client.

## Tests

```bash
pytest task4_model_router
```

- `test_rate_limiter.py` — admission at the limit, sliding (not resetting) window, no boundary double-spend, exact `Retry-After`, on-disk eviction bounded to 60 rows, tenant isolation, reconcile/release, state surviving a restart, **100 threads → exactly 50 admitted**, **4 processes sharing the file → exactly 50 admitted**, hashed keys.
- `test_router.py` — failover on 429 / slow / 5xx / connection error, real 3.0 s deadline, cancellation of the abandoned call, no failover on 4xx, attempt history, circuit breaker open / skip / half-open / recover / `Retry-After`.
- `test_app.py` — end to end over ASGI: headers, reconciliation, fallback, sanitized 503 / 500, 429 then recovery after the window slides, **100 concurrent HTTP requests → exactly 50 succeed**, auth before validation, clear messages for invalid JSON and missing fields, standardized 404 errors.
- `test_admin.py` — the tenant CLI prints a working key, and duplicate tenants or non-positive limits fail with a clear message instead of a traceback.

## Production notes

- **Several gateway nodes:** move the same algorithm to Redis (a Lua script performing evict-sum-compare-record atomically on a sorted set or hash per tenant). The SQLite version is right for a single node or several workers sharing a disk.
- **Streaming:** apply the 3 s deadline to the first byte, and only fail over before any byte has been sent to the client.
- **Tokenizer:** replace the heuristic with the provider's tokenizer (for example `tiktoken`) behind `tokens.estimate_request_tokens`.
- **Routing policy:** per-model limits, weighted routing, cost-aware selection and hedged requests for latency-critical tenants.
